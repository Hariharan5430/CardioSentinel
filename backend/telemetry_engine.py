"""
CardioSentinel - Multi-Bed Telemetry Engine & Playback Controller
Provides authoritative session state for each bed (BED-01, BED-02, BED-03, REMOTE-01),
supporting synchronized 250 Hz multi-lead streaming from real MIT-BIH recordings or
synthetic simulations, interactive playback controls (seek, pause, resume, speed 0.5x-10x, event jump).
"""

import math
import random
import time
import numpy as np
from typing import Dict, List, Any, Optional

class CardiacRhythmType:
    NSR = "Normal Sinus Rhythm"
    BRADYCARDIA = "Sinus Bradycardia"
    TACHYCARDIA = "Sinus Tachycardia"
    AFIB = "Atrial Fibrillation"
    PVC = "Premature Ventricular Contractions"
    VT = "Ventricular Tachycardia"
    VFIB = "Ventricular Fibrillation"
    ASYSTOLE = "Asystole / Cardiac Arrest"
    ARTIFACT = "Lead Disconnect / Motion Artifact"

class PatientTelemetryState:
    """Authoritative session state for an individual hospital bed or remote patient."""

    def __init__(
        self,
        patient_id: str,
        name: str,
        age: int,
        gender: str,
        location: str,
        baseline_rhythm: str,
        default_record_id: Optional[str] = "100"
    ):
        self.patient_id = patient_id
        self.name = name
        self.is_demonstration_patient = True
        self.age = age
        self.gender = gender
        self.location = location
        self.baseline_rhythm = baseline_rhythm
        self.current_rhythm = baseline_rhythm

        # Playback control state
        self.data_mode = "MIT-BIH" # "MIT-BIH" or "SYNTHETIC"
        self.playback_status = "PLAYING" # "PLAYING" or "PAUSED"
        self.playback_speed = 1.0 # 0.5, 1.0, 2.0, 5.0, 10.0
        self.time_position_sec = 0.0
        self.mitbih_record_id = default_record_id
        self.sample_rate = 250
        self.sample_index = 0

        # Generator parameters
        self.phase = 0.0
        self.time_elapsed = 0.0
        self.current_hr = 72.0
        self.target_hr = 72.0
        self.spo2 = 98.0
        self.bp_sys = 120.0
        self.bp_dia = 80.0
        self.resp_rate = 16.0
        self.temp_c = 36.8
        self.perfusion_index = 4.2

        self.pvc_counter = 0
        self.pvc_interval = 8
        self.override_duration = 0.0

        # Signal buffer for real-time DSP analysis (~1500 samples = 6s)
        self.lead2_buffer: List[float] = []
        self.lead_v1_buffer: List[float] = []
        self.buffer_max_len = 1500

        # Cached record pointer
        self._cached_record: Optional[Dict[str, Any]] = None
        self._cached_l1_resampled: Optional[np.ndarray] = None
        self._cached_l2_resampled: Optional[np.ndarray] = None
        self._cached_annotations: List[Dict[str, Any]] = []

        self.sync_vitals_to_rhythm()
        if self.mitbih_record_id:
            self._load_mitbih_record(self.mitbih_record_id)

    def _load_mitbih_record(self, record_id: str):
        """Loads and pre-resamples record to 250 Hz for zero-latency streaming."""
        try:
            from backend.dataset_manager import dataset_manager
            rec = dataset_manager.load_record(record_id)
            self._cached_record = rec
            native_fs = rec.get("fs", 360)

            # Resample full buffers to 250 Hz
            if native_fs == 360:
                self._cached_l1_resampled = dataset_manager.resample_to_250hz(rec["lead1_raw"])
                self._cached_l2_resampled = dataset_manager.resample_to_250hz(rec["lead2_raw"])
            else:
                self._cached_l1_resampled = rec["lead1_raw"]
                self._cached_l2_resampled = rec["lead2_raw"]

            # Store annotations with 250Hz aligned sample indices
            self._cached_annotations = rec.get("annotations", [])
            self.sample_index = 0
            self.time_position_sec = 0.0
        except Exception as e:
            print(f"[!] Warning loading MIT-BIH record {record_id} for {self.patient_id}: {e}")
            self.data_mode = "SYNTHETIC"

    def set_data_mode(self, mode: str, record_id: Optional[str] = None):
        """Switches bed data source between MIT-BIH recorded session and synthetic simulation."""
        self.data_mode = mode.upper()
        if record_id:
            self.mitbih_record_id = str(record_id)
            self._load_mitbih_record(self.mitbih_record_id)

    def set_playback_status(self, status: str):
        """Controls playback (PLAYING / PAUSED)."""
        if status in ["PLAYING", "PAUSED"]:
            self.playback_status = status

    def set_playback_speed(self, speed: float):
        """Sets playback speed multiplier (0.5, 1.0, 2.0, 5.0, 10.0)."""
        if speed in [0.5, 1.0, 2.0, 5.0, 10.0]:
            self.playback_speed = float(speed)

    def seek_to_time(self, seconds: float):
        """Seeks playback to specified time offset in seconds."""
        if self._cached_l1_resampled is not None and len(self._cached_l1_resampled) > 0:
            total_sec = len(self._cached_l1_resampled) / float(self.sample_rate)
            clamped_sec = max(0.0, min(total_sec - 1.0, seconds))
            self.time_position_sec = clamped_sec
            self.sample_index = int(clamped_sec * self.sample_rate)
        else:
            self.time_position_sec = max(0.0, seconds)
            self.sample_index = int(self.time_position_sec * self.sample_rate)

    def jump_to_event(self, direction: str = "next") -> Optional[Dict[str, Any]]:
        """Jumps playhead to next or previous annotated cardiac event."""
        if not self._cached_annotations:
            return None

        current_t = self.time_position_sec
        if direction == "next":
            candidates = [a for a in self._cached_annotations if a["time_sec"] > (current_t + 0.5)]
            target = candidates[0] if candidates else self._cached_annotations[0]
        else:
            candidates = [a for a in self._cached_annotations if a["time_sec"] < (current_t - 0.5)]
            target = candidates[-1] if candidates else self._cached_annotations[-1]

        target_time = max(0.0, target["time_sec"] - 1.0) # 1 sec before event for lead-in context
        self.seek_to_time(target_time)
        return target

    def sync_vitals_to_rhythm(self):
        r = self.current_rhythm
        if r == CardiacRhythmType.NSR:
            self.target_hr = 72.0
            self.spo2 = 98.0
            self.bp_sys = 120.0
            self.bp_dia = 80.0
            self.resp_rate = 15.0
        elif r == CardiacRhythmType.BRADYCARDIA:
            self.target_hr = 38.0
            self.spo2 = 93.0
            self.bp_sys = 96.0
            self.bp_dia = 62.0
            self.resp_rate = 11.0
        elif r == CardiacRhythmType.TACHYCARDIA:
            self.target_hr = 152.0
            self.spo2 = 95.0
            self.bp_sys = 142.0
            self.bp_dia = 90.0
            self.resp_rate = 24.0
        elif r == CardiacRhythmType.AFIB:
            self.target_hr = 126.0
            self.spo2 = 93.0
            self.bp_sys = 132.0
            self.bp_dia = 86.0
            self.resp_rate = 20.0
        elif r == CardiacRhythmType.PVC:
            self.target_hr = 78.0
            self.spo2 = 96.0
            self.bp_sys = 122.0
            self.bp_dia = 80.0
            self.resp_rate = 16.0
        elif r == CardiacRhythmType.VT:
            self.target_hr = 188.0
            self.spo2 = 86.0
            self.bp_sys = 74.0
            self.bp_dia = 48.0
            self.resp_rate = 28.0
        elif r == CardiacRhythmType.VFIB:
            self.target_hr = 260.0
            self.spo2 = 70.0
            self.bp_sys = 40.0
            self.bp_dia = 24.0
            self.resp_rate = 6.0
        elif r == CardiacRhythmType.ASYSTOLE:
            self.target_hr = 0.0
            self.spo2 = 50.0
            self.bp_sys = 20.0
            self.bp_dia = 10.0
            self.resp_rate = 0.0

    def set_rhythm(self, rhythm: str, duration: float = 0.0):
        self.current_rhythm = rhythm
        self.override_duration = duration
        self.sync_vitals_to_rhythm()

    def generate_next_sample(self) -> Dict[str, Any]:
        """Generates or replays the next synchronized 250 Hz sample point."""
        dt = 1.0 / self.sample_rate

        # If paused, hold current waveform point
        if self.playback_status == "PAUSED":
            last_l2 = self.lead2_buffer[-1] if self.lead2_buffer else 0.0
            last_v1 = self.lead_v1_buffer[-1] if self.lead_v1_buffer else 0.0
            return {
                "lead2": round(last_l2, 4),
                "lead_v1": round(last_v1, 4),
                "is_r_peak": False,
                "patient_id": self.patient_id,
                "rhythm": self.current_rhythm,
                "data_mode": self.data_mode,
                "record_id": self.mitbih_record_id,
                "time_sec": round(self.time_position_sec, 2),
                "hr": round(self.current_hr, 1),
                "spo2": round(self.spo2, 1),
                "bp_sys": round(self.bp_sys),
                "bp_dia": round(self.bp_dia),
                "resp_rate": round(self.resp_rate, 1),
                "temp_c": round(self.temp_c, 1),
                "perfusion_index": round(self.perfusion_index, 2),
                "ref_annotation": None
            }

        # Advance playhead taking playback speed into account
        self.time_elapsed += dt * self.playback_speed
        self.time_position_sec += dt * self.playback_speed

        # Handle transient acute simulation timers
        if self.override_duration > 0:
            self.override_duration -= dt
            if self.override_duration <= 0:
                self.current_rhythm = self.baseline_rhythm
                self.sync_vitals_to_rhythm()

        # Smooth HR convergence
        self.current_hr += (self.target_hr - self.current_hr) * 0.02

        # 1. Real MIT-BIH recorded replay mode
        if self.data_mode == "MIT-BIH" and self._cached_l1_resampled is not None:
            l1_arr = self._cached_l1_resampled
            l2_arr = self._cached_l2_resampled

            # Wrap around when reaching recording end
            idx = int(self.sample_index) % len(l1_arr)
            lead2_val = float(l1_arr[idx])
            lead_v1_val = float(l2_arr[idx]) if l2_arr is not None else lead2_val

            # Advance sample index by speed factor
            self.sample_index = (self.sample_index + self.playback_speed) % len(l1_arr)

            # Check for aligned reference annotation
            current_t = self.time_position_sec
            active_ann = None
            is_r_peak = False
            for a in self._cached_annotations:
                if abs(a["time_sec"] - current_t) < 0.04: # within 40ms window
                    active_ann = a
                    is_r_peak = True
                    break

            # Buffer updates
            self.lead2_buffer.append(lead2_val)
            self.lead_v1_buffer.append(lead_v1_val)
            if len(self.lead2_buffer) > self.buffer_max_len:
                self.lead2_buffer.pop(0)
                self.lead_v1_buffer.pop(0)

            return {
                "lead2": round(lead2_val, 4),
                "lead_v1": round(lead_v1_val, 4),
                "is_r_peak": is_r_peak,
                "patient_id": self.patient_id,
                "rhythm": self.current_rhythm,
                "data_mode": "MIT-BIH",
                "record_id": self.mitbih_record_id,
                "time_sec": round(self.time_position_sec, 2),
                "hr": round(self.current_hr, 1),
                "spo2": round(self.spo2 + random.uniform(-0.1, 0.1), 1),
                "bp_sys": round(self.bp_sys + random.uniform(-0.5, 0.5)),
                "bp_dia": round(self.bp_dia + random.uniform(-0.4, 0.4)),
                "resp_rate": round(self.resp_rate, 1),
                "temp_c": round(self.temp_c, 1),
                "perfusion_index": round(self.perfusion_index, 2),
                "ref_annotation": active_ann.get("symbol") if active_ann else None
            }

        # 2. Synthetic ECG generator mode
        self.sample_index += int(self.playback_speed)
        t = self.time_elapsed
        rhythm = self.current_rhythm
        is_r_peak = False

        resp_wander = 0.05 * math.sin(2 * math.pi * 0.25 * t)
        emg_noise = random.gauss(0, 0.012)
        powerline_hum = 0.008 * math.sin(2 * math.pi * 50.0 * t)

        if rhythm == CardiacRhythmType.ASYSTOLE:
            lead2 = random.gauss(0, 0.02) + resp_wander * 0.3
            lead_v1 = lead2 * 0.8
        elif rhythm == CardiacRhythmType.VFIB:
            f1 = 0.42 * math.sin(2 * math.pi * 4.8 * t)
            f2 = 0.35 * math.sin(2 * math.pi * 7.2 * t + 1.2)
            f3 = 0.28 * math.sin(2 * math.pi * 9.5 * t + 2.5)
            lead2 = f1 + f2 + f3 + emg_noise * 2.5
            lead_v1 = lead2 * 0.85
        elif rhythm == CardiacRhythmType.VT:
            vt_hr = 188.0
            self.phase += (vt_hr / 60.0) * dt * self.playback_speed
            p = self.phase % 1.0
            val = 1.7 * math.sin(p * math.pi * 2) ** 3
            if 0.20 <= p <= 0.24:
                is_r_peak = True
            lead2 = val + emg_noise + powerline_hum
            lead_v1 = -val * 0.8 + emg_noise
        elif rhythm == CardiacRhythmType.AFIB:
            jitter = 1.0 + 0.35 * math.sin(3.7 * t) + 0.2 * math.cos(5.9 * t)
            afib_hr = max(60.0, self.current_hr * jitter)
            self.phase += (afib_hr / 60.0) * dt * self.playback_speed
            p = self.phase % 1.0
            f_wave = 0.08 * (math.sin(2 * math.pi * 7.0 * t) + math.cos(2 * math.pi * 9.5 * t))
            val = f_wave
            if 0.22 <= p < 0.28:
                val += 1.25 * math.sin((p - 0.22) / 0.06 * math.pi)
                if 0.24 <= p <= 0.26:
                    is_r_peak = True
            elif 0.45 <= p < 0.65:
                val += 0.28 * math.sin((p - 0.45) / 0.20 * math.pi)
            lead2 = val + resp_wander + emg_noise
            lead_v1 = val * 0.75 + emg_noise
        else:
            # NSR / PVC
            hr = max(30.0, self.current_hr)
            self.phase += (hr / 60.0) * dt * self.playback_speed
            is_pvc = False
            if rhythm == CardiacRhythmType.PVC:
                if int(self.phase) > self.pvc_counter:
                    self.pvc_counter = int(self.phase)
                    if self.pvc_counter % self.pvc_interval == 0:
                        is_pvc = True

            p = self.phase % 1.0
            if is_pvc:
                val = 1.6 * math.sin(p / 0.32 * math.pi) ** 1.2 if p < 0.32 else -0.65 * math.sin((p - 0.32) / 0.36 * math.pi) if p < 0.68 else 0.0
                if 0.14 <= p <= 0.18:
                    is_r_peak = True
                lead2 = val + emg_noise
                lead_v1 = -val * 0.85 + emg_noise
            else:
                val = 0.0
                if 0.08 <= p < 0.18:
                    val += 0.18 * math.sin((p - 0.08) / 0.10 * math.pi)
                elif 0.25 <= p < 0.30:
                    val += 1.35 * math.sin((p - 0.25) / 0.05 * math.pi)
                    if 0.265 <= p <= 0.285:
                        is_r_peak = True
                elif 0.48 <= p < 0.68:
                    val += 0.32 * math.sin((p - 0.48) / 0.20 * math.pi)
                lead2 = val + resp_wander + emg_noise
                lead_v1 = (val * 0.6 - 0.15 * math.sin(p * math.pi * 2)) + emg_noise

        self.lead2_buffer.append(lead2)
        self.lead_v1_buffer.append(lead_v1)
        if len(self.lead2_buffer) > self.buffer_max_len:
            self.lead2_buffer.pop(0)
            self.lead_v1_buffer.pop(0)

        return {
            "lead2": round(lead2, 4),
            "lead_v1": round(lead_v1, 4),
            "is_r_peak": is_r_peak,
            "patient_id": self.patient_id,
            "rhythm": self.current_rhythm,
            "data_mode": "SYNTHETIC",
            "record_id": None,
            "time_sec": round(self.time_position_sec, 2),
            "hr": round(self.current_hr, 1),
            "spo2": round(self.spo2, 1),
            "bp_sys": round(self.bp_sys),
            "bp_dia": round(self.bp_dia),
            "resp_rate": round(self.resp_rate, 1),
            "temp_c": round(self.temp_c, 1),
            "perfusion_index": round(self.perfusion_index, 2),
            "ref_annotation": None
        }

class TelemetryEngine:
    """Manages multi-bed cardiac telemetry and synchronization."""

    def __init__(self):
        self.patients: Dict[str, PatientTelemetryState] = {}
        self.setup_default_beds()

    def setup_default_beds(self):
        # Bed 1: Eleanor Vance (Post-CABG ICU) - Record 100 (Normal Sinus Rhythm)
        self.patients["BED-01"] = PatientTelemetryState(
            patient_id="BED-01",
            name="Eleanor Vance",
            age=68,
            gender="Female",
            location="Cardiac ICU - Bed 01",
            baseline_rhythm=CardiacRhythmType.NSR,
            default_record_id="100"
        )

        # Bed 2: Marcus Brody (Cardiomyopathy ICU) - Record 200 (VT & PVCs)
        self.patients["BED-02"] = PatientTelemetryState(
            patient_id="BED-02",
            name="Marcus Brody",
            age=72,
            gender="Male",
            location="Cardiac ICU - Bed 02",
            baseline_rhythm=CardiacRhythmType.VT,
            default_record_id="200"
        )

        # Bed 3: Sarah Lin (Telemetry Stepdown) - Record 209 (Atrial Fibrillation)
        self.patients["BED-03"] = PatientTelemetryState(
            patient_id="BED-03",
            name="Sarah Lin",
            age=45,
            gender="Female",
            location="Telemetry Stepdown - Bed 03",
            baseline_rhythm=CardiacRhythmType.AFIB,
            default_record_id="209"
        )

        # Remote 1: Arthur Pendelton (Ambulatory Wearable) - Record 207 (Ventricular Flutter/Fibrillation)
        self.patients["REMOTE-01"] = PatientTelemetryState(
            patient_id="REMOTE-01",
            name="Arthur Pendelton",
            age=59,
            gender="Male",
            location="Remote Home Care (Wearable Patch)",
            baseline_rhythm=CardiacRhythmType.VFIB,
            default_record_id="207"
        )

    def get_patient(self, patient_id: str) -> Optional[PatientTelemetryState]:
        return self.patients.get(patient_id)

    def switch_bed_record(self, patient_id: str, record_id: str) -> bool:
        """Assigns an official MIT-BIH recorded session to a bed."""
        p = self.patients.get(patient_id)
        if p:
            p.set_data_mode("MIT-BIH", record_id)
            return True
        return False

    def trigger_event(self, patient_id: str, rhythm: str, duration: float = 30.0) -> bool:
        p = self.patients.get(patient_id)
        if p:
            p.set_rhythm(rhythm, duration)
            return True
        return False

    def generate_batch_for_all(self, num_samples: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        output = {}
        for pid, patient in self.patients.items():
            output[pid] = [patient.generate_next_sample() for _ in range(num_samples)]
        return output

# Global singleton
telemetry_engine = TelemetryEngine()
