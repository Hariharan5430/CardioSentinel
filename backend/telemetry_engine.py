"""
CardioSentinel - Telemetry & Cardiac Rhythm Generator Engine
Generates high-fidelity, medically realistic multi-lead ECG signals (Lead II and Lead V1)
at 250 Hz for multiple hospital beds and remote patients.
Simulates Normal Sinus Rhythm, Atrial Fibrillation, Ventricular Tachycardia,
Ventricular Fibrillation, Premature Ventricular Contractions, Bradycardia, Tachycardia, and Asystole.
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
    def __init__(self, patient_id: str, name: str, age: int, gender: str, location: str, baseline_rhythm: str):
        self.patient_id = patient_id
        self.name = name
        self.age = age
        self.gender = gender
        self.location = location
        self.baseline_rhythm = baseline_rhythm
        self.current_rhythm = baseline_rhythm
        
        # Internal generation state
        self.phase = 0.0
        self.time_elapsed = 0.0
        self.sample_rate = 250
        
        # Clinical vitals
        self.target_hr = 72.0
        self.current_hr = 72.0
        self.spo2 = 98.0
        self.bp_sys = 120.0
        self.bp_dia = 80.0
        self.resp_rate = 16.0
        self.temp_c = 36.8
        self.perfusion_index = 4.2
        
        # Arrhythmia parameters
        self.pvc_counter = 0
        self.pvc_interval = 8 # PVC every N beats if PVC rhythm
        self.override_duration = 0.0 # timer for transient simulated acute events
        
        # Historical buffer for signal processing (holds last 1500 samples ~ 6 seconds)
        self.lead2_buffer = []
        self.lead_v1_buffer = []
        self.buffer_max_len = 1500
        
        # Real PhysioNet MIT-BIH Dataset playback support
        self.data_mode = "SYNTHETIC"  # "SYNTHETIC" or "MIT-BIH"
        self.mitbih_record_id: Optional[str] = None
        self.mitbih_sample_index = 0

        # Initialize default vitals per rhythm
        self.sync_vitals_to_rhythm()

    def set_data_mode(self, mode: str, record_id: Optional[str] = None):
        """Switches between synthetic ECG generation and real PhysioNet MIT-BIH playback."""
        self.data_mode = mode.upper()
        if record_id:
            self.mitbih_record_id = str(record_id)
            self.mitbih_sample_index = 0

    def sync_vitals_to_rhythm(self):
        r = self.current_rhythm
        if r == CardiacRhythmType.NSR:
            self.target_hr = 72.0 + random.uniform(-3, 3)
            self.spo2 = 98.0
            self.bp_sys = 122.0
            self.bp_dia = 78.0
            self.resp_rate = 15.0
        elif r == CardiacRhythmType.BRADYCARDIA:
            self.target_hr = 38.0 + random.uniform(-2, 2)
            self.spo2 = 94.0
            self.bp_sys = 96.0
            self.bp_dia = 62.0
            self.resp_rate = 11.0
        elif r == CardiacRhythmType.TACHYCARDIA:
            self.target_hr = 155.0 + random.uniform(-5, 5)
            self.spo2 = 95.0
            self.bp_sys = 142.0
            self.bp_dia = 92.0
            self.resp_rate = 24.0
        elif r == CardiacRhythmType.AFIB:
            self.target_hr = 128.0 + random.uniform(-10, 10)
            self.spo2 = 93.0
            self.bp_sys = 134.0
            self.bp_dia = 88.0
            self.resp_rate = 20.0
        elif r == CardiacRhythmType.PVC:
            self.target_hr = 78.0 + random.uniform(-4, 4)
            self.spo2 = 96.0
            self.bp_sys = 124.0
            self.bp_dia = 80.0
            self.resp_rate = 16.0
        elif r == CardiacRhythmType.VT:
            self.target_hr = 188.0 + random.uniform(-4, 4)
            self.spo2 = 86.0
            self.bp_sys = 75.0
            self.bp_dia = 48.0
            self.resp_rate = 28.0
        elif r == CardiacRhythmType.VFIB:
            self.target_hr = 260.0
            self.spo2 = 70.0
            self.bp_sys = 42.0
            self.bp_dia = 25.0
            self.resp_rate = 6.0
        elif r == CardiacRhythmType.ASYSTOLE:
            self.target_hr = 0.0
            self.spo2 = 50.0
            self.bp_sys = 20.0
            self.bp_dia = 10.0
            self.resp_rate = 0.0
        elif r == CardiacRhythmType.ARTIFACT:
            self.target_hr = 80.0

    def set_rhythm(self, rhythm: str, duration: float = 0.0):
        self.current_rhythm = rhythm
        self.override_duration = duration
        self.sync_vitals_to_rhythm()

    def generate_next_sample(self) -> Dict[str, float]:
        """
        Synthesizes the next telemetry sample (at 250 Hz) for Lead II and Lead V1,
        along with physiological noise, respiratory modulation, and current vitals.
        """
        dt = 1.0 / self.sample_rate
        self.time_elapsed += dt
        
        # Check if running in real PhysioNet MIT-BIH stream mode
        if self.data_mode == "MIT-BIH" and self.mitbih_record_id:
            try:
                from backend.dataset_manager import dataset_manager
                stream = dataset_manager.get_signal_stream(self.mitbih_record_id)
                if stream and len(stream.get("lead2", [])) > 0:
                    l2_arr = stream["lead2"]
                    v1_arr = stream["lead_v1"]
                    idx = self.mitbih_sample_index % len(l2_arr)
                    lead2 = float(l2_arr[idx])
                    lead_v1 = float(v1_arr[idx])
                    self.mitbih_sample_index = (idx + 1) % len(l2_arr)

                    # Update buffer
                    self.lead2_buffer.append(lead2)
                    self.lead_v1_buffer.append(lead_v1)
                    if len(self.lead2_buffer) > self.buffer_max_len:
                        self.lead2_buffer.pop(0)
                        self.lead_v1_buffer.pop(0)

                    return {
                        "lead2": round(lead2, 4),
                        "lead_v1": round(lead_v1, 4),
                        "is_r_peak": False,
                        "patient_id": self.patient_id,
                        "rhythm": f"MIT-BIH Record {self.mitbih_record_id}",
                        "data_mode": "MIT-BIH",
                        "hr": round(self.current_hr, 1),
                        "spo2": round(self.spo2 + random.uniform(-0.1, 0.1), 1),
                        "bp_sys": round(self.bp_sys),
                        "bp_dia": round(self.bp_dia),
                        "resp_rate": round(self.resp_rate, 1),
                        "temp_c": round(self.temp_c, 1),
                        "perfusion_index": round(self.perfusion_index, 1)
                    }
            except Exception:
                pass

        # Countdown override duration if acute event was triggered temporarily
        if self.override_duration > 0:
            self.override_duration -= dt
            if self.override_duration <= 0:
                self.current_rhythm = self.baseline_rhythm
                self.sync_vitals_to_rhythm()
                
        # Smoothly track target HR
        self.current_hr += (self.target_hr - self.current_hr) * 0.02
        
        # Baseline wander (respiration 0.25 Hz)
        resp_wander = 0.06 * math.sin(2 * math.pi * 0.25 * self.time_elapsed)
        # Powerline 50Hz hum
        powerline_hum = 0.015 * math.sin(2 * math.pi * 50.0 * self.time_elapsed)
        # Muscle tremor noise
        emg_noise = random.gauss(0, 0.015)
        
        rhythm = self.current_rhythm
        lead2 = 0.0
        lead_v1 = 0.0
        is_r_peak = False
        
        if rhythm == CardiacRhythmType.ASYSTOLE:
            # Flatline with slight sensor noise
            lead2 = random.gauss(0, 0.02) + resp_wander * 0.2
            lead_v1 = random.gauss(0, 0.02) + resp_wander * 0.2
            self.phase = 0.0
            
        elif rhythm == CardiacRhythmType.ARTIFACT:
            # Violent swings & disconnected baseline
            lead2 = 1.8 * math.sin(2 * math.pi * 3.2 * self.time_elapsed) + random.gauss(0, 0.35)
            lead_v1 = 1.5 * math.cos(2 * math.pi * 2.8 * self.time_elapsed) + random.gauss(0, 0.3)
            
        elif rhythm == CardiacRhythmType.VFIB:
            # Ventricular Fibrillation: chaotic sinusoidal oscillations, 4-7 Hz
            f1 = 4.8 + 0.8 * math.sin(0.9 * self.time_elapsed)
            f2 = 6.2 + 0.6 * math.cos(1.2 * self.time_elapsed)
            vfib_amp = 0.75 + 0.3 * math.sin(0.4 * self.time_elapsed)
            val = vfib_amp * (0.6 * math.sin(2 * math.pi * f1 * self.time_elapsed) + 
                              0.4 * math.sin(2 * math.pi * f2 * self.time_elapsed) + 
                              random.gauss(0, 0.12))
            lead2 = val
            lead_v1 = val * 0.85 + random.gauss(0, 0.08)
            
        elif rhythm == CardiacRhythmType.VT:
            # Ventricular Tachycardia: wide monomorphic QRS at 180+ bpm without P wave
            vt_freq = (self.current_hr / 60.0)
            self.phase += vt_freq * dt
            p = self.phase % 1.0
            
            # Broad sine-like inverted/wide monomorphic beat
            # QRS is wide (> 140 ms = > 0.42 fraction of cycle at 180bpm)
            if p < 0.35:
                # Wide R wave peak
                w = math.sin(p / 0.35 * math.pi)
                val = 1.4 * (w ** 1.4)
                if 0.16 <= p <= 0.20:
                    is_r_peak = True
            elif p < 0.65:
                # Wide deep S/T inversion
                w = math.sin((p - 0.35) / 0.30 * math.pi)
                val = -0.55 * w
            else:
                val = 0.0
            lead2 = val + emg_noise * 1.5 + powerline_hum
            lead_v1 = -val * 0.9 + emg_noise * 1.2
            
        elif rhythm == CardiacRhythmType.AFIB:
            # Atrial Fibrillation: Irregularly irregular RR intervals, no P wave, fibrillatory f-waves (6-8Hz)
            # Modulate phase step dynamically to create chaotic RR variability
            jitter = 1.0 + 0.42 * math.sin(3.7 * self.time_elapsed) + 0.25 * math.cos(5.9 * self.time_elapsed)
            afib_hr = max(60.0, self.current_hr * jitter)
            self.phase += (afib_hr / 60.0) * dt
            p = self.phase % 1.0
            
            # High-frequency fibrillatory baseline (f-waves)
            f_wave = 0.09 * (math.sin(2 * math.pi * 7.0 * self.time_elapsed) + 
                             math.cos(2 * math.pi * 9.5 * self.time_elapsed))
            
            # QRS & T wave only (NO distinct P wave)
            val = f_wave
            if 0.18 <= p < 0.22:
                # Q wave
                val += -0.15 * math.sin((p - 0.18) / 0.04 * math.pi)
            elif 0.22 <= p < 0.28:
                # Narrow normal R wave
                val += 1.25 * math.sin((p - 0.22) / 0.06 * math.pi)
                if 0.24 <= p <= 0.26:
                    is_r_peak = True
            elif 0.28 <= p < 0.33:
                # S wave
                val += -0.32 * math.sin((p - 0.28) / 0.05 * math.pi)
            elif 0.45 <= p < 0.65:
                # T wave
                val += 0.28 * math.sin((p - 0.45) / 0.20 * math.pi)
                
            lead2 = val + resp_wander + emg_noise + powerline_hum
            lead_v1 = val * 0.75 + resp_wander * 0.8 + emg_noise
            
        else:
            # Sinus Rhythm (NSR, Bradycardia, Tachycardia, or baseline with occasional PVC)
            hr = max(30.0, self.current_hr)
            self.phase += (hr / 60.0) * dt
            
            # Check if this beat is an ectopic PVC
            is_pvc_beat = False
            if rhythm == CardiacRhythmType.PVC:
                if int(self.phase) > self.pvc_counter:
                    self.pvc_counter = int(self.phase)
                    if self.pvc_counter % self.pvc_interval == 0:
                        is_pvc_beat = True
                        
            p = self.phase % 1.0
            
            if is_pvc_beat:
                # Wide, bizarre, premature QRS complex (>140ms), deep T inversion, no P wave
                if p < 0.32:
                    val = 1.6 * math.sin(p / 0.32 * math.pi) ** 1.2
                    if 0.14 <= p <= 0.18:
                        is_r_peak = True
                elif p < 0.68:
                    val = -0.65 * math.sin((p - 0.32) / 0.36 * math.pi)
                else:
                    val = 0.0
                lead2 = val + emg_noise * 1.5 + powerline_hum
                lead_v1 = -val * 0.85 + emg_noise
            else:
                # Classical P-Q-R-S-T morphology
                val = 0.0
                # P wave (atrial depolarization: 80-100 ms)
                if 0.08 <= p < 0.18:
                    val += 0.18 * math.sin((p - 0.08) / 0.10 * math.pi)
                # PR segment: flat isoelectric
                # Q wave
                elif 0.22 <= p < 0.25:
                    val += -0.12 * math.sin((p - 0.22) / 0.03 * math.pi)
                # R peak (ventricular depolarization: sharp, tall)
                elif 0.25 <= p < 0.30:
                    val += 1.35 * math.sin((p - 0.25) / 0.05 * math.pi)
                    if 0.265 <= p <= 0.285:
                        is_r_peak = True
                # S wave
                elif 0.30 <= p < 0.35:
                    val += -0.28 * math.sin((p - 0.30) / 0.05 * math.pi)
                # ST segment / T wave (ventricular repolarization)
                elif 0.48 <= p < 0.68:
                    val += 0.32 * math.sin((p - 0.48) / 0.20 * math.pi)
                    
                lead2 = val + resp_wander + emg_noise + powerline_hum
                # Lead V1 has biphasic/smaller R and distinct S
                lead_v1 = (val * 0.6 - 0.15 * math.sin(p * math.pi * 2)) + resp_wander * 0.7 + emg_noise
        
        # Buffer appending
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
            "hr": round(self.current_hr, 1),
            "spo2": round(self.spo2 + random.uniform(-0.2, 0.2), 1),
            "bp_sys": round(self.bp_sys + random.uniform(-1, 1)),
            "bp_dia": round(self.bp_dia + random.uniform(-0.8, 0.8)),
            "resp_rate": round(self.resp_rate + random.uniform(-0.2, 0.2), 1),
            "temp_c": round(self.temp_c + random.uniform(-0.02, 0.02), 1),
            "perfusion_index": round(self.perfusion_index + random.uniform(-0.1, 0.1), 2)
        }

class TelemetryEngine:
    """Manages multi-patient cardiac telemetry streaming."""
    def __init__(self):
        self.patients: Dict[str, PatientTelemetryState] = {}
        self.setup_default_patients()
        
    def setup_default_patients(self):
        # Patient 1: Eleanor Vance - Post-CABG ICU Bed 1 (High Early Warning Risk, Frequent PVCs)
        self.patients["BED-01"] = PatientTelemetryState(
            patient_id="BED-01",
            name="Eleanor Vance",
            age=68,
            gender="Female",
            location="Cardiac ICU - Bed 01",
            baseline_rhythm=CardiacRhythmType.PVC
        )
        self.patients["BED-01"].pvc_interval = 6 # frequent PVCs
        
        # Patient 2: Marcus Brody - Dilated Cardiomyopathy ICU Bed 2 (Chronic AFib)
        self.patients["BED-02"] = PatientTelemetryState(
            patient_id="BED-02",
            name="Marcus Brody",
            age=72,
            gender="Male",
            location="Cardiac ICU - Bed 02",
            baseline_rhythm=CardiacRhythmType.AFIB
        )
        
        # Patient 3: Sarah Lin - Post-MI Recovery (Stable NSR)
        self.patients["BED-03"] = PatientTelemetryState(
            patient_id="BED-03",
            name="Sarah Lin",
            age=45,
            gender="Female",
            location="Telemetry Stepdown - Bed 03",
            baseline_rhythm=CardiacRhythmType.NSR
        )
        
        # Patient 4: Arthur Pendelton - Ambulatory Remote Wearable
        self.patients["REMOTE-01"] = PatientTelemetryState(
            patient_id="REMOTE-01",
            name="Arthur Pendelton",
            age=59,
            gender="Male",
            location="Remote Home Care (Wearable Patch)",
            baseline_rhythm=CardiacRhythmType.NSR
        )

        # Assign default MIT-BIH record pairings
        self.patients["BED-01"].mitbih_record_id = "100"  # Normal Sinus Rhythm
        self.patients["BED-02"].mitbih_record_id = "200"  # Ventricular Tachycardia & PVCs
        self.patients["BED-03"].mitbih_record_id = "209"  # Atrial Fibrillation
        self.patients["REMOTE-01"].mitbih_record_id = "207" # Ventricular Flutter / Fibrillation

    def get_patient(self, patient_id: str) -> Optional[PatientTelemetryState]:
        return self.patients.get(patient_id)

    def trigger_event(self, patient_id: str, rhythm: str, duration: float = 30.0) -> bool:
        p = self.patients.get(patient_id)
        if p:
            p.set_rhythm(rhythm, duration)
            return True
        return False

    def set_patient_data_mode(self, patient_id: str, mode: str, record_id: Optional[str] = None) -> bool:
        """Configures data mode (SYNTHETIC vs MIT-BIH) for an individual bed."""
        p = self.patients.get(patient_id)
        if p:
            p.set_data_mode(mode, record_id)
            return True
        return False

    def set_global_data_mode(self, mode: str):
        """Sets data mode across all hospital beds."""
        for p in self.patients.values():
            p.set_data_mode(mode)

    def generate_batch_for_all(self, num_samples: int = 10) -> Dict[str, List[Dict]]:
        """Generates synchronized chunks of telemetry for all beds (for low-latency batch streaming)."""
        output = {}
        for pid, patient in self.patients.items():
            batch = [patient.generate_next_sample() for _ in range(num_samples)]
            output[pid] = batch
        return output
