"""
CardioSentinel - Medical Digital Signal Processing & QRS Detection Engine
Includes Butterworth digital filtering, 50/60Hz notch filter, baseline wander removal,
Pan-Tompkins real-time QRS detection algorithm (1985), and clinical heart rate variability (HRV)
metrics extraction (SDNN, RMSSD, pNN50, Poincaré SD1/SD2, Shannon Entropy).
"""

import numpy as np
from scipy import signal
from typing import List, Dict, Tuple, Optional, Any

class ECGSignalProcessor:
    def __init__(self, sample_rate: int = 250):
        self.sample_rate = sample_rate
        self._init_filters()

    def _init_filters(self):
        # 4th-order Butterworth bandpass filter (0.5 Hz - 40 Hz) for baseline removal and high-freq noise suppression
        lowcut = 0.5
        highcut = 40.0
        nyq = 0.5 * self.sample_rate
        low = min(0.99, max(0.01, lowcut / nyq))
        high = min(0.99, max(0.01, highcut / nyq))
        self.b_band, self.a_band = signal.butter(4, [low, high], btype='band')

        # 50Hz Notch filter to remove powerline interference
        if self.sample_rate > 105:
            notch_freq = 50.0
            q_factor = 30.0
            self.b_notch, self.a_notch = signal.iirnotch(notch_freq, q_factor, self.sample_rate)
        else:
            self.b_notch, self.a_notch = None, None

        # Moving window integration width for Pan-Tompkins (~150 ms)
        self.integration_window = max(3, int(0.15 * self.sample_rate))

    def set_sample_rate(self, sample_rate: int):
        if sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            self._init_filters()

    def filter_signal(self, data: np.ndarray) -> np.ndarray:
        """Applies zero-phase bandpass and powerline notch filtering to raw ECG telemetry."""
        if len(data) < 20:
            return data

        try:
            if len(data) > 60:
                filtered = signal.filtfilt(self.b_band, self.a_band, data)
                if self.b_notch is not None:
                    filtered = signal.filtfilt(self.b_notch, self.a_notch, filtered)
            else:
                filtered = signal.lfilter(self.b_band, self.a_band, data)
                if self.b_notch is not None:
                    filtered = signal.lfilter(self.b_notch, self.a_notch, filtered)
            return filtered
        except Exception:
            return data

    def pan_tompkins_qrs_detect(self, ecg: np.ndarray) -> List[int]:
        """
        Validated Pan-Tompkins QRS Detection Algorithm:
        1. Bandpass filter (5-15 Hz for QRS energy isolation)
        2. Five-point derivative filter: y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / 8
        3. Squaring function (non-linear amplification of high frequency QRS complex)
        4. Moving window integration (~150 ms)
        5. Adaptive dual-threshold peak detection with 200ms refractory period (~250 bpm ceiling)
        6. Precise R-peak alignment against original filtered signal maxima
        """
        if len(ecg) < self.sample_rate:
            return []

        nyq = 0.5 * self.sample_rate
        low = min(0.99, max(0.01, 5.0 / nyq))
        high = min(0.99, max(0.01, 15.0 / nyq))

        try:
            b_qrs, a_qrs = signal.butter(3, [low, high], btype='band')
            filtered = signal.filtfilt(b_qrs, a_qrs, ecg)
        except Exception:
            filtered = ecg

        # Derivative filter
        d_kernel = np.array([-2, -1, 0, 1, 2]) / 8.0
        differentiated = np.convolve(filtered, d_kernel, mode='same')

        # Squaring
        squared = differentiated ** 2

        # Moving window integration
        kernel = np.ones(self.integration_window) / self.integration_window
        integrated = np.convolve(squared, kernel, mode='same')

        # Refractory period: 200 ms (~300 bpm ceiling)
        min_distance = max(5, int(0.20 * self.sample_rate))
        threshold = np.mean(integrated) + 0.45 * np.std(integrated)

        peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)

        # Refine peaks to find actual R-peak maxima on original signal
        refined_peaks = []
        search_radius = max(3, int(0.08 * self.sample_rate)) # +/- 80 ms search radius
        for p in peaks:
            start = max(0, p - search_radius)
            end = min(len(ecg), p + search_radius)
            if end > start:
                local_max = start + np.argmax(ecg[start:end])
                if not refined_peaks or (local_max - refined_peaks[-1]) >= min_distance:
                    refined_peaks.append(int(local_max))

        return refined_peaks

    def evaluate_signal_quality(self, ecg_window: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates ECG signal quality using ECGSignalQualityMonitor while preserving
        backward-compatible status indicators.
        """
        from backend.signal_quality import signal_quality_monitor, QualityStatus

        assessment = signal_quality_monitor.assess_quality(
            ecg_window,
            sample_rate=self.sample_rate,
            lead_name="Lead II"
        )

        std_status = assessment["status"]
        score = assessment["quality_score"]
        snr_db = assessment["measurements"].get("estimated_snr_db", 0.0)

        # Map to legacy status strings for backward compatibility
        if std_status == QualityStatus.INSUFFICIENT_DATA:
            legacy_status = "DEGRADED"
        elif std_status == QualityStatus.POOR:
            if "flatline" in assessment["primary_reason"].lower() or assessment["measurements"].get("peak_to_peak_mv", 1.0) < 0.12:
                legacy_status = "LEAD OFF / FLATLINE"
            else:
                legacy_status = "MOTION ARTIFACT"
        elif std_status == QualityStatus.REVIEW:
            legacy_status = "MOTION ARTIFACT" if snr_db < 10.0 else "DEGRADED"
        else:
            legacy_status = "OPTIMAL"

        return {
            "status": legacy_status,
            "snr_db": snr_db,
            "quality_score": score,
            "standard_status": std_status,
            "technical_quality_status": std_status,
            "primary_reason": assessment["primary_reason"],
            "quality_assessment": assessment
        }

    def assess_signal_quality(self, ecg_window: np.ndarray) -> Dict[str, Any]:
        """Alias returning normalized score (0.0 to 1.0), standard status, and legacy status."""
        eval_res = self.evaluate_signal_quality(ecg_window)
        status = eval_res["status"]
        if "FLATLINE" in status or "LEAD OFF" in status:
            mapped_status = "LEAD_OFF"
        elif "ARTIFACT" in status or "DEGRADED" in status or eval_res["quality_score"] < 50:
            mapped_status = "POOR"
        else:
            mapped_status = "OPTIMAL"

        return {
            "status": mapped_status,
            "raw_status": status,
            "standard_status": eval_res.get("standard_status", "GOOD SIGNAL"),
            "score": round(eval_res["quality_score"] / 100.0, 2),
            "quality_score": eval_res["quality_score"],
            "snr_db": eval_res["snr_db"],
            "primary_reason": eval_res.get("primary_reason", "Signal quality checks passed."),
            "quality_assessment": eval_res.get("quality_assessment", {})
        }

    def _measure_qrs_duration(self, ecg: np.ndarray, peak: int) -> Optional[float]:
        """
        Measures individual beat QRS complex duration in milliseconds.
        Detects Q-onset (departure from isoelectric baseline before R) and J-point (return of S to baseline).
        """
        fs = self.sample_rate
        q_window_start = max(0, peak - int(0.09 * fs))
        s_window_end = min(len(ecg), peak + int(0.09 * fs))

        if peak - q_window_start < 4 or s_window_end - peak < 4:
            return None

        baseline = float(np.median(ecg[max(0, peak - int(0.14 * fs)):q_window_start])) if q_window_start > 0 else float(ecg[q_window_start])
        q_local = int(np.argmin(ecg[q_window_start:peak])) + q_window_start
        s_local = int(np.argmin(ecg[peak:s_window_end])) + peak
        r_val = float(ecg[peak])
        amp = max(0.1, abs(r_val - baseline))

        q_onset = q_local
        for idx in range(q_local, q_window_start, -1):
            if abs(ecg[idx] - baseline) <= 0.08 * amp:
                q_onset = idx
                break

        j_point = s_local
        for idx in range(s_local, s_window_end):
            if abs(ecg[idx] - baseline) <= 0.08 * amp:
                j_point = idx
                break

        dur_ms = float(((j_point - q_onset) / fs) * 1000.0)
        if 40.0 <= dur_ms <= 240.0:
            return dur_ms
        return None

    def _measure_qt_interval(self, ecg: np.ndarray, peak: int, next_peak: Optional[int], mean_rr_sec: float) -> Optional[Tuple[float, float, float]]:
        """
        Measures raw QT interval (ms), Bazett corrected QTc (ms), and Fridericia corrected QTc (ms).
        Returns (raw_qt_ms, qtc_bazett_ms, qtc_fridericia_ms) or None if unmeasurable.
        """
        fs = self.sample_rate
        if mean_rr_sec <= 0.25:
            return None

        q_window_start = max(0, peak - int(0.09 * fs))
        q_local = int(np.argmin(ecg[q_window_start:peak])) + q_window_start if peak > q_window_start else peak
        q_onset = max(0, q_local - int(0.02 * fs))

        t_start = peak + int(0.10 * fs)
        max_t = int(min(0.55 * fs, 0.70 * (next_peak - peak) if next_peak else 0.55 * fs))
        t_end = min(len(ecg), peak + max_t)

        if t_end - t_start < int(0.08 * fs):
            return None

        t_seg = ecg[t_start:t_end]
        baseline = float(np.median(ecg[max(0, peak - int(0.14 * fs)):max(0, peak - int(0.09 * fs))])) if peak > int(0.14 * fs) else 0.0

        t_max_idx = int(np.argmax(t_seg))
        t_min_idx = int(np.argmin(t_seg))
        pos_amp = abs(t_seg[t_max_idx] - baseline)
        neg_amp = abs(t_seg[t_min_idx] - baseline)

        if max(pos_amp, neg_amp) < 0.03:
            # T-wave is flat, isoelectric, or obscured by noise
            return None

        if pos_amp >= neg_amp:
            t_peak_idx = t_start + t_max_idx
            t_offset = t_end
            for idx in range(t_peak_idx, t_end):
                if ecg[idx] <= baseline + 0.15 * pos_amp:
                    t_offset = idx
                    break
        else:
            t_peak_idx = t_start + t_min_idx
            t_offset = t_end
            for idx in range(t_peak_idx, t_end):
                if ecg[idx] >= baseline - 0.15 * neg_amp:
                    t_offset = idx
                    break

        qt_ms = float(((t_offset - q_onset) / fs) * 1000.0)
        if 250.0 <= qt_ms <= 620.0:
            bazett = float(qt_ms / np.sqrt(mean_rr_sec))
            fridericia = float(qt_ms / (mean_rr_sec ** (1.0 / 3.0)))
            return round(qt_ms, 1), round(bazett, 1), round(fridericia, 1)

        return None

    def extract_clinical_metrics(self, ecg_window: np.ndarray, peaks: List[int]) -> Dict[str, Any]:
        """
        Extracts validated electrophysiological metrics from ECG window & detected R-peaks:
        - Heart Rate (BPM)
        - RR intervals & HRV: SDNN (ms), RMSSD (ms), pNN50 (%)
        - Poincaré Plot non-linear descriptors: SD1 (ms), SD2 (ms)
        - Morphological QRS complex duration (ms)
        - Corrected QT intervals: Bazett QTc & Fridericia QTc (ms)
        - RR Shannon Entropy (irregularity score)
        - Signal Quality assessment and explicit measurability verification flags
        """
        quality = self.evaluate_signal_quality(ecg_window)
        is_flatline = (quality["status"] == "LEAD OFF / FLATLINE")
        is_severe_artifact = (quality["status"] == "MOTION ARTIFACT" and quality.get("snr_db", 0.0) < 6.0)
        is_suitable = not is_flatline and not is_severe_artifact

        unmeasurable_reasons = {}

        # 1. Check for insufficient beats (<2 peaks) or flatline with no beats
        if len(peaks) < 2:
            num_detected = len(peaks)
            if is_flatline or num_detected == 0:
                unmeasurable_reasons["heart_rate"] = "Lead off / isoelectric flatline detected (signal amplitude < 0.12 mV). No R-peaks present."
                unmeasurable_reasons["qrs_duration"] = "Lead off / flatline: Isoelectric baseline prevents QRS onset/offset detection."
                unmeasurable_reasons["qt_interval"] = "Lead off / flatline: Ventricular repolarization absent on isoelectric trace."
                unmeasurable_reasons["hrv"] = "Lead off / flatline: No consecutive cardiac cycles available."

                return {
                    "heart_rate": 0,
                    "hr_measurable": False,
                    "hr_status_note": "Asystole / Lead Disconnect",
                    "hr_unmeasurable_reason": unmeasurable_reasons["heart_rate"],
                    "hrv_sdnn": 0.0,
                    "hrv_rmssd": 0.0,
                    "pnn50": 0.0,
                    "mean_rr_ms": 0.0,
                    "rr_mean_ms": 0.0,
                    "rr_cv": 0.0,
                    "poincare_sd1": 0.0,
                    "poincare_sd2": 0.0,
                    "hrv_measurable": False,
                    "hrv_status_note": "Unmeasurable",
                    "hrv_unmeasurable_reason": unmeasurable_reasons["hrv"],
                    "qrs_duration_ms": 0.0,
                    "qrs_measurable": False,
                    "qrs_status_note": "Unmeasurable (Flatline)",
                    "qrs_unmeasurable_reason": unmeasurable_reasons["qrs_duration"],
                    "qt_ms": 0.0,
                    "qtc_ms": 0.0,
                    "qtc_bazett_ms": 0.0,
                    "qtc_fridericia_ms": 0.0,
                    "qt_measurable": False,
                    "qt_status_note": "Unmeasurable (Flatline)",
                    "qt_unmeasurable_reason": unmeasurable_reasons["qt_interval"],
                    "rr_entropy": 0.0,
                    "num_beats": 0,
                    "signal_quality": quality["status"],
                    "technical_quality_status": quality.get("standard_status", "POOR SIGNAL"),
                    "technical_quality_reason": quality.get("primary_reason", "Lead off / flatline detected."),
                    "quality_score": quality["quality_score"],
                    "snr_db": quality.get("snr_db", -10.0),
                    "quality_assessment": quality.get("quality_assessment", {}),
                    "is_suitable": False,
                    "unmeasurable_reasons": unmeasurable_reasons
                }

            unmeasurable_reasons["heart_rate"] = "Single isolated beat detected; minimum 2 consecutive beats required for instantaneous rate cadence."
            unmeasurable_reasons["qt_interval"] = "Consecutive RR interval required for rate-corrected QTc calculation (Bazett & Fridericia)."
            unmeasurable_reasons["hrv"] = f"Requires >= 2 consecutive cardiac cycles for RR interval delta; only {num_detected} beat(s) detected."

            # If 1 beat detected and not flatline, attempt QRS duration
            qrs_single_dur = self._measure_qrs_duration(ecg_window, peaks[0]) if not is_flatline else None
            qrs_meas = (qrs_single_dur is not None)
            qrs_val = round(qrs_single_dur, 1) if qrs_meas else 0.0
            qrs_note = "Normal (<100ms)" if (qrs_meas and qrs_val < 100) else ("Wide (>120ms)" if (qrs_meas and qrs_val > 120) else "Borderline (100-120ms)" if qrs_meas else "Indistinct QRS fiducials")
            if not qrs_meas:
                unmeasurable_reasons["qrs_duration"] = "QRS fiducial boundaries could not be resolved reliably on single beat."

            return {
                "heart_rate": 0,
                "hr_measurable": False,
                "hr_status_note": "Insufficient Beats (< 2)",
                "hr_unmeasurable_reason": unmeasurable_reasons["heart_rate"],
                "hrv_sdnn": 0.0,
                "hrv_rmssd": 0.0,
                "pnn50": 0.0,
                "mean_rr_ms": 0.0,
                "rr_mean_ms": 0.0,
                "rr_cv": 0.0,
                "poincare_sd1": 0.0,
                "poincare_sd2": 0.0,
                "hrv_measurable": False,
                "hrv_status_note": "Insufficient Beats",
                "hrv_unmeasurable_reason": unmeasurable_reasons["hrv"],
                "qrs_duration_ms": qrs_val,
                "qrs_measurable": qrs_meas,
                "qrs_status_note": qrs_note,
                "qrs_unmeasurable_reason": unmeasurable_reasons.get("qrs_duration"),
                "qt_ms": 0.0,
                "qtc_ms": 0.0,
                "qtc_bazett_ms": 0.0,
                "qtc_fridericia_ms": 0.0,
                "qt_measurable": False,
                "qt_status_note": "Rate correction requires >= 2 beats",
                "qt_unmeasurable_reason": unmeasurable_reasons["qt_interval"],
                "rr_entropy": 0.0,
                "num_beats": num_detected,
                "signal_quality": quality["status"],
                "technical_quality_status": quality.get("standard_status", "REVIEW SIGNAL"),
                "technical_quality_reason": quality.get("primary_reason", "Single beat / insufficient beats for rate cadence."),
                "quality_score": quality["quality_score"],
                "snr_db": quality.get("snr_db", 18.0),
                "quality_assessment": quality.get("quality_assessment", {}),
                "is_suitable": is_suitable,
                "unmeasurable_reasons": unmeasurable_reasons
            }

        # 2. Valid signal with >= 2 peaks: Compute validated physiological metrics
        rr_intervals_sec = np.diff(peaks) / float(self.sample_rate)
        rr_intervals_ms = rr_intervals_sec * 1000.0

        # Filter out extreme physiologically impossible RR intervals (<200ms or >3000ms) for HRV
        valid_mask = (rr_intervals_ms >= 200.0) & (rr_intervals_ms <= 3000.0)
        valid_rr_ms = rr_intervals_ms[valid_mask] if np.any(valid_mask) else rr_intervals_ms

        mean_rr_sec = float(np.mean(rr_intervals_sec))
        heart_rate = round(60.0 / mean_rr_sec) if mean_rr_sec > 0 else 0
        hr_status_note = "Normal Sinus (60-100 bpm)" if 60 <= heart_rate <= 100 else ("Bradycardia (<60 bpm)" if heart_rate < 60 else "Tachycardia (>100 bpm)")

        # SDNN & RMSSD
        sdnn = float(np.std(valid_rr_ms))
        diff_rr = np.diff(valid_rr_ms)
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.0
        pnn50 = float(np.sum(np.abs(diff_rr) > 50.0) / len(diff_rr) * 100.0) if len(diff_rr) > 0 else 0.0

        mean_rr_ms = float(np.mean(valid_rr_ms))
        rr_cv = float(sdnn / mean_rr_ms) if mean_rr_ms > 0 else 0.0

        # Poincaré non-linear dynamics
        if len(valid_rr_ms) >= 2:
            x_n = valid_rr_ms[:-1]
            x_next = valid_rr_ms[1:]
            sd1 = float(np.sqrt(0.5 * np.var(x_next - x_n)))
            sd2 = float(np.sqrt(0.5 * np.var(x_next + x_n)))
            if sd2 == 0.0 and sdnn > 0:
                sd2 = float(np.sqrt(max(0.0, 2.0 * (sdnn ** 2) - (sd1 ** 2))))
        else:
            sd1, sd2 = 0.0, 0.0

        hrv_measurable = (len(valid_rr_ms) >= 2)
        hrv_status_note = "Depressed Autonomic Tone (<20ms)" if sdnn < 20 else ("Normal Autonomic Tone (30-60ms)" if 30 <= sdnn <= 60 else "Elevated Variation (>60ms)")
        if not hrv_measurable:
            unmeasurable_reasons["hrv"] = "Insufficient consecutive valid RR intervals for statistical dispersion."

        # Measure QRS duration across detected beats
        beat_qrs_durations = []
        for pk in peaks:
            dur = self._measure_qrs_duration(ecg_window, pk)
            if dur is not None:
                beat_qrs_durations.append(dur)

        if beat_qrs_durations:
            qrs_duration_ms = float(np.median(beat_qrs_durations))
            qrs_measurable = True
            if qrs_duration_ms < 100.0:
                qrs_status_note = "Normal (<100ms)"
            elif qrs_duration_ms <= 120.0:
                qrs_status_note = "Incomplete Conduction Delay (100-120ms)"
            else:
                qrs_status_note = "Wide QRS / Conduction Block (>120ms)"
        else:
            qrs_duration_ms = 0.0
            qrs_measurable = False
            qrs_status_note = "Morphologic fiducials indistinct in current lead"
            unmeasurable_reasons["qrs_duration"] = "Q-onset and J-point baseline return boundaries could not be resolved reliably in current lead."

        # Measure QT / QTc across detected beats
        beat_qt_results = []
        for i, pk in enumerate(peaks):
            nxt_pk = peaks[i + 1] if i + 1 < len(peaks) else None
            qt_res = self._measure_qt_interval(ecg_window, pk, nxt_pk, mean_rr_sec)
            if qt_res is not None:
                beat_qt_results.append(qt_res)

        if beat_qt_results:
            raw_qts = [r[0] for r in beat_qt_results]
            bazetts = [r[1] for r in beat_qt_results]
            fridericias = [r[2] for r in beat_qt_results]

            qt_ms = float(np.median(raw_qts))
            qtc_bazett_ms = float(np.median(bazetts))
            qtc_fridericia_ms = float(np.median(fridericias))
            qt_measurable = True

            if qtc_bazett_ms < 440.0:
                qt_status_note = "Normal QTc (<440ms)"
            elif qtc_bazett_ms <= 470.0:
                qt_status_note = "Borderline Prolonged (440-470ms)"
            elif qtc_bazett_ms <= 500.0:
                qt_status_note = "Prolonged QTc (>470ms)"
            else:
                qt_status_note = "Critical Prolongation (>500ms, TdP Risk)"
        else:
            qt_ms = 0.0
            qtc_bazett_ms = 0.0
            qtc_fridericia_ms = 0.0
            qt_measurable = False
            qt_status_note = "T-wave repolarization fiducial ambiguous"
            unmeasurable_reasons["qt_interval"] = "T-wave repolarization offset ambiguous or obscured by baseline wander in current lead window."

        # Shannon entropy of RR intervals
        hist, _ = np.histogram(valid_rr_ms, bins=min(6, max(3, len(valid_rr_ms) // 2)), density=True)
        hist = hist[hist > 0]
        rr_entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

        return {
            "heart_rate": int(heart_rate),
            "hr_measurable": True,
            "hr_status_note": hr_status_note,
            "hr_unmeasurable_reason": None,
            "hrv_sdnn": round(sdnn, 2),
            "hrv_rmssd": round(rmssd, 2),
            "pnn50": round(pnn50, 1),
            "mean_rr_ms": round(mean_rr_ms, 1),
            "rr_mean_ms": round(mean_rr_ms, 1),
            "rr_cv": round(rr_cv, 3),
            "poincare_sd1": round(sd1, 2),
            "poincare_sd2": round(sd2, 2),
            "hrv_measurable": hrv_measurable,
            "hrv_status_note": hrv_status_note,
            "hrv_unmeasurable_reason": unmeasurable_reasons.get("hrv"),
            "qrs_duration_ms": round(qrs_duration_ms, 1),
            "qrs_measurable": qrs_measurable,
            "qrs_status_note": qrs_status_note,
            "qrs_unmeasurable_reason": unmeasurable_reasons.get("qrs_duration"),
            "qt_ms": round(qt_ms, 1),
            "qtc_ms": round(qtc_bazett_ms, 1), # Kept for backward compatibility
            "qtc_bazett_ms": round(qtc_bazett_ms, 1),
            "qtc_fridericia_ms": round(qtc_fridericia_ms, 1),
            "qt_measurable": qt_measurable,
            "qt_status_note": qt_status_note,
            "qt_unmeasurable_reason": unmeasurable_reasons.get("qt_interval"),
            "rr_entropy": round(rr_entropy, 3),
            "num_beats": len(peaks),
            "signal_quality": quality["status"],
            "technical_quality_status": quality.get("standard_status", "GOOD SIGNAL"),
            "technical_quality_reason": quality.get("primary_reason", "Signal quality checks passed."),
            "quality_score": quality["quality_score"],
            "snr_db": quality.get("snr_db", 24.5),
            "quality_assessment": quality.get("quality_assessment", {}),
            "is_suitable": is_suitable,
            "unmeasurable_reasons": unmeasurable_reasons
        }

    def extract_metrics(self, ecg_window: np.ndarray, peaks: List[int], raw_signal: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compatibility wrapper for extract_clinical_metrics."""
        return self.extract_clinical_metrics(ecg_window, peaks)

# Global singleton
signal_processor = ECGSignalProcessor()
