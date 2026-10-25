"""
CardioSentinel - Technical ECG Signal Quality Monitor
Provides lightweight, configurable, non-invasive assessment of ECG signal integrity
prior to automated rhythm classification and clinical alert evaluation.

Evaluates:
1. Missing, non-finite (NaN, Inf), or invalid samples.
2. Flatline or near-flatline segments (lead off / disconnection).
3. Excessive high-frequency noise and baseline drift / wander.
4. Implausible signal amplitudes and sensor saturation / railing.
5. Insufficient recording duration or sample count.
6. Electrophysiological rate/QRS consistency checks.

Returns standardized statuses:
- GOOD SIGNAL: All configured technical quality checks passed.
- REVIEW SIGNAL: Potential quality concerns identified; interpretation should proceed with caution.
- POOR SIGNAL: Severe technical degradation; signal unsuitable for reliable automated interpretation.
- INSUFFICIENT DATA: Incomplete or empty data; insufficient samples to perform assessment.

DISCLAIMER:
This module evaluates technical signal parameters for engineering and telemetry reliability.
It does not constitute medical certification or clinical safety validation.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union


from enum import Enum


class QualityStatus(str, Enum):
    GOOD = "GOOD SIGNAL"
    REVIEW = "REVIEW SIGNAL"
    POOR = "POOR SIGNAL"
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"

    # Convenient aliases matching prompt conventions
    GOOD_SIGNAL = "GOOD SIGNAL"
    REVIEW_SIGNAL = "REVIEW SIGNAL"
    POOR_SIGNAL = "POOR SIGNAL"


SignalQualityStatus = QualityStatus


class ECGSignalQualityMonitor:
    """
    Lightweight, configurable ECG signal quality monitoring engine.
    Designed for 250 Hz continuous telemetry windows and 360 Hz benchmark records.
    """

    def __init__(
        self,
        sample_rate: int = 250,
        min_duration_sec: float = 1.0,
        flatline_ptp_threshold_mv: float = 0.10,
        flatline_var_threshold_mv2: float = 0.003,
        saturation_ptp_threshold_mv: float = 6.0,
        max_abs_amplitude_mv: float = 8.0,
        min_snr_db: float = 10.0,
        poor_snr_db: float = 4.0,
        max_baseline_wander_mv: float = 1.50,
        max_consecutive_flatline_sec: float = 0.80,
    ):
        """
        Configurable threshold parameters. Documented for engineering transparency.
        - sample_rate: Expected ECG digitization rate in Hz (default: 250 Hz)
        - min_duration_sec: Minimum window length in seconds required for valid analysis (1.0s)
        - flatline_ptp_threshold_mv: Peak-to-peak amplitude below which a flatline is flagged (0.10 mV)
        - flatline_var_threshold_mv2: Variance floor for isoelectric trace detection (0.003 mV^2)
        - saturation_ptp_threshold_mv: Upper peak-to-peak amplitude indicating clipping/saturation (6.0 mV)
        - max_abs_amplitude_mv: Upper absolute limit for physiologically plausible ECG telemetry (8.0 mV)
        - min_snr_db: Minimum estimated Signal-to-Noise Ratio for uncompromised status (10.0 dB)
        - poor_snr_db: SNR threshold below which signal is categorized as POOR (4.0 dB)
        - max_baseline_wander_mv: Maximum allowable low-frequency wander before review flag (1.50 mV)
        - max_consecutive_flatline_sec: Maximum duration of sustained zero/near-zero delta (0.80s)
        """
        self.sample_rate = int(sample_rate)
        self.min_duration_sec = float(min_duration_sec)
        self.flatline_ptp_threshold_mv = float(flatline_ptp_threshold_mv)
        self.flatline_var_threshold_mv2 = float(flatline_var_threshold_mv2)
        self.saturation_ptp_threshold_mv = float(saturation_ptp_threshold_mv)
        self.max_abs_amplitude_mv = float(max_abs_amplitude_mv)
        self.min_snr_db = float(min_snr_db)
        self.poor_snr_db = float(poor_snr_db)
        self.max_baseline_wander_mv = float(max_baseline_wander_mv)
        self.max_consecutive_flatline_sec = float(max_consecutive_flatline_sec)

    def set_sample_rate(self, sample_rate: int):
        if sample_rate > 0:
            self.sample_rate = int(sample_rate)

    def assess_quality(
        self,
        ecg_data: Optional[Union[np.ndarray, List[float]]],
        sample_rate: Optional[int] = None,
        lead_name: str = "Lead II",
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes lightweight technical quality checks on actual loaded ECG samples.
        Never returns a hardcoded status.
        Safely handles None, empty arrays, NaN/Inf values, short recordings, and extreme amplitudes.
        """
        fs = int(sample_rate) if (sample_rate and sample_rate > 0) else self.sample_rate

        # -------------------------------------------------------------
        # 1. Null, Empty, and Type Safety Check
        # -------------------------------------------------------------
        if ecg_data is None:
            return self._build_insufficient_data_result(
                reason="No ECG signal array provided (null input).",
                lead_name=lead_name,
                sample_rate=fs,
                sample_count=0
            )

        try:
            arr = np.asarray(ecg_data, dtype=np.float64)
        except Exception as e:
            return self._build_insufficient_data_result(
                reason=f"Failed to parse signal array into numeric samples: {e}",
                lead_name=lead_name,
                sample_rate=fs,
                sample_count=0
            )

        sample_count = len(arr)
        if sample_count == 0:
            return self._build_insufficient_data_result(
                reason="Recording window contains 0 samples (empty array).",
                lead_name=lead_name,
                sample_rate=fs,
                sample_count=0
            )

        duration_sec = sample_count / float(fs)
        min_required_samples = max(20, int(self.min_duration_sec * fs))

        # Check for minimum required window length
        if sample_count < min_required_samples:
            return self._build_insufficient_data_result(
                reason=f"Insufficient signal duration ({sample_count} samples, {duration_sec:.2f}s; minimum {self.min_duration_sec:.1f}s / {min_required_samples} samples required).",
                lead_name=lead_name,
                sample_rate=fs,
                sample_count=sample_count,
                duration_sec=duration_sec
            )

        # -------------------------------------------------------------
        # 2. Non-Finite (NaN / Inf) Sample Validation
        # -------------------------------------------------------------
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        invalid_mask = nan_mask | inf_mask
        invalid_count = int(np.sum(invalid_mask))
        invalid_ratio = invalid_count / float(sample_count)

        detected_issues: List[str] = []
        checks_log: List[Dict[str, Any]] = []

        is_severe_poor = False
        is_review_flag = False

        if invalid_count > 0:
            detected_issues.append(
                f"Invalid numerical samples: {invalid_count} non-finite (NaN/Inf) values detected ({invalid_ratio * 100:.1f}% of window)."
            )
            is_severe_poor = True
            checks_log.append({
                "name": "Finite Numerical Validity",
                "passed": False,
                "severity": "POOR",
                "measurement": f"{invalid_count} NaN/Inf samples ({invalid_ratio * 100:.1f}%)",
                "threshold": "0 non-finite samples",
                "detail": "Non-finite data corrupts digital filtering and feature extraction."
            })
        else:
            checks_log.append({
                "name": "Finite Numerical Validity",
                "passed": True,
                "severity": "GOOD",
                "measurement": f"{sample_count} valid finite samples (0 NaN/Inf)",
                "threshold": "0 non-finite samples",
                "detail": "All telemetry points represent finite real numbers."
            })

        # Sanitize signal for statistical evaluation if isolated NaNs exist
        if invalid_count > 0:
            valid_samples = arr[~invalid_mask]
            if len(valid_samples) < max(20, int(0.5 * min_required_samples)):
                return self._build_poor_signal_result(
                    primary_reason="Missing or non-finite samples comprise the recording.",
                    detected_issues=detected_issues,
                    checks_log=checks_log,
                    lead_name=lead_name,
                    sample_rate=fs,
                    sample_count=sample_count,
                    duration_sec=duration_sec,
                    invalid_count=invalid_count,
                    score=10
                )
            eval_arr = valid_samples
        else:
            eval_arr = arr

        # -------------------------------------------------------------
        # 3. Flatline & Isoelectric Lead-Off Assessment
        # -------------------------------------------------------------
        ptp_val = float(np.ptp(eval_arr))
        variance_val = float(np.var(eval_arr))
        max_abs_val = float(np.max(np.abs(eval_arr)))
        consecutive_flatline = {"detected": False, "length_samples": 0}

        # Global flatline check across entire window
        if ptp_val < self.flatline_ptp_threshold_mv or variance_val < self.flatline_var_threshold_mv2:
            detected_issues.append(
                f"Global flatline or sensor disconnect: peak-to-peak amplitude {ptp_val:.3f} mV (< {self.flatline_ptp_threshold_mv:.2f} mV floor), variance {variance_val:.5f} mV²."
            )
            is_severe_poor = True
            checks_log.append({
                "name": "Flatline & Lead-Off Detection",
                "passed": False,
                "severity": "POOR",
                "measurement": f"PTP {ptp_val:.3f} mV, Var {variance_val:.5f} mV²",
                "threshold": f"PTP >= {self.flatline_ptp_threshold_mv:.2f} mV, Var >= {self.flatline_var_threshold_mv2:.4f} mV²",
                "detail": "Near-zero amplitude indicates disconnected electrode or asystole."
            })
        else:
            # Check for sustained consecutive sub-segment flatline (e.g. 0.8s flatline run within 3s window)
            max_flatline_samples = int(self.max_consecutive_flatline_sec * fs)
            consecutive_flatline = self._detect_consecutive_flatline(eval_arr, max_flatline_samples)
            if consecutive_flatline["detected"]:
                flat_dur = consecutive_flatline["length_samples"] / float(fs)
                detected_issues.append(
                    f"Sustained flatline segment detected: {flat_dur:.2f}s ({consecutive_flatline['length_samples']} samples) without physiological deflection."
                )
                is_severe_poor = True
                checks_log.append({
                    "name": "Flatline & Lead-Off Detection",
                    "passed": False,
                    "severity": "POOR",
                    "measurement": f"Flatline run of {flat_dur:.2f}s",
                    "threshold": f"Max allowable flatline: {self.max_consecutive_flatline_sec:.2f}s",
                    "detail": "Sustained isoelectric pause detected within recording window."
                })
            else:
                checks_log.append({
                    "name": "Flatline & Lead-Off Detection",
                    "passed": True,
                    "severity": "GOOD",
                    "measurement": f"PTP {ptp_val:.2f} mV, Var {variance_val:.4f} mV²",
                    "threshold": f"PTP >= {self.flatline_ptp_threshold_mv:.2f} mV",
                    "detail": "Adequate dynamic range; no isoelectric disconnection detected."
                })

        # -------------------------------------------------------------
        # 4. Amplitude Plausibility & Sensor Saturation Check
        # -------------------------------------------------------------
        if ptp_val > self.saturation_ptp_threshold_mv or max_abs_val > self.max_abs_amplitude_mv:
            detected_issues.append(
                f"Implausible signal amplitude / sensor saturation: PTP {ptp_val:.2f} mV (limit {self.saturation_ptp_threshold_mv:.1f} mV), Max |V| {max_abs_val:.2f} mV (limit {self.max_abs_amplitude_mv:.1f} mV)."
            )
            is_severe_poor = True
            checks_log.append({
                "name": "Amplitude Bounds & Saturation",
                "passed": False,
                "severity": "POOR",
                "measurement": f"PTP {ptp_val:.2f} mV, Peak |V| {max_abs_val:.2f} mV",
                "threshold": f"PTP <= {self.saturation_ptp_threshold_mv:.1f} mV, Peak <= {self.max_abs_amplitude_mv:.1f} mV",
                "detail": "Excessive electrical voltage indicates motion artifact, rail saturation, or defibrillation discharge."
            })
        elif ptp_val > (0.75 * self.saturation_ptp_threshold_mv):
            detected_issues.append(
                f"High amplitude excursion detected: PTP {ptp_val:.2f} mV approaching saturation ceiling ({self.saturation_ptp_threshold_mv:.1f} mV)."
            )
            is_review_flag = True
            checks_log.append({
                "name": "Amplitude Bounds & Saturation",
                "passed": False,
                "severity": "REVIEW",
                "measurement": f"PTP {ptp_val:.2f} mV",
                "threshold": f"PTP <= {self.saturation_ptp_threshold_mv:.1f} mV",
                "detail": "Borderline elevated amplitude; monitor for electrode movement or baseline shift."
            })
        else:
            checks_log.append({
                "name": "Amplitude Bounds & Saturation",
                "passed": True,
                "severity": "GOOD",
                "measurement": f"PTP {ptp_val:.2f} mV (Max: {max_abs_val:.2f} mV)",
                "threshold": f"PTP <= {self.saturation_ptp_threshold_mv:.1f} mV",
                "detail": "Physiologically plausible ECG voltage range."
            })

        # -------------------------------------------------------------
        # 5. High-Frequency Noise & Signal-to-Noise Ratio (SNR)
        # -------------------------------------------------------------
        snr_db = self._estimate_snr_db(eval_arr, fs)
        if snr_db < self.poor_snr_db:
            detected_issues.append(
                f"Excessive signal noise / low SNR: estimated {snr_db:.1f} dB (floor {self.poor_snr_db:.1f} dB)."
            )
            is_severe_poor = True
            checks_log.append({
                "name": "High-Frequency Noise & SNR",
                "passed": False,
                "severity": "POOR",
                "measurement": f"SNR {snr_db:.1f} dB",
                "threshold": f"SNR >= {self.poor_snr_db:.1f} dB",
                "detail": "Severe electromyographic (EMG) or 50/60 Hz interference degrades QRS resolution."
            })
        elif snr_db < self.min_snr_db:
            detected_issues.append(
                f"Moderate signal noise detected: estimated SNR {snr_db:.1f} dB (recommended >= {self.min_snr_db:.1f} dB)."
            )
            is_review_flag = True
            checks_log.append({
                "name": "High-Frequency Noise & SNR",
                "passed": False,
                "severity": "REVIEW",
                "measurement": f"SNR {snr_db:.1f} dB",
                "threshold": f"SNR >= {self.min_snr_db:.1f} dB",
                "detail": "Elevated background noise may lower rhythm classifier confidence."
            })
        else:
            checks_log.append({
                "name": "High-Frequency Noise & SNR",
                "passed": True,
                "severity": "GOOD",
                "measurement": f"SNR {snr_db:.1f} dB",
                "threshold": f"SNR >= {self.min_snr_db:.1f} dB",
                "detail": "Low background interference; sharp QRS fiducial peaks."
            })

        # -------------------------------------------------------------
        # 6. Baseline Wander / Low-Frequency Respiration Drift
        # -------------------------------------------------------------
        wander_mv = self._estimate_baseline_wander_mv(eval_arr, fs)
        if wander_mv > (1.8 * self.max_baseline_wander_mv):
            detected_issues.append(
                f"Excessive baseline drift: {wander_mv:.2f} mV wander (ceiling {self.max_baseline_wander_mv:.2f} mV)."
            )
            is_severe_poor = True
            checks_log.append({
                "name": "Baseline Wander & Low-Frequency Drift",
                "passed": False,
                "severity": "POOR",
                "measurement": f"Drift {wander_mv:.2f} mV",
                "threshold": f"Drift <= {self.max_baseline_wander_mv:.2f} mV",
                "detail": "Severe isoelectric drift may shift ST segments and obscure P/T waves."
            })
        elif wander_mv > self.max_baseline_wander_mv:
            detected_issues.append(
                f"Noticeable baseline wander detected: {wander_mv:.2f} mV (recommended <= {self.max_baseline_wander_mv:.2f} mV)."
            )
            is_review_flag = True
            checks_log.append({
                "name": "Baseline Wander & Low-Frequency Drift",
                "passed": False,
                "severity": "REVIEW",
                "measurement": f"Drift {wander_mv:.2f} mV",
                "threshold": f"Drift <= {self.max_baseline_wander_mv:.2f} mV",
                "detail": "Patient respiration or perspiration causing moderate baseline oscillation."
            })
        else:
            checks_log.append({
                "name": "Baseline Wander & Low-Frequency Drift",
                "passed": True,
                "severity": "GOOD",
                "measurement": f"Drift {wander_mv:.2f} mV",
                "threshold": f"Drift <= {self.max_baseline_wander_mv:.2f} mV",
                "detail": "Stable isoelectric baseline line with minimal drift."
            })

        # -------------------------------------------------------------
        # 7. Optional Electrophysiological Cadence Consistency Check
        # -------------------------------------------------------------
        if additional_metrics and isinstance(additional_metrics, dict):
            hr = additional_metrics.get("heart_rate")
            num_beats = additional_metrics.get("num_beats")
            if hr is not None and isinstance(hr, (int, float)):
                if hr > 280 and ptp_val < 3.0:
                    detected_issues.append(
                        f"Implausible detected heart rate: {int(hr)} bpm exceeding physiological cadence ceiling."
                    )
                    is_review_flag = True
                    checks_log.append({
                        "name": "QRS Rate Consistency",
                        "passed": False,
                        "severity": "REVIEW",
                        "measurement": f"Detected HR {int(hr)} bpm",
                        "threshold": "HR <= 280 bpm",
                        "detail": "High-frequency spike artifact may cause Pan-Tompkins double-counting."
                    })
                elif hr == 0 and ptp_val > 0.8:
                    detected_issues.append(
                        "Rate detection mismatch: 0 beats detected despite presence of dynamic electrical signal (> 0.8 mV)."
                    )
                    is_review_flag = True
                    checks_log.append({
                        "name": "QRS Rate Consistency",
                        "passed": False,
                        "severity": "REVIEW",
                        "measurement": "0 beats detected on active signal",
                        "threshold": "Fiducial sync",
                        "detail": "Atypical complex morphology or wide conduction delay."
                    })

        # -------------------------------------------------------------
        # 8. Status Synthesis & Objective Quality Score Calculation
        # -------------------------------------------------------------
        # Documented calculation:
        # Start at 100 points, apply explicit deductions per failing check
        score = 100.0

        # Non-finite deductions
        if invalid_count > 0:
            score -= min(40.0, invalid_ratio * 200.0)

        # Flatline deductions
        if ptp_val < self.flatline_ptp_threshold_mv or variance_val < self.flatline_var_threshold_mv2:
            score -= 85.0
        elif consecutive_flatline.get("detected", False):
            score -= 60.0

        # Amplitude saturation deductions
        if ptp_val > self.saturation_ptp_threshold_mv:
            score -= min(60.0, 30.0 + (ptp_val - self.saturation_ptp_threshold_mv) * 10.0)
        elif ptp_val > (0.75 * self.saturation_ptp_threshold_mv):
            score -= 15.0

        # Noise / SNR deductions
        if snr_db < 20.0:
            snr_deficit = max(0.0, 20.0 - snr_db)
            score -= min(45.0, snr_deficit * 2.5)

        # Baseline wander deductions
        if wander_mv > self.max_baseline_wander_mv:
            drift_excess = wander_mv - self.max_baseline_wander_mv
            score -= min(25.0, drift_excess * 15.0)

        score = max(5.0, min(98.0, round(score, 1)))

        # Final Status Decision based on actual technical criteria
        if is_severe_poor or score < 50.0:
            status = QualityStatus.POOR
            primary_reason = detected_issues[0] if detected_issues else "Significant technical signal degradation detected."
        elif is_review_flag or score < 80.0:
            status = QualityStatus.REVIEW
            primary_reason = detected_issues[0] if detected_issues else "Moderate technical noise or baseline deviation identified."
        else:
            status = QualityStatus.GOOD
            primary_reason = "Signal quality checks passed."
            score = max(85.0, score)

        checks_dict = {
            "finite_samples": {
                "passed": invalid_count == 0,
                "invalid_count": invalid_count,
                "invalid_ratio": round(invalid_ratio, 4)
            },
            "flatline": {
                "passed": not (ptp_val < self.flatline_ptp_threshold_mv or variance_val < self.flatline_var_threshold_mv2 or consecutive_flatline.get("detected", False)),
                "ptp_mv": round(ptp_val, 3),
                "variance": round(variance_val, 5),
                "threshold_mv": self.flatline_ptp_threshold_mv,
                "consecutive_flatline_samples": consecutive_flatline.get("length_samples", 0)
            },
            "saturation": {
                "passed": not (ptp_val > self.saturation_ptp_threshold_mv or max_abs_val > self.max_abs_amplitude_mv),
                "ptp_mv": round(ptp_val, 3),
                "max_abs_mv": round(max_abs_val, 3),
                "threshold_mv": self.saturation_ptp_threshold_mv
            },
            "noise_and_drift": {
                "passed": not (snr_db < self.min_snr_db or wander_mv > self.max_baseline_wander_mv),
                "snr_db": round(snr_db, 1),
                "baseline_drift_mv": round(wander_mv, 3),
                "min_snr_threshold_db": self.min_snr_db,
                "max_drift_threshold_mv": self.max_baseline_wander_mv
            },
            "duration": {
                "passed": sample_count >= int(self.min_duration_sec * fs),
                "duration_sec": round(duration_sec, 2),
                "samples": sample_count,
                "min_duration_sec": self.min_duration_sec
            }
        }

        return {
            "status": status,
            "quality_score": round(score),
            "score": round(score),
            "primary_reason": primary_reason,
            "detected_issues": detected_issues,
            "is_suitable_for_classification": (status != QualityStatus.POOR and status != QualityStatus.INSUFFICIENT_DATA),
            "is_suitable_for_analysis": (status != QualityStatus.POOR and status != QualityStatus.INSUFFICIENT_DATA),
            "measurements": {
                "duration_sec": round(duration_sec, 2),
                "sample_count": sample_count,
                "sample_rate_hz": fs,
                "invalid_samples_count": invalid_count,
                "peak_to_peak_mv": round(ptp_val, 3),
                "variance": round(variance_val, 5),
                "estimated_snr_db": round(snr_db, 1),
                "baseline_wander_mv": round(wander_mv, 3),
                "lead_channel": lead_name
            },
            "configured_thresholds": {
                "min_duration_sec": self.min_duration_sec,
                "flatline_ptp_mv": self.flatline_ptp_threshold_mv,
                "flatline_variance_floor": self.flatline_var_threshold_mv2,
                "saturation_ptp_mv": self.saturation_ptp_threshold_mv,
                "max_abs_amplitude_mv": self.max_abs_amplitude_mv,
                "min_snr_db": self.min_snr_db,
                "poor_snr_db": self.poor_snr_db,
                "max_baseline_wander_mv": self.max_baseline_wander_mv
            },
            "checks": checks_dict,
            "checks_log": checks_log,
            "disclaimer": "Technical signal-quality check only; does not establish clinical safety or replace physician ECG review."
        }

    def _detect_consecutive_flatline(self, data: np.ndarray, max_samples: int) -> Dict[str, Any]:
        """Detects consecutive isoelectric samples within the window."""
        if len(data) < max_samples:
            return {"detected": False, "length_samples": 0}

        diffs = np.abs(np.diff(data))
        near_zero = diffs < 0.005 # Less than 5 microvolts delta

        max_run = 0
        current_run = 0
        for is_flat in near_zero:
            if is_flat:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0

        return {
            "detected": max_run >= max_samples,
            "length_samples": max_run
        }

    def _estimate_snr_db(self, data: np.ndarray, fs: int) -> float:
        """
        Estimates signal-to-noise ratio in decibels.
        Approximates noise via high-frequency successive difference dispersion.
        """
        if len(data) < 20:
            return 10.0

        var_total = float(np.var(data))
        if var_total <= 1e-6:
            return -5.0

        # Noise estimate using second-order difference filter (isolates high-frequency noise)
        diff2 = np.diff(data, n=2)
        # Scale factor for Gaussian white noise on 2nd difference is sqrt(6)
        noise_std = float(np.std(diff2) / np.sqrt(6.0)) if len(diff2) > 0 else 0.05
        var_noise = max(1e-6, noise_std ** 2)

        var_signal = max(1e-6, var_total - var_noise)
        snr_linear = var_signal / var_noise
        snr_db = 10.0 * np.log10(max(0.1, snr_linear))

        return float(np.clip(snr_db, -10.0, 32.0))

    def _estimate_baseline_wander_mv(self, data: np.ndarray, fs: int) -> float:
        """
        Estimates low-frequency isoelectric baseline wander in millivolts
        via moving average / median baseline subtraction.
        """
        if len(data) < max(10, int(0.5 * fs)):
            return 0.0

        # Moving window of ~0.6 seconds to estimate respiration baseline
        win_size = max(5, int(0.6 * fs))
        kernel = np.ones(win_size) / float(win_size)
        baseline = np.convolve(data, kernel, mode="same")
        wander_range = float(np.ptp(baseline))

        return float(round(wander_range, 3))

    def _build_insufficient_data_result(
        self,
        reason: str,
        lead_name: str,
        sample_rate: int,
        sample_count: int,
        duration_sec: float = 0.0
    ) -> Dict[str, Any]:
        """Constructs safe INSUFFICIENT DATA payload."""
        checks_dict = {
            "finite_samples": {"passed": sample_count > 0, "invalid_count": 0},
            "flatline": {"passed": False, "ptp_mv": 0.0},
            "saturation": {"passed": False},
            "noise_and_drift": {"passed": False, "snr_db": 0.0, "baseline_drift_mv": 0.0},
            "duration": {"passed": False, "duration_sec": round(duration_sec, 2), "samples": sample_count}
        }
        checks_log = [
            {
                "name": "Data Completeness & Duration",
                "passed": False,
                "severity": "INSUFFICIENT_DATA",
                "measurement": f"{sample_count} samples ({duration_sec:.2f}s)",
                "threshold": f">= {self.min_duration_sec:.1f}s",
                "detail": reason
            }
        ]
        return {
            "status": QualityStatus.INSUFFICIENT_DATA,
            "quality_score": 0,
            "score": 0,
            "primary_reason": reason,
            "detected_issues": [reason],
            "is_suitable_for_classification": False,
            "is_suitable_for_analysis": False,
            "measurements": {
                "duration_sec": round(duration_sec, 2),
                "sample_count": sample_count,
                "sample_rate_hz": sample_rate,
                "invalid_samples_count": 0,
                "peak_to_peak_mv": 0.0,
                "variance": 0.0,
                "estimated_snr_db": 0.0,
                "baseline_wander_mv": 0.0,
                "lead_channel": lead_name
            },
            "configured_thresholds": {
                "min_duration_sec": self.min_duration_sec,
                "flatline_ptp_mv": self.flatline_ptp_threshold_mv,
                "min_snr_db": self.min_snr_db
            },
            "checks": checks_dict,
            "checks_log": checks_log,
            "disclaimer": "Technical signal-quality check only; does not establish clinical safety or replace physician ECG review."
        }

    def _build_poor_signal_result(
        self,
        primary_reason: str,
        detected_issues: List[str],
        checks_log: List[Dict[str, Any]],
        lead_name: str,
        sample_rate: int,
        sample_count: int,
        duration_sec: float,
        invalid_count: int,
        score: int = 10
    ) -> Dict[str, Any]:
        """Constructs safe POOR SIGNAL payload for severe corruption."""
        checks_dict = {
            "finite_samples": {"passed": invalid_count == 0, "invalid_count": invalid_count},
            "flatline": {"passed": not any("flatline" in i.lower() or "lead-off" in i.lower() for i in detected_issues)},
            "saturation": {"passed": not any("saturation" in i.lower() or "amplitude" in i.lower() for i in detected_issues)},
            "noise_and_drift": {"passed": not any("noise" in i.lower() or "wander" in i.lower() for i in detected_issues)},
            "duration": {"passed": sample_count >= int(self.min_duration_sec * sample_rate), "duration_sec": round(duration_sec, 2)}
        }
        return {
            "status": QualityStatus.POOR,
            "quality_score": score,
            "score": score,
            "primary_reason": primary_reason,
            "detected_issues": detected_issues,
            "is_suitable_for_classification": False,
            "is_suitable_for_analysis": False,
            "measurements": {
                "duration_sec": round(duration_sec, 2),
                "sample_count": sample_count,
                "sample_rate_hz": sample_rate,
                "invalid_samples_count": invalid_count,
                "peak_to_peak_mv": 0.0,
                "variance": 0.0,
                "estimated_snr_db": -5.0,
                "baseline_wander_mv": 0.0,
                "lead_channel": lead_name
            },
            "configured_thresholds": {
                "min_duration_sec": self.min_duration_sec,
                "flatline_ptp_mv": self.flatline_ptp_threshold_mv,
                "min_snr_db": self.min_snr_db
            },
            "checks": checks_dict,
            "checks_log": checks_log,
            "disclaimer": "Technical signal-quality check only; does not establish clinical safety or replace physician ECG review."
        }


# Global singleton instance
signal_quality_monitor = ECGSignalQualityMonitor()
