"""
Comprehensive Automated Test Suite for ECG Signal Quality Monitor
Validates technical quality checks:
1. Clean, valid ECG-like signal -> GOOD SIGNAL
2. Excessive noise / low SNR -> REVIEW SIGNAL or POOR SIGNAL
3. Flatline / lead-off signal -> POOR SIGNAL
4. Non-finite / NaN / Inf samples -> POOR SIGNAL
5. Empty recording -> INSUFFICIENT DATA
6. Short recording (< min_duration_sec) -> INSUFFICIENT DATA
7. Saturation / excessive amplitude -> POOR SIGNAL / REVIEW SIGNAL
8. MIT-BIH dataset recording integration
"""

import numpy as np
import pytest

from backend.signal_quality import (
    ECGSignalQualityMonitor,
    SignalQualityStatus,
    signal_quality_monitor
)
from backend.signal_processor import signal_processor
from backend.dataset_manager import dataset_manager


def generate_synthetic_clean_ecg(duration_sec=3.0, fs=250.0, hr_bpm=75.0):
    """Generates synthetic clean ECG-like signal using sinus rhythm simulation."""
    n_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    
    # Baseline + periodic QRS peaks and T waves
    rr_sec = 60.0 / hr_bpm
    ecg = np.zeros(n_samples, dtype=float)
    
    # Generate R peaks every rr_sec
    peak_times = np.arange(0.2, duration_sec, rr_sec)
    for pt in peak_times:
        # QRS complex (narrow biphasic spike ~80ms)
        qrs = 1.2 * np.exp(-((t - pt) ** 2) / (2 * (0.02 ** 2)))
        q_wave = -0.2 * np.exp(-((t - (pt - 0.03)) ** 2) / (2 * (0.015 ** 2)))
        s_wave = -0.3 * np.exp(-((t - (pt + 0.03)) ** 2) / (2 * (0.015 ** 2)))
        # T wave (broader positive deflection ~160ms)
        t_wave = 0.35 * np.exp(-((t - (pt + 0.22)) ** 2) / (2 * (0.06 ** 2)))
        # P wave
        p_wave = 0.15 * np.exp(-((t - (pt - 0.16)) ** 2) / (2 * (0.04 ** 2)))
        ecg += (qrs + q_wave + s_wave + t_wave + p_wave)
        
    return ecg


class TestECGSignalQualityMonitor:

    def setup_method(self):
        self.monitor = ECGSignalQualityMonitor(sample_rate=250.0)

    def test_clean_ecg_passes_as_good_signal(self):
        """A valid, clean ECG-like signal must receive GOOD SIGNAL status with passing checks."""
        clean_sig = generate_synthetic_clean_ecg(duration_sec=3.0, fs=250.0)
        res = self.monitor.assess_quality(clean_sig)

        assert res["status"] == SignalQualityStatus.GOOD_SIGNAL.value
        assert res["is_suitable_for_analysis"] is True
        assert res["score"] >= 80
        assert "checks passed" in res["primary_reason"].lower() or "suitable" in res["primary_reason"].lower()
        assert len(res["detected_issues"]) == 0
        assert res["checks"]["finite_samples"]["passed"] is True
        assert res["checks"]["flatline"]["passed"] is True
        assert res["checks"]["saturation"]["passed"] is True
        assert res["checks"]["duration"]["passed"] is True

    def test_flatline_signal_detected(self):
        """A zero or flat constant signal must trigger flatline check and POOR SIGNAL."""
        # Exact flatline
        flat_sig = np.zeros(750)
        res = self.monitor.assess_quality(flat_sig)

        assert res["status"] == SignalQualityStatus.POOR_SIGNAL.value
        assert res["is_suitable_for_analysis"] is False
        assert res["score"] <= 50
        assert res["checks"]["flatline"]["passed"] is False
        assert any("flatline" in issue.lower() or "lead-off" in issue.lower() for issue in res["detected_issues"])

    def test_near_flatline_minimal_amplitude(self):
        """A near-zero signal with PTP < threshold must fail flatline check."""
        near_flat = np.full(750, 0.02) + np.random.normal(0, 0.005, 750)
        res = self.monitor.assess_quality(near_flat)

        assert res["status"] in [SignalQualityStatus.POOR_SIGNAL.value, SignalQualityStatus.REVIEW_SIGNAL.value]
        assert res["checks"]["flatline"]["passed"] is False

    def test_excessive_noise_low_snr(self):
        """A signal dominated by high-frequency white noise must fail SNR check."""
        clean_sig = generate_synthetic_clean_ecg(duration_sec=3.0, fs=250.0)
        noisy_sig = clean_sig + np.random.normal(0, 2.0, len(clean_sig)) # heavy noise
        res = self.monitor.assess_quality(noisy_sig)

        assert res["status"] in [SignalQualityStatus.REVIEW_SIGNAL.value, SignalQualityStatus.POOR_SIGNAL.value]
        assert res["checks"]["noise_and_drift"]["snr_db"] < 10.0 or res["score"] < 75

    def test_severe_baseline_drift(self):
        """A signal with massive low-frequency wander must be flagged in baseline drift."""
        clean_sig = generate_synthetic_clean_ecg(duration_sec=3.0, fs=250.0)
        t = np.linspace(0, 3.0, len(clean_sig))
        drift = 3.5 * np.sin(2 * np.pi * 0.2 * t) # 3.5 mV swing at 0.2 Hz
        drifted_sig = clean_sig + drift
        res = self.monitor.assess_quality(drifted_sig)

        assert res["checks"]["noise_and_drift"]["baseline_drift_mv"] > 1.50
        assert res["checks"]["noise_and_drift"]["passed"] is False
        assert res["status"] in [SignalQualityStatus.REVIEW_SIGNAL.value, SignalQualityStatus.POOR_SIGNAL.value]

    def test_missing_or_non_finite_samples(self):
        """Samples containing NaN or Inf must produce POOR SIGNAL with safety flag."""
        sig_with_nan = generate_synthetic_clean_ecg(duration_sec=3.0, fs=250.0)
        sig_with_nan[100:105] = np.nan
        res = self.monitor.assess_quality(sig_with_nan)

        assert res["status"] == SignalQualityStatus.POOR_SIGNAL.value
        assert res["is_suitable_for_analysis"] is False
        assert res["checks"]["finite_samples"]["passed"] is False
        assert res["checks"]["finite_samples"]["invalid_count"] == 5

        # Also test Inf
        sig_with_inf = generate_synthetic_clean_ecg(duration_sec=3.0, fs=250.0)
        sig_with_inf[50] = np.inf
        res_inf = self.monitor.assess_quality(sig_with_inf)
        assert res_inf["status"] == SignalQualityStatus.POOR_SIGNAL.value
        assert res_inf["checks"]["finite_samples"]["passed"] is False

    def test_empty_recording(self):
        """Empty array must return INSUFFICIENT DATA without crashing."""
        empty_sig = np.array([])
        res = self.monitor.assess_quality(empty_sig)

        assert res["status"] == SignalQualityStatus.INSUFFICIENT_DATA.value
        assert res["is_suitable_for_analysis"] is False
        assert "insufficient" in res["primary_reason"].lower() or "empty" in res["primary_reason"].lower()

    def test_too_short_recording(self):
        """A signal shorter than minimum duration (1.0 sec = 250 samples) must be INSUFFICIENT DATA."""
        short_sig = np.ones(100) # 0.4 seconds at 250 Hz
        res = self.monitor.assess_quality(short_sig)

        assert res["status"] == SignalQualityStatus.INSUFFICIENT_DATA.value
        assert res["checks"]["duration"]["passed"] is False

    def test_saturation_and_excessive_amplitude(self):
        """A signal exceeding physical ADC limits or with clipped rails must be detected."""
        clean_sig = generate_synthetic_clean_ecg(duration_sec=3.0, fs=250.0)
        # Saturate signal to 10 mV
        sat_sig = clean_sig * 8.0
        res = self.monitor.assess_quality(sat_sig)

        assert res["checks"]["saturation"]["passed"] is False
        assert res["status"] in [SignalQualityStatus.POOR_SIGNAL.value, SignalQualityStatus.REVIEW_SIGNAL.value]

    def test_signal_processor_integration(self):
        """ECGSignalProcessor.evaluate_signal_quality should return quality_assessment and compatible keys."""
        clean_sig = generate_synthetic_clean_ecg(duration_sec=3.0, fs=250.0)
        res = signal_processor.evaluate_signal_quality(clean_sig)

        assert "technical_quality_status" in res
        assert "quality_assessment" in res
        assert "quality_score" in res
        assert "snr_db" in res
        assert res["technical_quality_status"] == SignalQualityStatus.GOOD_SIGNAL.value

    def test_extract_clinical_metrics_includes_quality(self):
        """extract_clinical_metrics must carry technical quality assessment."""
        clean_sig = generate_synthetic_clean_ecg(duration_sec=3.0, fs=250.0)
        peaks = signal_processor.pan_tompkins_qrs_detect(clean_sig)
        metrics = signal_processor.extract_clinical_metrics(clean_sig, peaks)

        assert "technical_quality_status" in metrics
        assert "technical_quality_reason" in metrics
        assert "quality_assessment" in metrics
        assert metrics["technical_quality_status"] == SignalQualityStatus.GOOD_SIGNAL.value

    def test_dataset_manager_record_analysis_has_signal_quality(self):
        """dataset_manager.get_record_analysis on MIT-BIH segment returns real signal_quality."""
        analysis = dataset_manager.get_record_analysis("100", start_sec=0.0, duration_sec=5.0)
        if analysis.get("success"):
            assert "signal_quality" in analysis
            sq = analysis["signal_quality"]
            assert "status" in sq
            assert "score" in sq
            assert sq["status"] in [s.value for s in SignalQualityStatus]
