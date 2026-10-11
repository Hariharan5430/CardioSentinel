"""
Unit & Algorithmic Tests for ECG Digital Signal Processor (DSP)
Covers:
- Butterworth Bandpass (0.5 - 40 Hz) & 50 Hz Notch filter stability
- Pan-Tompkins QRS detector on controlled synthetic and physiological signals
- Heart Rate (bpm) derivation and physiological range bounds
- HRV Time-Domain Biomarkers: SDNN (ms), RMSSD (ms), pNN50 (%)
- HRV Non-Linear Geometry: Poincare SD1, SD2, SD1/SD2 ratio, Shannon Entropy
- Insufficient data handling, flatline/lead-off detection, and high-frequency noise rejection
"""

import unittest
import numpy as np
from backend.signal_processor import ECGSignalProcessor, signal_processor

class TestSignalProcessingModule(unittest.TestCase):

    def setUp(self):
        self.sp = signal_processor
        self.fs = 250 # Standard 250 Hz sample rate

    def test_01_filter_stability_and_noise_suppression(self):
        """Verify Butterworth bandpass and notch filters attenuate baseline wander and 50 Hz powerline hum."""
        t = np.linspace(0, 4, 4 * self.fs)
        # ECG proxy (1.2 Hz) + Baseline wander (0.1 Hz) + 50 Hz powerline noise + high freq noise (80 Hz)
        clean_comp = 1.0 * np.sin(2 * np.pi * 1.2 * t)
        wander = 0.5 * np.sin(2 * np.pi * 0.1 * t)
        hum_50hz = 0.4 * np.sin(2 * np.pi * 50.0 * t)
        noise = 0.1 * np.random.randn(len(t))

        raw = clean_comp + wander + hum_50hz + noise
        filtered = self.sp.filter_signal(raw)

        self.assertEqual(len(filtered), len(raw))
        self.assertFalse(np.isnan(filtered).any())
        self.assertFalse(np.isinf(filtered).any())
        # Filtered output variance should be significantly reduced compared to noisy composite
        self.assertLess(np.var(filtered), np.var(raw))

    def test_02_pan_tompkins_synthetic_peaks(self):
        """Verify Pan-Tompkins detector locates QRS peaks on periodic synthetic pulse train."""
        # 5 seconds at 60 bpm -> exactly 5 beats (1 beat every 1.0 sec = 250 samples)
        duration_sec = 5.0
        total_samples = int(duration_sec * self.fs)
        signal = np.zeros(total_samples)

        expected_peaks = []
        for beat in range(5):
            center = int((beat + 0.5) * self.fs) # 125, 375, 625, 875, 1125
            expected_peaks.append(center)
            # Inject sharp QRS spike (Gaussian, sigma = 8 samples ~ 32ms)
            for i in range(max(0, center - 25), min(total_samples, center + 25)):
                signal[i] += 1.8 * np.exp(-0.5 * ((i - center) / 8.0) ** 2)

        filtered = self.sp.filter_signal(signal)
        detected_peaks = self.sp.pan_tompkins_qrs_detect(filtered)

        self.assertGreaterEqual(len(detected_peaks), 4)
        # Verify detected peak positions are within 15 samples (~60ms) of true peaks
        for exp in expected_peaks[1:]: # Skip first if filter warmup boundary
            matches = [p for p in detected_peaks if abs(p - exp) <= 15]
            self.assertTrue(len(matches) >= 1, f"Expected peak near {exp} was not detected")

    def test_03_heart_rate_and_rr_intervals(self):
        """Verify correct heart rate calculation and RR interval millisecond conversion."""
        # Known peak positions with constant RR = 200 samples at 250 Hz = 800 ms (75 bpm)
        peaks = np.array([200, 400, 600, 800, 1000, 1200, 1400])
        # Generate signal matching length
        dummy_signal = np.zeros(1600)
        for p in peaks:
            dummy_signal[p-5:p+5] = 1.5

        metrics = self.sp.extract_clinical_metrics(dummy_signal, peaks)
        
        self.assertIn("heart_rate", metrics)
        self.assertAlmostEqual(metrics["heart_rate"], 75.0, delta=2.0)
        self.assertAlmostEqual(metrics["rr_mean_ms"], 800.0, delta=10.0)

    def test_04_hrv_sdnn_rmssd_pnn50(self):
        """Verify SDNN, RMSSD, and pNN50 calculations using a known varying interval series."""
        # RR intervals in ms: [800, 860, 780, 890, 770, 880, 790, 850]
        # Successive differences: [+60, -80, +110, -120, +110, -90, +60] -> all > 50 ms!
        # Thus pNN50 should be 100%
        rr_ms_known = np.array([800, 860, 780, 890, 770, 880, 790, 850], dtype=float)
        peak_indices = [0]
        for rr in rr_ms_known:
            peak_indices.append(int(peak_indices[-1] + (rr * self.fs / 1000.0)))
        peaks = np.array(peak_indices)
        
        dummy_signal = np.zeros(peaks[-1] + 100)
        metrics = self.sp.extract_clinical_metrics(dummy_signal, peaks)

        self.assertGreater(metrics["hrv_sdnn"], 30.0)
        self.assertGreater(metrics["hrv_rmssd"], 40.0)
        self.assertEqual(metrics["pnn50"], 100.0) # all differences > 50ms

    def test_05_poincare_geometry_and_entropy(self):
        """Verify Poincare plot SD1, SD2, and Shannon entropy metrics are generated."""
        peaks = np.array([100, 300, 520, 710, 930, 1120, 1340, 1530, 1750, 1940])
        dummy_signal = np.zeros(2100)
        metrics = self.sp.extract_clinical_metrics(dummy_signal, peaks)

        self.assertIn("poincare_sd1", metrics)
        self.assertIn("poincare_sd2", metrics)
        self.assertIn("rr_entropy", metrics)
        self.assertGreater(metrics["poincare_sd1"], 0.0)
        self.assertGreater(metrics["poincare_sd2"], 0.0)
        self.assertGreater(metrics["rr_entropy"], 0.0)

    def test_06_insufficient_data_and_flatline(self):
        """Verify robust fallback to safe defaults on zero peaks or flatline signal."""
        flatline = np.zeros(1000)
        peaks_none = np.array([])
        metrics = self.sp.extract_clinical_metrics(flatline, peaks_none)

        self.assertEqual(metrics["heart_rate"], 0.0)
        self.assertEqual(metrics["hrv_sdnn"], 0.0)
        self.assertEqual(metrics["hrv_rmssd"], 0.0)
        self.assertEqual(metrics["pnn50"], 0.0)

    def test_07_signal_quality_and_leadoff(self):
        """Verify signal quality indicator flags flatline as POOR / LEAD-OFF."""
        flatline = np.zeros(500)
        sq_flat = self.sp.assess_signal_quality(flatline)
        self.assertIn(sq_flat["status"], ["POOR", "LEAD_OFF"])
        self.assertLess(sq_flat["score"], 0.5)

        # High quality sinusoidal pulse train
        t = np.linspace(0, 2, 500)
        good_ecg = 1.0 * np.sin(2 * np.pi * 1.5 * t)
        sq_good = self.sp.assess_signal_quality(good_ecg)
        self.assertGreaterEqual(sq_good["score"], 0.5)

if __name__ == "__main__":
    unittest.main()
