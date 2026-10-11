"""
Comprehensive Integration & Unit Test Suite for CardioSentinel Telemetry System
Tests:
1. Multi-Patient Telemetry Signal Generator
2. Digital Filtering & Pan-Tompkins QRS Detection
3. AI Arrhythmia Machine Learning Classifier
4. Cardiac Early Warning Index (CEWI) Synthesis
5. IEC 60601-1-8 Clinical Alarm System
6. REST API Endpoints & Telemetry Streaming Server
"""

import unittest
import numpy as np
import time

from backend.telemetry_engine import TelemetryEngine, CardiacRhythmType, PatientTelemetryState
from backend.signal_processor import ECGSignalProcessor
from backend.ai_engine import AIEarlyWarningEngine
from backend.alert_system import ClinicalAlertSystem, AlarmPriority
from backend.server import app, telemetry_engine, signal_processor, ai_engine, alert_system

class TestCardioSentinelPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.telemetry_engine = telemetry_engine
        cls.signal_processor = signal_processor
        cls.ai_engine = ai_engine
        cls.alert_system = alert_system
        cls.test_client = app.test_client()

    def test_01_telemetry_engine_initialization(self):
        """Verify all default hospital beds and remote patients are initialized."""
        patients = self.telemetry_engine.patients
        self.assertIn("BED-01", patients)
        self.assertIn("BED-02", patients)
        self.assertIn("BED-03", patients)
        self.assertIn("REMOTE-01", patients)
        
        bed1 = self.telemetry_engine.get_patient("BED-01")
        self.assertEqual(bed1.name, "Eleanor Vance")
        self.assertEqual(bed1.sample_rate, 250)

    def test_02_telemetry_signal_generation(self):
        """Verify 250 Hz signal output and physiological vitals dictionary."""
        patient = self.telemetry_engine.get_patient("BED-03")
        sample = patient.generate_next_sample()
        
        required_keys = ["lead2", "lead_v1", "is_r_peak", "patient_id", "rhythm", "hr", "spo2", "bp_sys", "bp_dia"]
        for key in required_keys:
            self.assertIn(key, sample)
            
        self.assertIsInstance(sample["lead2"], float)
        self.assertIsInstance(sample["lead_v1"], float)
        self.assertGreater(sample["hr"], 30)

    def test_03_arrhythmia_trigger_and_reversion(self):
        """Verify event injection and transient duration mechanics."""
        success = self.telemetry_engine.trigger_event("BED-03", CardiacRhythmType.VT, duration=10.0)
        self.assertTrue(success)
        
        patient = self.telemetry_engine.get_patient("BED-03")
        self.assertEqual(patient.current_rhythm, CardiacRhythmType.VT)
        self.assertGreater(patient.override_duration, 0)

        # Restore baseline
        self.telemetry_engine.trigger_event("BED-03", CardiacRhythmType.NSR, duration=0)
        self.assertEqual(patient.current_rhythm, CardiacRhythmType.NSR)

    def test_04_signal_filtering_stability(self):
        """Verify Butterworth bandpass and notch filters do not explode or produce NaNs."""
        t = np.linspace(0, 2, 500)
        raw_ecg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(500)
        filtered = self.signal_processor.filter_signal(raw_ecg)
        
        self.assertEqual(len(filtered), len(raw_ecg))
        self.assertFalse(np.isnan(filtered).any())
        self.assertFalse(np.isinf(filtered).any())

    def test_05_pan_tompkins_qrs_detection(self):
        """Verify Pan-Tompkins detector identifies R-peaks on synthesized ECG."""
        patient = PatientTelemetryState("TEST-01", "Test Subject", 50, "Male", "Test Lab", CardiacRhythmType.NSR)
        for _ in range(750): # 3 seconds of data
            patient.generate_next_sample()
            
        signal = np.array(patient.lead2_buffer)
        filtered = self.signal_processor.filter_signal(signal)
        peaks = self.signal_processor.pan_tompkins_qrs_detect(filtered)
        
        # In 3 seconds at 72 bpm, expect ~3-4 beats
        self.assertGreaterEqual(len(peaks), 2)
        
        # Verify clinical metrics extraction
        metrics = self.signal_processor.extract_clinical_metrics(filtered, peaks)
        self.assertIn("heart_rate", metrics)
        self.assertIn("qrs_duration_ms", metrics)
        self.assertIn("qtc_ms", metrics)
        self.assertIn("hrv_sdnn", metrics)
        self.assertGreater(metrics["heart_rate"], 40)
        self.assertLess(metrics["heart_rate"], 120)

    def test_06_ai_model_arrhythmia_classification(self):
        """Verify AI model performs inference and returns valid probabilities."""
        metrics = {
            "heart_rate": 72,
            "hrv_sdnn": 45.0,
            "hrv_rmssd": 38.0,
            "pnn50": 18.0,
            "rr_cv": 0.06,
            "qrs_duration_ms": 86.0,
            "qtc_ms": 408.0,
            "rr_entropy": 0.65
        }
        dummy_signal = np.sin(np.linspace(0, 10, 500))
        result = self.ai_engine.classify_rhythm(metrics, dummy_signal)
        
        self.assertIn("diagnosis", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(len(result["probabilities"]), 9)

    def test_07_cewi_early_warning_synthesis(self):
        """Verify CEWI score calculates risk and shock index appropriately."""
        # Stable baseline case
        vitals_stable = {"hr": 72, "bp_sys": 120, "bp_dia": 80, "spo2": 98, "resp_rate": 16}
        metrics_stable = {"qrs_duration_ms": 85, "qtc_ms": 410, "hrv_sdnn": 45}
        cewi_stable = self.ai_engine.compute_cewi_score(vitals_stable, metrics_stable, "Normal Sinus Rhythm")
        
        self.assertLess(cewi_stable["cewi_score"], 35)
        self.assertIn("LOW RISK", cewi_stable["risk_level"])
        self.assertAlmostEqual(cewi_stable["shock_index"], 0.60, delta=0.05)

        # Critical collapse case (Hypotension, Tachycardia, Shock Index > 1.0, prolonged QTc)
        vitals_crit = {"hr": 165, "bp_sys": 80, "bp_dia": 50, "spo2": 88, "resp_rate": 28}
        metrics_crit = {"qrs_duration_ms": 145, "qtc_ms": 510, "hrv_sdnn": 12}
        cewi_crit = self.ai_engine.compute_cewi_score(vitals_crit, metrics_crit, "Ventricular Tachycardia")
        
        self.assertGreaterEqual(cewi_crit["cewi_score"], 80)
        self.assertIn("CRITICAL", cewi_crit["risk_level"])
        self.assertGreater(cewi_crit["shock_index"], 1.0)

    def test_08_clinical_alert_system_and_silencing(self):
        """Verify tiered alarm dispatch and IEC 60601-1-8 120s silence timer."""
        alert_sys = ClinicalAlertSystem()
        
        cewi_crit = {"cewi_score": 92, "shock_index": 1.4}
        metrics_crit = {"heart_rate": 180, "qrs_duration_ms": 150, "qtc_ms": 510, "hrv_sdnn": 15}
        
        alert = alert_sys.evaluate_and_trigger(
            patient_id="BED-01",
            patient_name="Eleanor Vance",
            location="ICU Bed 01",
            diagnosis="Ventricular Tachycardia",
            cewi_data=cewi_crit,
            metrics=metrics_crit,
            ecg_snapshot=[0.1] * 300
        )
        
        self.assertIsNotNone(alert)
        self.assertEqual(alert["priority"], AlarmPriority.CRISIS)
        self.assertTrue(alert["audio_alarm"])
        self.assertIn("channels", alert["dispatch_status"])

        # Test silencing
        alert_sys.silence_patient("BED-01", duration_sec=120.0)
        self.assertIn("BED-01", alert_sys.silenced_patients)
        self.assertFalse(alert_sys.active_alerts["BED-01"]["audio_alarm"])

        # Test acknowledgment
        ack_res = alert_sys.acknowledge_alert(alert["alert_id"], clinician_name="Dr. Vance")
        self.assertTrue(ack_res)

    def test_09_api_endpoints(self):
        """Verify all REST API endpoints return 200 and proper JSON structures."""
        # GET /
        res = self.test_client.get("/")
        self.assertEqual(res.status_code, 200)

        # GET /api/health
        res = self.test_client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        health_data = res.get_json()
        self.assertEqual(health_data["status"], "healthy")
        self.assertEqual(health_data["num_patients"], 4)

        # GET /api/patients
        res = self.test_client.get("/api/patients")
        self.assertEqual(res.status_code, 200)
        patients_data = res.get_json()
        self.assertTrue(patients_data["success"])
        self.assertEqual(len(patients_data["patients"]), 4)

        # GET /api/patient/BED-01
        res = self.test_client.get("/api/patient/BED-01")
        self.assertEqual(res.status_code, 200)
        pt_data = res.get_json()
        self.assertTrue(pt_data["success"])
        self.assertEqual(pt_data["patient"]["patient_id"], "BED-01")

        # POST /api/trigger_arrhythmia
        res = self.test_client.post("/api/trigger_arrhythmia", json={
            "patient_id": "BED-01",
            "rhythm": "Atrial Fibrillation",
            "duration": 15.0
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])

        # GET /api/export_report/BED-01
        res = self.test_client.get("/api/export_report/BED-01")
        self.assertEqual(res.status_code, 200)
        rpt = res.get_json()
        self.assertTrue(rpt["success"])
        self.assertIn("clinical_summary", rpt["report"])

        # GET /api/alerts
        res = self.test_client.get("/api/alerts")
        self.assertEqual(res.status_code, 200)
        alerts_data = res.get_json()
        self.assertTrue(alerts_data["success"])

    def test_10_mitbih_dataset_manager(self):
        """Verify PhysioNet MIT-BIH dataset manager, records, and extracted biomarkers."""
        from backend.dataset_manager import dataset_manager
        summary = dataset_manager.get_dataset_summary()
        self.assertIn("records", summary)
        self.assertIn("100", summary["records"])
        self.assertIn("200", summary["records"])
        
        # Verify signal stream
        stream = dataset_manager.get_signal_stream("100")
        self.assertIsNotNone(stream)
        self.assertIn("lead2", stream)
        self.assertGreater(len(stream["lead2"]), 1000)

    def test_11_dataset_api_endpoints(self):
        """Verify REST API endpoints for PhysioNet dataset metadata and mode toggling."""
        # GET /api/dataset/summary
        res = self.test_client.get("/api/dataset/summary")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("dataset_name", data["dataset"])

        # POST /api/dataset/set_mode (single patient)
        res = self.test_client.post("/api/dataset/set_mode", json={
            "patient_id": "BED-01",
            "mode": "MIT-BIH",
            "record_id": "100"
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])

        bed1 = self.telemetry_engine.get_patient("BED-01")
        self.assertEqual(bed1.data_mode, "MIT-BIH")
        
        # Switch back to SYNTHETIC
        res = self.test_client.post("/api/dataset/set_mode", json={
            "patient_id": "BED-01",
            "mode": "SYNTHETIC"
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(bed1.data_mode, "SYNTHETIC")

    def test_12_mitbih_ai_training(self):
        """Verify training and validation of ML model on real PhysioNet extracted biomarkers."""
        res = self.test_client.post("/api/dataset/train_mitbih")
        self.assertEqual(res.status_code, 200)
        result = res.get_json()
        self.assertTrue(result["success"])
        self.assertGreater(result["samples_total"], 50)
        self.assertGreater(result["test_accuracy"], 70.0)

if __name__ == "__main__":
    unittest.main()
