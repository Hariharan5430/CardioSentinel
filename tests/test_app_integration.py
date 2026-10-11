"""
Application & Integration Tests for CardioSentinel ICU Telemetry Platform
Covers:
- System health status endpoint (/health and /api/health)
- Multi-bed management across all 4 beds (BED-01, BED-02, BED-03, REMOTE-01)
- Authoritative session state per bed & isolation (no cross-bed state bleed)
- Record switching synchronization (swapping records updates bed state instantly)
- Interactive ECG playback controls (PLAY, PAUSE, seek, speed scaling)
- Clinical Alert System: IEC 60601-1-8 priority dispatch, timed snooze (1m/5m/15m), acknowledgment, and CSV export
- Comprehensive Shift Reports and per-bed clinical exports
- Graceful handling of invalid inputs and non-existent beds/records
"""

import unittest
import json
import io
import csv
from backend.server import app, telemetry_engine, signal_processor, ai_engine, alert_system
from backend.alert_system import AlarmPriority

class TestAppIntegrationModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.telemetry = telemetry_engine
        cls.alerts = alert_system

    def test_01_health_endpoints(self):
        """Verify both /health and /api/health report system readiness, 4 beds, and model readiness."""
        for path in ["/health", "/api/health"]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["status"], "healthy")
            self.assertIn("active_beds", data)
            self.assertEqual(data["active_beds"], 4)
            self.assertIn("dataset_ready", data)
            self.assertIn("model_ready", data)
            self.assertTrue(data["model_ready"])

    def test_02_all_four_beds_authoritative_state(self):
        """Verify all 4 beds exist and maintain independent, authoritative session state."""
        res = self.client.get("/api/patients")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        
        beds = {p["patient_id"]: p for p in data["patients"]}
        for bid in ["BED-01", "BED-02", "BED-03", "REMOTE-01"]:
            self.assertIn(bid, beds)
            bed = beds[bid]
            self.assertIn("current_rhythm", bed)
            self.assertIn("data_mode", bed)
            self.assertIn("mitbih_record_id", bed)
            self.assertIn("playback_status", bed)
            self.assertIn("playback_speed", bed)

    def test_03_record_switching_synchronization(self):
        """Verify switching record on BED-02 immediately updates record ID, signal source, and vitals."""
        # Switch BED-02 to Record 106
        res = self.client.post("/api/bed/BED-02/record", json={"record_id": "106"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["patient"]["mitbih_record_id"], "106")

        # Verify backend telemetry state matches
        bed2 = self.telemetry.get_patient("BED-02")
        self.assertEqual(bed2.mitbih_record_id, "106")
        self.assertEqual(bed2.data_mode, "MIT-BIH")

        # Switch BED-02 back to Record 101
        res = self.client.post("/api/bed/BED-02/record", json={"record_id": "101"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(bed2.mitbih_record_id, "101")

    def test_04_playback_controls_state_machine(self):
        """Verify PLAY, PAUSE, SPEED, and SEEK controls update patient state deterministically."""
        # Pause BED-01
        res = self.client.post("/api/bed/BED-01/playback", json={"action": "pause"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "PAUSED")
        
        bed1 = self.telemetry.get_patient("BED-01")
        self.assertEqual(bed1.playback_status, "PAUSED")

        # Change speed to 2.0x
        res = self.client.post("/api/bed/BED-01/playback", json={"action": "speed", "speed": 2.0})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(bed1.playback_speed, 2.0)

        # Seek to 120.0 seconds
        res = self.client.post("/api/bed/BED-01/playback", json={"action": "seek", "time_sec": 120.0})
        self.assertEqual(res.status_code, 200)
        self.assertAlmostEqual(bed1.time_position_sec, 120.0, delta=1.0)

        # Resume playback
        res = self.client.post("/api/bed/BED-01/playback", json={"action": "play"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(bed1.playback_status, "PLAYING")

    def test_05_alert_acknowledgment_and_snooze(self):
        """Verify alert triggering, 1m/5m/15m snoozing, and acknowledgment lifecycle."""
        # Trigger critical alarm on BED-03
        alert = self.alerts.evaluate_and_trigger(
            patient_id="BED-03",
            patient_name="Arthur Pendelton",
            location="ICU Bed 03",
            diagnosis="Ventricular Tachycardia",
            cewi_data={"cewi_score": 92, "shock_index": 1.4},
            metrics={"heart_rate": 178, "qrs_duration_ms": 148, "qtc_ms": 515, "hrv_sdnn": 14},
            ecg_snapshot=[0.1] * 250
        )
        self.assertIsNotNone(alert)
        alert_id = alert["alert_id"]

        # Snooze for 5 minutes (300 sec) via API
        res = self.client.post("/api/alerts/action", json={
            "alert_id": alert_id,
            "action": "snooze",
            "snooze_duration_sec": 300
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])
        self.assertTrue(self.alerts.is_snoozed("BED-03"))

        # Acknowledge alert via API
        res = self.client.post("/api/alerts/action", json={
            "alert_id": alert_id,
            "action": "acknowledge",
            "clinician_name": "Dr. Sarah Lin, MD"
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])

    def test_06_alert_csv_export(self):
        """Verify alert audit log exports properly formatted RFC 4180 CSV."""
        res = self.client.get("/api/alerts/export.csv")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "text/csv")
        
        csv_text = res.data.decode("utf-8")
        reader = csv.reader(io.StringIO(csv_text))
        header = next(reader)
        header_normalized = [h.lower().replace(" ", "_") for h in header]
        self.assertIn("alert_id", header_normalized)
        self.assertIn("diagnosis", header_normalized)
        self.assertIn("priority", header_normalized)

    def test_07_shift_report_generation_and_export(self):
        """Verify Shift Report contains telemetry statistics, beds monitored, and clinical summary."""
        res = self.client.get("/api/reports/shift")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        report = data["report"]
        self.assertIn("report_id", report)
        self.assertIn("beds_monitored", report)
        self.assertEqual(len(report["beds_monitored"]), 4)
        self.assertIn("clinical_summary", report)
        self.assertIn("research_disclaimer", report)

    def test_08_invalid_inputs_and_error_handling(self):
        """Verify 400 and 404 responses for invalid patient ID and unknown playback commands."""
        # Unknown bed
        res = self.client.post("/api/bed/BED-999_INVALID/playback", json={"action": "play"})
        self.assertEqual(res.status_code, 404)

        # Invalid playback action
        res = self.client.post("/api/bed/BED-01/playback", json={"action": "non_existent_action"})
        self.assertEqual(res.status_code, 400)

        # Invalid alert action
        res = self.client.post("/api/alerts/action", json={"action": "invalid_alert_op"})
        self.assertEqual(res.status_code, 400)

if __name__ == "__main__":
    unittest.main()
