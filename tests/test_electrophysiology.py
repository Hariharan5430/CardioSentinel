"""
Tests for View 4: Current Bed Electrophysiology Breakdown
Validates:
1. Endpoint /api/patient/<patient_id>/electrophysiology responses and data structures.
2. Default /api/electrophysiology query routing.
3. Distinction between historical PhysioNet MIT-BIH recordings and live telemetry.
4. Non-ECG channels (BP, SpO2, Temperature) are explicitly marked unavailable with scientific rationale.
5. Signal suitability, flatline/lead-off detection, and avoidance of fabricated values.
6. Algorithmic fiducials (Pan-Tompkins QRS, Bazett/Fridericia QTc, HRV biomarkers).
7. Informational research disclaimer present on all payloads.
8. Frontend DOM and script integration hooks.
"""

import unittest
import numpy as np
from backend.server import app, telemetry_engine, build_electrophysiology_payload
from backend.signal_processor import signal_processor

class TestElectrophysiologyBreakdown(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_01_electrophysiology_endpoint_valid_patient(self):
        """Verify /api/patient/BED-01/electrophysiology returns comprehensive validated payload."""
        response = self.client.get("/api/patient/BED-01/electrophysiology")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertTrue(data["success"])
        self.assertEqual(data["patient_id"], "BED-01")
        self.assertIn("patient_name", data)
        self.assertIn("signal_quality", data)
        self.assertIn("measurements", data)
        self.assertIn("research_disclaimer", data)

        m = data["measurements"]
        self.assertIn("heart_rate", m)
        self.assertIn("qrs_duration", m)
        self.assertIn("qt_interval", m)
        self.assertIn("hrv", m)

        # Heart rate
        self.assertEqual(m["heart_rate"]["unit"], "bpm")
        self.assertIn("reference_range", m["heart_rate"])

        # QRS Duration
        self.assertEqual(m["qrs_duration"]["unit"], "ms")
        self.assertIn("Pan-Tompkins", m["qrs_duration"]["method"])

        # QT interval
        self.assertEqual(m["qt_interval"]["unit"], "ms")
        self.assertIn("qtc_bazett_ms", m["qt_interval"])
        self.assertIn("qtc_fridericia_ms", m["qt_interval"])
        self.assertIn("formulas", m["qt_interval"])

        # HRV
        self.assertIn("sdnn", m["hrv"])
        self.assertIn("rmssd", m["hrv"])
        self.assertIn("pnn50", m["hrv"])
        self.assertIn("poincare_sd1", m["hrv"])
        self.assertIn("poincare_sd2", m["hrv"])
        self.assertIn("shannon_entropy", m["hrv"])

    def test_02_electrophysiology_endpoint_404(self):
        """Verify 404 response for non-existent patient ID."""
        response = self.client.get("/api/patient/NON_EXISTENT_BED/electrophysiology")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_03_default_electrophysiology_route(self):
        """Verify /api/electrophysiology handles default bed and query param ?patient_id=BED-02."""
        res_default = self.client.get("/api/electrophysiology")
        self.assertEqual(res_default.status_code, 200)
        self.assertTrue(res_default.get_json()["success"])

        res_bed2 = self.client.get("/api/electrophysiology?patient_id=BED-02")
        self.assertEqual(res_bed2.status_code, 200)
        data2 = res_bed2.get_json()
        self.assertEqual(data2["patient_id"], "BED-02")

    def test_04_historical_mitbih_labeling_and_omitted_channels(self):
        """Verify PhysioNet MIT-BIH recordings are clearly labeled as historical and do NOT invent BP/SpO2."""
        response = self.client.get("/api/patient/BED-01/electrophysiology")
        data = response.get_json()

        if data["data_mode"] == "MIT-BIH":
            self.assertTrue(data["is_historical_recorded"])
            self.assertIn("RECORDED-DATA ANALYSIS", data["mode_label"])
            self.assertIn("PhysioNet MIT-BIH Record", data["mode_label"])

            # Non-ECG channels omitted by design
            unavailable = data.get("unavailable_measurements", [])
            self.assertGreaterEqual(len(unavailable), 2)
            param_names = [u["parameter"] for u in unavailable]
            self.assertTrue(any("Blood Pressure" in p for p in param_names))
            self.assertTrue(any("Pulse Oximetry" in p or "SpO2" in p for p in param_names))

            for u in unavailable:
                self.assertEqual(u["status"], "Omitted by Design")
                self.assertIn("PhysioNet MIT-BIH", u["reason"])

    def test_05_flatline_unsuitable_signal_no_fabricated_values(self):
        """Verify that on flatline/lead-off signal, measurements are unmeasurable rather than fabricated."""
        flatline = np.zeros(750)
        peaks = []
        metrics = signal_processor.extract_clinical_metrics(flatline, peaks)

        self.assertFalse(metrics["is_suitable"])
        self.assertFalse(metrics["hr_measurable"])
        self.assertFalse(metrics["qrs_measurable"])
        self.assertFalse(metrics["qt_measurable"])
        self.assertFalse(metrics["hrv_measurable"])

        # Crucial requirement: Must NOT fabricate 85ms QRS or 410ms QTc on flatline
        self.assertEqual(metrics["heart_rate"], 0)
        self.assertEqual(metrics["qrs_duration_ms"], 0.0)
        self.assertEqual(metrics["qt_ms"], 0.0)
        self.assertEqual(metrics["qtc_bazett_ms"], 0.0)
        self.assertEqual(metrics["hrv_sdnn"], 0.0)

        # Clear reasons provided
        reasons = metrics["unmeasurable_reasons"]
        self.assertIn("heart_rate", reasons)
        self.assertIn("qrs_duration", reasons)
        self.assertIn("qt_interval", reasons)
        self.assertIn("hrv", reasons)

    def test_06_research_disclaimer_present(self):
        """Verify informational research disclaimer is explicitly included."""
        payload = build_electrophysiology_payload("BED-01")
        self.assertIsNotNone(payload)
        self.assertIn("research_disclaimer", payload)
        self.assertIn("Research", payload["research_disclaimer"])
        self.assertIn("clinical diagnosis", payload["research_disclaimer"].lower())

    def test_07_frontend_dom_elements_exist(self):
        """Verify index.html contains the necessary electrophysiology elements and controls."""
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn('id="analysis-bed-breakdown"', html)
        self.assertIn('id="analysis-bed-select"', html)
        self.assertIn('id="btn-analysis-refresh"', html)
        self.assertIn('id="analysis-source-badge"', html)

    def test_08_frontend_js_wiring_exists(self):
        """Verify static/js/app.js contains electrophysiology methods and routing."""
        with open("static/js/app.js", "r", encoding="utf-8") as f:
            js = f.read()

        self.assertIn("renderElectrophysiologyBreakdown", js)
        self.assertIn("displayElectrophysiologyData", js)
        self.assertIn("populateAnalysisBedDropdown", js)
        self.assertTrue(('switchView("analysis")' in js) or ('viewName === "analysis"' in js))
        self.assertIn("/api/patient/", js)
        self.assertIn("/electrophysiology", js)

if __name__ == "__main__":
    unittest.main()
