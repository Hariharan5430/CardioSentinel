"""
CardioSentinel - Unit & Integration Tests for ECG Record Browser Classification
Tests the verified MIT-BIH reference annotation classification:
1. Normal Record Classification (exclusively normal beats, e.g. Record 115, Record 122)
2. Abnormal Record Classification (verified arrhythmia beats/rhythms, e.g. Records 100, 101, 102, 106, 207)
3. Unclassified Handling (missing, unsupported, or insufficient annotations)
4. Record Browser Filters (All, Normal, Abnormal, Unclassified)
5. Independent Separation of Waveform, Reference Annotations, and Random Forest ML Prediction
6. Safety Boundary: Reference classification does not trigger patient emergency alarms or SMS
"""

import unittest
import numpy as np
from backend.dataset_manager import (
    MITBIHDatasetManager,
    dataset_manager,
    NORMAL_BEAT_SYMBOLS,
    ABNORMAL_BEAT_SYMBOLS,
    UNCLASSIFIED_BEAT_SYMBOLS
)
from backend.server import app
from backend.alert_system import alert_system
from backend.database import get_db_connection

class TestRecordClassification(unittest.TestCase):

    def setUp(self):
        self.dm = dataset_manager
        self.client = app.test_client()

    def test_01_normal_record_classification(self):
        """Verify records with only normal reference beats are classified as Normal."""
        normal_records = ["115", "122"]
        for rec_id in normal_records:
            meta = self.dm.get_record_metadata(rec_id)
            self.assertIsNotNone(meta, f"Metadata missing for Record {rec_id}")

            classif = self.dm.classify_record(meta)
            self.assertEqual(classif["classification"], "NORMAL")
            self.assertEqual(classif["classification_label"], "Normal")
            self.assertEqual(classif["badge_class"], "stable")
            self.assertIn("N", classif["reason"])
            self.assertIn("exclusively normal", classif["reason"].lower())

            # Verify actual numbers
            sc = meta.get("symbol_counts", {})
            self.assertGreater(sc.get("N", 0), 0)
            for abn_sym in ABNORMAL_BEAT_SYMBOLS:
                self.assertEqual(sc.get(abn_sym, 0), 0, f"Record {rec_id} contains unexpected {abn_sym}")

            # Print actual verified dataset results
            print(f"\n[TEST RESULT] Record ID: {rec_id}")
            print(f"  Reference Classification: {classif['classification_label']}")
            print(f"  Reason: {classif['reason']}")
            print(f"  Patient Info: {meta.get('patient_summary')}")

    def test_02_abnormal_record_classification_ectopic_beats(self):
        """Verify records with verified ectopic or arrhythmia beats are classified as Abnormal."""
        # Test records with specific known pathologies
        test_cases = [
            ("100", "A", "Atrial Premature Beat"),
            ("101", "A", "Atrial Premature Beat"),
            ("105", "V", "Premature Ventricular Contraction"),
            ("106", "V", "Premature Ventricular Contraction"),
            ("109", "L", "Left Bundle Branch Block"),
            ("118", "R", "Right Bundle Branch Block"),
            ("207", "!", "Ventricular Flutter Wave"),
        ]

        for rec_id, expected_symbol, expected_desc in test_cases:
            meta = self.dm.get_record_metadata(rec_id)
            self.assertIsNotNone(meta, f"Metadata missing for Record {rec_id}")

            classif = self.dm.classify_record(meta)
            self.assertEqual(classif["classification"], "ABNORMAL", f"Record {rec_id} should be classified as ABNORMAL")
            self.assertEqual(classif["classification_label"], "Abnormal")
            self.assertEqual(classif["badge_class"], "crisis")
            self.assertIn("verified abnormal", classif["reason"].lower())
            self.assertIn(expected_symbol, classif["details"]["abnormal_symbols"])

            # Print actual verified dataset results
            print(f"\n[TEST RESULT] Record ID: {rec_id}")
            print(f"  Reference Classification: {classif['classification_label']}")
            print(f"  Reason: {classif['reason']}")
            print(f"  Abnormal Beats Found: {classif['details']['abnormal_symbols']}")

    def test_03_abnormal_record_classification_paced(self):
        """Verify pacemaker records with paced beats are classified as Abnormal."""
        paced_cases = ["102", "104", "107", "217"]
        for rec_id in paced_cases:
            meta = self.dm.get_record_metadata(rec_id)
            self.assertIsNotNone(meta)
            classif = self.dm.classify_record(meta)
            self.assertEqual(classif["classification"], "ABNORMAL")
            self.assertEqual(classif["classification_label"], "Abnormal")
            
            # Print actual verified dataset results
            print(f"\n[TEST RESULT] Paced Record ID: {rec_id}")
            print(f"  Reference Classification: {classif['classification_label']}")
            print(f"  Reason: {classif['reason']}")

    def test_04_unclassified_handling_missing_or_insufficient(self):
        """Verify records with missing or insufficient annotations are classified as Unclassified without guessing."""
        # Case 1: Missing annotations flag
        dummy_unannotated = {
            "record_id": "DUMMY_NO_ANN",
            "has_annotations": False,
            "beat_count": 0,
            "symbol_counts": {},
            "rhythms": []
        }
        res1 = self.dm.classify_record(dummy_unannotated)
        self.assertEqual(res1["classification"], "UNCLASSIFIED")
        self.assertEqual(res1["classification_label"], "Unclassified")
        self.assertEqual(res1["badge_class"], "advisory")
        self.assertIn("missing, unsupported, or insufficient", res1["reason"].lower())

        # Case 2: Only unclassifiable 'Q' symbols without normal or verified abnormal beats
        dummy_q_only = {
            "record_id": "DUMMY_Q_ONLY",
            "has_annotations": True,
            "beat_count": 2,
            "symbol_counts": {"Q": 2},
            "rhythms": []
        }
        res2 = self.dm.classify_record(dummy_q_only)
        self.assertEqual(res2["classification"], "UNCLASSIFIED")
        self.assertEqual(res2["classification_label"], "Unclassified")

        # Case 3: Empty metadata
        res3 = self.dm.classify_record({})
        self.assertEqual(res3["classification"], "UNCLASSIFIED")

        print(f"\n[TEST RESULT] Unclassified Cases:")
        print(f"  Missing Annotations: {res1['classification_label']} -> {res1['reason']}")
        print(f"  Unclassifiable Symbols Only: {res2['classification_label']} -> {res2['reason']}")

    def test_05_record_filters_api(self):
        """Verify API filtering by All, Normal, Abnormal, and Unclassified."""
        # 1. Filter: ALL
        res_all = self.client.get("/api/dataset/records?filter=ALL")
        self.assertEqual(res_all.status_code, 200)
        data_all = res_all.get_json()
        self.assertTrue(data_all["success"])
        self.assertEqual(data_all["total"], 48)
        self.assertEqual(len(data_all["records"]), 48)

        # Verify counts in payload
        counts = data_all.get("counts", {})
        self.assertEqual(counts.get("all"), 48)
        self.assertEqual(counts.get("normal"), 2)
        self.assertEqual(counts.get("abnormal"), 46)
        self.assertEqual(counts.get("unclassified"), 0)

        # 2. Filter: NORMAL
        res_norm = self.client.get("/api/dataset/records?filter=NORMAL")
        self.assertEqual(res_norm.status_code, 200)
        data_norm = res_norm.get_json()
        self.assertEqual(len(data_norm["records"]), 2)
        for r in data_norm["records"]:
            self.assertEqual(r["record_classification"], "Normal")
            self.assertIn(r["record_id"], ["115", "122"])

        # 3. Filter: ABNORMAL
        res_abn = self.client.get("/api/dataset/records?filter=ABNORMAL")
        self.assertEqual(res_abn.status_code, 200)
        data_abn = res_abn.get_json()
        self.assertEqual(len(data_abn["records"]), 46)
        for r in data_abn["records"]:
            self.assertEqual(r["record_classification"], "Abnormal")

        # 4. Filter: UNCLASSIFIED
        res_unc = self.client.get("/api/dataset/records?filter=UNCLASSIFIED")
        self.assertEqual(res_unc.status_code, 200)
        data_unc = res_unc.get_json()
        self.assertEqual(len(data_unc["records"]), 0)

        print("\n[TEST RESULT] API Filters Verification:")
        print(f"  All Records: {len(data_all['records'])}")
        print(f"  Normal Records: {[r['record_id'] for r in data_norm['records']]}")
        print(f"  Abnormal Records Count: {len(data_abn['records'])}")
        print(f"  Unclassified Records Count: {len(data_unc['records'])}")

    def test_06_separate_waveform_annotations_and_model_prediction(self):
        """Verify endpoint returns waveform, reference annotations, and RF prediction distinctly."""
        res = self.client.get("/api/dataset/record_analysis?record_id=100&start_sec=0&duration_sec=10")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertTrue(data["success"])
        self.assertIn("waveform", data)
        self.assertIn("reference_classification", data)
        self.assertIn("model_prediction", data)
        self.assertIn("patient_info", data)

        # 1. Waveform validation
        wf = data["waveform"]
        self.assertIn("lead1", wf)
        self.assertIn("lead2", wf)
        self.assertEqual(wf["sample_rate"], 250)
        self.assertAlmostEqual(len(wf["lead1"]), 2500, delta=5)

        # 2. Reference Classification validation (Ground Truth)
        ref = data["reference_classification"]
        self.assertEqual(ref["classification_label"], "Abnormal")
        self.assertIn("symbol_counts", ref)
        self.assertIn("segment_annotations", ref)
        self.assertTrue(ref["is_verified_ground_truth"])

        # 3. Random Forest Model Prediction validation
        ml = data["model_prediction"]
        self.assertIn("Random Forest", ml["model_name"])
        self.assertIn("diagnosis", ml)
        self.assertIn("confidence_percent", ml)
        self.assertIn("features", ml)
        self.assertIn("probabilities", ml)

        # Biomarkers extracted from signal
        feats = ml["features"]
        self.assertIn("heart_rate", feats)
        self.assertIn("qrs_duration_ms", feats)
        self.assertIn("qtc_ms", feats)

        # Print actual values
        print(f"\n[TEST RESULT] Record 100 Multi-Panel Inspection:")
        print(f"  Waveform Samples: {len(wf['lead1'])} at {wf['sample_rate']} Hz")
        print(f"  Reference Ground Truth: {ref['classification_label']} ({ref['reason']})")
        print(f"  Reference Slice Annotations: {len(ref['segment_annotations'])} beats")
        print(f"  Random Forest ML Diagnosis: {ml['diagnosis']} ({ml['confidence_percent']}% confidence)")
        print(f"  Extracted Electrophysiology: HR={feats['heart_rate']} bpm, QRS={feats['qrs_duration_ms']} ms, QTc={feats['qtc_ms']} ms")

    def test_07_safety_boundary_no_emergency_trigger(self):
        """Verify opening/classifying an Abnormal record does NOT trigger patient emergency alerts."""
        db = get_db_connection()
        initial_episodes = db.execute("SELECT COUNT(*) FROM emergency_episodes").fetchone()[0]
        initial_alerts_count = len(alert_system.get_alert_history())

        # Inspect multiple abnormal records (e.g. 100, 102, 106, 207)
        for rec_id in ["100", "102", "106", "207"]:
            res = self.client.get(f"/api/dataset/record_analysis?record_id={rec_id}&start_sec=0&duration_sec=10")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["reference_classification"]["classification_label"], "Abnormal")

        # Verify NO emergency episode was created in SQLite database
        final_episodes = db.execute("SELECT COUNT(*) FROM emergency_episodes").fetchone()[0]
        self.assertEqual(final_episodes, initial_episodes, "Offline record inspection must NOT create emergency episodes!")

        # Verify NO alerts were added to active alert history
        final_alerts_count = len(alert_system.get_alert_history())
        self.assertEqual(final_alerts_count, initial_alerts_count, "Offline record inspection must NOT dispatch alerts!")

        print("\n[TEST RESULT] Clinical Safety Separation Verified:")
        print(f"  Initial Emergency Episodes: {initial_episodes} -> Final Emergency Episodes: {final_episodes} (Delta: 0)")
        print(f"  Initial Active Alerts: {initial_alerts_count} -> Final Active Alerts: {final_alerts_count} (Delta: 0)")

if __name__ == "__main__":
    unittest.main()
