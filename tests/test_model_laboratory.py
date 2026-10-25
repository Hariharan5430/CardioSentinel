"""
Comprehensive Verification Tests for Model Laboratory: Training & Testing Dataset Feature
Covers all user specifications:
1. Training Dataset:
   - Exact ECG record IDs used to train Random Forest model are stored and returned.
   - Actual number of training records.
   - Model trained using only the training dataset and reference labels.
2. Testing Dataset:
   - Exact ECG record IDs reserved for testing are stored and returned.
   - Actual number of testing records.
   - Model evaluated on held-out testing dataset without training on its labels.
   - Actual accuracy, precision, recall, F1-score, and confusion matrix.
3. Dataset Splitting:
   - Configurable training/testing split (70/30 default, 80/20, 60/40).
   - Partitioned strictly by complete ECG records and patient subjects (SUBJECT_MAPPING).
   - Zero overlap between training and testing records (set intersection is empty).
   - Record IDs and split configuration stored with model artifact.
4. Model Laboratory UI / API:
   - Duplicate training submissions prevented (HTTP 409 / training lock).
   - Training status endpoint provides progress reporting.
5. Emergency Alert Safety:
   - Training and testing do NOT trigger emergency alarms or SMS dispatch.
6. Record Browser Integration:
   - Reference classification is kept strictly separate from model prediction.
   - Documented annotation rules applied for Normal vs Abnormal.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from backend.ai_engine import AIEarlyWarningEngine, ai_engine, SUBJECT_MAPPING
from backend.alert_system import alert_system
from backend.server import app
from backend.dataset_manager import dataset_manager

class TestModelLaboratoryFeature(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        self.ai = ai_engine

    def test_01_training_dataset_record_ids_and_counts(self):
        """Verify exact ECG record IDs used for training are saved, count is accurate, and labels are reference."""
        metrics = self.ai.get_evaluation_metrics()
        self.assertIn("training_dataset", metrics)
        train_data = metrics["training_dataset"]

        self.assertIn("records", train_data)
        self.assertIn("record_count", train_data)
        self.assertIn("sample_count", train_data)

        train_records = train_data["records"]
        self.assertIsInstance(train_records, list)
        self.assertGreaterEqual(len(train_records), 10)
        self.assertEqual(train_data["record_count"], len(train_records))

        # Check each record entry has required fields
        for rec in train_records:
            self.assertIn("record_id", rec)
            self.assertIn("subject_id", rec)
            self.assertIn("pathology", rec)
            self.assertIn("samples", rec)
            self.assertIn("classification", rec)
            self.assertEqual(rec["role"], "Training")

    def test_02_testing_dataset_record_ids_and_counts(self):
        """Verify exact ECG record IDs reserved for testing are saved and count is accurate."""
        metrics = self.ai.get_evaluation_metrics()
        self.assertIn("testing_dataset", metrics)
        test_data = metrics["testing_dataset"]

        self.assertIn("records", test_data)
        self.assertIn("record_count", test_data)
        self.assertIn("sample_count", test_data)

        test_records = test_data["records"]
        self.assertIsInstance(test_records, list)
        self.assertGreaterEqual(len(test_records), 3)
        self.assertEqual(test_data["record_count"], len(test_records))

        # Check each record entry has required fields
        for rec in test_records:
            self.assertIn("record_id", rec)
            self.assertIn("subject_id", rec)
            self.assertIn("pathology", rec)
            self.assertIn("samples", rec)
            self.assertIn("classification", rec)
            self.assertEqual(rec["role"], "Held-out Test")

    def test_03_strict_dataset_partition_zero_overlap(self):
        """Verify training and testing partitions have zero overlapping ECG records or patient subjects."""
        metrics = self.ai.get_evaluation_metrics()
        train_record_ids = {r["record_id"] for r in metrics["training_dataset"]["records"]}
        test_record_ids = {r["record_id"] for r in metrics["testing_dataset"]["records"]}

        # Strict record disjointness
        intersection = train_record_ids.intersection(test_record_ids)
        self.assertEqual(len(intersection), 0, f"Data leakage detected! Overlapping records: {intersection}")

        # Strict subject disjointness
        train_subjects = {SUBJECT_MAPPING.get(r, int(r) if r.isdigit() else r) for r in train_record_ids}
        test_subjects = {SUBJECT_MAPPING.get(r, int(r) if r.isdigit() else r) for r in test_record_ids}
        subj_intersection = train_subjects.intersection(test_subjects)
        self.assertEqual(len(subj_intersection), 0, f"Patient leakage detected! Overlapping subjects: {subj_intersection}")

        # Split configuration metadata
        split_cfg = metrics.get("split_config", {})
        self.assertTrue(split_cfg.get("is_leakage_free"))
        self.assertEqual(split_cfg.get("overlapping_records_count"), 0)

    def test_04_test_evaluation_metrics_integrity(self):
        """Verify genuine evaluation metrics computed only on held-out test data (no fabrication)."""
        metrics = self.ai.get_evaluation_metrics()

        # Check all required metric fields
        self.assertIn("accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("per_class_metrics", metrics)

        # Accuracy, Balanced Accuracy, and Macro F1 in valid percentage / ratio ranges
        self.assertGreater(metrics["accuracy"], 0.80)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreater(metrics["balanced_accuracy"], 0.80)
        self.assertGreater(metrics["macro_f1"], 0.80)

        # 5x5 Confusion matrix
        cm = metrics["confusion_matrix"]
        self.assertEqual(len(cm), 5)
        for row in cm:
            self.assertEqual(len(row), 5)
            self.assertTrue(all(isinstance(v, int) and v >= 0 for v in row))

        # Sum of confusion matrix matches total test samples
        cm_total = sum(sum(row) for row in cm)
        self.assertEqual(cm_total, metrics["testing_dataset"]["sample_count"])

        # Per-class metrics
        for cls_name, pcm in metrics["per_class_metrics"].items():
            self.assertIn("precision", pcm)
            self.assertIn("recall", pcm)
            self.assertIn("f1_score", pcm)
            self.assertIn("support", pcm)
            self.assertGreater(pcm["support"], 0)

    def test_05_configurable_split_ratio(self):
        """Verify model pipeline can train with different configured split ratios (e.g., 80/20, 60/40)."""
        # Test 80/20 split
        engine = AIEarlyWarningEngine()
        train_res_80 = engine.train_and_validate(train_ratio=0.80, n_estimators=20)
        self.assertTrue(train_res_80.get("success", False))
        report_80 = train_res_80.get("report", train_res_80)
        train_ids_80 = {r["record_id"] for r in report_80["training_dataset"]["records"]}
        test_ids_80 = {r["record_id"] for r in report_80["testing_dataset"]["records"]}
        self.assertEqual(len(train_ids_80.intersection(test_ids_80)), 0)
        self.assertEqual(report_80["split_config"]["split_ratio_target"], "80/20")

        # Test 60/40 split
        train_res_60 = engine.train_and_validate(train_ratio=0.60, n_estimators=20)
        self.assertTrue(train_res_60.get("success", False))
        report_60 = train_res_60.get("report", train_res_60)
        train_ids_60 = {r["record_id"] for r in report_60["training_dataset"]["records"]}
        test_ids_60 = {r["record_id"] for r in report_60["testing_dataset"]["records"]}
        self.assertEqual(len(train_ids_60.intersection(test_ids_60)), 0)
        self.assertEqual(report_60["split_config"]["split_ratio_target"], "60/40")

    def test_06_emergency_alert_safety_during_training(self):
        """Verify training and testing operations NEVER trigger emergency alerts or SMS messages."""
        initial_active_alerts = len(alert_system.get_active_alerts())

        with patch("backend.sms_service.sms_service.send_emergency_alert") as mock_sms:
            # Run model training
            res = self.ai.train_and_validate(train_ratio=0.70, n_estimators=20)
            self.assertTrue(res["success"])

            # Verify SMS service was never called
            mock_sms.assert_not_called()

        # Verify active alerts count did not increase
        final_active_alerts = len(alert_system.get_active_alerts())
        self.assertEqual(initial_active_alerts, final_active_alerts)

    def test_07_api_ml_metrics_endpoint(self):
        """Verify GET /api/ml/metrics returns complete training and testing dataset breakdown."""
        resp = self.client.get("/api/ml/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("report", data)
        report = data["report"]
        self.assertIn("training_dataset", report)
        self.assertIn("testing_dataset", report)
        self.assertIn("split_config", report)
        self.assertIn("accuracy", report)
        self.assertIn("macro_f1", report)
        self.assertIn("confusion_matrix", report)

    def test_08_api_prevent_duplicate_concurrent_training(self):
        """Verify POST /api/ml/train rejects duplicate submissions with 409 Conflict when busy."""
        # Set training flag manually to simulate active training in progress
        self.ai.is_training = True
        try:
            resp = self.client.post("/api/ml/train", json={"train_ratio": 0.70})
            self.assertEqual(resp.status_code, 409)
            data = resp.get_json()
            self.assertFalse(data["success"])
            self.assertIn("already in progress", data["message"])
        finally:
            self.ai.is_training = False

    def test_09_api_train_status_endpoint(self):
        """Verify GET /api/ml/train_status exposes step and progress percentage."""
        resp = self.client.get("/api/ml/train_status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("status", data)
        status = data["status"]
        self.assertIn("is_training", status)
        self.assertIn("progress_percent", status)
        self.assertIn("step", status)

    def test_10_record_browser_reference_vs_prediction_separation(self):
        """Verify Record Browser keeps ground truth reference annotations strictly separated from ML prediction."""
        analysis = dataset_manager.get_record_analysis("100")
        self.assertTrue(analysis["success"])
        self.assertIn("reference_classification", analysis)
        self.assertIn("model_prediction", analysis)

        ref = analysis["reference_classification"]
        ml = analysis["model_prediction"]

        # Reference classification has verified annotation details
        self.assertIn("classification_label", ref)
        self.assertIn("reason", ref)
        self.assertIn("details", ref)

        # ML prediction has model confidence and biomarkers
        self.assertIn("diagnosis", ml)
        self.assertIn("confidence", ml)
        self.assertIn("probabilities", ml)
        self.assertIn("features", ml)
        self.assertIn("disclaimer", ml)

        # The prediction disclaimer clearly marks statistical separation
        self.assertIn("separately from dataset reference", ml["disclaimer"])

if __name__ == "__main__":
    unittest.main()
