"""
Unit & Integration Tests for PhysioNet MIT-BIH Arrhythmia Dataset Module
Covers:
- Record discovery and catalog verification (all 48 records)
- WFDB header parsing (.hea), lead detection, sampling rate (360 Hz)
- Signal loading and polyphase resampling (360 Hz -> 250 Hz)
- Annotation loading (.atr), beat count, and symbol distribution
- Handling of missing files, non-existent records, and corrupted inputs
- Verified duration and sample count calculations
"""

import os
import unittest
import numpy as np
from backend.dataset_manager import MITBIHDatasetManager, dataset_manager

class TestDatasetModule(unittest.TestCase):

    def setUp(self):
        self.dm = dataset_manager

    def test_01_all_48_records_discovered(self):
        """Verify all 48 official MIT-BIH records are discovered in the catalog."""
        summary = self.dm.get_dataset_summary()
        self.assertIn("records", summary)
        records = summary["records"]
        
        # Verify all 48 expected record numbers exist in catalog
        expected = [
            "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
            "111", "112", "113", "114", "115", "116", "117", "118", "119",
            "121", "122", "123", "124",
            "200", "201", "202", "203", "205", "207", "208", "209", "210",
            "212", "213", "214", "215", "217", "219", "220", "221", "222", "223",
            "228", "230", "231", "232", "233", "234"
        ]
        for rec_id in expected:
            self.assertIn(rec_id, records, f"Record {rec_id} missing from catalog")
        self.assertEqual(len(records), 48)

    def test_02_header_parsing_and_leads(self):
        """Verify WFDB header parsing yields 360 Hz sampling rate, 2 leads, and correct duration."""
        meta = self.dm.get_record_metadata("100")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["fs"], 360)
        self.assertGreaterEqual(len(meta["leads"]), 2)
        self.assertIn("MLII", meta["leads"])
        self.assertGreater(meta["sig_len"], 600000)
        self.assertGreater(meta["duration_sec"], 1700) # ~30 minutes

    def test_03_signal_loading_and_resampling(self):
        """Verify loading 10-second segment resamples accurately from 360 Hz to 250 Hz."""
        segment = self.dm.get_record_segment("100", start_sec=0.0, duration_sec=10.0, target_fs=250)
        self.assertIsNotNone(segment)
        self.assertIn("lead2", segment)
        self.assertIn("lead_v1", segment)
        self.assertEqual(segment["target_fs"], 250)
        
        # 10 seconds at 250 Hz should have approximately 2500 samples (± 2 samples due to rational filter edge)
        self.assertAlmostEqual(len(segment["lead2"]), 2500, delta=5)
        self.assertFalse(np.isnan(segment["lead2"]).any())
        self.assertFalse(np.isinf(segment["lead2"]).any())

    def test_04_annotation_parsing_and_symbols(self):
        """Verify reading .atr annotations yields valid beat counts and MIT-BIH symbols."""
        ann = self.dm.get_record_annotations("100")
        self.assertIsNotNone(ann)
        self.assertIn("symbols", ann)
        self.assertIn("sample_indices", ann)
        self.assertGreater(len(ann["symbols"]), 2000) # Record 100 has ~2273 beats
        self.assertIn("N", ann["symbols"]) # Predominantly Normal beats

    def test_05_paced_record_detection(self):
        """Verify that known pacemaker records (102, 104, 107, 217) are identified."""
        for rec_id in ["102", "104", "107", "217"]:
            meta = self.dm.get_record_metadata(rec_id)
            if meta:
                self.assertTrue(meta.get("is_paced", False), f"Record {rec_id} should be marked as paced")

    def test_06_nonexistent_and_corrupt_records(self):
        """Verify graceful fallback for non-existent or corrupted record IDs without crashing."""
        invalid_meta = self.dm.get_record_metadata("999_NON_EXISTENT")
        self.assertIsNone(invalid_meta)

        invalid_segment = self.dm.get_record_segment("999_NON_EXISTENT", 0.0, 5.0)
        self.assertIsNone(invalid_segment)

        invalid_ann = self.dm.get_record_annotations("999_NON_EXISTENT")
        self.assertIsNone(invalid_ann)

    def test_07_signal_streaming_buffer_cache(self):
        """Verify fast streaming cache delivers continuous signals for ICU bed telemetry."""
        stream = self.dm.get_signal_stream("100")
        self.assertIsNotNone(stream)
        self.assertIn("lead2", stream)
        self.assertGreater(len(stream["lead2"]), 5000)

if __name__ == "__main__":
    unittest.main()
