
"""
CardioSentinel - Official PhysioNet MIT-BIH Arrhythmia Database Manager
Provides unified record discovery, WFDB file parsing (.hea, .dat, .atr),
proper polyphase resampling (360 Hz to 250 Hz), segment caching, annotation
navigation, and streaming access for all 48 official MIT-BIH recordings.
"""

import os
import json
import time
import glob
import numpy as np
from scipy import signal as sp_signal
from typing import Dict, List, Optional, Any, Tuple

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "mitbih_arrhythmia_database")
LEGACY_CSV_DIR = os.path.join(BASE_DIR, "data", "mitbih", "csv")
PHYSIONET_URL = "https://physionet.org/files/mitdb/1.0.0"

# Documented MIT-BIH Pacemaker Records
PACED_RECORDS = {"102", "104", "107", "217"}

# Documented Subject mapping (Records 201 & 202 share Subject 201)
SUBJECT_MAP = {
    "100": 100, "101": 101, "102": 102, "103": 103, "104": 104, "105": 105,
    "106": 106, "107": 107, "108": 108, "109": 109, "111": 111, "112": 112,
    "113": 113, "114": 114, "115": 115, "116": 116, "117": 117, "118": 118,
    "119": 119, "121": 121, "122": 122, "123": 123, "124": 124, "200": 200,
    "201": 201, "202": 201, "203": 203, "205": 205, "207": 207, "208": 208,
    "209": 209, "210": 210, "212": 212, "213": 213, "214": 214, "215": 215,
    "217": 217, "219": 219, "220": 220, "221": 221, "222": 222, "223": 223,
    "228": 228, "230": 230, "231": 231, "232": 232, "233": 233, "234": 234
}

class MITBIHDatasetManager:
    """Enterprise-grade manager for MIT-BIH Arrhythmia Database."""

    def __init__(self, dataset_dir: Optional[str] = None):
        self.dataset_dir = dataset_dir or DATASET_DIR
        self.catalog_file = os.path.join(self.dataset_dir, "dataset_catalog.json")
        self.catalog: Dict[str, Dict[str, Any]] = {}
        self.loaded_cache: Dict[str, Dict[str, Any]] = {} # record_id -> parsed data
        self.refresh_catalog()

    def refresh_catalog(self):
        """Discovers all available MIT-BIH records and reads/builds catalog."""
        if os.path.exists(self.catalog_file):
            try:
                with open(self.catalog_file, "r", encoding="utf-8") as f:
                    self.catalog = json.load(f)
            except Exception as e:
                print(f"[!] Warning reading catalog JSON: {e}")

        # Scan directory for newly downloaded or existing files
        hea_files = glob.glob(os.path.join(self.dataset_dir, "*.hea"))
        for h_path in hea_files:
            rec_id = os.path.splitext(os.path.basename(h_path))[0]
            if rec_id in ["RECORDS", "dataset_summary"]:
                continue

            dat_path = os.path.join(self.dataset_dir, f"{rec_id}.dat")
            atr_path = os.path.join(self.dataset_dir, f"{rec_id}.atr")
            csv_path = self._find_csv_path(rec_id)

            has_dat = os.path.exists(dat_path) and os.path.getsize(dat_path) > 0
            has_atr = os.path.exists(atr_path) and os.path.getsize(atr_path) > 0
            has_csv = csv_path is not None

            is_playable = has_dat or has_csv

            if rec_id not in self.catalog:
                self.catalog[rec_id] = {
                    "record_id": rec_id,
                    "fs": 360,
                    "num_channels": 2,
                    "leads": ["MLII", "V1"],
                    "sig_len": 650000,
                    "duration_sec": 1805.5,
                    "duration_formatted": "30m 05s",
                    "is_paced": rec_id in PACED_RECORDS,
                    "subject_id": SUBJECT_MAP.get(rec_id, int(rec_id) if rec_id.isdigit() else 0),
                    "primary_diagnosis": "PhysioNet Clinical Telemetry",
                    "beat_count": 0,
                    "has_header": True,
                    "has_signal": has_dat,
                    "has_annotation": has_atr,
                    "has_csv": has_csv,
                    "is_playable": is_playable,
                    "file_status": "Fully Playable (Binary WFDB)" if has_dat else ("Playable (Extracted CSV)" if has_csv else "Metadata Cataloged")
                }
            else:
                self.catalog[rec_id]["has_signal"] = has_dat
                self.catalog[rec_id]["has_annotation"] = has_atr
                self.catalog[rec_id]["has_csv"] = has_csv
                self.catalog[rec_id]["is_playable"] = is_playable
                if has_dat:
                    self.catalog[rec_id]["file_status"] = "Fully Playable (Binary WFDB)"
                elif has_csv:
                    self.catalog[rec_id]["file_status"] = "Playable (Extracted CSV)"

    def _find_csv_path(self, record_id: str) -> Optional[str]:
        """Locates CSV representation if available."""
        matches = glob.glob(os.path.join(self.dataset_dir, f"record_{record_id}*.csv"))
        if matches:
            return matches[0]
        legacy_path = os.path.join(LEGACY_CSV_DIR, f"record_{record_id}.csv")
        if os.path.exists(legacy_path):
            return legacy_path
        return None

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Returns catalog summary metadata for dataset tests and API."""
        catalog_list = self.get_catalog_list()
        playable = [c for c in catalog_list if c.get("is_playable")]
        return {
            "dataset_name": "PhysioNet MIT-BIH Arrhythmia Database",
            "database_name": "MIT-BIH Arrhythmia Database",
            "source_repository": "PhysioNet (https://physionet.org/content/mitdb/1.0.0/)",
            "sampling_frequency_hz": 360,
            "resampled_frequency_hz": 250,
            "total_cataloged_records": len(catalog_list),
            "fully_playable_records": len(playable),
            "records": {c["record_id"]: c for c in catalog_list}
        }

    def get_signal_stream(self, record_id: str) -> Dict[str, List[float]]:
        """Returns resampled lead2 and lead_v1 lists for compatibility."""
        rec = self.load_record(record_id)
        native_fs = rec.get("fs", 360)
        l1 = self.resample_to_250hz(rec["lead1_raw"]) if native_fs == 360 else rec["lead1_raw"]
        l2 = self.resample_to_250hz(rec["lead2_raw"]) if native_fs == 360 else rec["lead2_raw"]
        return {
            "lead2": l1.tolist() if isinstance(l1, np.ndarray) else list(l1),
            "lead_v1": l2.tolist() if isinstance(l2, np.ndarray) else list(l2)
        }

    def get_catalog_list(self) -> List[Dict[str, Any]]:
        """Returns sorted list of cataloged records."""
        self.refresh_catalog()
        records = list(self.catalog.values())
        records.sort(key=lambda r: int(r["record_id"]) if r["record_id"].isdigit() else 999)
        return records

    def get_record_metadata(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Returns catalog metadata for a record or None if non-existent."""
        self.refresh_catalog()
        return self.catalog.get(str(record_id))

    def get_record_annotations(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Returns annotations dictionary with symbols and samples, or None if invalid."""
        if str(record_id) not in self.catalog:
            return None
        rec = self.load_record(record_id)
        if not rec:
            return None
        annotations = rec.get("annotations", [])
        return {
            "record_id": record_id,
            "beat_count": len(annotations),
            "symbols": [a["symbol"] for a in annotations],
            "sample_indices": [a.get("sample_360", 0) for a in annotations],
            "annotations": annotations
        }

    def get_record_segment(
        self,
        record_id: str,
        start_sec: float = 0.0,
        duration_sec: float = 10.0,
        target_fs: int = 250
    ) -> Optional[Dict[str, Any]]:
        """Convenience method for segment retrieval with error safety."""
        if str(record_id) not in self.catalog:
            return None
        return self.get_segment(record_id, start_sec, duration_sec, target_fs)

    def load_record(self, record_id: str, fallback_synthetic: bool = True) -> Optional[Dict[str, Any]]:
        """
        Loads signal data and annotations for a record.
        Prioritizes:
        1. Memory Cache
        2. Local WFDB Binary (.dat + .hea)
        3. Local CSV file
        4. Realistic synthetic fallback (if enabled)
        """
        record_id = str(record_id)
        if record_id in self.loaded_cache:
            return self.loaded_cache[record_id]

        dat_path = os.path.join(self.dataset_dir, f"{record_id}.dat")
        hea_path = os.path.join(self.dataset_dir, f"{record_id}.hea")
        csv_path = self._find_csv_path(record_id)

        if record_id not in self.catalog and not os.path.exists(hea_path) and not csv_path:
            if not fallback_synthetic:
                return None

        meta = self.catalog.get(record_id, {
            "record_id": record_id,
            "fs": 360,
            "leads": ["MLII", "V1"],
            "primary_diagnosis": "MIT-BIH Recorded Session"
        })

        # 1. Try WFDB native binary read
        if WFDB_AVAILABLE and os.path.exists(dat_path) and os.path.exists(hea_path):
            try:
                rec_stem = os.path.join(self.dataset_dir, record_id)
                rec = wfdb.rdrecord(rec_stem)
                fs = rec.fs
                sig_names = rec.sig_name if rec.sig_name else ["MLII", "V1"]
                signals = rec.p_signal # shape (N, channels)

                # Read annotations if available
                annotations = []
                atr_path = os.path.join(self.dataset_dir, f"{record_id}.atr")
                if os.path.exists(atr_path):
                    try:
                        ann = wfdb.rdann(rec_stem, "atr")
                        for idx, sym in zip(ann.sample, ann.symbol):
                            annotations.append({
                                "sample_360": int(idx),
                                "time_sec": round(float(idx) / fs, 3),
                                "symbol": str(sym),
                                "is_ventricular": sym in ["V", "E", "!"],
                                "is_atrial": sym in ["A", "a", "J", "S"]
                            })
                    except Exception as e:
                        print(f"[!] Warning reading annotations for {record_id}: {e}")

                loaded = {
                    "record_id": record_id,
                    "source": "MIT-BIH Recorded Data (Binary WFDB)",
                    "fs": fs,
                    "leads": sig_names,
                    "sig_len": len(signals),
                    "duration_sec": round(len(signals) / fs, 2),
                    "lead1_raw": signals[:, 0].astype(float),
                    "lead2_raw": signals[:, 1].astype(float) if signals.shape[1] > 1 else signals[:, 0].astype(float),
                    "annotations": annotations,
                    "meta": meta
                }
                self.loaded_cache[record_id] = loaded
                return loaded
            except Exception as e:
                print(f"[!] Error reading WFDB binary for {record_id}: {e}")

        # 2. Try CSV file read
        if csv_path and os.path.exists(csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                l2_col = [c for c in df.columns if "lead2" in c.lower() or "mlii" in c.lower() or "val1" in c.lower()]
                v1_col = [c for c in df.columns if "v1" in c.lower() or "v5" in c.lower() or "lead_v1" in c.lower()]

                l2 = df[l2_col[0]].values.astype(float) if l2_col else df.iloc[:, 0].values.astype(float)
                v1 = df[v1_col[0]].values.astype(float) if v1_col else l2

                fs = 250 if "sample_rate" in df.columns and df["sample_rate"].iloc[0] == 250 else 360
                duration_sec = round(len(l2) / fs, 2)

                # Parse R-peak indicators if present
                annotations = []
                if "is_r_peak" in df.columns:
                    peak_indices = np.where(df["is_r_peak"] == 1)[0]
                    for idx in peak_indices:
                        annotations.append({
                            "sample_360": int(idx),
                            "time_sec": round(float(idx) / fs, 3),
                            "symbol": "N",
                            "is_ventricular": False,
                            "is_atrial": False
                        })

                loaded = {
                    "record_id": record_id,
                    "source": "MIT-BIH Recorded Data (Pre-processed CSV)",
                    "fs": fs,
                    "leads": ["MLII", "V1"],
                    "sig_len": len(l2),
                    "duration_sec": duration_sec,
                    "lead1_raw": l2,
                    "lead2_raw": v1,
                    "annotations": annotations,
                    "meta": meta
                }
                self.loaded_cache[record_id] = loaded
                return loaded
            except Exception as e:
                print(f"[!] Error loading CSV for {record_id}: {e}")

        # 3. Fallback: Generate clinically faithful signal for this record
        print(f"[*] Generating clinical synthetic stream for Record {record_id}...")
        return self._generate_synthetic_record(record_id, meta)

    def _generate_synthetic_record(self, record_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Generates realistic synthetic multi-lead ECG for offline research demonstration."""
        fs = 250
        duration_sec = 60.0 # 60 seconds loop
        total_samples = int(fs * duration_sec)
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)

        hr = 75
        diag = meta.get("primary_diagnosis", "")
        if "Tachycardia" in diag or "VT" in diag:
            hr = 165
        elif "Bradycardia" in diag:
            hr = 45
        elif "Fibrillation" in diag:
            hr = 135

        beat_period = 60.0 / hr
        l1 = np.zeros(total_samples)
        l2 = np.zeros(total_samples)
        annotations = []

        num_beats = int(duration_sec / beat_period)
        for b in range(num_beats):
            center_t = (b + 0.5) * beat_period
            center_idx = int(center_t * fs)
            if center_idx >= total_samples:
                break

            annotations.append({
                "sample_360": int(center_idx * (360.0 / 250.0)),
                "time_sec": round(center_t, 3),
                "symbol": "N",
                "is_ventricular": False,
                "is_atrial": False
            })

            # Gaussian QRS complex synthesis
            sigma_qrs = 0.025
            qrs = 1.6 * np.exp(-0.5 * ((t - center_t) / sigma_qrs) ** 2)
            # P wave
            p_wave = 0.22 * np.exp(-0.5 * ((t - (center_t - 0.16)) / 0.04) ** 2)
            # T wave
            t_wave = 0.38 * np.exp(-0.5 * ((t - (center_t + 0.22)) / 0.07) ** 2)

            l1 += qrs + p_wave + t_wave
            l2 += (qrs * 0.7) - (p_wave * 0.5) + (t_wave * 0.8)

        # Baseline noise
        l1 += np.random.normal(0, 0.02, total_samples)
        l2 += np.random.normal(0, 0.02, total_samples)

        loaded = {
            "record_id": record_id,
            "source": "Synthetic Demonstration Session",
            "fs": fs,
            "leads": ["MLII", "V1"],
            "sig_len": total_samples,
            "duration_sec": duration_sec,
            "lead1_raw": l1,
            "lead2_raw": l2,
            "annotations": annotations,
            "meta": meta
        }
        self.loaded_cache[record_id] = loaded
        return loaded

    def resample_to_250hz(self, signal_360: np.ndarray) -> np.ndarray:
        """
        Converts 360 Hz MIT-BIH signal to standard 250 Hz telemetry stream
        using polyphase rational resampling: 250 / 360 = 25 / 36.
        """
        if len(signal_360) == 0:
            return signal_360
        try:
            return sp_signal.resample_poly(signal_360, 25, 36)
        except Exception:
            # Fallback linear interpolation
            num_orig = len(signal_360)
            num_target = int(round(num_orig * (250.0 / 360.0)))
            return np.interp(
                np.linspace(0, num_orig, num_target, endpoint=False),
                np.arange(num_orig),
                signal_360
            )

    def get_segment(
        self,
        record_id: str,
        start_sec: float,
        duration_sec: float,
        target_fs: int = 250
    ) -> Dict[str, Any]:
        """
        Retrieves a windowed slice of Lead 1 and Lead 2 resampled to target_fs (250 Hz)
        with aligned annotations.
        """
        rec = self.load_record(record_id)
        native_fs = rec["fs"]

        start_sample = int(max(0, start_sec * native_fs))
        end_sample = int(min(rec["sig_len"], (start_sec + duration_sec) * native_fs))

        if start_sample >= rec["sig_len"]:
            start_sample = 0
            end_sample = min(rec["sig_len"], int(duration_sec * native_fs))

        raw_l1 = rec["lead1_raw"][start_sample:end_sample]
        raw_l2 = rec["lead2_raw"][start_sample:end_sample]

        # Resample to 250 Hz if needed
        if native_fs == 360 and target_fs == 250:
            l1_resampled = self.resample_to_250hz(raw_l1)
            l2_resampled = self.resample_to_250hz(raw_l2)
        else:
            l1_resampled = raw_l1
            l2_resampled = raw_l2

        # Annotations in this window
        window_annotations = [
            a for a in rec.get("annotations", [])
            if start_sec <= a["time_sec"] < (start_sec + duration_sec)
        ]

        return {
            "record_id": record_id,
            "source": rec["source"],
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "sample_rate": target_fs,
            "target_fs": target_fs,
            "lead1_name": rec["leads"][0] if rec["leads"] else "MLII",
            "lead2_name": rec["leads"][1] if len(rec["leads"]) > 1 else "V1",
            "lead1": l1_resampled.tolist(),
            "lead2": l2_resampled.tolist(),
            "lead_v1": l2_resampled.tolist(),
            "annotations": window_annotations,
            "total_duration_sec": rec["duration_sec"],
            "meta": rec.get("meta", {})
        }

# Global singleton
dataset_manager = MITBIHDatasetManager()
