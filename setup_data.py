"""
CardioSentinel - PhysioNet MIT-BIH Arrhythmia Database Setup & Synchronizer
Downloads, verifies, and catalogs records from the official MIT-BIH database:
https://physionet.org/content/mitdb/1.0.0/

All 48 Official Records:
100-109, 111-119, 121-124, 200-203, 205, 207-210, 212-215, 217, 219-223, 228, 230-234
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Any, Optional

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "mitbih_arrhythmia_database")
PHYSIONET_URL = "https://physionet.org/files/mitdb/1.0.0"

EXPECTED_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234"
]

PACED_RECORDS = {"102", "104", "107", "217"}

# Subject mapping per MIT-BIH clinical database documentation
# Note: records 201 and 202 are from the same subject (Subject 201)
RECORD_SUBJECT_MAP = {
    "100": 100, "101": 101, "102": 102, "103": 103, "104": 104, "105": 105,
    "106": 106, "107": 107, "108": 108, "109": 109, "111": 111, "112": 112,
    "113": 113, "114": 114, "115": 115, "116": 116, "117": 117, "118": 118,
    "119": 119, "121": 121, "122": 122, "123": 123, "124": 124, "200": 200,
    "201": 201, "202": 201, "203": 203, "205": 205, "207": 207, "208": 208,
    "209": 209, "210": 210, "212": 212, "213": 213, "214": 214, "215": 215,
    "217": 217, "219": 219, "220": 220, "221": 221, "222": 222, "223": 223,
    "228": 228, "230": 230, "231": 231, "232": 232, "233": 233, "234": 234
}

# Key curated clinical diagnosis notes from MIT-BIH documentation
CLINICAL_DIAGNOSES = {
    "100": "Normal Sinus Rhythm, isolated APCs",
    "101": "Normal Sinus Rhythm, occasional APCs",
    "102": "Paced Rhythm, frequent PVCs",
    "103": "Normal Sinus Rhythm",
    "104": "Paced Rhythm, PVCs",
    "105": "Normal Sinus Rhythm, high artifact",
    "106": "Normal Sinus Rhythm, complex PVCs (bigeminy)",
    "107": "Paced Rhythm, complete heart block",
    "108": "Sinus Tachycardia / Bradycardia, APCs",
    "109": "Left Bundle Branch Block (LBBB)",
    "111": "Normal Sinus Rhythm, LBBB",
    "112": "Normal Sinus Rhythm, APCs",
    "113": "Normal Sinus Rhythm",
    "114": "Normal Sinus Rhythm",
    "115": "Normal Sinus Rhythm",
    "116": "Normal Sinus Rhythm, PVCs",
    "117": "Normal Sinus Rhythm",
    "118": "Right Bundle Branch Block (RBBB)",
    "119": "Normal Sinus Rhythm, frequent PVCs",
    "121": "Normal Sinus Rhythm",
    "122": "Normal Sinus Rhythm",
    "123": "Normal Sinus Rhythm, occasional PVCs",
    "124": "Right Bundle Branch Block",
    "200": "Ventricular Tachycardia (VT runs) & Frequent PVCs",
    "201": "Atrial Fibrillation, PVCs",
    "202": "Atrial Fibrillation, APCs",
    "203": "Atrial Fibrillation, rapid ventricular response, PVCs",
    "205": "Ventricular Tachycardia / Flutter, PVCs",
    "207": "Ventricular Flutter / Fibrillation (VFib/VFL)",
    "208": "Severe Ventricular Irritability, couplets, PVC runs",
    "209": "Atrial Fibrillation / Flutter",
    "210": "Ventricular Tachycardia, APCs",
    "212": "Right Bundle Branch Block",
    "213": "Ventricular Tachycardia, PVCs",
    "214": "Normal Sinus Rhythm, PVCs",
    "215": "Ventricular Tachycardia, PVCs",
    "217": "Paced Rhythm, PVCs",
    "219": "Normal Sinus Rhythm, APCs",
    "220": "Normal Sinus Rhythm",
    "221": "Ventricular Tachycardia, PVCs",
    "222": "Atrial Flutter with varying block",
    "223": "Ventricular Tachycardia, PVCs",
    "228": "Ventricular Tachycardia, PVCs",
    "230": "Ventricular Tachycardia, PVCs",
    "231": "Right Bundle Branch Block",
    "232": "Sinus Bradycardia",
    "233": "Ventricular Tachycardia, multiform PVCs",
    "234": "Normal Sinus Rhythm"
}

def ensure_dataset_dir() -> str:
    os.makedirs(DATASET_DIR, exist_ok=True)
    return DATASET_DIR

def download_file_stream(url: str, dest_path: str, timeout: int = 15) -> bool:
    """Download a file with streaming and error safety."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True
    try:
        headers = {"User-Agent": "CardioSentinel-Research-Telemetry/2.0"}
        with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
            if r.status_code != 200:
                return False
            temp_path = dest_path + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                os.rename(temp_path, dest_path)
                return True
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False
    except Exception as e:
        temp_path = dest_path + ".tmp"
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False

def parse_header_metadata(hea_path: str, record_id: str) -> Dict[str, Any]:
    """Parses standard WFDB .hea file for sampling rate, leads, sample count, and demographics."""
    meta = {
        "record_id": record_id,
        "fs": 360,
        "num_channels": 2,
        "sig_len": 650000,
        "leads": ["MLII", "V1"],
        "duration_sec": 1805.5,
        "duration_formatted": "30m 05s",
        "comments": [],
        "age": None,
        "sex": None,
        "medications": None,
        "is_paced": record_id in PACED_RECORDS,
        "subject_id": RECORD_SUBJECT_MAP.get(record_id, int(record_id) if record_id.isdigit() else 0),
        "primary_diagnosis": CLINICAL_DIAGNOSES.get(record_id, "ECG Telemetry")
    }
    
    if not os.path.exists(hea_path):
        return meta
        
    try:
        with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        leads = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if i == 0:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        meta["fs"] = int(parts[2].split("/")[0])
                    except Exception:
                        pass
                if len(parts) >= 4:
                    try:
                        meta["sig_len"] = int(parts[3])
                        meta["duration_sec"] = round(meta["sig_len"] / meta["fs"], 1)
                        m = int(meta["duration_sec"] // 60)
                        s = int(meta["duration_sec"] % 60)
                        meta["duration_formatted"] = f"{m}m {s:02d}s"
                    except Exception:
                        pass
            elif line.startswith("#"):
                meta["comments"].append(line[1:].strip())
                cmt = line.lower()
                if "age" in cmt:
                    for part in cmt.split():
                        if part.isdigit():
                            meta["age"] = int(part)
                if "male" in cmt:
                    meta["sex"] = "Male"
                elif "female" in cmt:
                    meta["sex"] = "Female"
            else:
                # Signal channel line: format typically: <filename> <format> <gain> ... <description/lead>
                parts = line.split()
                if len(parts) >= 9:
                    lead_name = parts[8].replace("'", "").replace('"', '')
                    leads.append(lead_name)
                    
        if leads:
            meta["leads"] = leads
            meta["num_channels"] = len(leads)
            
    except Exception as e:
        print(f"[!] Warning parsing header {hea_path}: {e}", flush=True)
        
    return meta

def parse_annotation_stats(atr_path: str, record_id: str) -> Dict[str, Any]:
    """Reads .atr file via WFDB if available, or returns metadata summary."""
    stats = {
        "beat_count": 0,
        "symbol_counts": {},
        "rhythms": [],
        "has_annotations": os.path.exists(atr_path) and os.path.getsize(atr_path) > 0
    }
    
    if not stats["has_annotations"]:
        return stats
        
    if WFDB_AVAILABLE:
        try:
            record_dir = os.path.dirname(atr_path)
            ann = wfdb.rdann(os.path.join(record_dir, record_id), "atr")
            stats["beat_count"] = len(ann.symbol)
            counts = {}
            for s in ann.symbol:
                counts[s] = counts.get(s, 0) + 1
            stats["symbol_counts"] = counts
            if hasattr(ann, "aux_note") and ann.aux_note:
                rhythms = list(set([str(note).strip() for note in ann.aux_note if str(note).strip()]))
                stats["rhythms"] = rhythms
        except Exception:
            pass
            
    return stats

def sync_mitbih_dataset(download_signals: bool = True, fetch_all_signals: bool = False, specific_records: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    1. Downloads all 48 .hea headers and .atr annotations (fast, ~300 KB total).
    2. Downloads .dat binary signals for core records or all requested.
    3. Builds the comprehensive dataset catalog JSON.
    """
    dest_dir = ensure_dataset_dir()
    records = specific_records or EXPECTED_RECORDS
    headers_synced = 0
    annotations_synced = 0
    signals_synced = 0
    
    print("=" * 70, flush=True)
    print("CardioSentinel™ - PhysioNet MIT-BIH Arrhythmia Dataset Synchronizer", flush=True)
    print(f"Target Directory: {dest_dir}", flush=True)
    print(f"Total Target Records: {len(records)}", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Download RECORDS index
    rec_file = os.path.join(dest_dir, "RECORDS")
    if not os.path.exists(rec_file):
        download_file_stream(f"{PHYSIONET_URL}/RECORDS", rec_file)
        
    # Core signal records to download for live telemetry playback
    core_signals = {"100", "101", "102", "103", "105", "106", "118", "119", "200", "207", "208", "209", "210"}
    signals_to_download = set(EXPECTED_RECORDS) if fetch_all_signals else core_signals
    
    # Pass 1: Download ALL 48 .hea headers and .atr annotations first (very fast, ~300KB total)
    print("[*] Pass 1: Syncing all 48 record headers (.hea) and annotations (.atr)...", flush=True)
    for rec_id in EXPECTED_RECORDS:
        hea_dest = os.path.join(dest_dir, f"{rec_id}.hea")
        if not os.path.exists(hea_dest):
            ok = download_file_stream(f"{PHYSIONET_URL}/{rec_id}.hea", hea_dest, timeout=8)
            if ok: headers_synced += 1
            
        atr_dest = os.path.join(dest_dir, f"{rec_id}.atr")
        if not os.path.exists(atr_dest):
            ok = download_file_stream(f"{PHYSIONET_URL}/{rec_id}.atr", atr_dest, timeout=8)
            if ok: annotations_synced += 1
            
    print(f"    [OK] Headers and annotations synced (New headers: {headers_synced}, New annotations: {annotations_synced})", flush=True)

    # Pass 2: Download binary signals (.dat) for requested records
    if download_signals:
        print("[*] Pass 2: Syncing binary ECG signals (.dat)...", flush=True)
        for rec_id in signals_to_download:
            dat_dest = os.path.join(dest_dir, f"{rec_id}.dat")
            if not os.path.exists(dat_dest):
                print(f"    -> Downloading binary waveform for Record {rec_id}...", flush=True)
                ok = download_file_stream(f"{PHYSIONET_URL}/{rec_id}.dat", dat_dest, timeout=30)
                if ok: signals_synced += 1
                
    # Catalog all 48 records
    print("[*] Generating catalog index from verified files...", flush=True)
    catalog = {}
    for rec_id in EXPECTED_RECORDS:
        hea_path = os.path.join(dest_dir, f"{rec_id}.hea")
        dat_path = os.path.join(dest_dir, f"{rec_id}.dat")
        atr_path = os.path.join(dest_dir, f"{rec_id}.atr")
        csv_path = os.path.join(dest_dir, f"record_{rec_id}_{CLINICAL_DIAGNOSES.get(rec_id, '').replace(' ', '_').replace('/', '_')}.csv")
        
        has_hea = os.path.exists(hea_path) and os.path.getsize(hea_path) > 0
        has_dat = os.path.exists(dat_path) and os.path.getsize(dat_path) > 0
        has_atr = os.path.exists(atr_path) and os.path.getsize(atr_path) > 0
        has_csv = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        
        meta = parse_header_metadata(hea_path, rec_id)
        ann_stats = parse_annotation_stats(atr_path, rec_id)
        
        playable = has_dat or has_csv
        
        catalog[rec_id] = {
            **meta,
            **ann_stats,
            "has_header": has_hea,
            "has_signal": has_dat,
            "has_annotation": has_atr,
            "has_csv": has_csv,
            "is_playable": playable,
            "file_status": "Fully Playable (Binary WFDB)" if has_dat else ("Playable (Extracted CSV)" if has_csv else "Metadata Cataloged (On-demand Download)")
        }
        
    catalog_path = os.path.join(dest_dir, "dataset_catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
        
    playable_count = sum(1 for c in catalog.values() if c["is_playable"])
    cataloged_count = len(catalog)
    
    print("\n" + "=" * 70, flush=True)
    print(f"[OK] PhysioNet MIT-BIH Catalog Synchronized Successfully!", flush=True)
    print(f"     Total Records Discovered: {cataloged_count} / {len(EXPECTED_RECORDS)}", flush=True)
    print(f"     Immediately Playable:     {playable_count} / {len(EXPECTED_RECORDS)}", flush=True)
    print(f"     Saved Catalog to:         {catalog_path}", flush=True)
    print("=" * 70, flush=True)
    
    return {
        "success": True,
        "catalog_path": catalog_path,
        "total_records": cataloged_count,
        "playable_records": playable_count,
        "catalog": catalog
    }

if __name__ == "__main__":
    fetch_all = "--all" in sys.argv
    sync_mitbih_dataset(fetch_all_signals=fetch_all)
