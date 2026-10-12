"""
CardioSentinel - Central ICU Telemetry & Early Warning Alerting Web Server
Integrates Telemetry Engine, Digital Signal Processor, AI Arrhythmia Classifier,
and IEC 60601-1-8 Clinical Alert System with real-time SSE streaming and REST API.
"""

import time
import json
import threading
import os
import csv
import io
from typing import Dict, List, Optional, Any
from flask import Flask, render_template, jsonify, request, Response, send_from_directory, make_response
import numpy as np

from backend.telemetry_engine import TelemetryEngine, CardiacRhythmType, telemetry_engine
from backend.signal_processor import ECGSignalProcessor, signal_processor
from backend.ai_engine import AIEarlyWarningEngine, ai_engine
from backend.alert_system import ClinicalAlertSystem, AlarmPriority, alert_system
from backend.dataset_manager import MITBIHDatasetManager, dataset_manager

# Flask app configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
    static_url_path="/static"
)

# Thread safety lock & state store
state_lock = threading.Lock()
latest_patient_analysis: Dict[str, Dict] = {}
live_telemetry_queues: Dict[str, List] = {} # patient_id -> list of latest samples

# Initialize analysis storage for each patient
for pid, patient in telemetry_engine.patients.items():
    live_telemetry_queues[pid] = []
    latest_patient_analysis[pid] = {
        "patient_id": pid,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "location": patient.location,
        "current_rhythm": patient.current_rhythm,
        "baseline_rhythm": patient.baseline_rhythm,
        "data_mode": patient.data_mode,
        "mitbih_record_id": patient.mitbih_record_id,
        "playback_status": patient.playback_status,
        "playback_speed": patient.playback_speed,
        "time_position_sec": patient.time_position_sec,
        "vitals": {
            "hr": patient.current_hr,
            "spo2": patient.spo2,
            "bp_sys": patient.bp_sys,
            "bp_dia": patient.bp_dia,
            "resp_rate": patient.resp_rate,
            "temp_c": patient.temp_c,
            "perfusion_index": patient.perfusion_index
        },
        "metrics": {
            "heart_rate": int(patient.current_hr),
            "hrv_sdnn": 45.0,
            "hrv_rmssd": 35.0,
            "pnn50": 15.0,
            "qrs_duration_ms": 88.0,
            "qtc_ms": 412.0,
            "rr_entropy": 0.65,
            "signal_quality": "OPTIMAL",
            "quality_score": 95
        },
        "ai_diagnosis": {
            "diagnosis": patient.current_rhythm,
            "confidence": 0.95,
            "is_critical": False,
            "probabilities": {patient.current_rhythm: 0.95}
        },
        "cewi": {
            "cewi_score": 10,
            "risk_level": "LOW RISK (HEMODYNAMICALLY STABLE)",
            "alert_color": "#10b981",
            "estimated_lead_time": "Continuous routine monitoring",
            "shock_index": 0.60,
            "reasons": ["Electrophysiological parameters within normal baseline limits."]
        },
        "active_alert": None,
        "recent_peaks": []
    }

SYMBOL_CLINICAL_NAMES = {
    "N": "Normal Beat (N)",
    "V": "Premature Ventricular Contraction (V)",
    "A": "Atrial Premature Beat (A)",
    "F": "Fusion Beat (F)",
    "/": "Paced Beat (/)",
    "L": "Left Bundle Branch Block (L)",
    "R": "Right Bundle Branch Block (R)",
    "a": "Aberrated Atrial Premature (a)",
    "J": "Nodal Premature Beat (J)",
    "S": "Supraventricular Premature (S)",
    "E": "Ventricular Escape Beat (E)",
    "j": "Nodal Escape Beat (j)",
    "Q": "Unclassifiable Beat (Q)",
    "+": "Rhythm Change (+)",
    "~": "Signal Quality Change (~)",
    "!": "Ventricular Flutter Wave (!)"
}

def get_patient_annotation_context(patient):
    anns = getattr(patient, "_cached_annotations", None) or []
    curr_t = float(patient.time_position_sec) if patient else 0.0

    if not anns:
        return {
            "current": {
                "symbol": "N",
                "name": "Normal Beat (N)",
                "time_sec": round(curr_t, 2),
                "offset_sec": 0.0,
                "is_ventricular": False,
                "is_atrial": False
            },
            "upcoming": []
        }

    past = [a for a in anns if a["time_sec"] <= (curr_t + 0.15)]
    current = past[-1] if past else anns[0]
    upcoming = [a for a in anns if a["time_sec"] > (curr_t + 0.15)][:5]

    def format_ann(a):
        sym = str(a.get("symbol", "N"))
        t = round(float(a.get("time_sec", 0.0)), 2)
        diff = round(max(0.0, t - curr_t), 2)
        return {
            "symbol": sym,
            "name": SYMBOL_CLINICAL_NAMES.get(sym, f"Beat ({sym})"),
            "time_sec": t,
            "offset_sec": diff,
            "is_ventricular": bool(a.get("is_ventricular", False)),
            "is_atrial": bool(a.get("is_atrial", False))
        }

    return {
        "current": format_ann(current),
        "upcoming": [format_ann(a) for a in upcoming]
    }

def background_telemetry_worker():
    """
    Background worker that runs at 25 Hz (every 40ms generating 10 samples = 250 Hz real-time),
    and triggers signal processing / AI inference / Alert evaluation at 2 Hz (every 500ms).
    """
    sample_interval = 0.04 # 40ms loop interval
    analysis_counter = 0

    while True:
        loop_start = time.time()
        analysis_counter += 1

        # 1. Generate 10 samples (at 250Hz, 10 samples = 40ms) for each bed
        batch_output = telemetry_engine.generate_batch_for_all(num_samples=10)

        with state_lock:
            for pid, samples in batch_output.items():
                if pid not in live_telemetry_queues:
                    live_telemetry_queues[pid] = []
                live_telemetry_queues[pid].extend(samples)
                if len(live_telemetry_queues[pid]) > 500:
                    live_telemetry_queues[pid] = live_telemetry_queues[pid][-500:]

        # 2. Every ~500ms (every 12 cycles of 40ms), perform signal filtering, Pan-Tompkins, AI inference & CEWI
        if analysis_counter >= 12:
            analysis_counter = 0
            for pid, patient in telemetry_engine.patients.items():
                with state_lock:
                    lead2_buf = list(patient.lead2_buffer)
                    latest_vitals = {
                        "hr": patient.current_hr,
                        "spo2": patient.spo2,
                        "bp_sys": patient.bp_sys,
                        "bp_dia": patient.bp_dia,
                        "resp_rate": patient.resp_rate,
                        "temp_c": patient.temp_c,
                        "perfusion_index": patient.perfusion_index
                    }

                if len(lead2_buf) >= 120:
                    raw_signal = np.array(lead2_buf[-750:]) # 3 seconds window
                    filtered_signal = signal_processor.filter_signal(raw_signal)
                    peaks = signal_processor.pan_tompkins_qrs_detect(filtered_signal)
                    metrics = signal_processor.extract_clinical_metrics(filtered_signal, peaks)

                    # Update HR from validated Pan-Tompkins peaks if reliable
                    if metrics["heart_rate"] > 0:
                        latest_vitals["hr"] = float(metrics["heart_rate"])

                    # Run ML classification
                    ai_result = ai_engine.classify_rhythm(metrics, filtered_signal)

                    # Compute CEWI Early Warning score
                    cewi_result = ai_engine.compute_cewi_score(latest_vitals, metrics, ai_result["diagnosis"])

                    # Evaluate clinical alert rules
                    triggered_alert = alert_system.evaluate_and_trigger(
                        patient_id=pid,
                        patient_name=patient.name,
                        location=patient.location,
                        source_record=patient.mitbih_record_id or "SYNTHETIC",
                        diagnosis=ai_result["diagnosis"],
                        confidence=ai_result["confidence"],
                        cewi_data=cewi_result,
                        metrics=metrics,
                        signal_quality=metrics.get("signal_quality", "OPTIMAL"),
                        event_time_sec=patient.time_position_sec
                    )

                    ann_context = get_patient_annotation_context(patient)

                    with state_lock:
                        latest_patient_analysis[pid] = {
                            "patient_id": pid,
                            "name": patient.name,
                            "age": patient.age,
                            "gender": patient.gender,
                            "location": patient.location,
                            "current_rhythm": patient.current_rhythm,
                            "baseline_rhythm": patient.baseline_rhythm,
                            "data_mode": patient.data_mode,
                            "mitbih_record_id": patient.mitbih_record_id,
                            "playback_status": patient.playback_status,
                            "playback_speed": patient.playback_speed,
                            "time_position_sec": round(patient.time_position_sec, 2),
                            "vitals": latest_vitals,
                            "metrics": metrics,
                            "ai_diagnosis": ai_result,
                            "cewi": cewi_result,
                            "active_alert": triggered_alert,
                            "ref_annotation": ann_context["current"],
                            "upcoming_annotations": ann_context["upcoming"],
                            "recent_peaks": [int(p) for p in peaks[-8:]]
                        }

        # Maintain 40ms timing loop
        elapsed = time.time() - loop_start
        sleep_time = max(0.005, sample_interval - elapsed)
        time.sleep(sleep_time)

# Start background telemetry loop
worker_thread = threading.Thread(target=background_telemetry_worker, daemon=True)
worker_thread.start()


# --- ROUTES & ENDPOINTS ---

@app.route("/")
def index():
    """Renders the main ICU Central Telemetry Dashboard."""
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """Returns SVG telemetry favicon for browsers."""
    svg_icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#0b111e"/>'
        '<path d="M4 16h6l2.5-7 4 15 3.5-11 2 5.5 2-2.5h6" fill="none" stroke="#00e5ff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )
    return Response(svg_icon, mimetype="image/svg+xml")


@app.route("/health")
@app.route("/api/health")
def health_endpoint():
    """System health, pipeline metrics, dataset readiness, and model status."""
    with state_lock:
        active_count = len(alert_system.active_alerts)
        history_count = len(alert_system.alert_history)

    ml_report = ai_engine.evaluation_report or {}
    catalog_count = len(dataset_manager.catalog)

    return jsonify({
        "status": "healthy",
        "system": "CardioSentinel | ECG Intelligence",
        "version": "2.2.0-clinical-research",
        "sample_rate_hz": 250,
        "sample_rate": 250,
        "num_patients": len(telemetry_engine.patients),
        "monitored_beds": len(telemetry_engine.patients),
        "active_beds": len(telemetry_engine.patients),
        "dataset_ready": catalog_count > 0,
        "catalog_records": catalog_count,
        "model_ready": ai_engine.model is not None,
        "model_name": ml_report.get("model_name", "RandomForest v2.1"),
        "model_balanced_acc": ml_report.get("balanced_accuracy_percent", "Not evaluated"),
        "active_alerts_count": active_count,
        "total_alerts_recorded": history_count,
        "audio_muted": alert_system.audio_muted,
        "timestamp": time.time()
    })


@app.route("/api/patients", methods=["GET"])
def get_patients():
    """Returns single-source-of-truth status for all monitored patients."""
    with state_lock:
        data = list(latest_patient_analysis.values())
    return jsonify({"success": True, "patients": data})


@app.route("/api/patient/<patient_id>", methods=["GET"])
def get_patient_detail(patient_id):
    """Returns comprehensive telemetry detail for a single patient."""
    with state_lock:
        patient_data = latest_patient_analysis.get(patient_id)
    if not patient_data:
        return jsonify({"success": False, "error": "Patient not found"}), 404
    return jsonify({"success": True, "patient": patient_data})


@app.route("/api/patient/<patient_id>/copilot", methods=["GET"])
def get_patient_copilot(patient_id):
    """Returns real-time AI Clinical Copilot diagnostic narrative, morphological fiducials, and recommendations."""
    with state_lock:
        patient_data = latest_patient_analysis.get(patient_id)
    if not patient_data:
        return jsonify({"success": False, "error": "Patient not found"}), 404

    diag_info = patient_data.get("ai_diagnosis", {})
    copilot_data = ai_engine.generate_clinical_ai_copilot(
        patient_name=patient_data.get("name", "Unknown Patient"),
        patient_id=patient_id,
        vitals=patient_data.get("vitals", {}),
        metrics=patient_data.get("metrics", {}),
        diagnosis=diag_info.get("diagnosis", "Normal Sinus Rhythm"),
        confidence=diag_info.get("confidence", 0.95),
        cewi_data=patient_data.get("cewi", {})
    )
    return jsonify(copilot_data)


@app.route("/api/bed/<patient_id>/playback", methods=["POST"])
def bed_playback_control(patient_id):
    """
    Controls playback for a bed: play, pause, seek, speed, jump.
    Body JSON: {"action": "play"|"pause"|"seek"|"speed"|"jump", "value": ...}
    """
    patient = telemetry_engine.get_patient(patient_id)
    if not patient:
        return jsonify({"success": False, "error": "Bed not found"}), 404

    data = request.get_json() or {}
    action = data.get("action", "").lower()

    if action == "play":
        patient.set_playback_status("PLAYING")
    elif action == "pause":
        patient.set_playback_status("PAUSED")
    elif action == "seek":
        seconds = float(data.get("time_sec", data.get("value", 0.0)))
        patient.seek_to_time(seconds)
    elif action == "speed":
        speed = float(data.get("speed", data.get("value", 1.0)))
        patient.set_playback_speed(speed)
    elif action == "jump":
        direction = data.get("value", "next")
        evt = patient.jump_to_event(direction)
        return jsonify({
            "success": True,
            "action": action,
            "status": patient.playback_status,
            "target_event": evt,
            "time_position_sec": patient.time_position_sec
        })
    else:
        return jsonify({"success": False, "error": f"Unknown action: {action}"}), 400

    return jsonify({
        "success": True,
        "patient_id": patient_id,
        "status": patient.playback_status,
        "playback_status": patient.playback_status,
        "playback_speed": patient.playback_speed,
        "time_position_sec": patient.time_position_sec
    })


@app.route("/api/bed/<patient_id>/record", methods=["POST"])
def bed_assign_record(patient_id):
    """Assigns an MIT-BIH record to a bed."""
    patient = telemetry_engine.get_patient(patient_id)
    if not patient:
        return jsonify({"success": False, "error": "Bed not found"}), 404

    data = request.get_json() or {}
    record_id = str(data.get("record_id", "100"))
    success = telemetry_engine.switch_bed_record(patient_id, record_id)

    return jsonify({
        "success": success,
        "patient_id": patient_id,
        "mitbih_record_id": record_id,
        "data_mode": "MIT-BIH",
        "patient": {
            "patient_id": patient_id,
            "mitbih_record_id": record_id,
            "data_mode": "MIT-BIH"
        }
    })


@app.route("/api/trigger_arrhythmia", methods=["POST"])
def trigger_arrhythmia():
    """Injects an acute simulated arrhythmia event into a patient."""
    req = request.get_json() or {}
    patient_id = req.get("patient_id")
    rhythm = req.get("rhythm")
    duration = float(req.get("duration", 30.0))

    if not patient_id or not rhythm:
        return jsonify({"success": False, "error": "patient_id and rhythm required"}), 400

    success = telemetry_engine.trigger_event(patient_id, rhythm, duration)
    return jsonify({
        "success": success,
        "patient_id": patient_id,
        "rhythm": rhythm,
        "duration": duration
    })


# --- DATASET API ---

@app.route("/api/dataset/records", methods=["GET"])
def get_dataset_records():
    """Returns the full catalog of all 48 MIT-BIH Arrhythmia Database recordings."""
    catalog = dataset_manager.get_catalog_list()
    return jsonify({
        "success": True,
        "total": len(catalog),
        "playable_count": sum(1 for c in catalog if c.get("is_playable")),
        "records": catalog
    })


@app.route("/api/dataset/summary", methods=["GET"])
def get_dataset_summary():
    """Returns summary metadata for PhysioNet database."""
    summary_data = dataset_manager.get_dataset_summary()
    return jsonify({
        "success": True,
        "dataset_name": "PhysioNet MIT-BIH Arrhythmia Database",
        "dataset": summary_data,
        "database": "PhysioNet MIT-BIH Arrhythmia Database (v1.0.0)",
        "total_expected_records": 48,
        "cataloged_records": len(dataset_manager.catalog),
        "playable_records": summary_data.get("fully_playable_records", 1),
        "sample_rate_native": 360,
        "sample_rate_telemetry": 250,
        "disclaimer": "Academic PhysioNet MIT-BIH dataset."
    })


@app.route("/api/dataset/train_mitbih", methods=["POST"])
def train_mitbih_endpoint():
    """Triggers ML training and cross-validation on PhysioNet dataset."""
    result = ai_engine.train_and_validate()
    return jsonify({"success": True, **result})


@app.route("/api/export_report/<patient_id>", methods=["GET"])
def export_single_patient_report(patient_id):
    """Generates structured clinical report for a single patient."""
    patient = telemetry_engine.get_patient(patient_id)
    if not patient:
        return jsonify({"success": False, "error": "Patient not found"}), 404

    with state_lock:
        analysis = latest_patient_analysis.get(patient_id, {})
        strip = list(patient.lead2_buffer[-1000:])

    report = {
        "report_id": f"RPT-{int(time.time())}-{patient_id}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patient": {
            "id": patient.patient_id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "location": patient.location
        },
        "clinical_summary": {
            "primary_rhythm": analysis.get("ai_diagnosis", {}).get("diagnosis", patient.current_rhythm),
            "ai_confidence": analysis.get("ai_diagnosis", {}).get("confidence", 0.95),
            "cewi_score": analysis.get("cewi", {}).get("cewi_score", 10),
            "risk_level": analysis.get("cewi", {}).get("risk_level", "LOW RISK"),
            "shock_index": analysis.get("cewi", {}).get("shock_index", 0.6),
            "vitals": analysis.get("vitals", {}),
            "electrophysiology_metrics": analysis.get("metrics", {}),
            "clinical_reasons": analysis.get("cewi", {}).get("reasons", [])
        },
        "ecg_strip_sample": strip,
        "dispatched_alerts": [a for a in alert_system.get_alert_history() if a.get("patient_id") == patient_id][:5]
    }
    return jsonify({"success": True, "report": report})


@app.route("/api/dataset/segment", methods=["GET"])
def get_dataset_segment():
    """Returns windowed slice of signals and annotations for ECG viewer/playback."""
    record_id = request.args.get("record_id", "100")
    start_sec = float(request.args.get("start_sec", 0.0))
    duration_sec = float(request.args.get("duration_sec", 10.0))

    segment = dataset_manager.get_segment(record_id, start_sec, duration_sec, target_fs=250)
    return jsonify({"success": True, "segment": segment})


@app.route("/api/dataset/set_mode", methods=["POST"])
def set_dataset_mode():
    """Switches bed between MIT-BIH recorded stream and synthetic generator."""
    req = request.get_json() or {}
    mode = req.get("mode", "MIT-BIH").upper()
    patient_id = req.get("patient_id")
    record_id = req.get("record_id")

    if patient_id:
        p = telemetry_engine.get_patient(patient_id)
        if not p:
            return jsonify({"success": False, "error": "Patient not found"}), 404
        p.set_data_mode(mode, record_id)
        return jsonify({"success": True, "patient_id": patient_id, "mode": mode, "record_id": p.mitbih_record_id})
    else:
        for p in telemetry_engine.patients.values():
            p.set_data_mode(mode)
        return jsonify({"success": True, "mode": mode, "all_beds": True})


# --- MACHINE LEARNING API ---

@app.route("/api/ml/metrics", methods=["GET"])
def get_ml_metrics():
    """Returns honest evaluation metrics, confusion matrix, and feature importances."""
    report = ai_engine.evaluation_report
    if not report:
        return jsonify({"success": False, "status": "Not evaluated", "report": None})
    return jsonify({"success": True, "report": report})


@app.route("/api/ml/train", methods=["POST"])
def trigger_ml_training():
    """Triggers asynchronous model training and cross-validation."""
    req = request.get_json() or {}
    algorithm = req.get("algorithm", "RandomForest")
    n_estimators = int(req.get("n_estimators", 100))

    started, msg = ai_engine.start_async_training(algorithm=algorithm, n_estimators=n_estimators)
    return jsonify({"success": started, "message": msg})


@app.route("/api/ml/train_status", methods=["GET"])
def get_ml_train_status():
    """Returns background training progress and status."""
    return jsonify({"success": True, "status": ai_engine.training_status})


@app.route("/api/ml/explain", methods=["GET"])
def get_ml_explain():
    """Returns explainability features, Poincaré plot coordinates, and RR tachogram."""
    patient_id = request.args.get("patient_id", "BED-01")
    with state_lock:
        analysis = latest_patient_analysis.get(patient_id, {})
    metrics = analysis.get("metrics", {})
    diag = analysis.get("ai_diagnosis", {}).get("diagnosis", "Normal Sinus Rhythm")

    explain = ai_engine.get_explainability(metrics, diag)
    return jsonify({"success": True, "patient_id": patient_id, "explainability": explain})


# --- ALERTS API ---

@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Returns active alarms and audit history without flooding."""
    with state_lock:
        active = alert_system.get_active_alerts()
        history = alert_system.get_alert_history(limit=50)
        unread_count = alert_system.get_unread_count()
    return jsonify({
        "success": True,
        "active_alerts": active,
        "alert_history": history,
        "unread_count": unread_count,
        "audio_muted": alert_system.audio_muted
    })


@app.route("/api/alerts/action", methods=["POST"])
def alert_action():
    """Handles acknowledge, resolve, snooze, mark read, and audio toggle."""
    req = request.get_json() or {}
    action = req.get("action", "").lower()
    alert_id = req.get("alert_id")
    patient_id = req.get("patient_id")

    if action in ["ack", "acknowledge"] and alert_id:
        by = req.get("clinician_name", "Staff Nurse")
        ok = alert_system.acknowledge_alert(alert_id, by)
        return jsonify({"success": ok, "message": f"Alert {alert_id} acknowledged."})

    elif action in ["mark_read", "read"] and alert_id:
        ok = alert_system.mark_read(alert_id)
        return jsonify({"success": ok, "message": f"Alert {alert_id} marked as read.", "unread_count": alert_system.get_unread_count()})

    elif action in ["mark_all_read", "read_all"]:
        count = alert_system.mark_all_read()
        return jsonify({"success": True, "marked_count": count, "message": f"Marked {count} alerts as read.", "unread_count": 0})

    elif action == "resolve" and alert_id:
        ok = alert_system.resolve_alert(alert_id, resolved_by="Clinician Manual Action")
        return jsonify({"success": ok, "message": f"Alert {alert_id} resolved."})

    elif action == "snooze":
        if not patient_id and alert_id:
            for p, a in alert_system.active_alerts.items():
                if a.get("alert_id") == alert_id:
                    patient_id = a.get("patient_id")
                    break
            if not patient_id:
                for a in alert_system.alert_history:
                    if a.get("alert_id") == alert_id:
                        patient_id = a.get("patient_id")
                        break
            if not patient_id and "_" in alert_id:
                parts = alert_id.split("_")
                if len(parts) >= 2:
                    patient_id = parts[1]

        if patient_id:
            sec = float(req.get("snooze_duration_sec", 0.0))
            mins = float(req.get("minutes", (sec / 60.0) if sec > 0 else 5.0))
            res = alert_system.snooze_alert(patient_id, minutes=mins)
            return jsonify(res)

    elif action == "toggle_mute":
        muted = alert_system.toggle_audio_mute()
        return jsonify({"success": True, "audio_muted": muted})

    return jsonify({"success": False, "error": "Invalid action parameters"}), 400


@app.route("/api/alerts/export.csv", methods=["GET"])
def export_alerts_csv():
    """Exports clinical alert log as downloadable CSV file."""
    csv_text = alert_system.export_alerts_csv()
    response = make_response(csv_text)
    response.headers["Content-Disposition"] = "attachment; filename=cardiosentinel_alerts.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


# --- SHIFT REPORTS & EXPORTS ---

@app.route("/api/reports/shift", methods=["GET"])
def get_shift_report():
    """Generates shift telemetry summary for all beds or a specific bed."""
    patient_id = request.args.get("patient_id")
    with state_lock:
        data = dict(latest_patient_analysis)

    beds_summary = []
    for pid, info in data.items():
        if patient_id and pid != patient_id:
            continue
        vitals = info.get("vitals", {})
        metrics = info.get("metrics", {})
        cewi = info.get("cewi", {})
        ai = info.get("ai_diagnosis", {})

        beds_summary.append({
            "bed": pid,
            "patient_name": info.get("name"),
            "location": info.get("location"),
            "data_source": f"MIT-BIH Record {info.get('mitbih_record_id')}" if info.get("data_mode") == "MIT-BIH" else "Synthetic Simulation",
            "heart_rate": vitals.get("hr"),
            "spo2": vitals.get("spo2"),
            "bp": f"{vitals.get('bp_sys')}/{vitals.get('bp_dia')}",
            "diagnosis": ai.get("diagnosis"),
            "cewi_score": cewi.get("cewi_score"),
            "risk_level": cewi.get("risk_level"),
            "sdnn": metrics.get("hrv_sdnn"),
            "rmssd": metrics.get("hrv_rmssd"),
            "qrs_ms": metrics.get("qrs_duration_ms"),
            "qtc_ms": metrics.get("qtc_ms")
        })

    report_dict = {
        "report_id": f"SHIFT-{int(time.time())}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shift": "ICU Day/Night Telemetry Shift",
        "beds_monitored": beds_summary,
        "clinical_summary": f"All {len(beds_summary)} telemetry bed channels monitored. Continuous rhythm analysis and CEWI alerting active.",
        "research_disclaimer": "CardioSentinel | ECG Intelligence - Academic Educational Use Only. Not clinically validated."
    }

    return jsonify({
        "success": True,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shift": "ICU Day/Night Telemetry Shift",
        "beds_count": len(beds_summary),
        "beds": beds_summary,
        "beds_monitored": beds_summary,
        "report": report_dict,
        "disclaimer": "CardioSentinel | ECG Intelligence - Academic Educational Use Only."
    })


@app.route("/api/reports/shift/export", methods=["GET"])
def export_shift_report_csv():
    """Exports shift telemetry report as CSV."""
    with state_lock:
        data = dict(latest_patient_analysis)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Bed", "Patient Name", "Location", "Data Source", "Heart Rate (BPM)",
        "SpO2 (%)", "BP (Sys/Dia)", "Classifier Diagnosis", "CEWI Score", "Risk Level",
        "SDNN (ms)", "RMSSD (ms)", "QRS (ms)", "QTc (ms)"
    ])

    for pid, info in data.items():
        vitals = info.get("vitals", {})
        metrics = info.get("metrics", {})
        cewi = info.get("cewi", {})
        ai = info.get("ai_diagnosis", {})
        writer.writerow([
            pid,
            info.get("name"),
            info.get("location"),
            f"MIT-BIH {info.get('mitbih_record_id')}" if info.get("data_mode") == "MIT-BIH" else "Synthetic",
            vitals.get("hr"),
            vitals.get("spo2"),
            f"{vitals.get('bp_sys')}/{vitals.get('bp_dia')}",
            ai.get("diagnosis"),
            cewi.get("cewi_score"),
            cewi.get("risk_level"),
            metrics.get("hrv_sdnn"),
            metrics.get("hrv_rmssd"),
            metrics.get("qrs_duration_ms"),
            metrics.get("qtc_ms")
        ])

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=cardiosentinel_shift_report.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


# --- HIGH-PERFORMANCE SSE TELEMETRY STREAM ---

@app.route("/api/stream")
def telemetry_stream():
    """
    High-performance Server-Sent Events (SSE) telemetry stream.
    Delivers 40ms batches of Lead II and Lead V1 points to the browser,
    along with vitals, metrics, AI predictions, and active alerts.
    """
    def event_generator():
        while True:
            batch_payload = {}
            with state_lock:
                for pid in telemetry_engine.patients.keys():
                    queue = live_telemetry_queues.get(pid, [])
                    chunk = list(queue)
                    live_telemetry_queues[pid] = []

                    analysis = latest_patient_analysis.get(pid, {})
                    batch_payload[pid] = {
                        "samples": [
                            {
                                "l2": s["lead2"],
                                "v1": s["lead_v1"],
                                "r": s["is_r_peak"]
                            }
                            for s in chunk
                        ],
                        "vitals": analysis.get("vitals", {}),
                        "metrics": analysis.get("metrics", {}),
                        "ai": analysis.get("ai_diagnosis", {}),
                        "cewi": analysis.get("cewi", {}),
                        "active_alert": analysis.get("active_alert"),
                        "ref_annotation": analysis.get("ref_annotation"),
                        "upcoming_annotations": analysis.get("upcoming_annotations", []),
                        "playback": {
                            "status": analysis.get("playback_status", "PLAYING"),
                            "speed": analysis.get("playback_speed", 1.0),
                            "time_sec": analysis.get("time_position_sec", 0.0),
                            "record_id": analysis.get("mitbih_record_id"),
                            "data_mode": analysis.get("data_mode", "MIT-BIH")
                        }
                    }

            yield f"data: {json.dumps(batch_payload)}\n\n"
            time.sleep(0.04) # ~25 FPS

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import webbrowser
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 70)
    print("CardioSentinel | ECG Intelligence starting...")
    print(f"Hospital Beds Monitored: {list(telemetry_engine.patients.keys())}")
    print(f"Web Console (Plain HTTP): http://127.0.0.1:{port}")
    print("=" * 70)

    app.run(host=host, port=port, debug=False, threaded=True)
