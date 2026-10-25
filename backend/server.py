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
from backend.database import DatabaseManager, normalize_phone_number, mask_phone_number
from backend.sms_service import SMSService, sms_service
from dotenv import load_dotenv

# Flask app configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

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

                    # Evaluate clinical alert rules (Safety check: Never evaluate or trigger alerts during active ML model training)
                    if getattr(ai_engine, "is_training", False):
                        triggered_alert = None
                    else:
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

                    # Evaluate Guardian Emergency Notification
                    # Trigger notification ONLY when a NEW qualifying emergency episode is detected
                    if triggered_alert and triggered_alert.get("is_new_emergency", False) and not getattr(ai_engine, "is_training", False):
                        try:
                            guardian_eval = sms_service.send_emergency_alert(
                                alert_payload=triggered_alert,
                                patient_data_mode=patient.data_mode
                            )
                            if isinstance(guardian_eval, dict):
                                triggered_alert["guardian_dispatch"] = guardian_eval
                                if triggered_alert.get("episode_id"):
                                    notif_uuid = guardian_eval.get("notification_uuid") or (
                                        guardian_eval.get("notification", {}).get("notification_uuid")
                                    )
                                    notif_sent = bool(guardian_eval.get("eligible"))
                                    notif_status = guardian_eval.get("mode") or guardian_eval.get("status")
                                    alert_system.record_episode_notification(
                                        pid,
                                        triggered_alert["episode_id"],
                                        notification_uuid=notif_uuid,
                                        notification_sent=notif_sent,
                                        notification_status=notif_status
                                    )
                        except Exception as e:
                            # Guardian notification failures must never crash ECG monitoring
                            pass

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
                            "clinical_state": alert_system.get_patient_state(pid),
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


def build_electrophysiology_payload(patient_id: str) -> Optional[Dict[str, Any]]:
    """Builds comprehensive, validated electrophysiological breakdown payload for a monitored patient."""
    with state_lock:
        patient = telemetry_engine.get_patient(patient_id)
        patient_data = latest_patient_analysis.get(patient_id)

    if not patient:
        return None

    # Get metrics from latest analysis or compute on current lead2 buffer
    metrics = (patient_data.get("metrics") if patient_data else None)
    if not metrics:
        lead2_buf = list(patient.lead2_buffer)
        if len(lead2_buf) >= 40:
            raw_signal = np.array(lead2_buf[-750:])
            filtered_signal = signal_processor.filter_signal(raw_signal)
            peaks = signal_processor.pan_tompkins_qrs_detect(filtered_signal)
            metrics = signal_processor.extract_clinical_metrics(filtered_signal, peaks)
        else:
            metrics = signal_processor.extract_clinical_metrics(np.array([]), [])

    now_ts = time.time()
    iso_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))

    is_mitbih = (patient.data_mode == "MIT-BIH")
    source_label = f"PhysioNet MIT-BIH Arrhythmia Database (Record {patient.mitbih_record_id})" if is_mitbih else "Real-time Bedside Monitor Telemetry"
    mode_label = f"RECORDED-DATA ANALYSIS (PhysioNet MIT-BIH Record {patient.mitbih_record_id})" if is_mitbih else "LIVE TELEMETRY STREAM"

    unavailable = []
    if is_mitbih:
        unavailable = [
            {
                "parameter": "Blood Pressure (NIBP / Arterial)",
                "unit": "mmHg",
                "status": "Omitted by Design",
                "reason": "The PhysioNet MIT-BIH Arrhythmia Database contains 2-channel analog ECG digitized at 360 Hz. Arterial lines or NIBP cuffs were not recorded in this Holter study. Pressure values are not fabricated.",
                "scientific_integrity": True
            },
            {
                "parameter": "Pulse Oximetry (SpO2)",
                "unit": "%",
                "status": "Omitted by Design",
                "reason": "Photoplethysmography sensors were not part of the PhysioNet MIT-BIH research protocol. Oxygen saturation is omitted to prevent synthetic fabrication.",
                "scientific_integrity": True
            },
            {
                "parameter": "Core Body Temperature",
                "unit": "°C",
                "status": "Omitted by Design",
                "reason": "PhysioNet MIT-BIH ambulatory Holter recordings did not deploy temperature thermistors.",
                "scientific_integrity": True
            }
        ]

    diag_info = patient_data.get("ai_diagnosis", {}) if patient_data else {}

    return {
        "success": True,
        "patient_id": patient_id,
        "patient_name": patient.name,
        "location": patient.location,
        "age": patient.age,
        "gender": patient.gender,
        "data_mode": patient.data_mode,
        "mitbih_record_id": patient.mitbih_record_id,
        "source_label": source_label,
        "mode_label": mode_label,
        "is_historical_recorded": is_mitbih,
        "timestamp": now_ts,
        "formatted_timestamp": iso_ts,
        "playback": {
            "status": patient.playback_status,
            "speed": patient.playback_speed,
            "time_position_sec": round(patient.time_position_sec, 2),
            "record_id": patient.mitbih_record_id
        },
        "signal_quality": {
            "status": metrics.get("signal_quality", "OPTIMAL"),
            "technical_quality_status": metrics.get("technical_quality_status", "GOOD SIGNAL"),
            "technical_quality_reason": metrics.get("technical_quality_reason", "Signal quality checks passed."),
            "quality_score": metrics.get("quality_score", 95),
            "snr_db": metrics.get("snr_db", 24.5),
            "is_suitable": metrics.get("is_suitable", True),
            "quality_assessment": metrics.get("quality_assessment", {}),
            "lead": "Lead II (Bipolar Limb / Modified Chest)",
            "sample_rate_hz": signal_processor.sample_rate
        },
        "measurements": {
            "heart_rate": {
                "value": metrics.get("heart_rate", 0),
                "unit": "bpm",
                "measurable": metrics.get("hr_measurable", False),
                "status_note": metrics.get("hr_status_note", "Normal Sinus"),
                "reference_range": "60 – 100 bpm",
                "critical_threshold": "<40 or >150 bpm",
                "unmeasurable_reason": metrics.get("hr_unmeasurable_reason")
            },
            "qrs_duration": {
                "value": metrics.get("qrs_duration_ms", 0.0),
                "unit": "ms",
                "measurable": metrics.get("qrs_measurable", False),
                "status_note": metrics.get("qrs_status_note", "Normal (<100ms)"),
                "reference_range": "80 – 100 ms",
                "critical_threshold": ">120 ms (Wide QRS / Conduction Block)",
                "method": "Pan-Tompkins adaptive Q-onset to J-point baseline return",
                "unmeasurable_reason": metrics.get("qrs_unmeasurable_reason")
            },
            "qt_interval": {
                "raw_qt_ms": metrics.get("qt_ms", 0.0),
                "qtc_bazett_ms": metrics.get("qtc_bazett_ms", metrics.get("qtc_ms", 0.0)),
                "qtc_fridericia_ms": metrics.get("qtc_fridericia_ms", 0.0),
                "unit": "ms",
                "measurable": metrics.get("qt_measurable", False),
                "status_note": metrics.get("qt_status_note", "Normal (<440ms)"),
                "reference_range": "<440 ms (Male), <460 ms (Female)",
                "critical_threshold": ">480 ms (Elevated TdP Risk)",
                "formulas": {
                    "bazett": "QTc = QT / sqrt(RR_sec)",
                    "fridericia": "QTc = QT / cbrt(RR_sec)"
                },
                "unmeasurable_reason": metrics.get("qt_unmeasurable_reason")
            },
            "hrv": {
                "sdnn": metrics.get("hrv_sdnn", 0.0),
                "rmssd": metrics.get("hrv_rmssd", 0.0),
                "pnn50": metrics.get("pnn50", 0.0),
                "poincare_sd1": metrics.get("poincare_sd1", 0.0),
                "poincare_sd2": metrics.get("poincare_sd2", 0.0),
                "shannon_entropy": metrics.get("rr_entropy", 0.0),
                "mean_rr_ms": metrics.get("mean_rr_ms", 0.0),
                "unit": "ms",
                "measurable": metrics.get("hrv_measurable", False),
                "status_note": metrics.get("hrv_status_note", "Normal"),
                "reference_range": "SDNN: 30 – 60 ms | RMSSD: 20 – 50 ms",
                "critical_threshold": "SDNN < 20 ms (Severe Autonomic Depression)",
                "unmeasurable_reason": metrics.get("hrv_unmeasurable_reason")
            }
        },
        "conduction_diagnosis": {
            "current_rhythm": patient.current_rhythm,
            "baseline_rhythm": patient.baseline_rhythm,
            "ai_diagnosis": diag_info.get("diagnosis", "Normal Sinus Rhythm"),
            "ai_confidence": diag_info.get("confidence", 0.95),
            "ref_annotation": patient_data.get("ref_annotation") if patient_data else None
        },
        "unavailable_measurements": unavailable,
        "research_disclaimer": "Informational Research & Exploratory Metrics Only: Interval calculations (Pan-Tompkins QRS onset-to-J-point, Bazett/Fridericia QTc, Poincaré HRV descriptors) are validated algorithmic estimates intended for research, engineering verification, and continuous telemetry trending. They do not constitute a clinical diagnosis or medical conclusion. Clinical interventions require 12-lead diagnostic electrocardiography and certified physician evaluation."
    }


@app.route("/api/patient/<patient_id>/electrophysiology", methods=["GET"])
def get_patient_electrophysiology(patient_id):
    """Returns validated electrophysiological interval breakdown for a monitored bed."""
    payload = build_electrophysiology_payload(patient_id)
    if not payload:
        return jsonify({"success": False, "error": f"Patient '{patient_id}' not found"}), 404
    return jsonify(payload)


@app.route("/api/electrophysiology", methods=["GET"])
def get_current_electrophysiology():
    """Returns electrophysiology breakdown for specified bed (?patient_id=...) or first active bed."""
    pid = request.args.get("patient_id")
    if not pid:
        with state_lock:
            pids = list(telemetry_engine.patients.keys())
        pid = pids[0] if pids else "BED-01"

    payload = build_electrophysiology_payload(pid)
    if not payload:
        return jsonify({"success": False, "error": f"Patient '{pid}' not found"}), 404
    return jsonify(payload)


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
    """Returns the catalog of MIT-BIH recordings with reference classifications and filtering."""
    filter_cls = request.args.get("filter", "ALL").strip().upper()
    search = request.args.get("search", "").strip()

    catalog = dataset_manager.get_catalog_list()
    counts = {
        "all": len(catalog),
        "normal": sum(1 for c in catalog if (c.get("classification") or "").upper() == "NORMAL"),
        "abnormal": sum(1 for c in catalog if (c.get("classification") or "").upper() == "ABNORMAL"),
        "unclassified": sum(1 for c in catalog if (c.get("classification") or "").upper() == "UNCLASSIFIED")
    }

    filtered = dataset_manager.filter_catalog(classification_filter=filter_cls, search_term=search)

    return jsonify({
        "success": True,
        "total": len(catalog),
        "filtered_count": len(filtered),
        "counts": counts,
        "playable_count": sum(1 for c in catalog if c.get("is_playable")),
        "records": filtered
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
    meta = dataset_manager.get_record_metadata(record_id) or {}
    classif = dataset_manager.classify_record(meta)
    pat_info = dataset_manager.parse_patient_info(meta)

    return jsonify({
        "success": True,
        "segment": segment,
        "record_classification": classif["classification_label"],
        "classification": classif["classification"],
        "classification_reason": classif["reason"],
        "patient_info": pat_info
    })


@app.route("/api/dataset/record_analysis", methods=["GET"])
def get_dataset_record_analysis():
    """
    Returns separate ECG waveform, reference ground-truth annotations,
    and Random Forest model predictions for a selected MIT-BIH record.
    Keeps dataset reference classification strictly isolated from real-time alerting.
    """
    record_id = request.args.get("record_id", "100")
    start_sec = float(request.args.get("start_sec", 0.0))
    duration_sec = float(request.args.get("duration_sec", 10.0))

    analysis = dataset_manager.get_record_analysis(record_id, start_sec=start_sec, duration_sec=duration_sec, target_fs=250)
    status_code = 200 if analysis.get("success") else 404
    return jsonify(analysis), status_code


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
    """Returns honest evaluation metrics, confusion matrix, training & testing dataset records, and feature importances."""
    report = ai_engine.evaluation_report
    if not report:
        report = ai_engine.get_evaluation_metrics()
    return jsonify({"success": True, "report": report})


@app.route("/api/ml/train", methods=["POST"])
def trigger_ml_training():
    """Triggers asynchronous model training and evaluation with configurable split."""
    req = request.get_json() or {}
    algorithm = req.get("algorithm", "RandomForest")
    n_estimators = int(req.get("n_estimators", 100))
    train_ratio = float(req.get("train_ratio", 0.70))

    if train_ratio < 0.50 or train_ratio > 0.95:
        train_ratio = 0.70

    started, msg = ai_engine.start_async_training(
        algorithm=algorithm,
        n_estimators=n_estimators,
        train_ratio=train_ratio
    )
    if not started:
        return jsonify({"success": False, "message": msg}), 409
    return jsonify({"success": True, "message": msg})


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


@app.route("/api/alerts/<alert_id>/explanation", methods=["GET"])
def get_alert_explanation_endpoint(alert_id):
    """Returns structured smart alert explanation for an alert."""
    with state_lock:
        explanation = alert_system.get_alert_explanation(alert_id)
    if not explanation:
        return jsonify({"success": False, "error": f"Alert '{alert_id}' not found"}), 404
    return jsonify({
        "success": True,
        "alert_id": alert_id,
        "explanation": explanation
    })


@app.route("/api/alerts/action", methods=["POST"])
def alert_action():
    """Handles acknowledge, resolve, snooze, dismiss, mark read, and audio toggle."""
    req = request.get_json() or {}
    action = req.get("action", "").lower()
    alert_id = req.get("alert_id")
    patient_id = req.get("patient_id")

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

    if action in ["ack", "acknowledge"] and alert_id:
        by = req.get("clinician_name", "Staff Nurse")
        ok = alert_system.acknowledge_alert(alert_id, by)
        with state_lock:
            for pid, analysis in latest_patient_analysis.items():
                cur = analysis.get("active_alert")
                if cur and (cur.get("alert_id") == alert_id or pid == patient_id):
                    cur["acknowledged"] = True
                    cur["status"] = "ACKNOWLEDGED"
        return jsonify({
            "success": ok,
            "action": "acknowledge",
            "alert_id": alert_id,
            "patient_id": patient_id,
            "message": f"Alert {alert_id} acknowledged."
        })

    elif action in ["dismiss", "dismiss_banner"] and alert_id:
        by = req.get("clinician_name", "Clinician")
        ok = alert_system.dismiss_banner_alert(alert_id, by)
        with state_lock:
            for pid, analysis in latest_patient_analysis.items():
                cur = analysis.get("active_alert")
                if cur and (cur.get("alert_id") == alert_id or pid == patient_id):
                    cur["banner_dismissed"] = True
        return jsonify({
            "success": ok,
            "action": "dismiss",
            "alert_id": alert_id,
            "patient_id": patient_id,
            "message": f"Alert {alert_id} dismissed from banner."
        })

    elif action in ["mark_read", "read"] and alert_id:
        ok = alert_system.mark_read(alert_id)
        return jsonify({"success": ok, "message": f"Alert {alert_id} marked as read.", "unread_count": alert_system.get_unread_count()})

    elif action in ["mark_all_read", "read_all"]:
        count = alert_system.mark_all_read()
        return jsonify({"success": True, "marked_count": count, "message": f"Marked {count} alerts as read.", "unread_count": 0})

    elif action == "resolve" and alert_id:
        by = req.get("clinician_name", "Clinician Manual Action")
        ok = alert_system.resolve_alert(alert_id, resolved_by=by)
        with state_lock:
            for pid, analysis in latest_patient_analysis.items():
                cur = analysis.get("active_alert")
                if cur and (cur.get("alert_id") == alert_id or pid == patient_id):
                    analysis["active_alert"] = None
        return jsonify({
            "success": ok,
            "action": "resolve",
            "alert_id": alert_id,
            "patient_id": patient_id,
            "message": f"Alert {alert_id} resolved."
        })

    elif action == "snooze":
        if patient_id:
            sec = float(req.get("snooze_duration_sec", 0.0))
            mins = int(round(float(req.get("minutes", (sec / 60.0) if sec > 0 else 5.0))))
            by = req.get("clinician_name", "Clinician")
            res = alert_system.snooze_alert(patient_id, minutes=mins, snoozed_by=by, alert_id=alert_id)
            with state_lock:
                for pid, analysis in latest_patient_analysis.items():
                    cur = analysis.get("active_alert")
                    if cur and (cur.get("alert_id") == alert_id or pid == patient_id):
                        cur["status"] = "SNOOZED"
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


# --- GUARDIAN EMERGENCY NOTIFICATION API ---

@app.route("/api/guardian", methods=["GET"])
@app.route("/api/guardian/<patient_id>", methods=["GET"])
def get_guardian_endpoint(patient_id=None):
    """Retrieves guardian details, readiness checks, provider status, delivery metrics, and verification status."""
    pid = patient_id or request.args.get("patient_id", "BED-01")
    patient = telemetry_engine.get_patient(pid)
    data_mode = patient.data_mode if patient else "MIT-BIH"

    guardian = DatabaseManager.get_guardian(pid)
    readiness = sms_service.check_readiness(pid, data_mode=data_mode)
    provider_info = sms_service.get_provider_status()
    summary = DatabaseManager.get_delivery_summary(pid)
    verif_status = DatabaseManager.get_verification_status(pid, provider_info["is_configured"] or sms_service.is_mock_mode)

    return jsonify({
        "success": True,
        "patient_id": pid,
        "patient_name": patient.name if patient else pid,
        "guardian": guardian,
        "readiness": readiness,
        "provider": provider_info,
        "summary": summary,
        "verification_status": verif_status
    })


@app.route("/api/guardian", methods=["POST"])
def save_guardian_endpoint():
    """Saves or updates guardian settings with server-side validation and audit logging."""
    req = request.get_json() or {}
    pid = req.get("patient_id", "BED-01")

    ok, updated_guardian, msg = DatabaseManager.save_guardian(pid, req)
    if not ok:
        return jsonify({"success": False, "error": msg}), 400

    patient = telemetry_engine.get_patient(pid)
    data_mode = patient.data_mode if patient else "MIT-BIH"
    readiness = sms_service.check_readiness(pid, data_mode=data_mode)
    provider_info = sms_service.get_provider_status()
    verif_status = DatabaseManager.get_verification_status(pid, provider_info["is_configured"] or sms_service.is_mock_mode)

    is_dup = ("already up to date" in msg)
    return jsonify({
        "success": True,
        "message": msg,
        "duplicate": is_dup,
        "guardian": updated_guardian,
        "readiness": readiness,
        "verification_status": verif_status
    })




@app.route("/api/guardian/remove", methods=["POST"])
@app.route("/api/guardian/<patient_id>", methods=["DELETE"])
def guardian_remove_endpoint(patient_id=None):
    """Explicitly removes the registered guardian for a bed and disarms real notifications."""
    req = request.get_json() or {} if request.method == "POST" else {}
    pid = patient_id or req.get("patient_id", "BED-01")

    ok, msg = DatabaseManager.remove_guardian(pid)
    if not ok:
        return jsonify({"success": False, "error": msg}), 400

    patient = telemetry_engine.get_patient(pid)
    data_mode = patient.data_mode if patient else "MIT-BIH"
    readiness = sms_service.check_readiness(pid, data_mode=data_mode)
    provider_info = sms_service.get_provider_status()

    return jsonify({
        "success": True,
        "message": msg,
        "patient_id": pid,
        "guardian": None,
        "readiness": readiness,
        "provider": provider_info
    })


@app.route("/api/bed/<patient_id>/mode", methods=["POST"])
def set_bed_data_mode_endpoint(patient_id):
    """
    Toggles bed data source mode between MIT-BIH (Recorded Playback / Demonstration Mode)
    and LIVE (Live Telemetry Stream Mode with real emergency notification capability).
    """
    patient = telemetry_engine.get_patient(patient_id)
    if not patient:
        return jsonify({"success": False, "error": "Bed not found"}), 404

    data = request.get_json() or {}
    new_mode = str(data.get("data_mode", data.get("mode", "MIT-BIH"))).strip().upper()
    canonical_mode = "LIVE" if new_mode in ["LIVE", "REAL", "TELEMETRY", "SYNTHETIC"] else "MIT-BIH"

    patient.data_mode = canonical_mode
    readiness = sms_service.check_readiness(patient_id, data_mode=canonical_mode)
    provider_info = sms_service.get_provider_status()

    return jsonify({
        "success": True,
        "patient_id": patient_id,
        "data_mode": canonical_mode,
        "readiness": readiness,
        "provider": provider_info,
        "message": f"Data source for {patient_id} set to {canonical_mode}."
    })


@app.route("/api/guardian/readiness", methods=["GET"])
@app.route("/api/guardian/readiness/<patient_id>", methods=["GET"])
def guardian_readiness_endpoint(patient_id=None):
    """Returns 5-point readiness audit for guardian emergency dispatch."""
    pid = patient_id or request.args.get("patient_id", "BED-01")
    patient = telemetry_engine.get_patient(pid)
    data_mode = patient.data_mode if patient else "MIT-BIH"

    readiness = sms_service.check_readiness(pid, data_mode=data_mode)
    return jsonify({"success": True, "readiness": readiness})


@app.route("/api/guardian/test_sms_preview", methods=["GET"])
@app.route("/api/guardian/test_sms_preview/<patient_id>", methods=["GET"])
def guardian_test_sms_preview_endpoint(patient_id=None):
    """Returns safe preview of test SMS recipient, bed, and message without sending."""
    pid = patient_id or request.args.get("patient_id", "BED-01")
    preview = sms_service.get_test_sms_preview(pid)
    status_code = 200 if preview.get("success") else 404
    return jsonify(preview), status_code


@app.route("/api/guardian/test_sms", methods=["POST"])
def guardian_send_test_sms_endpoint():
    """Executes validated Test SMS dispatch to guardian contact."""
    req = request.get_json() or {}
    pid = req.get("patient_id", "BED-01")

    result = sms_service.send_test_sms(pid, triggered_by="Console User / Clinician")
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/guardian/preview_demo", methods=["POST"])
def guardian_preview_demo_endpoint():
    """Records demonstration notification preview for recorded ECG playback."""
    req = request.get_json() or {}
    pid = req.get("patient_id", "BED-01")
    condition = req.get("condition", "Ventricular Tachycardia (Recorded MIT-BIH Playback)")

    fake_alert = {
        "alert_id": f"DEMO-{int(time.time())}",
        "patient_id": pid,
        "priority": "HIGH PRIORITY (CRISIS / CODE BLUE)",
        "trigger_condition": condition,
        "location": "ICU Bed 01"
    }

    result = sms_service.send_emergency_alert(fake_alert, patient_data_mode="MIT-BIH")
    return jsonify({"success": True, **result})


@app.route("/api/guardian/notifications", methods=["GET"])
def get_guardian_notifications_endpoint():
    """Returns recent guardian notification activity log with filtering."""
    pid = request.args.get("patient_id")
    status_filter = request.args.get("status") or request.args.get("filter") or "ALL"
    limit = int(request.args.get("limit", 50))

    notifs = DatabaseManager.get_notifications(patient_id=pid, status_filter=status_filter, limit=limit)
    summary = DatabaseManager.get_delivery_summary(patient_id=pid)

    return jsonify({
        "success": True,
        "notifications": notifs,
        "summary": summary
    })


@app.route("/api/guardian/notifications/retry", methods=["POST"])
def retry_guardian_notification_endpoint():
    """Retries an eligible failed notification."""
    req = request.get_json() or {}
    notif_uuid = req.get("notification_uuid")

    if not notif_uuid:
        return jsonify({"success": False, "error": "notification_uuid is required."}), 400

    result = sms_service.retry_notification(notif_uuid)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/guardian/audit_logs", methods=["GET"])
def get_guardian_audit_logs_endpoint():
    """Returns immutable audit records of guardian actions."""
    limit = int(request.args.get("limit", 50))
    logs = DatabaseManager.get_audit_logs(limit=limit)
    return jsonify({"success": True, "logs": logs})


@app.route("/api/guardian/provider_status", methods=["GET"])
def get_guardian_provider_status_endpoint():
    """Returns non-sensitive provider configuration metadata."""
    return jsonify({"success": True, "provider": sms_service.get_provider_status()})


@app.route("/api/guardian/twilio_callback", methods=["POST"])
def twilio_callback_endpoint():
    """Twilio webhook endpoint for real-time delivery status callbacks."""
    form_data = request.form.to_dict() or request.get_json(silent=True) or {}
    result = sms_service.handle_twilio_callback(form_data)
    status_code = 200 if result.get("success") else 404
    return jsonify(result), status_code


@app.route("/api/guardian/notifications/<notif_uuid>/sync_status", methods=["POST"])
def sync_notification_status_endpoint(notif_uuid):
    """Polls provider to sync delivery status for a notification."""
    result = sms_service.sync_notification_status(notif_uuid)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


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
                        "patient_id": pid,
                        "name": analysis.get("name", pid),
                        "age": analysis.get("age", "--"),
                        "gender": analysis.get("gender", "--"),
                        "location": analysis.get("location", "--"),
                        "current_rhythm": analysis.get("current_rhythm", "Normal Sinus Rhythm"),
                        "data_mode": analysis.get("data_mode", "MIT-BIH"),
                        "mitbih_record_id": analysis.get("mitbih_record_id", "100"),
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
                        "clinical_state": analysis.get("clinical_state", "NORMAL"),
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
