"""
CardioSentinel - Central ICU Telemetry & Early Warning Alerting Web Server
Integrates Telemetry Engine, Digital Signal Processor, AI Arrhythmia Classifier,
and IEC 60601-1-8 Clinical Alert System with real-time SSE streaming and REST API.
"""

import time
import json
import threading
import os
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
import numpy as np

from backend.telemetry_engine import TelemetryEngine, CardiacRhythmType
from backend.signal_processor import ECGSignalProcessor
from backend.ai_engine import AIEarlyWarningEngine
from backend.alert_system import ClinicalAlertSystem, AlarmPriority

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

# Global engine singletons
telemetry_engine = TelemetryEngine()
signal_processor = ECGSignalProcessor(sample_rate=250)
ai_engine = AIEarlyWarningEngine()
alert_system = ClinicalAlertSystem(max_history=150)

# Thread safety lock & state store
state_lock = threading.Lock()
latest_patient_analysis = {}
live_telemetry_queues = {} # patient_id -> list of latest samples pending broadcast

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
            "rr_entropy": 0.65
        },
        "ai_diagnosis": {
            "diagnosis": patient.current_rhythm,
            "confidence": 0.96,
            "is_critical": False,
            "probabilities": {patient.current_rhythm: 0.96}
        },
        "cewi": {
            "cewi_score": 12,
            "risk_level": "LOW RISK (HEMODYNAMICALLY STABLE)",
            "alert_color": "#10b981",
            "estimated_lead_time": "Stable continuous monitoring",
            "shock_index": 0.60,
            "reasons": ["Electrophysiological parameters within normal baseline limits."]
        },
        "active_alert": None,
        "recent_peaks": []
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
                # Keep maximum 500 samples in queue
                if len(live_telemetry_queues[pid]) > 500:
                    live_telemetry_queues[pid] = live_telemetry_queues[pid][-500:]

        # 2. Every ~500ms (every 12 cycles of 40ms), perform signal filtering, Pan-Tompkins, AI inference & CEWI
        if analysis_counter >= 12:
            analysis_counter = 0
            for pid, patient in telemetry_engine.patients.items():
                # Extract copy of buffer
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

                if len(lead2_buf) >= 150:
                    raw_signal = np.array(lead2_buf[-750:]) # 3 seconds window for fast adaptation
                    filtered_signal = signal_processor.filter_signal(raw_signal)
                    peaks = signal_processor.pan_tompkins_qrs_detect(filtered_signal)
                    metrics = signal_processor.extract_clinical_metrics(filtered_signal, peaks)
                    
                    # Override HR from metrics if reliable peaks detected
                    if metrics["heart_rate"] > 0:
                        latest_vitals["hr"] = float(metrics["heart_rate"])

                    # Run AI arrhytmia classification
                    ai_result = ai_engine.classify_rhythm(metrics, filtered_signal)

                    # Compute CEWI Early Warning score
                    cewi_result = ai_engine.compute_cewi_score(latest_vitals, metrics, ai_result["diagnosis"])

                    # Evaluate clinical alert rules
                    triggered_alert = alert_system.evaluate_and_trigger(
                        patient_id=pid,
                        patient_name=patient.name,
                        location=patient.location,
                        diagnosis=ai_result["diagnosis"],
                        cewi_data=cewi_result,
                        metrics=metrics,
                        ecg_snapshot=list(filtered_signal)
                    )

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
                            "vitals": latest_vitals,
                            "metrics": metrics,
                            "ai_diagnosis": ai_result,
                            "cewi": cewi_result,
                            "active_alert": alert_system.active_alerts.get(pid),
                            "recent_peaks": [int(p) for p in peaks[-8:]]
                        }

        # Sleep remaining time to maintain 40ms timing
        elapsed = time.time() - loop_start
        sleep_time = max(0.005, sample_interval - elapsed)
        time.sleep(sleep_time)


# Start worker thread as daemon
worker_thread = threading.Thread(target=background_telemetry_worker, daemon=True)
worker_thread.start()


# --- ROUTES & ENDPOINTS ---

@app.route("/")
def index():
    """Renders the main ICU Central Telemetry Dashboard."""
    return render_template("index.html")


@app.route("/api/patients", methods=["GET"])
def get_patients():
    """Returns summarized cards for all monitored patients."""
    with state_lock:
        data = list(latest_patient_analysis.values())
    return jsonify({"success": True, "patients": data})


@app.route("/api/patient/<patient_id>", methods=["GET"])
def get_patient_detail(patient_id):
    """Returns comprehensive telemetry & diagnostic detail for a single patient."""
    with state_lock:
        patient_data = latest_patient_analysis.get(patient_id)
    if not patient_data:
        return jsonify({"success": False, "error": "Patient not found"}), 404
    return jsonify({"success": True, "patient": patient_data})


@app.route("/api/trigger_arrhythmia", methods=["POST"])
def trigger_arrhythmia():
    """
    Injects an acute cardiac rhythm event into a patient.
    Body JSON: {"patient_id": "BED-01", "rhythm": "Ventricular Tachycardia", "duration": 30.0}
    """
    req = request.get_json() or {}
    patient_id = req.get("patient_id")
    rhythm = req.get("rhythm")
    duration = float(req.get("duration", 30.0))

    if not patient_id or not rhythm:
        return jsonify({"success": False, "error": "patient_id and rhythm required"}), 400

    success = telemetry_engine.trigger_event(patient_id, rhythm, duration)
    if success:
        return jsonify({
            "success": True,
            "message": f"Successfully injected {rhythm} into patient {patient_id} for {duration} seconds."
        })
    return jsonify({"success": False, "error": "Patient not found"}), 404


@app.route("/api/silence_alarm", methods=["POST"])
def silence_alarm():
    """
    Silences audio alarms for a patient for 120 seconds per IEC 60601-1-8 standard.
    Body JSON: {"patient_id": "BED-01", "duration": 120.0}
    """
    req = request.get_json() or {}
    patient_id = req.get("patient_id")
    duration = float(req.get("duration", 120.0))

    if not patient_id:
        return jsonify({"success": False, "error": "patient_id required"}), 400

    alert_system.silence_patient(patient_id, duration)
    return jsonify({
        "success": True,
        "message": f"Patient {patient_id} alarm silenced for {duration} seconds."
    })


@app.route("/api/acknowledge_alarm", methods=["POST"])
def acknowledge_alarm():
    """
    Clinician acknowledges an active alarm.
    Body JSON: {"alert_id": "ALT-XXXX", "clinician_name": "Dr. Smith"}
    """
    req = request.get_json() or {}
    alert_id = req.get("alert_id")
    clinician_name = req.get("clinician_name", "Staff Cardiologist")

    if not alert_id:
        return jsonify({"success": False, "error": "alert_id required"}), 400

    ack = alert_system.acknowledge_alert(alert_id, clinician_name)
    return jsonify({"success": ack, "message": f"Alert {alert_id} acknowledged by {clinician_name}."})


@app.route("/api/dataset/summary", methods=["GET"])
def get_dataset_summary():
    """Returns the PhysioNet MIT-BIH Arrhythmia Database metadata and catalog."""
    from backend.dataset_manager import dataset_manager
    return jsonify({"success": True, "dataset": dataset_manager.get_dataset_summary()})


@app.route("/api/dataset/set_mode", methods=["POST"])
def set_dataset_mode():
    """
    Switches between Synthetic telemetry and real PhysioNet MIT-BIH stream playback.
    Body JSON: {"mode": "MIT-BIH" | "SYNTHETIC", "patient_id": "BED-01" (optional), "record_id": "100" (optional)}
    """
    req = request.get_json() or {}
    mode = req.get("mode", "SYNTHETIC").upper()
    patient_id = req.get("patient_id")
    record_id = req.get("record_id")

    if patient_id:
        success = telemetry_engine.set_patient_data_mode(patient_id, mode, record_id)
        if not success:
            return jsonify({"success": False, "error": f"Patient {patient_id} not found"}), 404
        return jsonify({
            "success": True,
            "message": f"Patient {patient_id} switched to {mode} mode (Record: {record_id or 'default'}).",
            "data_mode": mode,
            "record_id": record_id
        })
    else:
        telemetry_engine.set_global_data_mode(mode)
        return jsonify({
            "success": True,
            "message": f"All monitored beds switched to {mode} mode.",
            "data_mode": mode
        })


@app.route("/api/dataset/train_mitbih", methods=["POST"])
def train_mitbih_model():
    """Triggers ML model training and cross-validation on the extracted PhysioNet MIT-BIH dataset."""
    result = ai_engine.train_on_mitbih_dataset()
    return jsonify(result)


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Returns active alerts and alert history."""
    return jsonify({
        "success": True,
        "active_alerts": alert_system.get_active_alerts(),
        "alert_history": alert_system.get_alert_history(limit=50)
    })


@app.route("/api/export_report/<patient_id>", methods=["GET"])
def export_report(patient_id):
    """Generates a structured clinical shift telemetry report with ECG strip and CEWI breakdown."""
    patient = telemetry_engine.get_patient(patient_id)
    if not patient:
        return jsonify({"success": False, "error": "Patient not found"}), 404

    with state_lock:
        analysis = latest_patient_analysis.get(patient_id, {})
        strip = list(patient.lead2_buffer[-1000:]) # Last 4 seconds strip

    report = {
        "report_id": f"RPT-{int(time.time())}-{patient_id}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
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


@app.route("/api/health", methods=["GET"])
def health():
    """System health and pipeline operational metrics."""
    return jsonify({
        "status": "healthy",
        "system": "CardioSentinel Continuous Telemetry & AI Alert Engine",
        "sample_rate": 250,
        "num_patients": len(telemetry_engine.patients),
        "active_alerts_count": len(alert_system.active_alerts),
        "total_alerts_recorded": len(alert_system.alert_history),
        "timestamp": time.time()
    })


@app.route("/api/stream")
def telemetry_stream():
    """
    High-performance Server-Sent Events (SSE) telemetry stream.
    Delivers 40ms batches of Lead II and Lead V1 points to the browser,
    along with updated vitals, metrics, AI predictions, and active alerts.
    """
    def event_generator():
        while True:
            # Gather newly accumulated points for all patients
            batch_payload = {}
            with state_lock:
                for pid in telemetry_engine.patients.keys():
                    queue = live_telemetry_queues.get(pid, [])
                    # Drain queue
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
                        "alert": analysis.get("active_alert")
                    }

            # Format as SSE
            yield f"data: {json.dumps(batch_payload)}\n\n"
            time.sleep(0.04) # ~25 frames per second SSE dispatch

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
    print("================================================================")
    print("CardioSentinel ICU Telemetry & AI Warning Engine starting...")
    print("Hospital Beds Monitored: BED-01, BED-02, BED-03, REMOTE-01")
    print("Web Console: http://127.0.0.1:5000")
    print("================================================================")
    threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open("http://127.0.0.1:5000")), daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
