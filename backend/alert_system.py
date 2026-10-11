"""
CardioSentinel - IEC 60601-1-8 Clinical Alert & Event Management Engine
Manages medical telemetry alarms (Crisis, Warning, Advisory), acknowledgment lifecycle,
timed snooze (1 min, 5 min, 15 min), deduplication cooldowns, CSV export, and audit trails.
"""

import time
import uuid
import csv
import io
from typing import Dict, List, Optional, Any

class AlarmPriority:
    CRISIS = "HIGH PRIORITY (CRISIS / CODE BLUE)"
    WARNING = "MEDIUM PRIORITY (WARNING)"
    ADVISORY = "LOW PRIORITY (ADVISORY)"

class ClinicalAlertSystem:
    def __init__(self, max_history: int = 150):
        self.active_alerts: Dict[str, Dict[str, Any]] = {}  # alert_id -> alert dict
        self.bed_active_map: Dict[str, str] = {}            # patient_id -> latest alert_id
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history = max_history

        # Snooze map: patient_id -> (expiry_timestamp, snooze_reason)
        self.snoozed_beds: Dict[str, Any] = {}
        self.silenced_patients = self.snoozed_beds

        # Cooldown map: (patient_id, trigger_condition) -> last_triggered_timestamp
        self.cooldown_map: Dict[Tuple[str, str], float] = {}
        self.cooldown_seconds = 15.0 # Minimum seconds before identical condition can re-trigger

        self.audio_muted = False

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.alert_history[:limit]

    def silence_patient(self, patient_id: str, duration: float = 120.0, duration_sec: Optional[float] = None) -> Dict[str, Any]:
        actual_sec = duration_sec if duration_sec is not None else duration
        mins = max(1, int(round(actual_sec / 60.0)))
        res = self.snooze_alert(patient_id, minutes=mins)
        if patient_id in self.active_alerts:
            self.active_alerts[patient_id]["audio_alarm"] = False
        return res

    def toggle_audio_mute(self) -> bool:
        self.audio_muted = not self.audio_muted
        return self.audio_muted

    def snooze_alert(self, patient_id: str, minutes: int = 5, snoozed_by: str = "Clinician") -> Dict[str, Any]:
        """Snoozes alarms for a bed for 1, 5, or 15 minutes."""
        now = time.time()
        expiry = now + (minutes * 60.0)
        self.snoozed_beds[patient_id] = (expiry, f"Snoozed for {minutes}m by {snoozed_by}")

        # Update any active alert for this bed
        active_id = self.bed_active_map.get(patient_id)
        if active_id and active_id in self.active_alerts:
            self.active_alerts[active_id]["status"] = "SNOOZED"
            self.active_alerts[active_id]["snooze_expiry"] = expiry

        return {
            "success": True,
            "patient_id": patient_id,
            "minutes": minutes,
            "expiry_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(expiry))
        }

    def is_snoozed(self, patient_id: str) -> bool:
        """Returns True if the specified bed is currently within an active snooze window."""
        if patient_id in self.snoozed_beds:
            expiry, _ = self.snoozed_beds[patient_id]
            if time.time() < expiry:
                return True
            else:
                del self.snoozed_beds[patient_id]
        return False

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "ICU Nurse / Clinician", clinician_name: Optional[str] = None) -> bool:
        """Acknowledges an active alert."""
        by = clinician_name or acknowledged_by
        alert = self.active_alerts.get(alert_id)
        if not alert:
            # Check history
            for h in self.alert_history:
                if h["alert_id"] == alert_id:
                    h["acknowledged"] = True
                    h["acknowledged_by"] = by
                    h["acknowledged_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    return True
            return False

        alert["acknowledged"] = True
        alert["acknowledged_by"] = by
        alert["acknowledged_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        alert["status"] = "ACKNOWLEDGED"
        return True

    def resolve_alert(self, alert_id: str, resolved_by: str = "System Auto-Resolution", note: str = "") -> bool:
        """Resolves an alert and moves it completely to history."""
        alert = self.active_alerts.pop(alert_id, None)
        if not alert:
            return False

        patient_id = alert["patient_id"]
        if self.bed_active_map.get(patient_id) == alert_id:
            del self.bed_active_map[patient_id]

        alert["status"] = "RESOLVED"
        alert["resolved_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        alert["resolved_by"] = resolved_by
        alert["resolution_note"] = note or "Physiological parameters normalized."

        self.alert_history.insert(0, alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[:self.max_history]

        return True

    def evaluate_and_trigger(
        self,
        patient_id: str,
        patient_name: str,
        location: str,
        diagnosis: str,
        cewi_data: Dict[str, Any],
        metrics: Dict[str, Any],
        source_record: str = "SYNTHETIC",
        confidence: float = 0.95,
        signal_quality: str = "OPTIMAL",
        event_time_sec: float = 0.0,
        ecg_snapshot: Optional[List[float]] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates clinical conditions, priority tiers, snooze status, and cooldowns.
        Triggers new traceable alert if criteria are met.
        """
        now = time.time()

        # Check snooze
        if patient_id in self.snoozed_beds:
            expiry, reason = self.snoozed_beds[patient_id]
            if now < expiry:
                return None
            else:
                del self.snoozed_beds[patient_id]

        cewi_score = cewi_data.get("cewi_score", 0)
        hr = metrics.get("heart_rate", 72)

        priority = None
        trigger_condition = None

        if diagnosis in ["Ventricular Fibrillation", "Asystole / Cardiac Arrest"]:
            priority = AlarmPriority.CRISIS
            trigger_condition = f"CATASTROPHIC RHYTHM: {diagnosis.upper()}"
        elif diagnosis == "Ventricular Tachycardia":
            priority = AlarmPriority.CRISIS
            trigger_condition = "CRITICAL: Sustained Ventricular Tachycardia"
        elif cewi_score >= 80:
            priority = AlarmPriority.CRISIS
            trigger_condition = f"IMMINENT COLLAPSE RISK: CEWI Score {cewi_score}/100"
        elif cewi_score >= 55 or diagnosis in ["Premature Ventricular Contractions", "Atrial Fibrillation"]:
            priority = AlarmPriority.WARNING
            trigger_condition = f"PRE-ARRHYTHMIC WARNING: {diagnosis} (CEWI {cewi_score})"
        elif hr < 45 and hr > 0:
            priority = AlarmPriority.WARNING
            trigger_condition = f"SEVERE BRADYCARDIA: HR {hr} bpm"
        elif hr > 150:
            priority = AlarmPriority.WARNING
            trigger_condition = f"SEVERE TACHYCARDIA: HR {hr} bpm"
        elif cewi_score >= 35:
            priority = AlarmPriority.ADVISORY
            trigger_condition = f"HEMODYNAMIC INSTABILITY NOTICE: CEWI {cewi_score}"

        # If parameters normalized, resolve existing alert
        if not priority:
            active_id = self.bed_active_map.get(patient_id)
            if active_id:
                self.resolve_alert(active_id, note="Patient rhythm stabilized")
            return None

        # Check deduplication & cooldown
        cooldown_key = (patient_id, trigger_condition)
        last_time = self.cooldown_map.get(cooldown_key, 0.0)
        if (now - last_time) < self.cooldown_seconds:
            # Under cooldown, don't re-trigger identical event
            active_id = self.bed_active_map.get(patient_id)
            return self.active_alerts.get(active_id) if active_id else None

        # Create new unique alert
        alert_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"
        alert_payload = {
            "alert_id": alert_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "location": location,
            "source_record": str(source_record),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "epoch": now,
            "event_time_sec": round(event_time_sec, 2),
            "priority": priority,
            "trigger_condition": trigger_condition,
            "trigger_reason": trigger_condition, # compatibility
            "diagnosis": diagnosis,
            "confidence": round(confidence, 3),
            "cewi_score": cewi_score,
            "heart_rate": hr,
            "signal_quality": signal_quality,
            "status": "ACTIVE",
            "acknowledged": False,
            "acknowledged_by": None,
            "acknowledged_time": None,
            "dispatch_status": {
                "sms": "Simulated ICU Pager Dispatched",
                "channels": ["ICU Central Pager", "Telemetry Nurse Station", "Cardiologist On-Duty"]
            },
            "audio_alarm": not self.audio_muted,
            "disclaimer": "Research telemetry indicator - not an emergency medical directive."
        }

        # Clear any prior active alert for this bed
        prior_id = self.bed_active_map.get(patient_id)
        if prior_id and prior_id in self.active_alerts:
            self.resolve_alert(prior_id, note="Superseded by higher or newer alert")

        self.active_alerts[alert_id] = alert_payload
        self.active_alerts[patient_id] = alert_payload
        self.bed_active_map[patient_id] = alert_id
        self.cooldown_map[cooldown_key] = now

        return alert_payload

    def export_alerts_csv(self) -> str:
        """Exports active and historical alerts as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Alert ID", "Timestamp", "Bed", "Patient Name", "Location", "Record",
            "Priority", "Trigger Condition", "Diagnosis", "Confidence", "CEWI", "HR",
            "Signal Quality", "Status", "Acknowledged By", "Ack Time"
        ])

        all_alerts = list(self.active_alerts.values()) + self.alert_history
        for a in all_alerts:
            writer.writerow([
                a.get("alert_id"),
                a.get("timestamp"),
                a.get("patient_id"),
                a.get("patient_name"),
                a.get("location"),
                a.get("source_record"),
                a.get("priority"),
                a.get("trigger_condition"),
                a.get("diagnosis"),
                a.get("confidence"),
                a.get("cewi_score"),
                a.get("heart_rate"),
                a.get("signal_quality"),
                a.get("status"),
                a.get("acknowledged_by") or "Unacknowledged",
                a.get("acknowledged_time") or ""
            ])

        return output.getvalue()

# Global singleton
alert_system = ClinicalAlertSystem()
