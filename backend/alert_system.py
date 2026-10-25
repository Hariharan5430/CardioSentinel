"""
CardioSentinel - IEC 60601-1-8 Clinical Alert & Event Management Engine
Manages medical telemetry alarms (Crisis, Warning, Advisory), acknowledgment lifecycle,
timed snooze (1 min, 5 min, 15 min), deduplication cooldowns, CSV export, and audit trails.
"""

import time
import uuid
import csv
import io
from typing import Dict, List, Optional, Any, Tuple, Union

from backend.database import DatabaseManager

class AlarmPriority:
    CRISIS = "HIGH PRIORITY (CRISIS / CODE BLUE)"
    WARNING = "MEDIUM PRIORITY (WARNING)"
    ADVISORY = "LOW PRIORITY (ADVISORY)"

class ClinicalState:
    NORMAL = "NORMAL"
    ABNORMAL = "ABNORMAL"
    EMERGENCY = "EMERGENCY"
    RESOLVED = "RESOLVED"

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
        self.cooldown_map: Dict[Any, float] = {}
        self.cooldown_seconds = 15.0 # Minimum seconds before identical condition can re-trigger

        # Banner dismissal set: alert_ids dismissed from top banner
        self.dismissed_banner_alerts: set = set()

        self.audio_muted = False

        # State management & episode tracking
        self.patient_states: Dict[str, str] = {}          # patient_id -> ClinicalState
        self.normal_readings_count: Dict[str, int] = {}  # patient_id -> consecutive normal analysis cycles
        self.recovery_threshold: int = 5                  # consecutive normal cycles (~2.5s) to confirm recovery
        self.active_episodes: Dict[str, Dict[str, Any]] = {} # patient_id -> episode dict
        self._sync_active_episodes_from_db()

    def _sync_active_episodes_from_db(self):
        """Re-hydrates active episodes from database across application restarts."""
        try:
            episodes = DatabaseManager.get_active_episodes()
            for ep in episodes:
                pid = ep.get("patient_id")
                aid = ep.get("alert_id")
                if pid and aid:
                    self.active_episodes[pid] = ep
                    self.patient_states[pid] = ClinicalState.EMERGENCY
                    self.bed_active_map[pid] = aid
                    now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
                    alert_payload = {
                        "alert_id": aid,
                        "episode_id": ep.get("episode_id"),
                        "patient_id": pid,
                        "patient_name": "Monitored Patient",
                        "location": "ICU Bed",
                        "source_record": "SYNTHETIC",
                        "timestamp": ep.get("started_at", now_iso),
                        "last_updated": ep.get("last_updated_at", now_iso),
                        "epoch": time.time(),
                        "priority": ep.get("priority", AlarmPriority.CRISIS),
                        "trigger_condition": ep.get("trigger_condition", "Cardiac Emergency"),
                        "trigger_reason": ep.get("trigger_condition", "Cardiac Emergency"),
                        "diagnosis": ep.get("diagnosis", "Ventricular Emergency"),
                        "status": "ACTIVE",
                        "state": ClinicalState.EMERGENCY,
                        "is_new_emergency": False,
                        "is_new_alert": False,
                        "read": False,
                        "acknowledged": False,
                        "audio_alarm": not self.audio_muted,
                        "dispatch_status": {
                            "sms": "Simulated ICU Pager Dispatched",
                            "channels": ["ICU Central Pager", "Telemetry Nurse Station", "Cardiologist On-Duty"]
                        },
                        "disclaimer": "Research telemetry indicator - not an emergency medical directive."
                    }
                    self.active_alerts[aid] = alert_payload
                    self.active_alerts[pid] = alert_payload
        except Exception:
            pass

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        from backend.alert_explanation import smart_alert_explainer
        seen = set()
        res = []
        for a in self.active_alerts.values():
            aid = a.get("alert_id")
            if aid and aid not in seen:
                seen.add(aid)
                if "smart_explanation" not in a:
                    a["smart_explanation"] = smart_alert_explainer.explain_existing_alert(a)
                res.append(a)
        return res

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        from backend.alert_explanation import smart_alert_explainer
        res = self.alert_history[:limit]
        for a in res:
            if "smart_explanation" not in a:
                a["smart_explanation"] = smart_alert_explainer.explain_existing_alert(a)
        return res

    def mark_read(self, alert_id: str) -> bool:
        found = False
        for a in self.active_alerts.values():
            if a.get("alert_id") == alert_id:
                a["read"] = True
                found = True
        for h in self.alert_history:
            if h.get("alert_id") == alert_id:
                h["read"] = True
                found = True
        return found

    def mark_all_read(self) -> int:
        count = 0
        seen = set()
        for a in self.active_alerts.values():
            aid = a.get("alert_id")
            if aid and aid not in seen:
                seen.add(aid)
                if not a.get("read", False):
                    a["read"] = True
                    count += 1
        for h in self.alert_history:
            aid = h.get("alert_id")
            if aid and aid not in seen:
                seen.add(aid)
                if not h.get("read", False):
                    h["read"] = True
                    count += 1
        return count

    def get_unread_count(self) -> int:
        seen = set()
        unread = 0
        for a in self.active_alerts.values():
            aid = a.get("alert_id")
            if aid and aid not in seen:
                seen.add(aid)
                if not a.get("read", False):
                    unread += 1
        return unread

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

    def snooze_alert(self, patient_id: str, minutes: int = 5, snoozed_by: str = "Clinician", alert_id: Optional[str] = None) -> Dict[str, Any]:
        """Snoozes alarms for a bed for 1, 5, or 15 minutes, persisting audit trail and checking breakthrough."""
        now = time.time()
        expiry = now + (minutes * 60.0)

        active_id = alert_id or self.bed_active_map.get(patient_id)
        current_pri = None
        if active_id and active_id in self.active_alerts:
            current_pri = self.active_alerts[active_id].get("priority")
            self.active_alerts[active_id]["status"] = "SNOOZED"
            self.active_alerts[active_id]["snooze_expiry"] = expiry

        self.snoozed_beds[patient_id] = (expiry, f"Snoozed for {minutes}m by {snoozed_by}", current_pri)

        expiry_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(expiry))
        try:
            DatabaseManager.log_audit(
                action="ALERT_SNOOZED",
                actor=snoozed_by,
                patient_id=patient_id,
                details=f"Snoozed alerts for bed {patient_id} for {minutes}m until {expiry_iso}."
            )
        except Exception:
            pass

        return {
            "success": True,
            "patient_id": patient_id,
            "minutes": minutes,
            "expiry_iso": expiry_iso
        }

    def is_snoozed(self, patient_id: str) -> bool:
        """Returns True if the specified bed is currently within an active snooze window."""
        if patient_id in self.snoozed_beds:
            snooze_item = self.snoozed_beds[patient_id]
            expiry = snooze_item[0]
            if time.time() < expiry:
                return True
            else:
                del self.snoozed_beds[patient_id]
        return False

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "ICU Nurse / Clinician", clinician_name: Optional[str] = None) -> bool:
        """Acknowledges an active alert persistently and idempotently."""
        by = clinician_name or acknowledged_by
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")

        alert = self.active_alerts.get(alert_id)
        if not alert:
            # Check history
            for h in self.alert_history:
                if h["alert_id"] == alert_id:
                    if h.get("acknowledged"):
                        return True
                    h["acknowledged"] = True
                    h["acknowledged_by"] = by
                    h["acknowledged_time"] = now_iso
                    try:
                        DatabaseManager.log_audit(
                            action="ALERT_ACKNOWLEDGED",
                            actor=by,
                            patient_id=h.get("patient_id"),
                            details=f"Acknowledged historical alert {alert_id} ({h.get('trigger_condition')})."
                        )
                    except Exception:
                        pass
                    return True
            return False

        if alert.get("acknowledged"):
            return True

        alert["acknowledged"] = True
        alert["acknowledged_by"] = by
        alert["acknowledged_time"] = now_iso
        alert["status"] = "ACKNOWLEDGED"

        try:
            DatabaseManager.log_audit(
                action="ALERT_ACKNOWLEDGED",
                actor=by,
                patient_id=alert.get("patient_id"),
                details=f"Acknowledged active alert {alert_id} ({alert.get('trigger_condition')})."
            )
        except Exception:
            pass

        return True

    def get_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves active or historical alert matching alert_id."""
        from backend.alert_explanation import smart_alert_explainer
        target = None
        if alert_id in self.active_alerts:
            target = self.active_alerts[alert_id]
        if not target:
            for a in self.active_alerts.values():
                if isinstance(a, dict) and a.get("alert_id") == alert_id:
                    target = a
                    break
        if not target:
            for h in self.alert_history:
                if isinstance(h, dict) and h.get("alert_id") == alert_id:
                    target = h
                    break
        if target and "smart_explanation" not in target:
            target["smart_explanation"] = smart_alert_explainer.explain_existing_alert(target)
        return target

    def get_alert_explanation(self, alert_id: str) -> Dict[str, Any]:
        """Retrieves or reconstructs structured Smart Alert Explanation for active or historical alert."""
        from backend.alert_explanation import smart_alert_explainer
        alert = self.get_alert_by_id(alert_id)
        if not alert:
            return smart_alert_explainer._build_unavailable_explanation(f"Alert ID '{alert_id}' was not found in active alerts or history.")
        return smart_alert_explainer.explain_existing_alert(alert)

    def dismiss_banner_alert(self, alert_id: str, dismissed_by: str = "Clinician") -> bool:
        """Dismisses the top notification banner popup for this alert while retaining clinical history."""
        self.dismissed_banner_alerts.add(alert_id)
        pid = None
        for a in self.active_alerts.values():
            if a.get("alert_id") == alert_id:
                a["banner_dismissed"] = True
                pid = a.get("patient_id")
        for h in self.alert_history:
            if h.get("alert_id") == alert_id:
                h["banner_dismissed"] = True
                if not pid:
                    pid = h.get("patient_id")

        try:
            DatabaseManager.log_audit(
                action="ALERT_BANNER_DISMISSED",
                actor=dismissed_by,
                patient_id=pid,
                details=f"Dismissed top banner popup for alert {alert_id}."
            )
        except Exception:
            pass
        return True

    def resolve_alert(self, alert_id: str, resolved_by: str = "System Auto-Resolution", note: str = "") -> bool:
        """Resolves an alert and moves it completely to history with audit tracking."""
        alert = self.active_alerts.pop(alert_id, None)
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")

        if not alert:
            for h in self.alert_history:
                if h["alert_id"] == alert_id:
                    h["status"] = "RESOLVED"
                    h["state"] = ClinicalState.RESOLVED
                    h["resolved_time"] = now_iso
                    h["resolved_by"] = resolved_by
                    h["resolution_note"] = note or "Physiological parameters normalized."
                    try:
                        DatabaseManager.log_audit(
                            action="ALERT_RESOLVED",
                            actor=resolved_by,
                            patient_id=h.get("patient_id"),
                            details=f"Resolved historical alert {alert_id}: {h.get('resolution_note')}."
                        )
                    except Exception:
                        pass
                    return True
            return False

        patient_id = alert["patient_id"]
        # Clear the patient_id key as well if pointing to this alert
        if self.active_alerts.get(patient_id, {}).get("alert_id") == alert_id:
            self.active_alerts.pop(patient_id, None)

        if self.bed_active_map.get(patient_id) == alert_id:
            del self.bed_active_map[patient_id]

        alert["status"] = "RESOLVED"
        alert["state"] = ClinicalState.RESOLVED
        alert["resolved_time"] = now_iso
        alert["resolved_by"] = resolved_by
        alert["resolution_note"] = note or "Physiological parameters normalized."

        self.alert_history.insert(0, alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[:self.max_history]

        # Update episode tracking if active
        if patient_id in self.active_episodes:
            ep = self.active_episodes.pop(patient_id)
            try:
                DatabaseManager.resolve_episode(ep.get("episode_id"), resolved_at=alert["resolved_time"])
            except Exception:
                pass

        self.patient_states[patient_id] = ClinicalState.RESOLVED
        self.normal_readings_count[patient_id] = 0

        try:
            DatabaseManager.log_audit(
                action="ALERT_RESOLVED",
                actor=resolved_by,
                patient_id=patient_id,
                details=f"Resolved alert {alert_id} ({alert.get('trigger_condition')}): {alert['resolution_note']}."
            )
        except Exception:
            pass

        return True

    def evaluate_and_trigger(
        self,
        patient_id: str,
        patient_name: str = "Unknown Patient",
        location: str = "ICU Bed",
        diagnosis: str = "Normal Sinus Rhythm",
        cewi_data: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        source_record: str = "SYNTHETIC",
        confidence: float = 0.95,
        signal_quality: Optional[Union[str, Dict[str, Any]]] = "OPTIMAL",
        event_time_sec: float = 0.0,
        ecg_snapshot: Optional[List[float]] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates clinical conditions, priority tiers, snooze status, state transitions,
        and manages the single-alert-per-episode lifecycle with breakthrough protection.
        """
        now = time.time()

        # Handle kwargs for flexible invocation
        if "rhythm" in kwargs and (diagnosis == "Normal Sinus Rhythm" or not diagnosis):
            diagnosis = kwargs["rhythm"]
        if "ai_diagnosis" in kwargs and isinstance(kwargs["ai_diagnosis"], dict):
            diag_obj = kwargs["ai_diagnosis"]
            if diagnosis == "Normal Sinus Rhythm" or not diagnosis:
                diagnosis = diag_obj.get("diagnosis", diagnosis)
            if "confidence" in diag_obj and (confidence == 0.95 or confidence is None):
                confidence = diag_obj.get("confidence", 0.95)

        metrics = dict(metrics or {})
        if "heart_rate" in kwargs and "heart_rate" not in metrics:
            metrics["heart_rate"] = kwargs["heart_rate"]
        if "vitals" in kwargs and isinstance(kwargs["vitals"], dict):
            if "hr" in kwargs["vitals"] and "heart_rate" not in metrics:
                metrics["heart_rate"] = kwargs["vitals"]["hr"]

        cewi_data = dict(cewi_data or {})
        if "cewi_score" in kwargs:
            cewi_data["cewi_score"] = kwargs["cewi_score"]
        if "risk_level" not in cewi_data:
            cscore = cewi_data.get("cewi_score", 0)
            cewi_data["risk_level"] = "HIGH RISK" if cscore >= 65 else ("MODERATE RISK" if cscore >= 35 else "LOW RISK")

        cewi_score = cewi_data.get("cewi_score", 0)
        hr = metrics.get("heart_rate", 72)
        sig_q = str(signal_quality or metrics.get("signal_quality", "OPTIMAL")).upper()

        priority = None
        trigger_condition = None
        is_emergency = False

        # 1. Validated signal quality check before rhythm evaluation
        if "LEAD OFF" in sig_q or "FLATLINE" in sig_q or "POOR" in sig_q or "INSUFFICIENT" in sig_q:
            priority = AlarmPriority.WARNING
            trigger_condition = "LEAD OFF / FLATLINE: SENSOR DISCONNECTED"
            diagnosis = "Lead Disconnect / Artifact"
            is_emergency = False
        elif diagnosis in ["Ventricular Fibrillation", "Asystole / Cardiac Arrest"]:
            priority = AlarmPriority.CRISIS
            trigger_condition = f"CATASTROPHIC RHYTHM: {diagnosis.upper()}"
            is_emergency = True
        elif diagnosis == "Ventricular Tachycardia":
            priority = AlarmPriority.CRISIS
            trigger_condition = "CRITICAL: Sustained Ventricular Tachycardia"
            is_emergency = True
        elif cewi_score >= 80:
            priority = AlarmPriority.CRISIS
            trigger_condition = f"IMMINENT COLLAPSE RISK: CEWI Score {cewi_score}/100"
            is_emergency = True
        elif cewi_score >= 55 or diagnosis in ["Premature Ventricular Contractions", "Atrial Fibrillation"]:
            priority = AlarmPriority.WARNING
            trigger_condition = f"PRE-ARRHYTHMIC WARNING: {diagnosis} (CEWI {cewi_score})"
            is_emergency = False
        elif (hr < 45 and hr > 0) or "BRADYCARDIA" in str(diagnosis).upper():
            priority = AlarmPriority.WARNING
            trigger_condition = f"SEVERE BRADYCARDIA: HR {hr} bpm"
            is_emergency = False
        elif hr > 150 or "TACHYCARDIA" in str(diagnosis).upper():
            priority = AlarmPriority.WARNING
            trigger_condition = f"SEVERE TACHYCARDIA: HR {hr} bpm"
            is_emergency = False
        elif cewi_score >= 35:
            priority = AlarmPriority.ADVISORY
            trigger_condition = f"HEMODYNAMIC INSTABILITY NOTICE: CEWI {cewi_score}"
            is_emergency = False

        # 2. Check snooze status and critical breakthrough safety policy
        if patient_id in self.snoozed_beds:
            snooze_item = self.snoozed_beds[patient_id]
            expiry = snooze_item[0]
            reason = snooze_item[1]
            snoozed_pri = snooze_item[2] if len(snooze_item) > 2 else None

            if now < expiry:
                # Breakthrough: allow a genuinely new CRISIS condition to override snooze if snooze was for warning/advisory
                is_breakthrough = (priority == AlarmPriority.CRISIS and snoozed_pri != AlarmPriority.CRISIS)
                if not is_breakthrough:
                    active_id = self.bed_active_map.get(patient_id)
                    existing = self.active_alerts.get(active_id) if active_id else None
                    if existing:
                        existing["status"] = "SNOOZED"
                        existing["banner_dismissed"] = (existing.get("alert_id") in self.dismissed_banner_alerts)
                    return existing
                else:
                    del self.snoozed_beds[patient_id]
                    try:
                        DatabaseManager.log_audit(
                            action="SNOOZE_BREAKTHROUGH",
                            actor="Safety Policy",
                            patient_id=patient_id,
                            details=f"Snooze cleared by higher-severity crisis event: {trigger_condition}."
                        )
                    except Exception:
                        pass
            else:
                del self.snoozed_beds[patient_id]

        curr_state = self.patient_states.get(patient_id, ClinicalState.NORMAL)
        active_id = self.bed_active_map.get(patient_id)
        active_ep = self.active_episodes.get(patient_id)

        # ---------------------------------------------------------
        # CASE 1: NORMAL PHYSIOLOGICAL READING (priority is None)
        # ---------------------------------------------------------
        if not priority:
            self.normal_readings_count[patient_id] = self.normal_readings_count.get(patient_id, 0) + 1

            if active_id and active_id in self.active_alerts:
                # Active alert exists. Check if consecutive normal readings meet recovery criteria
                if self.normal_readings_count[patient_id] >= self.recovery_threshold:
                    self.resolve_alert(active_id, note="Patient rhythm stabilized and recovery criteria met.")
                    self.patient_states[patient_id] = ClinicalState.NORMAL
                    self.normal_readings_count[patient_id] = 0
                    return None
                else:
                    # Within recovery evaluation window (hysteresis to prevent rapid bouncing)
                    return self.active_alerts.get(active_id)
            else:
                self.patient_states[patient_id] = ClinicalState.NORMAL
                return None

        # ---------------------------------------------------------
        # CASE 2: EMERGENCY CONDITION (is_emergency is True)
        # ---------------------------------------------------------
        if is_emergency:
            self.normal_readings_count[patient_id] = 0

            # If already in an active EMERGENCY episode with an existing active alert:
            if curr_state == ClinicalState.EMERGENCY and (active_ep or (active_id and active_id in self.active_alerts)):
                if not active_id and active_ep:
                    active_id = active_ep.get("alert_id")
                    self.bed_active_map[patient_id] = active_id

                existing_alert = self.active_alerts.get(active_id)
                if not existing_alert:
                    existing_alert = {
                        "alert_id": active_id or f"ALT-{uuid.uuid4().hex[:8].upper()}",
                        "episode_id": active_ep.get("episode_id") if active_ep else f"EP-{patient_id}-{int(now)}",
                        "patient_id": patient_id,
                        "patient_name": patient_name,
                        "location": location,
                        "source_record": str(source_record),
                        "timestamp": active_ep.get("started_at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))) if active_ep else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                        "status": "ACTIVE",
                        "read": False,
                        "acknowledged": False
                    }
                    self.active_alerts[existing_alert["alert_id"]] = existing_alert
                    self.active_alerts[patient_id] = existing_alert
                    self.bed_active_map[patient_id] = existing_alert["alert_id"]

                # Update existing alert in-place (same alert_id, same episode_id)
                existing_alert["epoch"] = now
                existing_alert["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                existing_alert["heart_rate"] = hr
                existing_alert["cewi_score"] = cewi_score
                existing_alert["confidence"] = round(confidence, 3)
                existing_alert["signal_quality"] = signal_quality
                if "audio_alarm" not in existing_alert:
                    existing_alert["audio_alarm"] = not self.audio_muted
                if patient_name:
                    existing_alert["patient_name"] = patient_name
                if location:
                    existing_alert["location"] = location
                if source_record:
                    existing_alert["source_record"] = str(source_record)
                existing_alert["event_time_sec"] = round(event_time_sec, 2)
                existing_alert["trigger_condition"] = trigger_condition
                existing_alert["diagnosis"] = diagnosis
                existing_alert["priority"] = priority
                existing_alert["is_new_emergency"] = False
                existing_alert["is_new_alert"] = False
                existing_alert["state"] = ClinicalState.EMERGENCY
                existing_alert["banner_dismissed"] = (existing_alert.get("alert_id") in self.dismissed_banner_alerts)

                # Update Smart Alert Explanation
                from backend.alert_explanation import smart_alert_explainer
                expl_update = smart_alert_explainer.generate_explanation(
                    patient_id=patient_id,
                    patient_name=patient_name,
                    location=location,
                    source_record=str(source_record),
                    diagnosis=diagnosis,
                    confidence=confidence,
                    metrics=metrics,
                    cewi_data=cewi_data,
                    signal_quality=signal_quality,
                    trigger_condition=trigger_condition,
                    priority=priority,
                    timestamp=existing_alert.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))),
                    status=existing_alert.get("status", "ACTIVE"),
                    acknowledged=existing_alert.get("acknowledged", False),
                    acknowledged_by=existing_alert.get("acknowledged_by"),
                    acknowledged_time=existing_alert.get("acknowledged_time")
                )
                existing_alert["smart_explanation"] = expl_update
                existing_alert["explanation"] = expl_update.get("plain_english_explanation", existing_alert.get("explanation"))

                if active_ep:
                    try:
                        DatabaseManager.update_episode(
                            active_ep.get("episode_id"),
                            trigger_condition=trigger_condition,
                            diagnosis=diagnosis,
                            explanation_json=expl_update
                        )
                    except Exception:
                        pass

                return existing_alert

            # Otherwise: START A NEW EMERGENCY EPISODE
            self.patient_states[patient_id] = ClinicalState.EMERGENCY
            episode_id = f"EP-{patient_id}-{int(now)}-{uuid.uuid4().hex[:6].upper()}"
            alert_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"

            # Generate Smart Alert Explanation
            from backend.alert_explanation import smart_alert_explainer
            expl_data = smart_alert_explainer.generate_explanation(
                alert_id=alert_id,
                patient_id=patient_id,
                patient_name=patient_name,
                location=location,
                source_record=str(source_record),
                diagnosis=diagnosis,
                confidence=confidence,
                metrics=metrics,
                cewi_data=cewi_data,
                signal_quality=signal_quality,
                trigger_condition=trigger_condition,
                priority=priority,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                status="ACTIVE",
                acknowledged=False
            )

            alert_payload = {
                "alert_id": alert_id,
                "episode_id": episode_id,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "location": location,
                "source_record": str(source_record),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
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
                "state": ClinicalState.EMERGENCY,
                "read": False,
                "is_new_emergency": True,
                "is_new_alert": True,
                "banner_dismissed": (alert_id in self.dismissed_banner_alerts),
                "source_type": "RECORDED_BENCHMARK" if str(source_record) != "SYNTHETIC" else "TELEMETRY_SIMULATION",
                "source_label": f"PhysioNet MIT-BIH Record {source_record} (Recorded Research Signal)" if str(source_record) != "SYNTHETIC" else "Continuous Telemetry Sensor Simulation",
                "smart_explanation": expl_data,
                "explanation": expl_data.get("plain_english_explanation", f"Electrophysiological arrhythmia pattern ({diagnosis}) observed during continuous multi-lead monitoring."),
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

            # If there was a prior active alert for this bed (e.g. advisory), resolve it cleanly
            if active_id and active_id in self.active_alerts:
                self.resolve_alert(active_id, note="Superseded by emergency episode")

            self.active_alerts[alert_id] = alert_payload
            self.active_alerts[patient_id] = alert_payload
            self.bed_active_map[patient_id] = alert_id

            ep_record = {
                "episode_id": episode_id,
                "patient_id": patient_id,
                "alert_id": alert_id,
                "state": ClinicalState.EMERGENCY,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "trigger_condition": trigger_condition,
                "diagnosis": diagnosis,
                "priority": priority,
                "notification_sent": 0,
                "notification_status": None,
                "notification_uuid": None,
                "explanation_json": expl_data
            }
            self.active_episodes[patient_id] = ep_record
            try:
                DatabaseManager.create_episode(
                    episode_id=episode_id,
                    patient_id=patient_id,
                    alert_id=alert_id,
                    trigger_condition=trigger_condition,
                    diagnosis=diagnosis,
                    priority=priority,
                    state=ClinicalState.EMERGENCY,
                    explanation_json=expl_data
                )
            except Exception:
                pass

            self.cooldown_map[(patient_id, diagnosis)] = now
            return alert_payload

        # ---------------------------------------------------------
        # CASE 3: ABNORMAL (NON-EMERGENCY) CONDITION (WARNING / ADVISORY)
        # ---------------------------------------------------------
        self.normal_readings_count[patient_id] = 0

        # If previous state was EMERGENCY, do not downgrade immediately without recovery
        if curr_state == ClinicalState.EMERGENCY and active_id and active_id in self.active_alerts:
            existing_alert = self.active_alerts[active_id]
            existing_alert["heart_rate"] = hr
            existing_alert["cewi_score"] = cewi_score
            existing_alert["epoch"] = now
            existing_alert["is_new_emergency"] = False
            existing_alert["banner_dismissed"] = (existing_alert.get("alert_id") in self.dismissed_banner_alerts)
            return existing_alert

        # Otherwise patient is in ABNORMAL state
        self.patient_states[patient_id] = ClinicalState.ABNORMAL

        # If an active alert already exists for this bed, update it
        if active_id and active_id in self.active_alerts:
            existing_alert = self.active_alerts[active_id]
            existing_alert["heart_rate"] = hr
            existing_alert["cewi_score"] = cewi_score
            existing_alert["epoch"] = now
            existing_alert["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            existing_alert["is_new_emergency"] = False
            existing_alert["is_new_alert"] = False
            existing_alert["state"] = ClinicalState.ABNORMAL
            existing_alert["banner_dismissed"] = (existing_alert.get("alert_id") in self.dismissed_banner_alerts)
            return existing_alert

        # Cooldown check for new warning/advisory alerts
        cooldown_key = (patient_id, diagnosis)
        last_time = self.cooldown_map.get(cooldown_key, 0.0)
        if (now - last_time) < self.cooldown_seconds:
            return None

        # Create advisory/warning alert
        alert_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"

        from backend.alert_explanation import smart_alert_explainer
        expl_data = smart_alert_explainer.generate_explanation(
            patient_id=patient_id,
            patient_name=patient_name,
            location=location,
            source_record=str(source_record),
            diagnosis=diagnosis,
            confidence=confidence,
            metrics=metrics,
            cewi_data=cewi_data,
            signal_quality=signal_quality,
            trigger_condition=trigger_condition,
            priority=priority,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            status="ACTIVE",
            acknowledged=False
        )

        alert_payload = {
            "alert_id": alert_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "location": location,
            "source_record": str(source_record),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
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
            "state": ClinicalState.ABNORMAL,
            "read": False,
            "is_new_emergency": False,
            "is_new_alert": True,
            "banner_dismissed": (alert_id in self.dismissed_banner_alerts),
            "source_type": "RECORDED_BENCHMARK" if str(source_record) != "SYNTHETIC" else "TELEMETRY_SIMULATION",
            "source_label": f"PhysioNet MIT-BIH Record {source_record} (Recorded Research Signal)" if str(source_record) != "SYNTHETIC" else "Continuous Telemetry Sensor Simulation",
            "smart_explanation": expl_data,
            "explanation": expl_data.get("plain_english_explanation", f"Electrophysiological arrhythmia pattern ({diagnosis}) observed during continuous multi-lead monitoring."),
            "acknowledged": False,
            "acknowledged_by": None,
            "acknowledged_time": None,
            "dispatch_status": {
                "sms": "Simulated ICU Pager Dispatched",
                "channels": ["Telemetry Nurse Station"]
            },
            "audio_alarm": not self.audio_muted,
            "disclaimer": "Research telemetry indicator - not an emergency medical directive."
        }

        self.active_alerts[alert_id] = alert_payload
        self.active_alerts[patient_id] = alert_payload
        self.bed_active_map[patient_id] = alert_id
        self.cooldown_map[cooldown_key] = now
        return alert_payload

    def record_episode_notification(
        self,
        patient_id: str,
        episode_id: str,
        notification_uuid: Optional[str] = None,
        notification_sent: bool = True,
        notification_status: Optional[str] = None
    ):
        """Records notification dispatch status for an ongoing emergency episode."""
        if patient_id in self.active_episodes:
            ep = self.active_episodes[patient_id]
            if ep.get("episode_id") == episode_id:
                ep["notification_sent"] = 1 if notification_sent else 0
                ep["notification_uuid"] = notification_uuid
                ep["notification_status"] = notification_status
        try:
            DatabaseManager.update_episode(
                episode_id=episode_id,
                notification_sent=1 if notification_sent else 0,
                notification_status=notification_status,
                notification_uuid=notification_uuid
            )
        except Exception:
            pass

    def get_patient_state(self, patient_id: str) -> str:
        """Returns the current clinical state (NORMAL, ABNORMAL, EMERGENCY, RESOLVED)."""
        return self.patient_states.get(patient_id, ClinicalState.NORMAL)

    def set_recovery_threshold(self, threshold: int):
        """Configures consecutive normal reading threshold for episode resolution."""
        if threshold > 0:
            self.recovery_threshold = int(threshold)

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
