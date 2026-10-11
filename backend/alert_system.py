"""
CardioSentinel - IEC 60601-1-8 Clinical Alert & Multi-Channel Dispatch Engine
Handles tiered medical alarms (Crisis, Warning, Advisory), alarm acknowledgment/silencing,
audit logs, simulated SMS/Pager emergency dispatch, and clinical strip snapshots.
"""

import time
import uuid
from typing import Dict, List, Optional

class AlarmPriority:
    CRISIS = "HIGH PRIORITY (CRISIS / CODE BLUE)"
    WARNING = "MEDIUM PRIORITY (WARNING)"
    ADVISORY = "LOW PRIORITY (ADVISORY)"

class ClinicalAlertSystem:
    def __init__(self, max_history: int = 100):
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: List[Dict] = []
        self.max_history = max_history
        self.silenced_patients: Dict[str, float] = {} # patient_id -> silence_expiry_timestamp

    def evaluate_and_trigger(
        self,
        patient_id: str,
        patient_name: str,
        location: str,
        diagnosis: str,
        cewi_data: Dict,
        metrics: Dict,
        ecg_snapshot: List[float]
    ) -> Optional[Dict]:
        """
        Evaluates current status against clinical threshold rules and active alarm conditions.
        Suppresses redundant alarms if already active, or if currently silenced.
        """
        now = time.time()
        
        # Check if patient alarm is silenced
        if patient_id in self.silenced_patients:
            if now < self.silenced_patients[patient_id]:
                return None
            else:
                del self.silenced_patients[patient_id]

        cewi_score = cewi_data.get("cewi_score", 0)
        
        # Determine priority and need for alert
        priority = None
        trigger_reason = None
        audio_alarm = False
        
        if diagnosis in ["Ventricular Fibrillation", "Asystole / Cardiac Arrest"]:
            priority = AlarmPriority.CRISIS
            trigger_reason = f"CATASTROPHIC RHYTHM: {diagnosis.upper()} DETECTED"
            audio_alarm = True
        elif diagnosis == "Ventricular Tachycardia":
            priority = AlarmPriority.CRISIS
            trigger_reason = "CRITICAL: Sustained Ventricular Tachycardia (>160 bpm)"
            audio_alarm = True
        elif cewi_score >= 80:
            priority = AlarmPriority.CRISIS
            trigger_reason = f"IMMINENT CARDIAC COLLAPSE RISK: CEWI Score {cewi_score}/100"
            audio_alarm = True
        elif cewi_score >= 55 or diagnosis in ["Premature Ventricular Contractions", "Atrial Fibrillation"]:
            priority = AlarmPriority.WARNING
            trigger_reason = f"PRE-ARRHYTHMIC WARNING: {diagnosis} (CEWI {cewi_score})"
            audio_alarm = True
        elif metrics.get("heart_rate", 70) < 45:
            priority = AlarmPriority.WARNING
            trigger_reason = f"SEVERE BRADYCARDIA: HR {metrics.get('heart_rate')} bpm"
            audio_alarm = True
        elif metrics.get("heart_rate", 70) > 150:
            priority = AlarmPriority.WARNING
            trigger_reason = f"SEVERE TACHYCARDIA: HR {metrics.get('heart_rate')} bpm"
            audio_alarm = True
        elif cewi_score >= 35:
            priority = AlarmPriority.ADVISORY
            trigger_reason = f"HEMODYNAMIC INSTABILITY NOTICE: CEWI {cewi_score}"
            audio_alarm = False
            
        if not priority:
            # If previously had alert and now normal, clear it
            if patient_id in self.active_alerts:
                self.resolve_alert(patient_id, reason="Patient rhythm returned to stable parameters")
            return None
            
        # Avoid spamming if same alert is already active
        existing = self.active_alerts.get(patient_id)
        if existing and existing["priority"] == priority and existing["diagnosis"] == diagnosis:
            return None
            
        alert_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"
        alert_payload = {
            "alert_id": alert_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "location": location,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "epoch": now,
            "priority": priority,
            "trigger_reason": trigger_reason,
            "diagnosis": diagnosis,
            "cewi_score": cewi_score,
            "vitals_snapshot": {
                "heart_rate": metrics.get("heart_rate"),
                "qrs_ms": metrics.get("qrs_duration_ms"),
                "qtc_ms": metrics.get("qtc_ms"),
                "sdnn_ms": metrics.get("hrv_sdnn"),
                "shock_index": cewi_data.get("shock_index")
            },
            "audio_alarm": audio_alarm,
            "acknowledged": False,
            "acknowledged_by": None,
            "ecg_strip_sample": ecg_snapshot[-300:] if len(ecg_snapshot) >= 300 else ecg_snapshot,
            "dispatch_status": self._simulate_emergency_dispatch(patient_name, location, priority, trigger_reason)
        }
        
        self.active_alerts[patient_id] = alert_payload
        self.alert_history.insert(0, alert_payload)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop()
            
        return alert_payload

    def _simulate_emergency_dispatch(self, patient_name: str, location: str, priority: str, reason: str) -> Dict:
        """Simulates rapid multi-channel emergency notification (SMS / Hospital Pager / Code Blue)."""
        dispatched_channels = []
        if priority == AlarmPriority.CRISIS:
            dispatched_channels.append({
                "channel": "CODE BLUE TELEMETRY BROKER",
                "target": "Crash Cart & ICU Resuscitation Team",
                "status": "IMMEDIATE AUDIO BEACON ACTIVATED",
                "message": f"[CODE BLUE] {patient_name} @ {location}. {reason}."
            })
            dispatched_channels.append({
                "channel": "ON-CALL CARDIOLOGIST PAGER",
                "target": "Dr. Sarah Sterling, MD (Cardiac Electrophysiology)",
                "status": "DISPATCHED (Ack Priority 1)",
                "message": f"STAT EMERGENCY: {patient_name} ({location}) - {reason}."
            })
        elif priority == AlarmPriority.WARNING:
            dispatched_channels.append({
                "channel": "CHARGENURSE TELEMETRY CONSOLE",
                "target": "Central Nursing Station Console 4",
                "status": "ALERT DISPLAYED & LOGGED",
                "message": f"[WARNING] {patient_name} @ {location}. {reason}."
            })
            dispatched_channels.append({
                "channel": "MOBILE NURSE HANDSET (VoLTE/SMS)",
                "target": "Duty Telemetry Nurse Pager #108",
                "status": "SMS SENT",
                "message": f"Cardiac Warning: {patient_name} in {location} shows early decompensation."
            })
        else:
            dispatched_channels.append({
                "channel": "CENTRAL AUDIT LOG",
                "target": "Clinical Advisory Stream",
                "status": "LOGGED",
                "message": f"Advisory notice for {patient_name}: {reason}."
            })
            
        return {
            "dispatch_timestamp": time.strftime("%H:%M:%S", time.localtime()),
            "channels": dispatched_channels
        }

    def acknowledge_alert(self, alert_id: str, clinician_name: str = "Dr. On-Duty") -> bool:
        """Clinician acknowledges the alert, silencing audio alarm for that alert."""
        for pid, alert in self.active_alerts.items():
            if alert["alert_id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_by"] = clinician_name
                alert["audio_alarm"] = False
                return True
        for alert in self.alert_history:
            if alert["alert_id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_by"] = clinician_name
                alert["audio_alarm"] = False
                return True
        return False

    def silence_patient(self, patient_id: str, duration_sec: float = 120.0) -> bool:
        """Temporarily silences alarms for a specific patient for 2 minutes during clinical care."""
        self.silenced_patients[patient_id] = time.time() + duration_sec
        if patient_id in self.active_alerts:
            self.active_alerts[patient_id]["audio_alarm"] = False
        return True

    def resolve_alert(self, patient_id: str, reason: str = "Resolved by system") -> Optional[Dict]:
        if patient_id in self.active_alerts:
            resolved = self.active_alerts.pop(patient_id)
            resolved["resolved_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            resolved["resolution_reason"] = reason
            return resolved
        return None

    def get_active_alerts(self) -> List[Dict]:
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        return self.alert_history[:limit]
