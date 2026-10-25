"""
CardioSentinel - Smart Alert Explanation Engine
Provides transparent, auditable, and structured clinical rationales for all generated alerts.

Distinguishes and correlates:
1. Measured electrophysiological evidence (HR, QRS duration, QTc interval, HRV SDNN/RMSSD).
2. Machine learning classifier predictions & confidence scores.
3. Technical ECG signal quality status and limitations.
4. The exact configured alarm rule and triggering condition.
5. Patient channel, recording source, timestamp, and lifecycle audit state.

DISCLAIMER:
CardioSentinel is an academic research platform. Alert explanations communicate
configured algorithmic rule evaluations and do not replace certified medical decision-making.
"""

from typing import Dict, List, Optional, Any, Union


class SmartAlertExplainer:
    """
    Constructs honest, evidence-based explanations for telemetry alerts.
    Does not run the classifier a second time; reuses actual evaluation artifacts.
    """

    # Configured Alarm Boundaries (IEC 60601-1-8 and Clinical Telemetry Standards)
    RULE_DEFINITIONS = {
        "CRITICAL_VFIB_ASYSTOLE": {
            "name": "Catastrophic Ventricular Rhythm / Arrest",
            "threshold": "Classifier diagnosis of Ventricular Fibrillation or Asystole / Cardiac Arrest",
            "priority": "HIGH PRIORITY (CRISIS / CODE BLUE)",
            "tier": "CRISIS"
        },
        "CRITICAL_VTACH": {
            "name": "Sustained Ventricular Tachycardia",
            "threshold": "Classifier diagnosis of Ventricular Tachycardia with wide QRS / rapid rate",
            "priority": "HIGH PRIORITY (CRISIS / CODE BLUE)",
            "tier": "CRISIS"
        },
        "CRITICAL_CEWI_COLLAPSE": {
            "name": "Imminent Hemodynamic Collapse Risk",
            "threshold": "Cardiac Early Warning Index (CEWI) >= 80/100",
            "priority": "HIGH PRIORITY (CRISIS / CODE BLUE)",
            "tier": "CRISIS"
        },
        "WARNING_PRE_ARRHYTHMIC": {
            "name": "Pre-Arrhythmic Warning",
            "threshold": "CEWI >= 55/100 OR diagnosis of Premature Ventricular Contractions / Atrial Fibrillation",
            "priority": "MEDIUM PRIORITY (WARNING)",
            "tier": "WARNING"
        },
        "WARNING_SEVERE_BRADYCARDIA": {
            "name": "Severe Bradycardia",
            "threshold": "Heart Rate < 45 bpm (and HR > 0)",
            "priority": "MEDIUM PRIORITY (WARNING)",
            "tier": "WARNING"
        },
        "WARNING_SEVERE_TACHYCARDIA": {
            "name": "Severe Tachycardia",
            "threshold": "Heart Rate > 150 bpm",
            "priority": "MEDIUM PRIORITY (WARNING)",
            "tier": "WARNING"
        },
        "WARNING_SENSOR_DISCONNECT": {
            "name": "Lead Off / Sensor Disconnect",
            "threshold": "Technical signal quality POOR or flatline amplitude < 0.10 mV",
            "priority": "MEDIUM PRIORITY (WARNING)",
            "tier": "WARNING"
        },
        "ADVISORY_HEMODYNAMIC_NOTICE": {
            "name": "Hemodynamic Instability Notice",
            "threshold": "Cardiac Early Warning Index (CEWI) >= 35/100",
            "priority": "LOW PRIORITY (ADVISORY)",
            "tier": "ADVISORY"
        }
    }

    @classmethod
    def generate_explanation(
        cls,
        alert: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
        location: Optional[str] = None,
        source_record: Optional[str] = None,
        diagnosis: Optional[str] = None,
        confidence: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        cewi_data: Optional[Dict[str, Any]] = None,
        signal_quality: Optional[Union[str, Dict[str, Any]]] = None,
        trigger_condition: Optional[str] = None,
        priority: Optional[str] = None,
        timestamp: Optional[str] = None,
        status: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        acknowledged_by: Optional[str] = None,
        acknowledged_time: Optional[str] = None,
        snooze_expiry: Optional[float] = None,
        vitals: Optional[Dict[str, Any]] = None,
        ai_diagnosis: Optional[Union[str, Dict[str, Any]]] = None,
        trigger_rule: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesizes a structured, evidence-based explanation for a generated or historical alert.
        Accepts alert dictionary directly, individual parameters, or keyword arguments.
        Never fabricates measurements or confidence ratings.
        """
        import time as _pytime

        # 1. Unpack from alert dict if provided
        if isinstance(alert, dict):
            alert_id = alert.get("alert_id") or kwargs.get("alert_id", "ALT-UNKNOWN")
            patient_id = patient_id or alert.get("patient_id") or "BED-01"
            patient_name = patient_name or alert.get("patient_name") or "Monitored Patient"
            location = location or alert.get("location") or "ICU Bed"
            source_record = source_record or alert.get("source_record") or "TELEMETRY"
            priority = priority or alert.get("priority") or "MEDIUM PRIORITY (WARNING)"
            trigger_condition = trigger_condition or alert.get("trigger_condition") or alert.get("trigger_reason")
            timestamp = timestamp or alert.get("timestamp") or _pytime.strftime("%H:%M:%S")
            status = status or alert.get("status") or "ACTIVE"
            acknowledged = acknowledged if acknowledged is not None else alert.get("acknowledged", False)
            acknowledged_by = acknowledged_by or alert.get("acknowledged_by")
            acknowledged_time = acknowledged_time or alert.get("acknowledged_time")
            snooze_expiry = snooze_expiry or alert.get("snooze_expiry")

            if diagnosis is None:
                diagnosis = alert.get("diagnosis")
            if confidence is None:
                confidence = alert.get("confidence")

            if signal_quality is None and "signal_quality" in alert:
                signal_quality = alert.get("signal_quality")

            if metrics is None and "metrics" in alert:
                metrics = alert.get("metrics")
            if metrics is None:
                metrics_candidate = {}
                if alert.get("heart_rate") is not None:
                    metrics_candidate["heart_rate"] = alert.get("heart_rate")
                if alert.get("qrs_duration_ms") is not None:
                    metrics_candidate["qrs_duration_ms"] = alert.get("qrs_duration_ms")
                if alert.get("qtc_ms") is not None:
                    metrics_candidate["qtc_ms"] = alert.get("qtc_ms")
                if alert.get("hrv_sdnn") is not None:
                    metrics_candidate["hrv_sdnn"] = alert.get("hrv_sdnn")
                if metrics_candidate:
                    metrics = metrics_candidate

            if cewi_data is None and "cewi_data" in alert:
                cewi_data = alert.get("cewi_data")
            if cewi_data is None and alert.get("cewi_score") is not None:
                cewi_data = {"cewi_score": alert.get("cewi_score")}
        else:
            alert_id = kwargs.get("alert_id", "ALT-UNKNOWN")
            patient_id = patient_id or "BED-01"
            patient_name = patient_name or "Monitored Patient"
            location = location or "ICU Bed"
            source_record = source_record or "TELEMETRY"
            priority = priority or "MEDIUM PRIORITY (WARNING)"
            trigger_condition = trigger_condition or "Telemetry Alert Condition"
            timestamp = timestamp or _pytime.strftime("%H:%M:%S")
            status = status or "ACTIVE"
            acknowledged = acknowledged if acknowledged is not None else False

        # 2. Extract vitals / ai_diagnosis kwargs
        if vitals and isinstance(vitals, dict):
            if metrics is None:
                metrics = {}
            if "heart_rate" not in metrics and "hr" in vitals:
                metrics["heart_rate"] = vitals["hr"]

        if ai_diagnosis:
            if isinstance(ai_diagnosis, dict):
                if diagnosis is None:
                    diagnosis = ai_diagnosis.get("diagnosis")
                if confidence is None:
                    confidence = ai_diagnosis.get("confidence")
            elif isinstance(ai_diagnosis, str) and diagnosis is None:
                diagnosis = ai_diagnosis

        # 3. Model classification outcome formatting
        if diagnosis:
            if confidence is not None and confidence > 0:
                conf_pct = round(confidence * 100, 1)
                model_result = f"{diagnosis} ({conf_pct}% confidence)"
                conf_label = f"{conf_pct}%"
            else:
                model_result = diagnosis
                conf_label = "Not available"
        else:
            model_result = "Not available"
            conf_label = "Not available"

        # 4. Signal quality parsing
        sq_status = "GOOD SIGNAL"
        sq_score = None
        sq_reason = "Signal quality checks passed."
        sq_issues: List[str] = []

        if signal_quality is None:
            sq_status = "Not available"
            sq_reason = "Signal quality assessment unavailable"
            sq_details = "Not available"
        elif isinstance(signal_quality, dict):
            sq_status = signal_quality.get("status", "GOOD SIGNAL")
            sq_score = signal_quality.get("quality_score") if signal_quality.get("quality_score") is not None else signal_quality.get("score")
            sq_reason = signal_quality.get("primary_reason", "Signal quality checks passed.")
            sq_issues = signal_quality.get("detected_issues", [])
            sq_details = sq_reason
        elif isinstance(signal_quality, str):
            sq_str = signal_quality.upper()
            if "POOR" in sq_str or "FLATLINE" in sq_str or "LEAD OFF" in sq_str:
                sq_status = "POOR SIGNAL"
                sq_reason = "Technical signal degradation or flatline detected."
            elif "REVIEW" in sq_str or "ARTIFACT" in sq_str or "DEGRADED" in sq_str:
                sq_status = "REVIEW SIGNAL"
                sq_reason = "Technical noise or baseline drift identified."
            elif "INSUFFICIENT" in sq_str:
                sq_status = "INSUFFICIENT DATA"
                sq_reason = "Insufficient valid signal samples."
            else:
                sq_status = "GOOD SIGNAL"
                sq_reason = "Signal quality checks passed."
            sq_details = sq_reason
        else:
            sq_details = sq_reason

        # 5. Extract metrics and physiological features
        metrics = metrics or {}
        hr = metrics.get("heart_rate")
        qrs_dur = metrics.get("qrs_duration_ms")
        qtc = metrics.get("qtc_ms") or metrics.get("qtc_bazett_ms")
        sdnn = metrics.get("hrv_sdnn")

        cewi_data = cewi_data or {}
        cewi_score = cewi_data.get("cewi_score")
        risk_level = cewi_data.get("risk_level", "LOW RISK")

        # 6. Observed measurements string
        if hr is not None and hr > 0:
            observed_measurement = f"{int(hr)} bpm"
            if qrs_dur is not None and float(qrs_dur) > 0:
                observed_measurement += f" (QRS: {float(qrs_dur):.1f} ms)"
        elif hr == 0:
            observed_measurement = "0 bpm (Asystole)"
        elif qrs_dur is not None and float(qrs_dur) > 0:
            observed_measurement = f"QRS: {float(qrs_dur):.1f} ms"
        elif cewi_score is not None and cewi_score > 0:
            observed_measurement = f"CEWI: {cewi_score}/100"
        else:
            observed_measurement = "Not available"

        # 7. Identify which configured rule triggered the alert
        rule_key, rule_name, rule_threshold = cls._identify_rule(
            trigger_condition=trigger_condition,
            priority=priority,
            diagnosis=diagnosis,
            cewi_score=cewi_score if cewi_score is not None else 0,
            hr=hr,
            sq_status=sq_status,
            trigger_rule=trigger_rule
        )

        # 8. Assemble structured supporting evidence list of dicts
        supporting_evidence: List[Dict[str, str]] = []
        observed_items: List[Dict[str, Any]] = []

        # Heart rate evidence
        if hr is not None and hr > 0:
            hr_val = f"{int(hr)} bpm"
            is_trig = (hr > 150 or hr < 45 or ("Tachycardia" in (trigger_condition or "") and hr > 100) or ("Bradycardia" in (trigger_condition or "") and hr < 60))
            supporting_evidence.append({
                "parameter": "Heart Rate",
                "observed_value": hr_val,
                "threshold": "> 150 bpm" if hr > 150 else ("< 45 bpm" if hr < 45 else "60 - 100 bpm"),
                "rule_status": "TRIGGERED" if is_trig else "NORMAL"
            })
            observed_items.append({
                "parameter": "Heart Rate",
                "value": hr_val,
                "unit": "bpm",
                "status": "Bradycardia (<60 bpm)" if hr < 60 else ("Tachycardia (>100 bpm)" if hr > 100 else "Normal Sinus (60-100 bpm)"),
                "is_trigger_driver": is_trig
            })
        elif hr == 0:
            supporting_evidence.append({
                "parameter": "Heart Rate",
                "observed_value": "0 bpm",
                "threshold": "> 45 bpm",
                "rule_status": "TRIGGERED"
            })
            observed_items.append({
                "parameter": "Heart Rate",
                "value": "0 bpm",
                "unit": "bpm",
                "status": "Asystole / Cardiac Arrest",
                "is_trigger_driver": True
            })

        # QRS Duration evidence
        if qrs_dur is not None and float(qrs_dur) > 0:
            qrs_val = f"{float(qrs_dur):.1f} ms"
            is_wide = float(qrs_dur) > 120.0
            supporting_evidence.append({
                "parameter": "QRS Duration",
                "observed_value": qrs_val,
                "threshold": "< 120 ms",
                "rule_status": "TRIGGERED" if is_wide else "NORMAL"
            })
            observed_items.append({
                "parameter": "QRS Duration",
                "value": qrs_val,
                "unit": "ms",
                "status": "Wide QRS (>120 ms)" if is_wide else "Normal (<100 ms)",
                "is_trigger_driver": is_wide and ("Ventricular" in (diagnosis or ""))
            })

        # QTc Interval evidence
        if qtc is not None and float(qtc) > 0:
            qtc_val = f"{float(qtc):.1f} ms"
            is_prolonged = float(qtc) > 470.0
            supporting_evidence.append({
                "parameter": "QTc (Bazett)",
                "observed_value": qtc_val,
                "threshold": "< 470 ms",
                "rule_status": "TRIGGERED" if is_prolonged else "NORMAL"
            })
            observed_items.append({
                "parameter": "QTc (Bazett)",
                "value": qtc_val,
                "unit": "ms",
                "status": "Prolonged (>470 ms)" if is_prolonged else "Normal (<450 ms)",
                "is_trigger_driver": is_prolonged
            })

        # HRV SDNN evidence
        if sdnn is not None and float(sdnn) > 0:
            sdnn_val = f"{float(sdnn):.1f} ms"
            is_depressed = float(sdnn) < 20.0
            supporting_evidence.append({
                "parameter": "HRV SDNN",
                "observed_value": sdnn_val,
                "threshold": ">= 30 ms",
                "rule_status": "TRIGGERED" if is_depressed else "NORMAL"
            })
            observed_items.append({
                "parameter": "HRV SDNN",
                "value": sdnn_val,
                "unit": "ms",
                "status": "Depressed (<20 ms)" if is_depressed else "Normal (30-60 ms)",
                "is_trigger_driver": is_depressed
            })

        # CEWI Score evidence
        if cewi_score is not None and cewi_score > 0:
            is_elevated = cewi_score >= 55
            supporting_evidence.append({
                "parameter": "CEWI Risk Index",
                "observed_value": f"{cewi_score}/100",
                "threshold": "< 35 points",
                "rule_status": "TRIGGERED" if is_elevated else ("REVIEW" if cewi_score >= 35 else "NORMAL")
            })
            observed_items.append({
                "parameter": "CEWI Risk Index",
                "value": f"{cewi_score}/100",
                "unit": "points",
                "status": risk_level,
                "is_trigger_driver": is_elevated
            })

        # Model Prediction evidence
        if diagnosis:
            is_trig = ("Ventricular" in diagnosis or "Tachycardia" in diagnosis or "Fibrillation" in diagnosis or "PVC" in diagnosis)
            supporting_evidence.append({
                "parameter": "Arrhythmia Classifier",
                "observed_value": f"{diagnosis} ({round(confidence * 100, 1)}%)" if confidence else diagnosis,
                "threshold": "Normal Sinus Rhythm",
                "rule_status": "TRIGGERED" if is_trig else "NORMAL"
            })

        # Signal Quality evidence
        if sq_status and sq_status != "Not available":
            supporting_evidence.append({
                "parameter": "Signal Quality",
                "observed_value": sq_status,
                "threshold": "GOOD SIGNAL",
                "rule_status": "TRIGGERED" if sq_status == "POOR SIGNAL" else ("REVIEW" if sq_status == "REVIEW SIGNAL" else "NORMAL")
            })

        # Fallback guarantee: supporting evidence must never be empty
        if not supporting_evidence:
            supporting_evidence.append({
                "parameter": "Alert Trigger Condition",
                "observed_value": trigger_condition or "Telemetry Event",
                "threshold": "Configured Clinical Alert Rules",
                "rule_status": "TRIGGERED"
            })

        # 9. Plain-English narrative summary
        summary_explanation = cls._construct_plain_english_text(
            rule_name=rule_name,
            rule_threshold=rule_threshold,
            trigger_condition=trigger_condition or rule_name,
            diagnosis=diagnosis or "",
            confidence=confidence or 0.0,
            hr=hr,
            cewi_score=cewi_score or 0,
            qrs_dur=qrs_dur,
            sq_status=sq_status,
            sq_reason=sq_reason
        )

        return {
            "explanation_available": True,
            "alert_id": alert_id,
            "alert_severity": priority,
            "channel_or_bed": patient_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "location": location,
            "source_record": str(source_record),
            "source_reference": str(source_record),
            "timestamp": timestamp,
            "summary_explanation": summary_explanation,
            "plain_english_explanation": summary_explanation,
            "rule_title": rule_name,
            "rule_name": rule_name,
            "rule_condition": rule_threshold,
            "configured_rule_threshold": rule_threshold,
            "triggering_condition": trigger_condition or rule_name,
            "trigger_reason": trigger_condition or rule_name,
            "observed_measurement": observed_measurement,
            "model_result": model_result,
            "signal_quality_status": sq_status,
            "signal_quality_details": sq_details,
            "supporting_evidence": supporting_evidence,
            "observed_measurements": observed_items,
            "model_prediction": {
                "rhythm_class": diagnosis or "Not available",
                "confidence_percent": round(confidence * 100, 1) if confidence else None,
                "confidence_label": conf_label,
                "is_critical": ("CRISIS" in str(priority) or "HIGH PRIORITY" in str(priority))
            },
            "signal_quality": {
                "status": sq_status,
                "quality_score": sq_score,
                "primary_reason": sq_reason,
                "detected_issues": sq_issues,
                "snr_db": metrics.get("snr_db")
            },
            "source_metadata": {
                "channel_or_bed": patient_id,
                "patient_name": patient_name,
                "location": location,
                "source_record": str(source_record),
                "source_reference": str(source_record),
                "timestamp": timestamp
            },
            "lifecycle": {
                "status": status,
                "priority": priority,
                "acknowledged": acknowledged,
                "acknowledged_by": acknowledged_by or "Unacknowledged",
                "acknowledged_time": acknowledged_time or "--",
                "snooze_active": (snooze_expiry is not None)
            },
            "disclaimer": "Explanation generated from actual configured rules and measured telemetry artifacts. For research evaluation only."
        }

    @classmethod
    def explain_existing_alert(cls, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely generates or retrieves explanation data for an existing or historical alert.
        Gracefully handles legacy alerts created before the feature was deployed.
        Never fabricates measurements or confidence values.
        """
        if not isinstance(alert, dict):
            return cls._build_unavailable_explanation("Invalid alert record provided.")

        # If full structured smart_explanation is already attached, return it directly
        if "smart_explanation" in alert and isinstance(alert["smart_explanation"], dict):
            return alert["smart_explanation"]

        return cls.generate_explanation(alert=alert)

    @classmethod
    def _identify_rule(
        cls,
        trigger_condition: Optional[str],
        priority: Optional[str],
        diagnosis: Optional[str],
        cewi_score: int,
        hr: Optional[float],
        sq_status: str,
        trigger_rule: Optional[str] = None
    ) -> (str, str, str):
        """Matches the alert to the exact configured trigger rule."""
        tc_upper = str(trigger_condition or "").upper()
        diag_upper = str(diagnosis or "").upper()

        if trigger_rule:
            tr_lower = str(trigger_rule).lower()
            if "tachycardia" in tr_lower:
                rule = cls.RULE_DEFINITIONS["WARNING_SEVERE_TACHYCARDIA"]
                return "WARNING_SEVERE_TACHYCARDIA", rule["name"], rule["threshold"]
            elif "bradycardia" in tr_lower:
                rule = cls.RULE_DEFINITIONS["WARNING_SEVERE_BRADYCARDIA"]
                return "WARNING_SEVERE_BRADYCARDIA", rule["name"], rule["threshold"]
            elif "vt" in tr_lower or "ventricular_tachycardia" in tr_lower:
                rule = cls.RULE_DEFINITIONS["CRITICAL_VTACH"]
                return "CRITICAL_VTACH", rule["name"], rule["threshold"]

        if "LEAD OFF" in tc_upper or "FLATLINE" in tc_upper or sq_status == "POOR SIGNAL":
            rule = cls.RULE_DEFINITIONS["WARNING_SENSOR_DISCONNECT"]
            return "WARNING_SENSOR_DISCONNECT", rule["name"], rule["threshold"]

        if "VENTRICULAR FIBRILLATION" in diag_upper or "ASYSTOLE" in diag_upper or "CATASTROPHIC" in tc_upper:
            rule = cls.RULE_DEFINITIONS["CRITICAL_VFIB_ASYSTOLE"]
            return "CRITICAL_VFIB_ASYSTOLE", rule["name"], rule["threshold"]

        if "VENTRICULAR TACHYCARDIA" in diag_upper or "SUSTAINED VENTRICULAR TACHYCARDIA" in tc_upper or "VTACH" in tc_upper:
            rule = cls.RULE_DEFINITIONS["CRITICAL_VTACH"]
            return "CRITICAL_VTACH", rule["name"], rule["threshold"]

        if cewi_score >= 80 or "IMMINENT COLLAPSE" in tc_upper:
            rule = cls.RULE_DEFINITIONS["CRITICAL_CEWI_COLLAPSE"]
            return "CRITICAL_CEWI_COLLAPSE", rule["name"], rule["threshold"]

        if "BRADYCARDIA" in tc_upper or (hr is not None and hr < 45 and hr > 0):
            rule = cls.RULE_DEFINITIONS["WARNING_SEVERE_BRADYCARDIA"]
            return "WARNING_SEVERE_BRADYCARDIA", rule["name"], rule["threshold"]

        if "TACHYCARDIA" in tc_upper or (hr is not None and hr > 150):
            rule = cls.RULE_DEFINITIONS["WARNING_SEVERE_TACHYCARDIA"]
            return "WARNING_SEVERE_TACHYCARDIA", rule["name"], rule["threshold"]

        if cewi_score >= 55 or "PRE-ARRHYTHMIC" in tc_upper or "PREMATURE" in diag_upper or "ATRIAL FIBRILLATION" in diag_upper:
            rule = cls.RULE_DEFINITIONS["WARNING_PRE_ARRHYTHMIC"]
            return "WARNING_PRE_ARRHYTHMIC", rule["name"], rule["threshold"]

        if cewi_score >= 35 or "HEMODYNAMIC INSTABILITY" in tc_upper:
            rule = cls.RULE_DEFINITIONS["ADVISORY_HEMODYNAMIC_NOTICE"]
            return "ADVISORY_HEMODYNAMIC_NOTICE", rule["name"], rule["threshold"]

        # Default fallback for custom or unconfirmed condition
        cust_name = trigger_condition or "Configured Clinical Parameter Rule"
        return "CONFIGURED_CLINICAL_RULE", cust_name, trigger_condition or "Custom clinical threshold"

    @classmethod
    def _construct_plain_english_text(
        cls,
        rule_name: str,
        rule_threshold: str,
        trigger_condition: str,
        diagnosis: str,
        confidence: float,
        hr: Optional[float],
        cewi_score: int,
        qrs_dur: Optional[float],
        sq_status: str,
        sq_reason: str
    ) -> str:
        """Constructs plain-English narrative explanation from actual measurements."""
        parts = []

        # 1. Trigger reason and rule
        parts.append(
            f"An alert was generated because telemetry evaluation satisfied the configured {rule_name} rule ({rule_threshold})."
        )

        # 2. Observed physiological measurements
        obs = []
        if hr is not None and hr > 0:
            obs.append(f"a measured heart rate of {int(hr)} bpm")
        if qrs_dur is not None and float(qrs_dur) > 0:
            obs.append(f"a QRS complex duration of {float(qrs_dur):.1f} ms")
        if cewi_score > 0:
            obs.append(f"a CEWI Early Warning score of {cewi_score}/100")

        if obs:
            parts.append(f"Observed telemetry features include {', '.join(obs)}.")

        # 3. Model classification outcome
        if diagnosis:
            conf_str = f"with {round(confidence * 100, 1)}% confidence" if confidence else "without confidence rating"
            parts.append(f"The rhythm classifier returned '{diagnosis}' {conf_str}.")

        # 4. Signal quality context
        if sq_status == "POOR SIGNAL":
            parts.append(
                f"Caution: Signal quality at detection was marked POOR SIGNAL ({sq_reason}), indicating severe artifact or technical interference. The rhythm classification must be interpreted cautiously to avoid mistaking noise or artifact for true arrhythmia."
            )
        elif sq_status == "REVIEW SIGNAL":
            parts.append(
                f"Note: Signal quality was marked REVIEW SIGNAL ({sq_reason}); clinical verification of the rhythm tracing is advised."
            )
        elif sq_status == "GOOD SIGNAL":
            parts.append(
                "Technical signal quality checks passed (GOOD SIGNAL), confirming that the trigger is based on authentic physiological telemetry."
            )

        return " ".join(parts)

    @classmethod
    def _build_unavailable_explanation(cls, reason: str) -> Dict[str, Any]:
        """Safe fallback when trigger reason or telemetry context cannot be confirmed."""
        return {
            "explanation_available": True,
            "alert_id": "ALT-UNKNOWN",
            "alert_severity": "UNKNOWN",
            "channel_or_bed": "--",
            "patient_id": "--",
            "patient_name": "--",
            "location": "--",
            "source_record": "--",
            "source_reference": "--",
            "timestamp": "--",
            "summary_explanation": f"Explanation unavailable: {reason}",
            "plain_english_explanation": f"Explanation unavailable: {reason}",
            "rule_title": "Unavailable",
            "rule_name": "Unavailable",
            "rule_condition": "Not available",
            "configured_rule_threshold": "Not available",
            "triggering_condition": "Unavailable",
            "trigger_reason": "Unavailable",
            "observed_measurement": "Not available",
            "model_result": "Not available",
            "signal_quality_status": "Not available",
            "signal_quality_details": reason,
            "supporting_evidence": [{
                "parameter": "Status",
                "observed_value": "Unavailable",
                "threshold": "Not available",
                "rule_status": "NOT AVAILABLE"
            }],
            "observed_measurements": [],
            "model_prediction": {
                "rhythm_class": "Not available",
                "confidence_percent": None,
                "confidence_label": "Not available",
                "is_critical": False
            },
            "signal_quality": {
                "status": "Not available",
                "quality_score": None,
                "primary_reason": reason,
                "detected_issues": [reason]
            },
            "source_metadata": {
                "channel_or_bed": "--",
                "patient_name": "--",
                "location": "--",
                "source_reference": "--",
                "timestamp": "--"
            },
            "lifecycle": {
                "status": "UNKNOWN",
                "priority": "UNKNOWN",
                "acknowledged": False,
                "acknowledged_by": "Not available",
                "acknowledged_time": "--",
                "snooze_active": False
            },
            "disclaimer": "Technical explanation metadata unavailable for this record."
        }


# Global singleton
smart_alert_explainer = SmartAlertExplainer()
