"""
CardioSentinel | ECG Intelligence - Guardian Emergency SMS Notification Service
Integrates configurable SMS providers (Twilio / Mock Adapter) with asynchronous queueing,
bounded exponential-backoff retries, phone masking, idempotency, and clinical mode isolation.
"""

import os
import time
import uuid
import queue
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple
import requests
from dotenv import load_dotenv

from backend.database import DatabaseManager, normalize_phone_number, mask_phone_number

# Load environment variables from .env at project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# Configure logger
logger = logging.getLogger("CardioSentinel.SMS")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [SMS] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SMSService:
    """Enterprise SMS Dispatcher for Guardian Emergency Telemetry Notifications."""

    def __init__(self):
        # Background delivery queue & worker thread
        self.dispatch_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.worker_thread = threading.Thread(target=self._delivery_worker, daemon=True)
        self.worker_thread.start()

        # Mock mode override (used for automated testing without real network dispatch)
        self._mock_mode: bool = os.environ.get("MOCK_SMS", "false").lower() in ("true", "1", "yes")

        # Rate limiting: destination phone -> list of recent dispatch epochs
        self._rate_limit_lock = threading.Lock()
        self._rate_limit_window_sec = 60.0 # 1 minute
        self._max_messages_per_window = 5
        self._dispatch_timestamps: Dict[str, List[float]] = {}

        # Cooldown map: (patient_id, alert_condition) -> last dispatch epoch
        self._cooldown_seconds = 60.0
        self._cooldown_map: Dict[Tuple[str, str], float] = {}

    @property
    def is_mock_mode(self) -> bool:
        return self._mock_mode

    def set_mock_mode(self, enabled: bool):
        """Enables or disables mock mode (vital for automated tests)."""
        self._mock_mode = bool(enabled)
        logger.info(f"SMS Service mock mode set to: {self._mock_mode}")

    def get_provider_status(self) -> Dict[str, Any]:
        """
        Inspects provider configuration and credentials without leaking secrets.
        Never exposes auth tokens or private credentials to browser.
        """
        sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
        token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
        from_num = os.environ.get("TWILIO_FROM_NUMBER", "").strip()

        # Identify missing or placeholder credentials
        missing_keys = []
        is_placeholder = False

        if not sid:
            missing_keys.append("TWILIO_ACCOUNT_SID")
        elif sid.startswith("AC_") or "placeholder" in sid.lower() or "your_" in sid.lower() or "acxxxx" in sid.lower() or len(sid) < 16:
            missing_keys.append("TWILIO_ACCOUNT_SID (contains placeholder)")
            is_placeholder = True

        if not token:
            missing_keys.append("TWILIO_AUTH_TOKEN")
        elif "your_" in token.lower() or "token_here" in token.lower() or "placeholder" in token.lower() or len(token) < 16:
            missing_keys.append("TWILIO_AUTH_TOKEN (contains placeholder)")
            is_placeholder = True

        if not from_num:
            missing_keys.append("TWILIO_FROM_NUMBER")
        else:
            is_valid_from, _, _ = normalize_phone_number(from_num)
            if not is_valid_from or "1234567" in from_num or "placeholder" in from_num.lower():
                missing_keys.append("TWILIO_FROM_NUMBER (invalid or placeholder)")
                is_placeholder = True

        is_configured = bool(sid and token and from_num and not is_placeholder and not missing_keys)

        if is_configured:
            status_label = "Configured & Operational"
            provider_name = "Twilio Telemetry Gateway"
        elif self._mock_mode:
            status_label = "Mock Emulation Active (Tests Only)"
            provider_name = "Twilio Mock Adapter (In-Memory)"
        else:
            status_label = "Provider Not Configured"
            provider_name = "SMS Provider Not Configured"

        return {
            "is_configured": is_configured,
            "provider_name": provider_name,
            "mock_mode_active": self._mock_mode,
            "account_sid_masked": (sid[:4] + "••••••••" + sid[-4:]) if (is_configured and len(sid) >= 8) else "--",
            "from_number_masked": mask_phone_number(from_num) if is_configured else "--",
            "status_label": status_label,
            "missing_keys": missing_keys
        }

    def check_readiness(self, patient_id: str, data_mode: str = "MIT-BIH") -> Dict[str, Any]:
        """
        Evaluates the mandatory readiness criteria for Guardian Notifications:
        1. Guardian Registration
        2. Phone Registration (Registered — Not Phone Verified)
        3. Provider Configuration
        4. Notification Permission (is_enabled && consent_confirmed)
        5. Data-Source Mode (Live vs Recorded Demo)
        """
        guardian = DatabaseManager.get_guardian(patient_id)
        provider_info = self.get_provider_status()

        reg_ok = bool(guardian is not None)
        phone_ok = bool(guardian and guardian.get("phone_normalized"))
        provider_ok = bool(provider_info["is_configured"] or self._mock_mode)
        perm_ok = bool(guardian and guardian.get("is_enabled") == 1 and guardian.get("consent_confirmed") == 1)
        is_live = (data_mode.upper() not in ["MIT-BIH", "RECORDED"])

        # Determine overall readiness
        if not reg_ok:
            overall = "REGISTRATION_REQUIRED"
            overall_label = "Guardian Not Registered"
            can_send_live = False
        elif not phone_ok:
            overall = "PHONE_MISSING"
            overall_label = "Phone Number Missing or Invalid"
            can_send_live = False
        elif not perm_ok:
            overall = "CONSENT_REQUIRED"
            overall_label = "Notification Consent Required"
            can_send_live = False
        elif not provider_ok:
            overall = "PROVIDER_UNCONFIGURED"
            overall_label = "SMS Provider Not Configured"
            can_send_live = False
        elif not is_live:
            overall = "DEMO_MODE_ACTIVE"
            overall_label = "Recorded ECG Playback (Demonstration Only)"
            can_send_live = False
        else:
            overall = "READY_LIVE"
            overall_label = "Ready for Real Emergency Notifications"
            can_send_live = True

        phone_check_data = {
            "passed": phone_ok,
            "label": "Registered — Not Phone Verified" if phone_ok else "Not Registered",
            "details": guardian.get("phone_masked") if guardian else "--"
        }

        return {
            "patient_id": patient_id,
            "overall_status": overall,
            "overall_label": overall_label,
            "can_send_live": can_send_live,
            "can_send_test": bool(reg_ok and phone_ok and provider_ok),
            "checks": {
                "guardian_registered": {
                    "passed": reg_ok,
                    "label": "Registered" if reg_ok else "Not Registered",
                    "details": guardian.get("full_name") if guardian else "No contact assigned"
                },
                "phone_registered": phone_check_data,
                "phone_verified": phone_check_data,  # Alias for backward compatibility
                "provider_configured": {
                    "passed": provider_ok,
                    "label": provider_info["status_label"],
                    "details": provider_info["provider_name"]
                },
                "notification_permission": {
                    "passed": perm_ok,
                    "label": "Enabled & Consented" if perm_ok else ("Consent Missing" if (guardian and not guardian.get("consent_confirmed")) else "Disabled"),
                    "details": "Guardian consent recorded" if perm_ok else "Explicit authorization needed"
                },
                "data_source_mode": {
                    "passed": is_live,
                    "label": "Live Telemetry Mode" if is_live else "RECORDED ECG — DEMONSTRATION MODE",
                    "details": "Continuous multi-lead sensor stream" if is_live else "PhysioNet MIT-BIH Historical Playback"
                }
            }
        }

    def _check_rate_limit(self, phone: str) -> bool:
        """Enforces rate limiting per phone number."""
        now = time.time()
        with self._rate_limit_lock:
            history = self._dispatch_timestamps.get(phone, [])
            history = [t for t in history if (now - t) < self._rate_limit_window_sec]
            if len(history) >= self._max_messages_per_window:
                self._dispatch_timestamps[phone] = history
                return False
            history.append(now)
            self._dispatch_timestamps[phone] = history
            return True

    def get_test_sms_preview(self, patient_id: str) -> Dict[str, Any]:
        """
        Prepares a safe preview of the recipient and message body before test dispatch.
        Does not send any message.
        """
        guardian = DatabaseManager.get_guardian(patient_id)
        if not guardian:
            return {
                "success": False,
                "error": "No registered guardian found for this bed."
            }

        provider_info = self.get_provider_status()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        test_body = (
            f"CardioSentinel test message: Bed: {patient_id} | Recipient: {guardian.get('full_name')} | "
            f"Time: {now_str}. Your registered guardian contact is configured for notification testing. "
            "This is a test only, not a medical emergency."
        )

        return {
            "success": True,
            "patient_id": patient_id,
            "recipient_name": guardian.get("full_name"),
            "phone_masked": guardian.get("phone_masked"),
            "relationship": guardian.get("relationship"),
            "is_enabled": guardian.get("is_enabled") == 1,
            "consent_confirmed": guardian.get("consent_confirmed") == 1,
            "phone_status": guardian.get("phone_status", "Registered — Not Phone Verified"),
            "preview_body": test_body,
            "provider_configured": bool(provider_info["is_configured"] or self._mock_mode),
            "provider_status_label": provider_info["status_label"],
            "missing_keys": provider_info.get("missing_keys", [])
        }

    def send_test_sms(self, patient_id: str, triggered_by: str = "Clinician") -> Dict[str, Any]:
        """
        Executes the safe Test SMS workflow.
        1. Validates guardian exists with valid registered phone number.
        2. Validates provider is configured (or mock mode active).
        3. Generates labeled test message explaining provider charges may apply.
        4. Dispatches synchronously through provider with bounded timeout.
        5. Records attempt in database and audit trail.
        """
        guardian = DatabaseManager.get_guardian(patient_id)
        if not guardian:
            return {
                "success": False,
                "status": "FAILED",
                "error": "No registered guardian found for this monitored bed. Please register guardian details first."
            }

        phone_to = guardian.get("phone_normalized")
        if not phone_to:
            return {
                "success": False,
                "status": "FAILED",
                "error": "Registered guardian has no valid phone number. Please update guardian phone number."
            }

        provider_info = self.get_provider_status()
        if not provider_info["is_configured"] and not self._mock_mode:
            missing = ", ".join(provider_info.get("missing_keys", [])) or "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER"
            return {
                "success": False,
                "status": "FAILED",
                "error": f"SMS provider not configured. Real delivery disabled. Missing configuration: {missing}. Configure credentials in .env to enable SMS dispatch."
            }

        if not self._check_rate_limit(phone_to):
            return {
                "success": False,
                "status": "FAILED",
                "error": f"Rate limit exceeded for {guardian.get('phone_masked')}. Please wait before sending another test message."
            }

        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        test_body = (
            f"CardioSentinel test message: Bed: {patient_id} | Recipient: {guardian.get('full_name')} | "
            f"Time: {now_str}. Your registered guardian contact is configured for notification testing. "
            "This is a test only, not a medical emergency."
        )

        idemp_key = f"TEST-{patient_id}-{int(time.time())}-{uuid.uuid4().hex[:6]}"

        # Record PENDING notification
        notif = DatabaseManager.create_notification(
            patient_id=patient_id,
            guardian_id=guardian.get("id"),
            recipient_phone_masked=guardian.get("phone_masked"),
            message_type="TEST",
            message_body=test_body,
            status="PENDING",
            provider="twilio" if provider_info["is_configured"] else "mock",
            idempotency_key=idemp_key
        )

        # Dispatch via provider
        result = self._dispatch_sms_provider(
            phone_to=phone_to,
            body=test_body,
            notification_uuid=notif["notification_uuid"]
        )

        return result

    def send_emergency_alert(
        self,
        alert_payload: Dict[str, Any],
        patient_data_mode: str = "MIT-BIH"
    ) -> Dict[str, Any]:
        """
        Evaluates clinical alert for guardian notification.
        Enforces:
        - Recorded ECG Demo Mode vs Real Monitoring Mode
        - Category preference matching (CRISIS / WARNING)
        - Idempotency & deduplication cooldown
        - Registered phone & consent
        """
        patient_id = alert_payload.get("patient_id", "BED-01")
        alert_id = alert_payload.get("alert_id", "ALT-UNKNOWN")
        episode_id = alert_payload.get("episode_id")
        priority = alert_payload.get("priority", "")
        condition = alert_payload.get("trigger_condition", alert_payload.get("diagnosis", "Critical Arrhythmia"))
        diagnosis = alert_payload.get("diagnosis", "Critical Arrhythmia")
        location = alert_payload.get("location", "ICU Telemetry")

        guardian = DatabaseManager.get_guardian(patient_id)
        if not guardian:
            logger.info(f"No guardian configured for {patient_id}. Skipping emergency SMS.")
            return {"eligible": False, "reason": "No guardian registered"}

        phone_to = guardian.get("phone_normalized")
        if not phone_to:
            logger.info(f"No phone number on file for guardian ({patient_id}). Skipping emergency SMS.")
            return {"eligible": False, "reason": "Guardian phone number missing"}

        # Episode-level and Alert-level idempotency / deduplication
        idemp_key = f"EPISODE-{episode_id}" if episode_id else f"ALERT-{alert_id}-{patient_id}"
        existing = DatabaseManager.check_idempotency(idemp_key)
        if existing:
            return {
                "eligible": False,
                "reason": "Notification already exists for this emergency episode",
                "already_sent": True,
                "notification_uuid": existing.get("notification_uuid")
            }

        # Also check fallback alert idempotency
        idemp_key_alt = f"ALERT-{alert_id}-{patient_id}"
        existing_alt = DatabaseManager.check_idempotency(idemp_key_alt)
        if existing_alt:
            return {
                "eligible": False,
                "reason": "Notification already exists for this alert ID",
                "already_sent": True,
                "notification_uuid": existing_alt.get("notification_uuid")
            }

        # Cooldown check: prevent re-sending identical condition/diagnosis to guardian within 60s
        cooldown_key = (patient_id, diagnosis)
        now = time.time()
        last_sent = max(self._cooldown_map.get(cooldown_key, 0.0), self._cooldown_map.get((patient_id, condition), 0.0))
        if (now - last_sent) < self._cooldown_seconds:
            logger.info(f"Guardian notification for ({patient_id}, {diagnosis}) suppressed under 60s cooldown.")
            return {"eligible": False, "reason": "Cooldown active for identical condition"}

        # Check preferred alert category
        prefs = [p.strip().upper() for p in str(guardian.get("preferred_categories", "CRISIS,WARNING")).split(",")]
        is_crisis = "CRISIS" in priority.upper() or "HIGH" in priority.upper() or "CRIT" in priority.upper() or alert_payload.get("state") == "EMERGENCY"
        is_warning = "WARNING" in priority.upper() or "MEDIUM" in priority.upper()

        if is_crisis and "CRISIS" not in prefs:
            return {"eligible": False, "reason": "Guardian preferences exclude Crisis alerts"}
        if is_warning and not is_crisis and "WARNING" not in prefs:
            return {"eligible": False, "reason": "Guardian preferences exclude Warning alerts"}
        if not is_crisis and not is_warning:
            return {"eligible": False, "reason": "Alert priority tier below guardian notification threshold"}

        # --- MODE ISOLATION ---
        is_recorded_demo = (patient_data_mode.upper() in ["MIT-BIH", "RECORDED"])
        event_time = alert_payload.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S")

        if is_recorded_demo:
            # DEMONSTRATION MODE: Never send real emergency SMS from recorded ECG playback!
            demo_body = (
                f"DEMO — Recorded ECG playback: Bed: {patient_id} | Severity: {priority} | "
                f"Event: {condition} | Time: {event_time}. "
                "This is a demonstration notification, not a live patient emergency."
            )

            notif = DatabaseManager.create_notification(
                patient_id=patient_id,
                alert_id=alert_id,
                guardian_id=guardian.get("id"),
                recipient_phone_masked=guardian.get("phone_masked"),
                message_type="DEMO_PREVIEW",
                message_body=demo_body,
                status="DEMO_RECORDED",
                provider="local_preview",
                provider_message_id=f"DEMO-{uuid.uuid4().hex[:8].upper()}",
                idempotency_key=idemp_key
            )

            self._cooldown_map[cooldown_key] = now
            self._cooldown_map[(patient_id, condition)] = now
            logger.info(f"Demonstration notification recorded for {patient_id} ({alert_id}). Real SMS suppressed.")
            return {
                "eligible": True,
                "mode": "DEMO_RECORDED",
                "notification": notif,
                "notification_uuid": notif.get("notification_uuid"),
                "message": "Demonstration notification recorded (no real SMS sent during recorded playback)."
            }

        # --- REAL MONITORING MODE ---
        readiness = self.check_readiness(patient_id, data_mode=patient_data_mode)
        # Check consent and registration
        if not readiness["checks"]["guardian_registered"]["passed"] or not readiness["checks"]["phone_registered"]["passed"] or not readiness["checks"]["notification_permission"]["passed"]:
            logger.warning(f"Real emergency SMS blocked for {patient_id}. Reason: {readiness['overall_label']}")
            return {
                "eligible": False,
                "reason": f"Real SMS blocked: {readiness['overall_label']}"
            }

        # Check provider configuration
        provider_info = self.get_provider_status()
        if not provider_info["is_configured"] and not self._mock_mode:
            # Provider is unconfigured: record FAILED status accurately without claiming delivery
            missing = ", ".join(provider_info.get("missing_keys", [])) or "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER"
            err_msg = f"SMS provider not configured ({missing}). Real delivery disabled."
            notif = DatabaseManager.create_notification(
                patient_id=patient_id,
                alert_id=alert_id,
                guardian_id=guardian.get("id"),
                recipient_phone_masked=guardian.get("phone_masked"),
                message_type="REAL_EMERGENCY",
                message_body=f"CardioSentinel Emergency Alert | Bed: {patient_id} | Severity: {priority} | Event: {condition} | Time: {event_time}. Delivery pending provider configuration.",
                status="FAILED",
                provider="none",
                error_message=err_msg,
                idempotency_key=idemp_key
            )
            self._cooldown_map[cooldown_key] = now
            self._cooldown_map[(patient_id, condition)] = now
            logger.warning(f"Recorded unconfigured provider failure for {patient_id} ({alert_id}): {err_msg}")
            return {
                "eligible": False,
                "status": "FAILED",
                "reason": f"Real SMS blocked: SMS provider not configured",
                "error": err_msg,
                "notification": notif,
                "notification_uuid": notif.get("notification_uuid")
            }

        emergency_body = (
            f"CardioSentinel monitoring alert: Bed: {patient_id} | Severity: {priority} | "
            f"Event: {condition} | Time: {event_time}. "
            "Please check clinical telemetry or contact responsible care team immediately. This message is not a diagnosis."
        )

        notif = DatabaseManager.create_notification(
            patient_id=patient_id,
            alert_id=alert_id,
            guardian_id=guardian.get("id"),
            recipient_phone_masked=guardian.get("phone_masked"),
            message_type="REAL_EMERGENCY",
            message_body=emergency_body,
            status="PENDING",
            provider="twilio" if provider_info["is_configured"] else "mock",
            idempotency_key=idemp_key
        )

        self._cooldown_map[cooldown_key] = now
        self._cooldown_map[(patient_id, condition)] = now

        if self._mock_mode:
            res = self._dispatch_sms_provider(
                phone_to=phone_to,
                body=emergency_body,
                notification_uuid=notif["notification_uuid"]
            )
            return {
                "eligible": True,
                "mode": "REAL_EMERGENCY_QUEUED",
                "status": res.get("status", "DELIVERED"),
                "notification_uuid": notif["notification_uuid"]
            }

        # Enqueue for asynchronous background delivery so ECG loop remains responsive
        self.dispatch_queue.put({
            "notification_uuid": notif["notification_uuid"],
            "phone_to": phone_to,
            "body": emergency_body,
            "retry_count": 0
        })

        return {
            "eligible": True,
            "mode": "REAL_EMERGENCY_QUEUED",
            "status": "PENDING",
            "notification_uuid": notif["notification_uuid"]
        }

    def _delivery_worker(self):
        """Background worker thread for resilient, asynchronous SMS dispatch."""
        while True:
            try:
                task = self.dispatch_queue.get()
                if task is None:
                    break

                notif_uuid = task["notification_uuid"]
                phone_to = task["phone_to"]
                body = task["body"]
                retry = task.get("retry_count", 0)

                result = self._dispatch_sms_provider(
                    phone_to=phone_to,
                    body=body,
                    notification_uuid=notif_uuid,
                    current_retry=retry
                )

                if not result.get("success") and retry < 2:
                    backoff = 1.0 * (2 ** retry)
                    logger.info(f"Retrying notification {notif_uuid} in {backoff}s (attempt {retry + 2}/3)")
                    time.sleep(backoff)
                    task["retry_count"] = retry + 1
                    self.dispatch_queue.put(task)

                self.dispatch_queue.task_done()
            except Exception as e:
                logger.error(f"Error in SMS delivery background worker: {e}", exc_info=True)
                time.sleep(0.5)

    def _dispatch_sms_provider(
        self,
        phone_to: str,
        body: str,
        notification_uuid: Optional[str] = None,
        current_retry: int = 0
    ) -> Dict[str, Any]:
        """
        Direct HTTP invocation of configured provider (Twilio or Mock Adapter).
        Applies strict validation, timeout, and records actual provider status.
        Never simulates delivery when real provider is configured or unconfigured.
        """
        if not notification_uuid:
            notification_uuid = f"NOTIF-{uuid.uuid4().hex[:8].upper()}"

        # If in mock mode (used in automated unit tests)
        if self._mock_mode:
            mock_msg_id = f"MOCK-SM-{uuid.uuid4().hex[:12].upper()}"
            DatabaseManager.update_notification_status(
                notification_uuid=notification_uuid,
                status="DELIVERED",
                provider_message_id=mock_msg_id,
                error_message=None,
                increment_retry=(current_retry > 0)
            )
            logger.info(f"[MOCK SMS] Delivered to {mask_phone_number(phone_to)}: {body[:60]}... [ID: {mock_msg_id}]")
            return {
                "success": True,
                "status": "DELIVERED",
                "provider": "mock",
                "message_id": mock_msg_id,
                "notification_uuid": notification_uuid,
                "message": "Mock SMS accepted and confirmed delivered."
            }

        # Check real Twilio credentials
        sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
        token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
        from_number = os.environ.get("TWILIO_FROM_NUMBER", "").strip()

        provider_status = self.get_provider_status()
        if not provider_status["is_configured"]:
            missing = ", ".join(provider_status.get("missing_keys", [])) or "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER"
            err_msg = f"SMS provider not configured ({missing}). Real delivery disabled."
            DatabaseManager.update_notification_status(
                notification_uuid=notification_uuid,
                status="FAILED",
                provider_message_id=None,
                error_message=err_msg,
                increment_retry=(current_retry > 0)
            )
            return {
                "success": False,
                "status": "FAILED",
                "provider": "none",
                "error": err_msg,
                "notification_uuid": notification_uuid
            }

        # Validate recipient and sender phone numbers
        is_valid_to, norm_to, err_to = normalize_phone_number(phone_to)
        if not is_valid_to:
            err_msg = f"Invalid recipient phone number: {err_to}"
            DatabaseManager.update_notification_status(
                notification_uuid=notification_uuid,
                status="FAILED",
                provider_message_id=None,
                error_message=err_msg,
                increment_retry=(current_retry > 0)
            )
            return {
                "success": False,
                "status": "FAILED",
                "provider": "twilio",
                "error": err_msg,
                "notification_uuid": notification_uuid
            }

        is_valid_from, norm_from, err_from = normalize_phone_number(from_number)
        if not is_valid_from:
            err_msg = f"Invalid sender phone number: {err_from}"
            DatabaseManager.update_notification_status(
                notification_uuid=notification_uuid,
                status="FAILED",
                provider_message_id=None,
                error_message=err_msg,
                increment_retry=(current_retry > 0)
            )
            return {
                "success": False,
                "status": "FAILED",
                "provider": "twilio",
                "error": err_msg,
                "notification_uuid": notification_uuid
            }

        # Real Twilio REST API invocation
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        payload = {
            "To": norm_to,
            "From": norm_from,
            "Body": body
        }
        status_callback_url = os.environ.get("TWILIO_STATUS_CALLBACK_URL", "").strip()
        if status_callback_url:
            payload["StatusCallback"] = status_callback_url

        try:
            resp = requests.post(
                url,
                data=payload,
                auth=(sid, token),
                timeout=5.0
            )

            if resp.status_code in [200, 201]:
                data = resp.json()
                msg_id = data.get("sid", f"SM-{uuid.uuid4().hex[:8]}")
                twilio_status = str(data.get("status", "accepted")).lower()

                # Accurately reflect provider state (never simulate delivered unless confirmed)
                if twilio_status in ["delivered"]:
                    final_status = "DELIVERED"
                elif twilio_status in ["sent"]:
                    final_status = "SENT"
                elif twilio_status in ["accepted"]:
                    final_status = "ACCEPTED"
                elif twilio_status in ["queued", "sending"]:
                    final_status = "PENDING"
                elif twilio_status in ["failed", "undelivered"]:
                    final_status = "FAILED"
                else:
                    final_status = "ACCEPTED"

                is_success = (final_status != "FAILED")
                err_msg = data.get("error_message") if not is_success else None

                DatabaseManager.update_notification_status(
                    notification_uuid=notification_uuid,
                    status=final_status,
                    provider_message_id=msg_id,
                    error_message=err_msg,
                    increment_retry=(current_retry > 0)
                )

                logger.info(f"Twilio message response: {msg_id} (status: {twilio_status} -> {final_status})")
                return {
                    "success": is_success,
                    "status": final_status,
                    "provider": "twilio",
                    "message_id": msg_id,
                    "provider_status": twilio_status,
                    "notification_uuid": notification_uuid,
                    "message": f"Message {twilio_status} by provider gateway (ID: {msg_id})."
                }
            else:
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("message", f"Twilio HTTP {resp.status_code} error.")
                except Exception:
                    err_msg = f"Provider rejected request with HTTP {resp.status_code}."

                DatabaseManager.update_notification_status(
                    notification_uuid=notification_uuid,
                    status="FAILED",
                    provider_message_id=None,
                    error_message=err_msg,
                    increment_retry=(current_retry > 0)
                )

                logger.warning(f"Twilio dispatch failed: {err_msg}")
                return {
                    "success": False,
                    "status": "FAILED",
                    "provider": "twilio",
                    "error": err_msg,
                    "notification_uuid": notification_uuid
                }

        except requests.exceptions.Timeout:
            err_msg = "Provider connection timed out after 5.0 seconds."
            DatabaseManager.update_notification_status(
                notification_uuid=notification_uuid,
                status="FAILED",
                error_message=err_msg,
                increment_retry=(current_retry > 0)
            )
            return {
                "success": False,
                "status": "FAILED",
                "error": err_msg,
                "notification_uuid": notification_uuid
            }
        except Exception as e:
            err_msg = f"Network or provider error: {str(e)}"
            DatabaseManager.update_notification_status(
                notification_uuid=notification_uuid,
                status="FAILED",
                error_message=err_msg,
                increment_retry=(current_retry > 0)
            )
            return {
                "success": False,
                "status": "FAILED",
                "error": err_msg,
                "notification_uuid": notification_uuid
            }

    def retry_notification(self, notification_uuid: str) -> Dict[str, Any]:
        """Allows safe manual retry for eligible failed notifications."""
        notif = DatabaseManager.get_notification_by_uuid(notification_uuid)
        if not notif:
            return {"success": False, "error": "Notification record not found."}

        if notif.get("status") not in ["FAILED", "PENDING"]:
            return {"success": False, "error": "Only failed or pending notifications can be retried."}

        retries = int(notif.get("retry_count", 0))
        max_retries = int(notif.get("max_retries", 3))
        if retries >= max_retries:
            return {"success": False, "error": f"Maximum retry limit ({max_retries}) reached for this notification."}

        guardian = DatabaseManager.get_guardian(notif.get("patient_id"))
        if not guardian:
            return {"success": False, "error": "Associated guardian record not found."}

        phone_to = guardian.get("phone_normalized")
        result = self._dispatch_sms_provider(
            phone_to=phone_to,
            body=notif.get("message_body"),
            notification_uuid=notification_uuid,
            current_retry=retries + 1
        )
        return result

    def sync_notification_status(self, notification_uuid: str) -> Dict[str, Any]:
        """
        Polls provider (Twilio) for current delivery status of a message.
        Accurately distinguishes NOT_CONFIGURED, FAILED, PENDING, SENT, ACCEPTED, and DELIVERED.
        """
        notif = DatabaseManager.get_notification_by_uuid(notification_uuid)
        if not notif:
            return {"success": False, "status": "NOT_FOUND", "error": "Notification record not found."}

        msg_id = notif.get("provider_message_id")
        if not msg_id:
            return {"success": False, "status": notif.get("status", "FAILED"), "error": "No provider message identifier on record."}

        if self._mock_mode or str(msg_id).startswith("MOCK-"):
            return {
                "success": True,
                "status": notif.get("status", "DELIVERED"),
                "provider_status": "delivered",
                "message": "Mock provider message verified."
            }

        provider_info = self.get_provider_status()
        if not provider_info["is_configured"]:
            return {"success": False, "status": "NOT_CONFIGURED", "error": "SMS Gateway Provider Not Configured."}

        sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
        token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages/{msg_id}.json"

        try:
            resp = requests.get(url, auth=(sid, token), timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                twilio_status = str(data.get("status", "")).lower()

                if twilio_status in ["delivered"]:
                    final_status = "DELIVERED"
                elif twilio_status in ["sent"]:
                    final_status = "SENT"
                elif twilio_status in ["accepted"]:
                    final_status = "ACCEPTED"
                elif twilio_status in ["queued", "sending"]:
                    final_status = "PENDING"
                elif twilio_status in ["failed", "undelivered"]:
                    final_status = "FAILED"
                else:
                    final_status = notif.get("status", "SENT")

                err_msg = data.get("error_message") if final_status == "FAILED" else None
                DatabaseManager.update_notification_status(
                    notification_uuid=notification_uuid,
                    status=final_status,
                    provider_message_id=msg_id,
                    error_message=err_msg
                )
                DatabaseManager.log_audit(
                    action="PROVIDER_STATUS_SYNC",
                    actor="Clinician / Poll",
                    patient_id=notif.get("patient_id", "UNKNOWN"),
                    details=f"Message {msg_id} synced: {twilio_status} -> {final_status}"
                )
                return {
                    "success": True,
                    "status": final_status,
                    "provider_status": twilio_status,
                    "error_message": err_msg
                }
            else:
                return {
                    "success": False,
                    "status": notif.get("status", "FAILED"),
                    "error": f"Twilio status check failed with HTTP {resp.status_code}."
                }
        except requests.exceptions.Timeout:
            return {"success": False, "status": notif.get("status", "FAILED"), "error": "Provider connection timed out after 5.0s."}
        except Exception as e:
            return {"success": False, "status": notif.get("status", "FAILED"), "error": f"Error polling status: {str(e)}"}

    def handle_twilio_callback(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes Twilio delivery status callback webhook.
        Twilio posts form fields: MessageSid, MessageStatus, ErrorCode, ErrorMessage, etc.
        """
        msg_sid = form_data.get("MessageSid") or form_data.get("SmsSid")
        if not msg_sid:
            return {"success": False, "error": "Missing MessageSid in callback."}

        raw_status = str(form_data.get("MessageStatus") or form_data.get("SmsStatus", "")).lower()
        error_msg = form_data.get("ErrorMessage")

        if raw_status in ["delivered"]:
            status = "DELIVERED"
        elif raw_status in ["sent"]:
            status = "SENT"
        elif raw_status in ["accepted"]:
            status = "ACCEPTED"
        elif raw_status in ["queued", "sending"]:
            status = "PENDING"
        elif raw_status in ["failed", "undelivered"]:
            status = "FAILED"
            if not error_msg and form_data.get("ErrorCode"):
                error_msg = f"Twilio ErrorCode {form_data.get('ErrorCode')}"
        else:
            status = "SENT"

        notif = DatabaseManager.get_notification_by_provider_id(msg_sid)
        if notif:
            DatabaseManager.update_notification_status(
                notification_uuid=notif["notification_uuid"],
                status=status,
                provider_message_id=msg_sid,
                error_message=error_msg
            )
            DatabaseManager.log_audit(
                action="PROVIDER_STATUS_CALLBACK",
                actor="Twilio Webhook",
                patient_id=notif.get("patient_id", "UNKNOWN"),
                details=f"Message {msg_sid} updated to {status} (Twilio status: {raw_status})"
            )
            return {"success": True, "notification_uuid": notif["notification_uuid"], "status": status}

        return {"success": False, "error": f"Notification not found for MessageSid {msg_sid}"}


# Global singleton instance
sms_service = SMSService()
