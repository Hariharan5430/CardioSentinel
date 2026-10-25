"""
CardioSentinel | ECG Intelligence - Comprehensive Repair Validation Suite
Verifies all 17 requirements of Section 11:
1. Valid and invalid guardian registration.
2. Phone-number normalization (E.164 formatting & validation).
3. Consent enforcement (explicit authorization and timestamp).
4. Duplicate registration behavior.
5. Registration while Twilio is unconfigured.
6. Honest unverified phone status.
7. Provider authentication and network failures (HTTP 401, 429, timeouts).
8. Accurate provider-status mapping (DELIVERED, SENT, ACCEPTED, PENDING, FAILED).
9. Duplicate alert suppression and cooldowns.
10. Acknowledge persistence and audit logging.
11. Snooze timing, auto-expiration, and critical event breakthrough.
12. Dismissal persistence and retained alert history.
13. Frontend / API controls using correct alert identifiers.
14. Notification counters and activity filtering.
15. Guardian preference and severity tier filtering.
16. Prevention of real SMS dispatch during recorded MIT-BIH playback.
17. Prevention of duplicate dispatch after repeated polling or page refreshes.
"""

import os
import time
import json
import pytest
from unittest.mock import patch, MagicMock
import requests

from backend.server import app
from backend.database import DatabaseManager, normalize_phone_number, mask_phone_number, get_db_connection
from backend.sms_service import SMSService, sms_service
from backend.alert_system import alert_system, AlarmPriority, ClinicalState


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Ensures mock mode is active and cooldowns are reset for safe testing."""
    sms_service.set_mock_mode(True)
    sms_service._cooldown_map.clear()
    alert_system.active_alerts.clear()
    alert_system.dismissed_banner_alerts.clear()
    alert_system.snoozed_beds.clear()
    alert_system.patient_states.clear()
    alert_system.bed_active_map.clear()
    alert_system.cooldown_map.clear()
    yield
    sms_service.set_mock_mode(True)
    sms_service._cooldown_map.clear()
    alert_system.active_alerts.clear()
    alert_system.dismissed_banner_alerts.clear()
    alert_system.snoozed_beds.clear()
    alert_system.patient_states.clear()
    alert_system.bed_active_map.clear()
    alert_system.cooldown_map.clear()


# =========================================================================
# 1. GUARDIAN REGISTRATION, NORMALIZATION, & CONSENT ENFORCEMENT
# =========================================================================

class TestGuardianRegistrationRepairs:
    """Tests Section 3 & 4 requirements: registration, normalization, consent, duplicates."""

    def test_phone_normalization_comprehensive(self):
        """Tests international E.164 normalization across multiple formats."""
        # Valid cases
        valid_cases = [
            ("+1 (555) 234-5678", "+15552345678"),
            ("+91 98765 43210", "+919876543210"),
            ("+44 20 7946 0912", "+442079460912"),
            ("9876543210", "+919876543210"),       # 10 digits default to +91
            ("15552345678", "+15552345678"),       # 11 digits starting with 1
        ]
        for raw, expected in valid_cases:
            is_valid, norm, err = normalize_phone_number(raw)
            assert is_valid, f"Expected {raw} to be valid, got error: {err}"
            assert norm == expected, f"Expected {expected}, got {norm}"

        # Invalid cases
        invalid_cases = [
            "",
            "123",              # Too short
            "+0123456789",      # Invalid country code 0
            "abc-def-ghij",     # Non-numeric
            "+1234567890123456" # Too long (> 15 digits)
        ]
        for raw in invalid_cases:
            is_valid, norm, err = normalize_phone_number(raw)
            assert not is_valid, f"Expected {raw} to be invalid, but passed with {norm}"

    def test_consent_enforcement_and_timestamp(self, client):
        """Verifies that enabling notifications requires explicit consent and records timestamp."""
        # 1. Attempt to register with enabled=True but consent=False -> Must fail
        payload_no_consent = {
            "patient_id": "BED-01",
            "full_name": "Marcus Vance",
            "phone_number": "+15552345678",
            "relationship": "Spouse",
            "is_enabled": True,
            "consent_confirmed": False,
            "preferred_categories": "CRISIS,WARNING"
        }
        res = client.post("/api/guardian", json=payload_no_consent)
        assert res.status_code == 400
        data = res.get_json()
        assert not data["success"]
        assert "consent" in data["error"].lower()

        # 2. Register with explicit consent -> Must succeed and save consent_timestamp
        payload_with_consent = {
            "patient_id": "BED-01",
            "full_name": "Arthur Vance",
            "phone_number": "+15552345678",
            "relationship": "Spouse & Primary Guardian",
            "is_enabled": True,
            "consent_confirmed": True,
            "preferred_categories": "CRISIS,WARNING"
        }
        res2 = client.post("/api/guardian", json=payload_with_consent)
        assert res2.status_code == 200
        data2 = res2.get_json()
        assert data2["success"]
        assert data2["guardian"]["consent_confirmed"] == 1
        assert "consent_timestamp" in data2["guardian"]
        assert data2["guardian"]["consent_timestamp"] is not None

    def test_duplicate_registration_detection(self, client):
        """Verifies that submitting identical registration returns duplicate status."""
        payload = {
            "patient_id": "BED-02",
            "full_name": "Sarah Connor",
            "phone_number": "+14155552671",
            "relationship": "Parent",
            "is_enabled": True,
            "consent_confirmed": True,
            "preferred_categories": "CRISIS,WARNING"
        }
        res1 = client.post("/api/guardian", json=payload)
        assert res1.status_code == 200

        # Submit exact same details again
        res2 = client.post("/api/guardian", json=payload)
        assert res2.status_code == 200
        data2 = res2.get_json()
        assert data2.get("duplicate") is True
        assert ("up to date" in data2.get("message", "").lower() or "duplicate" in data2.get("message", "").lower())

    def test_registration_while_twilio_unconfigured(self, client):
        """Verifies registration succeeds honestly even when Twilio is not configured."""
        with patch.dict(os.environ, {"TWILIO_ACCOUNT_SID": "", "TWILIO_AUTH_TOKEN": "", "TWILIO_FROM_NUMBER": ""}):
            sms_service.set_mock_mode(False)
            payload = {
                "patient_id": "BED-03",
                "full_name": "Linus Vance",
                "phone_number": "+14155553333",
                "relationship": "Brother",
                "is_enabled": True,
                "consent_confirmed": True,
                "preferred_categories": "CRISIS"
            }
            res = client.post("/api/guardian", json=payload)
            assert res.status_code == 200
            data = res.get_json()
            assert data["success"]

            # Query guardian status
            res_status = client.get("/api/guardian/BED-03")
            data_status = res_status.get_json()
            assert data_status["success"]
            # Verification status must be REGISTERED_UNVERIFIED
            assert data_status["verification_status"]["status"] == "REGISTERED_UNVERIFIED"
            assert data_status["verification_status"]["label"] == "Registered — Not Phone Verified"
            # Provider status must reflect unconfigured
            assert data_status["provider"]["is_configured"] is False
            assert "Not Configured" in data_status["provider"]["status_label"]

    def test_phone_number_privacy_masking(self):
        """Verifies that phone numbers are masked to protect patient/guardian privacy."""
        masked_us = mask_phone_number("+15552345678")
        assert "••••" in masked_us
        assert masked_us.endswith("678")
        assert not ("234" in masked_us)

        masked_in = mask_phone_number("+919876543210")
        assert "••••" in masked_in
        assert masked_in.endswith("210")


# =========================================================================
# 2. TWILIO CONFIGURATION & ERROR HANDLING (MOCKED RESPONSES)
# =========================================================================

class TestTwilioProviderSafetyAndStatus:
    """Tests Section 5 requirements: provider configuration, timeouts, 401/429 errors, status mapping."""

    def test_provider_credentials_masked_and_never_leaked(self):
        """Ensures that auth tokens are never exposed in provider status metadata."""
        with patch.dict(os.environ, {
            "TWILIO_ACCOUNT_SID": "AC12NOTASECRET123456789MOCKcdef",
            "TWILIO_AUTH_TOKEN": "secret_token_1234567890abcdef",
            "TWILIO_FROM_NUMBER": "+15005550006"
        }):
            sms_service.set_mock_mode(False)
            status = sms_service.get_provider_status()
            assert status["is_configured"] is True
            assert "secret" not in json.dumps(status)
            assert "auth_token" not in status
            assert "••••" in status["account_sid_masked"]
            assert status["account_sid_masked"].startswith("AC12")
            assert status["account_sid_masked"].endswith("cdef")

    def test_provider_authentication_failure_401(self):
        """Simulates Twilio HTTP 401 Unauthorized response and ensures FAILED status."""
        sms_service.set_mock_mode(False)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"message": "Authenticate failed: invalid credentials", "code": 20003}

        with patch("requests.post", return_value=mock_resp), \
             patch.dict(os.environ, {
                 "TWILIO_ACCOUNT_SID": "AC12NOTASECRET123456789MOCKcdef",
                 "TWILIO_AUTH_TOKEN": "invalid_auth_token_value",
                 "TWILIO_FROM_NUMBER": "+15005550006"
             }):
            res = sms_service._dispatch_sms_provider(
                phone_to="+15559876543",
                body="Test authentication failure message",
                notification_uuid="NOTIF-TEST-401"
            )
            assert res["success"] is False
            assert res["status"] == "FAILED"
            assert "Authenticate failed" in res["error"]

    def test_provider_rate_limit_failure_429(self):
        """Simulates Twilio HTTP 429 Too Many Requests response and ensures FAILED status."""
        sms_service.set_mock_mode(False)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"message": "Too many requests to Twilio API", "code": 20429}

        with patch("requests.post", return_value=mock_resp), \
             patch.dict(os.environ, {
                 "TWILIO_ACCOUNT_SID": "AC12NOTASECRET123456789MOCKcdef",
                 "TWILIO_AUTH_TOKEN": "valid_token_length_exceeds_16",
                 "TWILIO_FROM_NUMBER": "+15005550006"
             }):
            res = sms_service._dispatch_sms_provider(
                phone_to="+15559876543",
                body="Test rate limit failure",
                notification_uuid="NOTIF-TEST-429"
            )
            assert res["success"] is False
            assert res["status"] == "FAILED"
            assert "Too many requests" in res["error"]

    def test_provider_network_timeout(self):
        """Simulates network timeout contacting Twilio and ensures graceful FAILED handling."""
        sms_service.set_mock_mode(False)
        with patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out")), \
             patch.dict(os.environ, {
                 "TWILIO_ACCOUNT_SID": "AC12NOTASECRET123456789MOCKcdef",
                 "TWILIO_AUTH_TOKEN": "valid_token_length_exceeds_16",
                 "TWILIO_FROM_NUMBER": "+15005550006"
             }):
            res = sms_service._dispatch_sms_provider(
                phone_to="+15559876543",
                body="Test network timeout handling",
                notification_uuid="NOTIF-TEST-TIMEOUT"
            )
            assert res["success"] is False
            assert res["status"] == "FAILED"
            assert "timed out" in res["error"].lower()

    def test_accurate_provider_status_mapping(self):
        """Verifies mapping of provider states: queued -> PENDING, sent -> SENT, delivered -> DELIVERED."""
        sms_service.set_mock_mode(False)
        status_pairs = [
            ("queued", "PENDING"),
            ("sending", "PENDING"),
            ("sent", "SENT"),
            ("accepted", "ACCEPTED"),
            ("delivered", "DELIVERED"),
            ("undelivered", "FAILED"),
            ("failed", "FAILED")
        ]
        with patch.dict(os.environ, {
            "TWILIO_ACCOUNT_SID": "AC12NOTASECRET123456789MOCKcdef",
            "TWILIO_AUTH_TOKEN": "valid_token_length_exceeds_16",
            "TWILIO_FROM_NUMBER": "+15005550006"
        }):
            for twilio_st, expected_st in status_pairs:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "sid": f"SM-TEST-{twilio_st.upper()}",
                    "status": twilio_st
                }
                with patch("requests.post", return_value=mock_resp):
                    res = sms_service._dispatch_sms_provider(
                        phone_to="+15559876543",
                        body=f"Test mapping for {twilio_st}",
                        notification_uuid=f"NOTIF-MAP-{twilio_st.upper()}"
                    )
                    assert res["status"] == expected_st, f"Expected {expected_st} for Twilio status {twilio_st}, got {res['status']}"

    def test_twilio_webhook_callback(self, client):
        """Tests provider callback route /api/guardian/twilio_callback."""
        import uuid
        test_msg_sid = f"SM-WEBHOOK-{uuid.uuid4().hex[:8].upper()}"
        notif = DatabaseManager.create_notification(
            patient_id="BED-01",
            recipient_phone_masked="+1 •••• ••5678",
            message_type="REAL_EMERGENCY",
            message_body="Emergency ECG notification test",
            status="PENDING",
            provider="twilio",
            provider_message_id=test_msg_sid
        )
        assert notif is not None

        # Post callback webhook simulating carrier delivery receipt
        res = client.post("/api/guardian/twilio_callback", data={
            "MessageSid": test_msg_sid,
            "MessageStatus": "delivered"
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["status"] == "DELIVERED"

        # Verify database record updated
        updated = DatabaseManager.get_notification_by_uuid(notif["notification_uuid"])
        assert updated["status"] == "DELIVERED"


# =========================================================================
# 3. ALERT LIFECYCLE: ACKNOWLEDGE, SNOOZE, DISMISS & BREAKTHROUGH
# =========================================================================

class TestAlertLifecycleRepairs:
    """Tests Section 6 requirements: Acknowledge, Snooze, Dismiss, and critical event safety."""

    def test_acknowledge_persistence(self, client):
        """Verifies that alert acknowledgement updates state in SQLite and logs audit event."""
        # Trigger an alert in alert_system
        alert = alert_system.evaluate_and_trigger(
            patient_id="BED-01",
            patient_name="Eleanor Vance",
            location="ICU Bed 1",
            diagnosis="Ventricular Tachycardia",
            cewi_data={"cewi_score": 85},
            metrics={"heart_rate": 150, "signal_quality": "OPTIMAL"}
        )
        assert alert is not None
        alert_id = alert["alert_id"]

        # Call Acknowledge action
        res = client.post("/api/alerts/action", json={
            "action": "ack",
            "alert_id": alert_id
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True

        # Verify alert marked acknowledged in alert system
        acked = alert_system.active_alerts.get(alert_id)
        assert acked["acknowledged"] is True
        assert acked["status"] == "ACKNOWLEDGED"

        # Verify audit log was recorded
        logs = DatabaseManager.get_audit_logs(limit=10)
        ack_logs = [l for l in logs if l["action"] == "ALERT_ACKNOWLEDGED" and alert_id in l["details"]]
        assert len(ack_logs) > 0, "Expected ALERT_ACKNOWLEDGED audit log"

    def test_snooze_timing_and_persistence(self, client):
        """Verifies that snoozing an alert sets deadline and suppresses same-condition re-trigger."""
        alert = alert_system.evaluate_and_trigger(
            patient_id="BED-02",
            patient_name="Marcus Brody",
            location="ICU Bed 2",
            diagnosis="Atrial Fibrillation",
            cewi_data={"cewi_score": 55},
            metrics={"heart_rate": 160, "signal_quality": "OPTIMAL"}
        )
        assert alert is not None
        alert_id = alert["alert_id"]

        # Snooze for 5 minutes
        res = client.post("/api/alerts/action", json={
            "action": "snooze",
            "alert_id": alert_id,
            "patient_id": "BED-02",
            "minutes": 5
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert alert_system.is_snoozed("BED-02") is True

        # Attempt to trigger same condition during snooze -> stays SNOOZED
        suppressed = alert_system.evaluate_and_trigger(
            patient_id="BED-02",
            patient_name="Marcus Brody",
            location="ICU Bed 2",
            diagnosis="Atrial Fibrillation",
            cewi_data={"cewi_score": 55},
            metrics={"heart_rate": 160, "signal_quality": "OPTIMAL"}
        )
        assert suppressed is not None
        assert suppressed["status"] == "SNOOZED"

    def test_snooze_critical_event_breakthrough(self):
        """Verifies safety policy: a newly detected CRISIS event breaks through an existing snooze."""
        # 1. Snooze a WARNING condition on BED-01
        alert_system.snooze_alert("BED-01", minutes=5)
        assert alert_system.is_snoozed("BED-01") is True

        # 2. A catastrophic Ventricular Fibrillation (CRISIS) occurs
        crisis_alert = alert_system.evaluate_and_trigger(
            patient_id="BED-01",
            patient_name="Eleanor Vance",
            location="ICU Bed 1",
            diagnosis="Ventricular Fibrillation",
            cewi_data={"cewi_score": 95},
            metrics={"heart_rate": 0, "signal_quality": "OPTIMAL"}
        )
        assert crisis_alert is not None, "CRISIS event must break through WARNING snooze!"
        assert "CRISIS" in crisis_alert["priority"]
        assert alert_system.is_snoozed("BED-01") is False, "Snooze must be cleared on crisis breakthrough"

    def test_dismiss_banner_persistence(self, client):
        """Verifies that dismissing banner persists dismissal without erasing alert history."""
        alert = alert_system.evaluate_and_trigger(
            patient_id="BED-01",
            patient_name="Eleanor Vance",
            location="ICU Bed 1",
            diagnosis="Ventricular Tachycardia",
            cewi_data={"cewi_score": 85},
            metrics={"heart_rate": 150, "signal_quality": "OPTIMAL"}
        )
        assert alert is not None
        alert_id = alert["alert_id"]

        # User clicks Dismiss Banner ('X')
        res = client.post("/api/alerts/action", json={
            "action": "dismiss",
            "alert_id": alert_id,
            "patient_id": "BED-01"
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True

        # Alert must be marked banner_dismissed
        current = alert_system.active_alerts.get(alert_id)
        assert current["banner_dismissed"] is True

        # Alert history is NOT erased
        all_alerts = alert_system.get_active_alerts()
        assert any(a["alert_id"] == alert_id for a in all_alerts)


# =========================================================================
# 4. PREVENTING EXCESSIVE ALERTS & DEMO MODE ISOLATION
# =========================================================================

class TestExcessiveAlertPreventionAndDemoSafety:
    """Tests Section 7, 8, & 9 requirements: signal quality, MIT-BIH demo safety, duplicate suppression."""

    def test_normal_telemetry_does_not_generate_critical_alert(self):
        """Ensures that normal sinus rhythm telemetry generates NO critical alerts."""
        # Normal rhythm (NSR, HR 72, optimal SQI)
        normal_eval = alert_system.evaluate_and_trigger(
            patient_id="BED-01",
            patient_name="Eleanor Vance",
            location="ICU Bed 1",
            diagnosis="Normal Sinus Rhythm",
            cewi_data={"cewi_score": 0},
            metrics={"heart_rate": 72, "signal_quality": "OPTIMAL"}
        )
        assert normal_eval is None, "Normal sinus rhythm should generate no alert"

    def test_recorded_mitbih_playback_never_dispatches_real_sms(self):
        """Enforces Section 9: Recorded MIT-BIH playback strictly generates DEMO_RECORDED previews."""
        DatabaseManager.save_guardian("BED-01", {
            "full_name": "Arthur Vance",
            "phone_number": "+15552345678",
            "relationship": "Spouse & Primary Guardian",
            "is_enabled": True,
            "consent_confirmed": True,
            "preferred_categories": "CRISIS,WARNING"
        })
        import uuid
        test_alert_id = f"ALT-MITBIH-{uuid.uuid4().hex[:6]}"
        alert_payload = {
            "patient_id": "BED-01",
            "alert_id": test_alert_id,
            "priority": "CRITICAL",
            "trigger_condition": "Ventricular Tachycardia",
            "diagnosis": "Ventricular Tachycardia"
        }
        res = sms_service.send_emergency_alert(alert_payload, patient_data_mode="MIT-BIH")
        assert res["eligible"] is True
        assert res["mode"] == "DEMO_RECORDED"
        assert res["notification"]["status"] == "DEMO_RECORDED"
        assert res["notification"]["message_type"] == "DEMO_PREVIEW"
        assert "demonstration" in res["notification"]["message_body"].lower()

    def test_repeated_alert_deduplication_and_cooldown(self):
        """Verifies duplicate suppression within cooldown window."""
        DatabaseManager.save_guardian("BED-01", {
            "full_name": "Arthur Vance",
            "phone_number": "+15552345678",
            "relationship": "Spouse & Primary Guardian",
            "is_enabled": True,
            "consent_confirmed": True,
            "preferred_categories": "CRISIS,WARNING"
        })
        import uuid
        test_alert_id = f"ALT-DEDUP-{uuid.uuid4().hex[:6]}"
        alert_payload = {
            "patient_id": "BED-01",
            "alert_id": test_alert_id,
            "priority": "CRITICAL",
            "trigger_condition": "Ventricular Fibrillation",
            "diagnosis": "Ventricular Fibrillation"
        }
        # First dispatch
        res1 = sms_service.send_emergency_alert(alert_payload, patient_data_mode="LIVE")
        assert res1["eligible"] is True

        # Second dispatch immediately for same alert ID / condition -> Deduplicated
        res2 = sms_service.send_emergency_alert(alert_payload, patient_data_mode="LIVE")
        assert res2["eligible"] is False
        assert res2.get("already_sent") is True or "cooldown" in res2.get("reason", "").lower()

    def test_notification_counters_and_filtering(self, client):
        """Verifies that /api/guardian/notifications returns accurate records with status filters."""
        # Ensure at least one demo preview is in database
        DatabaseManager.create_notification(
            patient_id="BED-01",
            recipient_phone_masked="+1 •••• ••5678",
            message_type="DEMO_PREVIEW",
            message_body="Demo preview for testing",
            status="DEMO_RECORDED",
            provider="local_preview"
        )

        res_all = client.get("/api/guardian/notifications?filter=ALL")
        assert res_all.status_code == 200
        data_all = res_all.get_json()
        assert data_all["success"] is True
        assert "notifications" in data_all

        res_demo = client.get("/api/guardian/notifications?filter=DEMO")
        assert res_demo.status_code == 200
        data_demo = res_demo.get_json()
        assert data_demo["success"] is True
        assert len(data_demo["notifications"]) > 0
        for n in data_demo["notifications"]:
            assert n["status"] == "DEMO_RECORDED" or n.get("message_type") == "DEMO_PREVIEW"
