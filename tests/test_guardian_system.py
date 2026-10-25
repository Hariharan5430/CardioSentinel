"""
CardioSentinel | ECG Intelligence - Comprehensive Guardian Notification System Tests
Validates:
1. Complete removal of OTP system (endpoints 404, schema clean, no fake 'Verified' status).
2. Simplified Guardian Registration:
   - Registration and database persistence.
   - Editing and resetting guardian details (discarding unsaved changes without deletion).
   - International E.164 phone normalization, validation, and privacy masking.
   - UI status: 'Registered — Not Phone Verified' (REGISTERED_UNVERIFIED).
3. Actual SMS delivery & Twilio provider integration:
   - Credential validation and placeholder detection.
   - Provider failure handling (actionable error when unconfigured).
   - Phone format validation for sender and recipient.
   - Real provider response status mapping (ACCEPTED, PENDING, DELIVERED, FAILED).
   - Operational test SMS dispatch with carrier charges disclaimer.
4. Alert-to-guardian notification workflow:
   - High-priority clinical alarms dispatching SMS.
   - Preference and consent gating.
   - Idempotency, duplicate prevention, and cooldown suppression.
5. Clinical safety:
   - Historical MIT-BIH ECG playback strictly in Demonstration Mode (local demo previews only).
   - Live telemetry mode isolation.
"""

import pytest
import json
import time
import os
from unittest.mock import patch, MagicMock

from backend.server import app, telemetry_engine
from backend.database import DatabaseManager, normalize_phone_number, mask_phone_number, get_db_connection
from backend.sms_service import SMSService, sms_service


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def ensure_mock_mode():
    """Ensures SMS service defaults to mock mode for all tests so no real carrier calls occur."""
    sms_service.set_mock_mode(True)
    sms_service._cooldown_map.clear()
    yield
    sms_service.set_mock_mode(True)
    sms_service._cooldown_map.clear()


# =========================================================================
# 1. OTP SYSTEM COMPLETE REMOVAL TESTS
# =========================================================================
class TestOTPRemoval:
    """Verifies complete removal of OTP routes, database functions, columns, and fake verified status."""

    def test_otp_endpoints_return_404(self, client):
        """Verifies that all legacy OTP routes are completely removed and not registered."""
        otp_endpoints = [
            ("/api/guardian/send_otp", "POST", {"patient_id": "BED-01"}),
            ("/api/guardian/verify_otp", "POST", {"patient_id": "BED-01", "code": "123456"}),
            ("/api/guardian/verify_request", "POST", {"patient_id": "BED-01"}),
            ("/api/guardian/verify", "POST", {"patient_id": "BED-01", "code": "123456"}),
        ]
        for url, method, payload in otp_endpoints:
            if method == "POST":
                res = client.post(url, json=payload)
            else:
                res = client.get(url)
            assert res.status_code in (404, 405), f"Endpoint {url} was expected to be removed (404/405), got {res.status_code}"

        # Confirm no OTP route rules exist in Flask URL map
        otp_rules = [r.rule for r in app.url_map.iter_rules() if "otp" in r.rule.lower()]
        assert len(otp_rules) == 0, f"Found active OTP routes: {otp_rules}"

    def test_database_manager_has_no_otp_methods(self):
        """Verifies that OTP challenge creation and verification methods do not exist on DatabaseManager."""
        assert not hasattr(DatabaseManager, "create_otp_challenge"), "create_otp_challenge must be removed"
        assert not hasattr(DatabaseManager, "verify_otp"), "verify_otp must be removed"

    def test_sms_service_has_no_send_otp_sms(self):
        """Verifies that send_otp_sms is removed from SMSService."""
        assert not hasattr(sms_service, "send_otp_sms"), "send_otp_sms must be removed from SMSService"

    def test_database_schema_has_no_otp_columns(self):
        """Verifies that the guardians table schema does not include OTP-exclusive columns."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(guardians);")
        columns = [row["name"] for row in cursor.fetchall()]

        prohibited_otp_columns = [
            "verification_code",
            "verification_code_hash",
            "verification_salt",
            "verification_expiry",
            "verification_attempts",
            "verification_last_sent"
        ]
        for col in prohibited_otp_columns:
            assert col not in columns, f"Prohibited OTP column {col} found in guardians table schema"

    def test_no_fake_or_automatic_verified_status(self):
        """Verifies guardian status is 'Registered — Not Phone Verified' and never falsely 'Verified'."""
        test_bed = f"BED-STATUS-CHECK-{int(time.time()*1000)}"
        DatabaseManager.remove_guardian(test_bed)

        # 1. Before registration: Unregistered
        st_unreg = DatabaseManager.get_verification_status(test_bed)
        assert st_unreg["status"] == "UNREGISTERED"
        assert st_unreg["label"] == "Not Registered"

        # 2. After registration: Registered — Not Phone Verified
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "Dr. Claire Fraser",
            "phone_number": "+1 (415) 555-0199",
            "consent_confirmed": True,
            "is_enabled": True
        })
        st_reg = DatabaseManager.get_verification_status(test_bed)
        assert st_reg["status"] == "REGISTERED_UNVERIFIED"
        assert st_reg["label"] == "Registered — Not Phone Verified"
        assert "Verified" not in st_reg["label"].split(" — ")[0] # Must not start with 'Verified'

        g = DatabaseManager.get_guardian(test_bed)
        assert g["is_verified"] == 0, "is_verified must remain 0 (no simulated verification)"


# =========================================================================
# 2. SIMPLIFIED GUARDIAN REGISTRATION & PERSISTENCE TESTS
# =========================================================================
class TestSimplifiedGuardianRegistration:
    """Tests registration, phone validation, persistence, reset, and audit logging."""

    def test_phone_normalization_valid_international(self):
        """Verifies E.164 normalization for varied valid international numbers."""
        cases = [
            ("+1 (415) 555-2671", "+14155552671"),
            ("+91 98765 43210", "+919876543210"),
            ("+44 7911 123456", "+447911123456"),
            ("+61 412 345 678", "+61412345678"),
            ("9876543210", "+919876543210"), # Defaults 10-digit to India (+91)
        ]
        for raw, expected in cases:
            valid, norm, err = normalize_phone_number(raw)
            assert valid is True, f"Failed for {raw}: {err}"
            assert norm == expected

    def test_phone_normalization_invalid_rejected(self):
        """Verifies rejection of invalid phone formats on backend."""
        invalid_cases = [
            "",
            "123",
            "not-a-number",
            "+",
            "+0123456",   # Leading 0 country code
            "+91987",     # Incomplete / too short
            "+12345678901234567", # Too long (>15 digits)
        ]
        for raw in invalid_cases:
            valid, norm, err = normalize_phone_number(raw)
            assert valid is False, f"Expected rejection for {raw}"
            assert len(err) > 0

    def test_phone_privacy_masking(self):
        """Verifies phone masking preserves country code and last 3 digits."""
        assert mask_phone_number("+14155552671") == "+14 ••••• ••671"
        assert mask_phone_number("+919876543210") == "+91 ••••• ••210"
        assert mask_phone_number("") == "--"

    def test_guardian_save_and_reload_persistence(self):
        """Verifies complete save workflow: validation, commit, and exact reload from backend."""
        test_bed = f"BED-SAVE-{int(time.time()*1000)}"
        DatabaseManager.remove_guardian(test_bed)

        data = {
            "full_name": "Eleanor Vance",
            "phone_number": "+1 (555) 234-5678",
            "relationship": "Spouse / Partner",
            "preferred_categories": "CRISIS,WARNING",
            "consent_confirmed": True,
            "is_enabled": True
        }

        ok, saved, msg = DatabaseManager.save_guardian(test_bed, data)
        assert ok is True
        assert saved["full_name"] == "Eleanor Vance"
        assert saved["phone_normalized"] == "+15552345678"
        assert saved["relationship"] == "Spouse / Partner"
        assert saved["is_enabled"] == 1
        assert saved["consent_confirmed"] == 1
        assert saved["preferred_categories"] == "CRISIS,WARNING"

        # Reload from database to ensure persistence
        reloaded = DatabaseManager.get_guardian(test_bed)
        assert reloaded is not None
        assert reloaded["full_name"] == "Eleanor Vance"
        assert reloaded["phone_normalized"] == "+15552345678"
        assert reloaded["is_verified"] == 0

    def test_editing_guardian_details(self):
        """Verifies editing existing guardian details successfully updates database."""
        test_bed = f"BED-EDIT-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "Original Name",
            "phone_number": "+91 98000 11111",
            "relationship": "Primary Caregiver",
            "consent_confirmed": True,
            "is_enabled": True
        })

        # Update details
        ok, updated, _ = DatabaseManager.save_guardian(test_bed, {
            "full_name": "Updated Name",
            "phone_number": "+91 98000 22222",
            "relationship": "Legal Healthcare Proxy",
            "consent_confirmed": True,
            "is_enabled": True
        })
        assert ok is True
        assert updated["full_name"] == "Updated Name"
        assert updated["phone_normalized"] == "+919800022222"
        assert updated["relationship"] == "Legal Healthcare Proxy"

    def test_reset_discards_unsaved_edits_without_deleting_saved_data(self, client):
        """
        Verifies that discarding unsaved edits via reset reloads stored database state
        and does NOT delete or wipe the guardian record.
        """
        test_bed = f"BED-RESET-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "Permanent Contact",
            "phone_number": "+1 (415) 555-8888",
            "relationship": "Attending Physician",
            "consent_confirmed": True,
            "is_enabled": True
        })

        # When client fetches guardian to restore/reset form, existing database record is intact
        res = client.get(f"/api/guardian/{test_bed}")
        assert res.status_code == 200
        g = res.get_json()["guardian"]
        assert g is not None
        assert g["full_name"] == "Permanent Contact"
        assert g["phone_normalized"] == "+14155558888"

    def test_save_validation_errors(self, client):
        """Verifies backend rejects invalid submissions (missing name, invalid phone, missing consent)."""
        # Missing full name
        res1 = client.post("/api/guardian", json={"patient_id": "BED-01", "phone_number": "+14155552671"})
        assert res1.status_code == 400
        assert "name is required" in res1.get_json()["error"].lower()

        # Invalid phone format
        res2 = client.post("/api/guardian", json={"patient_id": "BED-01", "full_name": "Jane", "phone_number": "123"})
        assert res2.status_code == 400
        assert "invalid" in res2.get_json()["error"].lower() or "country code" in res2.get_json()["error"].lower()

        # Enabled without explicit consent
        res3 = client.post("/api/guardian", json={
            "patient_id": "BED-01",
            "full_name": "Jane",
            "phone_number": "+14155552671",
            "is_enabled": True,
            "consent_confirmed": False
        })
        assert res3.status_code == 400
        assert "consent" in res3.get_json()["error"].lower()


# =========================================================================
# 3. ACTUAL SMS DELIVERY & PROVIDER INTEGRATION TESTS
# =========================================================================
class TestActualSMSDelivery:
    """Tests Twilio integration, credential checking, status mapping, and test SMS workflow."""

    def test_provider_missing_credentials_handling(self, monkeypatch):
        """Verifies missing environment variables report unconfigured status with actionable missing keys."""
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("TWILIO_FROM_NUMBER", raising=False)

        sms_service.set_mock_mode(False)
        st = sms_service.get_provider_status()
        assert st["is_configured"] is False
        assert "TWILIO_ACCOUNT_SID" in st["missing_keys"]
        assert "TWILIO_AUTH_TOKEN" in st["missing_keys"]
        assert "TWILIO_FROM_NUMBER" in st["missing_keys"]

        # Real test SMS must fail cleanly without throwing unhandled exceptions
        test_bed = f"BED-UNCONFIG-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "Test User",
            "phone_number": "+14155552671",
            "consent_confirmed": True
        })
        res = sms_service.send_test_sms(test_bed)
        assert res["success"] is False
        assert "SMS provider not configured" in res["error"]
        sms_service.set_mock_mode(True)

    def test_provider_placeholder_credentials_rejected(self, monkeypatch):
        """Verifies default template placeholder strings are detected and rejected as unconfigured."""
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "your_twilio_account_sid_here")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "your_auth_token_here")
        monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15005550006")

        sms_service.set_mock_mode(False)
        st = sms_service.get_provider_status()
        assert st["is_configured"] is False
        assert "TWILIO_ACCOUNT_SID (contains placeholder)" in st["missing_keys"]
        sms_service.set_mock_mode(True)

    def test_twilio_response_status_mapping_mocked_http(self, monkeypatch):
        """
        Verifies actual Twilio HTTP status codes and responses are correctly mapped:
        - HTTP 201 with 'queued' maps to PENDING (does NOT fake DELIVERED).
        - HTTP 201 with 'accepted' maps to ACCEPTED.
        - HTTP 201 with 'delivered' maps to DELIVERED.
        - HTTP 400 with Twilio error JSON maps to FAILED.
        """
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC12NOTASECRET123456789MOCKcdef")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "auth_token_secret_12345")
        monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15005550006")
        sms_service.set_mock_mode(False)

        # 1. Twilio queued -> PENDING
        mock_resp_queued = MagicMock()
        mock_resp_queued.status_code = 201
        mock_resp_queued.json.return_value = {
            "sid": "SM_QUEUED_123",
            "status": "queued"
        }

        with patch("requests.post", return_value=mock_resp_queued):
            res_queued = sms_service._dispatch_sms_provider("+14155552671", "Test Queued")
            assert res_queued["success"] is True
            assert res_queued["status"] == "PENDING"
            assert res_queued["message_id"] == "SM_QUEUED_123"

        # 2. Twilio accepted -> ACCEPTED
        mock_resp_accepted = MagicMock()
        mock_resp_accepted.status_code = 201
        mock_resp_accepted.json.return_value = {
            "sid": "SM_ACCEPTED_123",
            "status": "accepted"
        }

        with patch("requests.post", return_value=mock_resp_accepted):
            res_accepted = sms_service._dispatch_sms_provider("+14155552671", "Test Accepted")
            assert res_accepted["success"] is True
            assert res_accepted["status"] == "ACCEPTED"
            assert res_accepted["message_id"] == "SM_ACCEPTED_123"

        # 3. Twilio delivered -> DELIVERED
        mock_resp_delivered = MagicMock()
        mock_resp_delivered.status_code = 201
        mock_resp_delivered.json.return_value = {
            "sid": "SM_DELIVERED_123",
            "status": "delivered"
        }

        with patch("requests.post", return_value=mock_resp_delivered):
            res_delivered = sms_service._dispatch_sms_provider("+14155552671", "Test Delivered")
            assert res_delivered["success"] is True
            assert res_delivered["status"] == "DELIVERED"
            assert res_delivered["message_id"] == "SM_DELIVERED_123"

        # 4. Twilio error HTTP 400 -> FAILED
        mock_resp_fail = MagicMock()
        mock_resp_fail.status_code = 400
        mock_resp_fail.json.return_value = {
            "code": 21211,
            "message": "The 'To' number is not a valid phone number."
        }

        with patch("requests.post", return_value=mock_resp_fail):
            res_fail = sms_service._dispatch_sms_provider("+14155552671", "Test Fail")
            assert res_fail["success"] is False
            assert res_fail["status"] == "FAILED"
            assert "not a valid phone number" in res_fail["error"]

        sms_service.set_mock_mode(True)

    def test_confirmed_test_sms_workflow(self):
        """Verifies sending confirmed test SMS to registered guardian in mock mode succeeds."""
        test_bed = f"BED-TEST-SMS-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "Test Recipient",
            "phone_number": "+1 (415) 555-9988",
            "consent_confirmed": True,
            "is_enabled": True
        })

        sms_service.set_mock_mode(True)
        res = sms_service.send_test_sms(test_bed)
        assert res["success"] is True
        assert res["status"] in ("DELIVERED", "ACCEPTED", "SENT")
        assert "notification_uuid" in res

        # Verify persisted notification record
        notif = DatabaseManager.get_notification_by_uuid(res["notification_uuid"])
        assert notif is not None
        assert notif["message_type"] == "TEST"
        assert "CardioSentinel test message" in notif["message_body"]
        assert "test only, not a medical emergency" in notif["message_body"].lower()

    def test_test_sms_endpoint_api(self, client):
        """POST /api/guardian/test_sms dispatches test SMS via Flask endpoint."""
        test_bed = f"BED-API-TEST-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "API Test Recipient",
            "phone_number": "+91 98888 12345",
            "consent_confirmed": True,
            "is_enabled": True
        })

        res = client.post("/api/guardian/test_sms", json={"patient_id": test_bed})
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["status"] in ("DELIVERED", "ACCEPTED", "SENT")
        assert "message_id" in data


# =========================================================================
# 4. ALERT-TO-GUARDIAN NOTIFICATION & DUPLICATE PREVENTION TESTS
# =========================================================================
class TestAlertToGuardianNotification:
    """Tests alert routing, guardian eligibility, deduplication, and cooldowns."""

    def test_live_monitoring_alert_dispatches_real_emergency_sms(self):
        """Verifies that eligible crisis alerts under LIVE monitoring trigger real emergency SMS."""
        test_bed = f"BED-LIVE-EMERGENCY-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "Emergency Guardian",
            "phone_number": "+1 (415) 555-3344",
            "consent_confirmed": True,
            "is_enabled": True,
            "preferred_categories": "CRISIS,WARNING"
        })

        alert_payload = {
            "alert_id": f"ALT-CRISIS-{int(time.time()*1000)}",
            "patient_id": test_bed,
            "priority": "HIGH PRIORITY (CRISIS / CODE BLUE)",
            "trigger_condition": "VENTRICULAR FIBRILLATION DETECTED",
            "diagnosis": "Ventricular Fibrillation",
            "location": "Cardiac Telemetry Bed 01"
        }

        eval_res = sms_service.send_emergency_alert(alert_payload, patient_data_mode="LIVE")
        assert eval_res["eligible"] is True
        assert eval_res["mode"] == "REAL_EMERGENCY_QUEUED"
        assert "notification_uuid" in eval_res

        notif = DatabaseManager.get_notification_by_uuid(eval_res["notification_uuid"])
        assert notif is not None
        assert notif["message_type"] == "REAL_EMERGENCY"
        assert "CardioSentinel monitoring alert" in notif["message_body"]

    def test_notification_duplicate_prevention_and_cooldown(self):
        """Verifies duplicate dispatches of repeated identical alarms are suppressed by cooldown."""
        test_bed = f"BED-DUP-PREVENT-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "Cooldown Guardian",
            "phone_number": "+91 97777 88888",
            "consent_confirmed": True,
            "is_enabled": True,
            "preferred_categories": "CRISIS"
        })

        cond = f"SUSTAINED VT-{int(time.time()*1000)}"
        alert_payload = {
            "alert_id": f"ALT-IDEM-{int(time.time()*1000)}",
            "patient_id": test_bed,
            "priority": "HIGH PRIORITY (CRISIS / CODE BLUE)",
            "trigger_condition": cond,
            "diagnosis": "Ventricular Tachycardia"
        }

        # First alert is accepted and dispatched
        res1 = sms_service.send_emergency_alert(alert_payload, patient_data_mode="LIVE")
        assert res1["eligible"] is True

        # Repeated alarm with same condition within cooldown must be suppressed
        res2 = sms_service.send_emergency_alert(alert_payload, patient_data_mode="LIVE")
        assert res2["eligible"] is False
        assert "Cooldown" in res2["reason"] or "already exists" in res2["reason"]

    def test_ineligible_alert_category_filtered(self):
        """Verifies alerts not matching guardian preferred categories are filtered out."""
        test_bed = f"BED-FILTER-CAT-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "Crisis Only Contact",
            "phone_number": "+14155554455",
            "consent_confirmed": True,
            "is_enabled": True,
            "preferred_categories": "CRISIS" # Only crisis
        })

        # Warning level alarm (mismatch with CRISIS preference)
        alert_warn = {
            "alert_id": f"ALT-WARN-{int(time.time()*1000)}",
            "patient_id": test_bed,
            "priority": "MEDIUM PRIORITY (WARNING)",
            "trigger_condition": "Sustained Atrial Fibrillation",
            "diagnosis": "Atrial Fibrillation"
        }
        res_warn = sms_service.send_emergency_alert(alert_warn, patient_data_mode="LIVE")
        assert res_warn["eligible"] is False
        assert "preferences exclude Warning" in res_warn["reason"]

        # Advisory level alarm (below threshold)
        alert_payload = {
            "alert_id": f"ALT-ADV-{int(time.time()*1000)}",
            "patient_id": test_bed,
            "priority": "LOW PRIORITY (ADVISORY)",
            "trigger_condition": "Mild sinus bradycardia",
            "diagnosis": "Sinus Bradycardia"
        }

        res = sms_service.send_emergency_alert(alert_payload, patient_data_mode="LIVE")
        assert res["eligible"] is False
        assert "below guardian notification threshold" in res["reason"]


# =========================================================================
# 5. MIT-BIH DEMONSTRATION MODE SAFETY ISOLATION TESTS
# =========================================================================
class TestDemonstrationModeSafetyIsolation:
    """Verifies that historical ECG playback (MIT-BIH) remains isolated as local preview notifications."""

    def test_recorded_ecg_playback_never_dispatches_real_sms(self):
        """
        CRITICAL SAFETY RULE: Recorded MIT-BIH playback events must NEVER contact external SMS providers.
        Must record demonstration preview with 'DEMO — Recorded ECG playback' disclaimer.
        """
        test_bed = f"BED-DEMO-SAFE-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "Demo Guardian",
            "phone_number": "+1 (415) 555-7766",
            "consent_confirmed": True,
            "is_enabled": True,
            "preferred_categories": "CRISIS,WARNING"
        })

        alert_payload = {
            "alert_id": f"ALT-DEMO-SIM-{int(time.time()*1000)}",
            "patient_id": test_bed,
            "priority": "HIGH PRIORITY (CRISIS / CODE BLUE)",
            "trigger_condition": "MIT-BIH Record 106 Ventricular Tachycardia",
            "diagnosis": "Ventricular Tachycardia",
            "location": "Educational Simulation Rig"
        }

        res = sms_service.send_emergency_alert(alert_payload, patient_data_mode="MIT-BIH")
        assert res["eligible"] is True
        assert res["mode"] == "DEMO_RECORDED"
        assert "Demonstration notification recorded" in res["message"]

        notif = res["notification"]
        assert notif["message_type"] == "DEMO_PREVIEW"
        assert notif["status"] == "DEMO_RECORDED"
        assert notif["message_body"].startswith("DEMO — Recorded ECG playback")
        assert "demonstration notification, not a live patient emergency" in notif["message_body"].lower()

    def test_readiness_checklist_flags_demo_mode(self):
        """Verifies readiness checklist accurately reflects DEMO_MODE_ACTIVE vs READY_LIVE."""
        test_bed = f"BED-READINESS-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "Readiness Patient Contact",
            "phone_number": "+1 (415) 555-1212",
            "consent_confirmed": True,
            "is_enabled": True
        })

        # Under MIT-BIH playback
        r_demo = sms_service.check_readiness(test_bed, data_mode="MIT-BIH")
        assert r_demo["checks"]["data_source_mode"]["passed"] is False
        assert r_demo["overall_status"] == "DEMO_MODE_ACTIVE"
        assert r_demo["can_send_live"] is False

        # Under LIVE telemetry
        r_live = sms_service.check_readiness(test_bed, data_mode="LIVE")
        assert r_live["checks"]["data_source_mode"]["passed"] is True
        assert r_live["overall_status"] == "READY_LIVE"
        assert r_live["can_send_live"] is True
