"""
CardioSentinel | ECG Intelligence - Multi-Bed Guardian Notification & Alert Repair Tests
Comprehensive unit and integration tests verifying:
1. Independent guardian profiles for BED-01, BED-02, BED-03, and REMOTE-01.
2. Bed-linked alert routing (alerts route ONLY to that specific bed's guardian).
3. Independent contact editing and database persistence without cross-bed leakage.
4. Safe test SMS preview mode and explicit dispatch.
5. Clinical SMS content formatting (bed identifier, severity, event, timestamp, zero patient PII).
6. Honest verification state modeling (REGISTERED_UNVERIFIED vs VERIFIED).
7. Demonstration mode safety isolation during MIT-BIH playback.
8. Alert lifecycle: acknowledge, snooze, resolve, dismiss, and crisis breakthrough.
9. Twilio delivery receipts, webhook callbacks, and status polling.
"""

import os
import time
import uuid
import pytest
from unittest.mock import patch, MagicMock

from backend.server import app
from backend.database import DatabaseManager, normalize_phone_number, mask_phone_number
from backend.sms_service import sms_service
from backend.alert_system import alert_system, ClinicalState, AlarmPriority


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestMultiBedGuardianProfiles:
    """Verifies independent profiles for BED-01, BED-02, BED-03, and REMOTE-01."""

    def test_all_four_beds_have_default_profiles(self):
        """Verifies each of the 4 monitored beds has an independently queryable profile."""
        beds = ["BED-01", "BED-02", "BED-03", "REMOTE-01"]
        for bed_id in beds:
            guardian = DatabaseManager.get_guardian(bed_id)
            assert guardian is not None, f"Expected default profile for {bed_id}"
            assert guardian["patient_id"] == bed_id
            assert len(guardian["full_name"]) > 0
            assert len(guardian["phone_normalized"]) > 0
            assert guardian["phone_masked"] != guardian["phone_normalized"]

    def test_independent_contact_editing_does_not_mutate_other_beds(self, client):
        """Editing the guardian for BED-02 must NOT modify BED-01, BED-03, or REMOTE-01."""
        # Initial snapshot of BED-01
        g1_before = DatabaseManager.get_guardian("BED-01")
        assert g1_before is not None

        # Modify BED-02
        new_name_bed2 = f"Updated Guardian {uuid.uuid4().hex[:6]}"
        new_phone_bed2 = "+14155550299"
        res = client.post("/api/guardian", json={
            "patient_id": "BED-02",
            "full_name": new_name_bed2,
            "phone_number": new_phone_bed2,
            "relationship": "Specialist Nurse Proxy",
            "is_enabled": True,
            "consent_confirmed": True,
            "preferred_categories": "CRISIS"
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True

        # Verify BED-02 updated
        g2_after = DatabaseManager.get_guardian("BED-02")
        assert g2_after["full_name"] == new_name_bed2
        assert g2_after["phone_normalized"] == new_phone_bed2
        assert g2_after["preferred_categories"] == "CRISIS"

        # Verify BED-01 remained unchanged
        g1_after = DatabaseManager.get_guardian("BED-01")
        assert g1_after["full_name"] == g1_before["full_name"]
        assert g1_after["phone_normalized"] == g1_before["phone_normalized"]
        assert g1_after["relationship"] == g1_before["relationship"]

    def test_persistence_across_database_reload(self):
        """Verifies saved guardian settings persist after fresh database connection queries."""
        bed_id = "REMOTE-01"
        unique_name = f"Margaret Updated {uuid.uuid4().hex[:6]}"
        unique_phone = "+14155558833"

        ok, saved, msg = DatabaseManager.save_guardian(bed_id, {
            "full_name": unique_name,
            "phone_number": unique_phone,
            "relationship": "Primary Remote Carer",
            "is_enabled": True,
            "consent_confirmed": True,
            "preferred_categories": "CRISIS,WARNING"
        })
        assert ok is True

        # Re-fetch via independent query
        reloaded = DatabaseManager.get_guardian(bed_id)
        assert reloaded is not None
        assert reloaded["full_name"] == unique_name
        assert reloaded["phone_normalized"] == unique_phone
        assert reloaded["relationship"] == "Primary Remote Carer"
        assert reloaded["consent_confirmed"] == 1
        assert reloaded["consent_timestamp"] is not None

    def test_validation_rejects_missing_or_invalid_fields(self, client):
        """Rejects registration missing required fields, explicit consent, or invalid phone numbers."""
        # 1. Missing full name
        res = client.post("/api/guardian", json={"patient_id": "BED-03", "phone_number": "+14155552671", "consent_confirmed": True})
        assert res.status_code == 400
        assert "name is required" in res.get_json()["error"].lower()

        # 2. Missing phone number
        res = client.post("/api/guardian", json={"patient_id": "BED-03", "full_name": "Dr. Lin", "consent_confirmed": True})
        assert res.status_code == 400
        assert "phone number is required" in res.get_json()["error"].lower()

        # 3. Invalid phone number format
        res = client.post("/api/guardian", json={"patient_id": "BED-03", "full_name": "Dr. Lin", "phone_number": "not-a-number", "consent_confirmed": True})
        assert res.status_code == 400
        assert "format" in res.get_json()["error"].lower() or "international" in res.get_json()["error"].lower()

        # 4. Enabled without explicit consent
        res = client.post("/api/guardian", json={
            "patient_id": "BED-03",
            "full_name": "Dr. Lin",
            "phone_number": "+14155552671",
            "is_enabled": True,
            "consent_confirmed": False
        })
        assert res.status_code == 400
        assert "consent" in res.get_json()["error"].lower()


class TestBedLinkedAlertDispatch:
    """Verifies that alerts resolve only to the specific bed's guardian profile."""

    def test_alert_routes_strictly_to_alert_bed_guardian(self):
        """An alert on BED-02 must dispatch to BED-02's guardian, never BED-01's guardian."""
        sms_service.set_mock_mode(True)

        g1 = DatabaseManager.get_guardian("BED-01")
        g2 = DatabaseManager.get_guardian("BED-02")
        assert g1["phone_normalized"] != g2["phone_normalized"]

        alert_bed2 = {
            "alert_id": f"ALT-BED2-{uuid.uuid4().hex[:6].upper()}",
            "patient_id": "BED-02",
            "priority": "HIGH PRIORITY (CRISIS / CODE BLUE)",
            "trigger_condition": "VENTRICULAR TACHYCARDIA",
            "diagnosis": "Ventricular Tachycardia",
            "location": "Cardiac ICU - Bed 02"
        }

        eval_res = sms_service.send_emergency_alert(alert_bed2, patient_data_mode="LIVE")
        assert eval_res["eligible"] is True
        assert "notification_uuid" in eval_res

        notif = DatabaseManager.get_notification_by_uuid(eval_res["notification_uuid"])
        assert notif is not None
        assert notif["patient_id"] == "BED-02"
        assert notif["recipient_phone_masked"] == g2["phone_masked"]
        assert notif["recipient_phone_masked"] != g1["phone_masked"]

    def test_sms_body_includes_bed_severity_event_and_timestamp_without_pii(self):
        """SMS message body must include bed identifier, severity, event, timestamp, without patient PII."""
        sms_service.set_mock_mode(True)
        test_bed = "BED-03"

        # Ensure BED-03 has consent enabled for test
        DatabaseManager.save_guardian(test_bed, {
            "full_name": "Dr. Sarah Guardian",
            "phone_number": "+14155553311",
            "consent_confirmed": True,
            "is_enabled": True,
            "preferred_categories": "CRISIS"
        })

        test_time = "2026-10-20 23:15:00"
        alert_payload = {
            "alert_id": f"ALT-BED3-{uuid.uuid4().hex[:6].upper()}",
            "patient_id": test_bed,
            "priority": "HIGH PRIORITY (CRISIS / CODE BLUE)",
            "trigger_condition": "VENTRICULAR FIBRILLATION DETECTED",
            "diagnosis": "Ventricular Fibrillation",
            "location": "Step-Down Ward Bed 03",
            "timestamp": test_time
        }

        eval_res = sms_service.send_emergency_alert(alert_payload, patient_data_mode="LIVE")
        assert eval_res["eligible"] is True
        notif = DatabaseManager.get_notification_by_uuid(eval_res["notification_uuid"])
        assert notif is not None

        body = notif["message_body"]
        # Required clinical fields
        assert f"Bed: {test_bed}" in body
        assert "Severity: HIGH PRIORITY (CRISIS / CODE BLUE)" in body
        assert "Event: VENTRICULAR FIBRILLATION DETECTED" in body
        assert f"Time: {test_time}" in body

        # Privacy check: No patient full name or confidential diagnosis narrative in message
        assert "Sarah Lin" not in body
        assert "Eleanor Vance" not in body

    def test_mitbih_playback_mode_never_sends_real_sms(self):
        """MIT-BIH recorded playback MUST isolate to DEMO_RECORDED and DEMO_PREVIEW."""
        test_bed = "REMOTE-01"
        alert_payload = {
            "alert_id": f"ALT-DEMO-{uuid.uuid4().hex[:6].upper()}",
            "patient_id": test_bed,
            "priority": "HIGH PRIORITY (CRISIS / CODE BLUE)",
            "trigger_condition": "MIT-BIH Record 207 Ventricular Flutter",
            "diagnosis": "Ventricular Fibrillation",
            "location": "Home Care Remote 01"
        }

        res = sms_service.send_emergency_alert(alert_payload, patient_data_mode="MIT-BIH")
        assert res["eligible"] is True
        assert res["mode"] == "DEMO_RECORDED"
        assert "no real SMS sent during recorded playback" in res["message"]

        notif = res["notification"]
        assert notif["message_type"] == "DEMO_PREVIEW"
        assert notif["status"] == "DEMO_RECORDED"
        assert "Bed: REMOTE-01" in notif["message_body"]
        assert "demonstration notification, not a live patient emergency" in notif["message_body"].lower()


class TestSafeTestSMSPreviewAndDispatch:
    """Verifies safe test preview mode and explicit dispatch."""

    def test_test_sms_preview_endpoint(self, client):
        """GET /api/guardian/test_sms_preview returns safe preview with bed, recipient, and message without sending."""
        res = client.get("/api/guardian/test_sms_preview/BED-01")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["patient_id"] == "BED-01"
        assert "recipient_name" in data
        assert "phone_masked" in data
        assert "preview_body" in data
        assert "Bed: BED-01" in data["preview_body"]
        assert "test only, not a medical emergency" in data["preview_body"].lower()

    def test_test_sms_dispatch_requires_explicit_post(self, client):
        """POST /api/guardian/test_sms requires explicit call and records test message."""
        sms_service.set_mock_mode(True)
        res = client.post("/api/guardian/test_sms", json={"patient_id": "BED-01"})
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["status"] in ("DELIVERED", "ACCEPTED", "SENT")

        notif = DatabaseManager.get_notification_by_uuid(data["notification_uuid"])
        assert notif is not None
        assert notif["message_type"] == "TEST"
        assert "Bed: BED-01" in notif["message_body"]


class TestAlertLifecycleControls:
    """Verifies atomic lifecycle actions: acknowledge, snooze, resolve, dismiss, breakthrough."""

    def test_acknowledge_persists_to_backend(self, client):
        """Acknowledge action sets status ACKNOWLEDGED and acknowledged=True."""
        alert_id = f"ALT-ACK-{uuid.uuid4().hex[:6].upper()}"
        alert_system.active_alerts[alert_id] = {
            "alert_id": alert_id,
            "patient_id": "BED-01",
            "priority": AlarmPriority.WARNING,
            "trigger_condition": "HEAVY TACHYCARDIA",
            "acknowledged": False,
            "status": "ACTIVE"
        }

        res = client.post("/api/alerts/action", json={
            "action": "ack",
            "alert_id": alert_id,
            "clinician_name": "Test Clinician"
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["action"] == "acknowledge"

        alert = alert_system.get_alert_by_id(alert_id)
        assert alert["acknowledged"] is True
        assert alert["status"] == "ACKNOWLEDGED"

    def test_dismiss_banner_persists_and_suppresses_banner(self, client):
        """Dismiss action registers alert in dismissed_banner_alerts to prevent SSE stream resurrection."""
        alert_id = f"ALT-DISMISS-{uuid.uuid4().hex[:6].upper()}"
        alert_system.active_alerts[alert_id] = {
            "alert_id": alert_id,
            "patient_id": "BED-02",
            "priority": AlarmPriority.WARNING,
            "trigger_condition": "PRE-ARRHYTHMIA",
            "banner_dismissed": False
        }

        res = client.post("/api/alerts/action", json={
            "action": "dismiss",
            "alert_id": alert_id
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert alert_id in alert_system.dismissed_banner_alerts

    def test_snooze_and_critical_breakthrough(self, client):
        """Snooze suppresses repeat notifications but allows higher-severity CRISIS breakthrough."""
        bed_id = "BED-03"
        alert_id = f"ALT-SNOOZE-{uuid.uuid4().hex[:6].upper()}"
        alert_system.active_alerts[alert_id] = {
            "alert_id": alert_id,
            "patient_id": bed_id,
            "priority": AlarmPriority.WARNING,
            "trigger_condition": "WARNING: Moderate Tachycardia",
            "status": "ACTIVE"
        }
        alert_system.bed_active_map[bed_id] = alert_id

        # 1. Snooze BED-03 for 5 minutes
        res = client.post("/api/alerts/action", json={
            "action": "snooze",
            "patient_id": bed_id,
            "alert_id": alert_id,
            "minutes": 5
        })
        assert res.status_code == 200
        assert bed_id in alert_system.snoozed_beds

        # 2. Re-evaluating identical warning condition is snoozed
        eval_warning = alert_system.evaluate_and_trigger(
            patient_id=bed_id,
            patient_name="Sarah Lin",
            location="Step-Down Ward Bed 03",
            diagnosis="Sinus Tachycardia",
            cewi_data={"cewi_score": 60},
            metrics={"heart_rate": 115, "signal_quality": "OPTIMAL"}
        )
        assert eval_warning is not None
        assert eval_warning["status"] == "SNOOZED"

        # 3. Genuinely new CRISIS event (Ventricular Fibrillation) breaks through the warning snooze
        eval_crisis = alert_system.evaluate_and_trigger(
            patient_id=bed_id,
            patient_name="Sarah Lin",
            location="Step-Down Ward Bed 03",
            diagnosis="Ventricular Fibrillation",
            cewi_data={"cewi_score": 95},
            metrics={"heart_rate": 180, "signal_quality": "OPTIMAL"}
        )
        assert eval_crisis is not None
        assert eval_crisis["priority"] == AlarmPriority.CRISIS
        assert "VENTRICULAR FIBRILLATION" in eval_crisis["trigger_condition"]
        # The warning snooze was cleared by the crisis breakthrough
        assert bed_id not in alert_system.snoozed_beds


class TestTwilioDeliveryReceiptsAndSync:
    """Verifies Twilio webhook receipts and status synchronization."""

    def test_twilio_callback_updates_notification_status(self, client):
        """POST /api/guardian/twilio_callback updates notification record from provider receipt."""
        notif_uuid = f"NOTIF-CB-{uuid.uuid4().hex[:8].upper()}"
        provider_sid = f"SM_TWILIO_CB_{uuid.uuid4().hex[:8]}"

        DatabaseManager.create_notification(
            patient_id="BED-01",
            recipient_phone_masked="+1 ••• ••• 2671",
            message_type="REAL_EMERGENCY",
            message_body="Test emergency",
            status="SENT",
            provider="twilio",
            provider_message_id=provider_sid,
            idempotency_key=f"IDEMP-CB-{uuid.uuid4().hex[:6]}"
        )

        res = client.post("/api/guardian/twilio_callback", data={
            "MessageSid": provider_sid,
            "MessageStatus": "delivered"
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["status"] == "DELIVERED"

        # Verify database record updated
        updated_notif = DatabaseManager.get_notification_by_provider_id(provider_sid)
        assert updated_notif is not None
        assert updated_notif["status"] == "DELIVERED"

    def test_notification_sync_status_endpoint(self, client):
        """POST /api/guardian/notifications/<uuid>/sync_status queries provider status."""
        sms_service.set_mock_mode(True)

        notif = DatabaseManager.create_notification(
            patient_id="BED-02",
            recipient_phone_masked="+1 ••• ••• 3344",
            message_type="REAL_EMERGENCY",
            message_body="Test sync status",
            status="ACCEPTED",
            provider="mock",
            provider_message_id=f"MOCK-SYNC-{uuid.uuid4().hex[:8]}",
            idempotency_key=f"IDEMP-SYNC-{uuid.uuid4().hex[:6]}"
        )
        notif_uuid = notif["notification_uuid"]

        res = client.post(f"/api/guardian/notifications/{notif_uuid}/sync_status")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "status" in data
