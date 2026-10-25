"""
CardioSentinel | ECG Intelligence - Event-Based Emergency Alert System Test Suite
Validates:
1. Normal ECG readings produce no emergency notifications.
2. Sustained emergency for 60 seconds results in exactly one notification attempt.
3. Episode resolution transitions active alert to RESOLVED after recovery criteria are satisfied.
4. A new emergency occurring after recovery creates a new alert and allows notification.
5. Unconfigured SMS provider records accurate FAILED status without claiming delivery.
6. Historical MIT-BIH playback produces demo alerts only with no real SMS.
7. Duplicate requests and application restarts prevent duplicate notifications via persistent episode IDs.
"""

import pytest
import time
import uuid
import os
from unittest.mock import patch, MagicMock

from backend.server import app, telemetry_engine
from backend.alert_system import ClinicalAlertSystem, ClinicalState, AlarmPriority, alert_system
from backend.sms_service import SMSService, sms_service
from backend.database import DatabaseManager, get_db_connection


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def reset_alert_and_sms_state():
    """Ensures a clean state before each test."""
    sms_service.set_mock_mode(True)
    sms_service._cooldown_map.clear()
    alert_system.active_alerts.clear()
    alert_system.bed_active_map.clear()
    alert_system.patient_states.clear()
    alert_system.normal_readings_count.clear()
    alert_system.active_episodes.clear()
    alert_system.cooldown_map.clear()
    alert_system.recovery_threshold = 5
    yield
    sms_service.set_mock_mode(True)
    sms_service._cooldown_map.clear()
    alert_system.active_alerts.clear()
    alert_system.bed_active_map.clear()
    alert_system.patient_states.clear()
    alert_system.normal_readings_count.clear()
    alert_system.active_episodes.clear()
    alert_system.cooldown_map.clear()


class TestEventBasedEmergencyAlertSystem:
    """Test suite for event-based emergency alerting and episode lifecycle."""

    def test_01_normal_ecg_no_emergency_notification(self):
        """Scenario: Normal ECG readings must not generate emergency alerts or notifications."""
        bed_id = f"BED-NORM-{int(time.time()*1000)}"

        # Evaluate normal readings over multiple cycles
        for cycle in range(10):
            alert = alert_system.evaluate_and_trigger(
                patient_id=bed_id,
                patient_name="Normal Patient",
                location="Cardiology Ward",
                diagnosis="Normal Sinus Rhythm",
                cewi_data={"cewi_score": 12, "shock_index": 0.6},
                metrics={"heart_rate": 72, "qrs_duration_ms": 88, "qtc_ms": 410, "hrv_sdnn": 45},
                source_record="100"
            )
            assert alert is None, f"Normal ECG should not generate an alert (cycle {cycle})"
            assert alert_system.get_patient_state(bed_id) == ClinicalState.NORMAL

        # Ensure no active alert exists in alert system
        assert bed_id not in alert_system.bed_active_map
        assert alert_system.get_active_alerts() == []

    def test_02_emergency_persists_60_seconds_one_notification_attempt(self):
        """Scenario: One emergency persisting for 60s produces 1 alert and 1 notification attempt."""
        bed_id = f"BED-VT-{int(time.time()*1000)}"

        # Register guardian with consent in LIVE mode
        DatabaseManager.save_guardian(bed_id, {
            "full_name": "Test Guardian",
            "phone_number": "+14155550199",
            "consent_confirmed": True,
            "is_enabled": True,
            "preferred_categories": "CRISIS,WARNING"
        })

        # Cycle 1: First emergency reading triggers new episode and alert
        alert_cycle1 = alert_system.evaluate_and_trigger(
            patient_id=bed_id,
            patient_name="Critical Patient",
            location="Cardiac ICU",
            diagnosis="Ventricular Tachycardia",
            cewi_data={"cewi_score": 92, "shock_index": 1.5},
            metrics={"heart_rate": 185, "qrs_duration_ms": 145, "qtc_ms": 520, "hrv_sdnn": 15},
            source_record="SYNTHETIC"
        )

        assert alert_cycle1 is not None
        assert alert_cycle1.get("is_new_emergency") is True
        initial_alert_id = alert_cycle1["alert_id"]
        initial_episode_id = alert_cycle1["episode_id"]

        # Attempt first notification
        dispatch_res1 = sms_service.send_emergency_alert(alert_cycle1, patient_data_mode="LIVE")
        assert dispatch_res1["eligible"] is True
        alert_system.record_episode_notification(
            bed_id,
            initial_episode_id,
            notification_uuid=dispatch_res1.get("notification_uuid"),
            notification_sent=True
        )

        # Simulate emergency persisting for 60 cycles (~60s or more)
        notification_attempts = 0
        for i in range(2, 62):
            alert_subsequent = alert_system.evaluate_and_trigger(
                patient_id=bed_id,
                patient_name="Critical Patient",
                location="Cardiac ICU",
                diagnosis="Ventricular Tachycardia",
                cewi_data={"cewi_score": 90 + (i % 5), "shock_index": 1.5},
                metrics={"heart_rate": 180 + (i % 6), "qrs_duration_ms": 145, "qtc_ms": 520, "hrv_sdnn": 15},
                source_record="SYNTHETIC"
            )

            # Alert ID and episode ID MUST remain identical (updated in-place, NOT recreated)
            assert alert_subsequent["alert_id"] == initial_alert_id
            assert alert_subsequent["episode_id"] == initial_episode_id
            assert alert_subsequent.get("is_new_emergency") is False
            assert alert_system.get_patient_state(bed_id) == ClinicalState.EMERGENCY

            # Only trigger notification if is_new_emergency is True (application rule)
            if alert_subsequent.get("is_new_emergency"):
                notification_attempts += 1

            # Even if send_emergency_alert were explicitly called again, database deduplication MUST reject it
            res_dup = sms_service.send_emergency_alert(alert_subsequent, patient_data_mode="LIVE")
            assert res_dup["eligible"] is False
            assert "already exists" in res_dup["reason"] or "Cooldown" in res_dup["reason"]

        assert notification_attempts == 0, "No additional notifications should be attempted while same emergency continues"

    def test_03_emergency_resolves_updates_alert_to_resolved(self):
        """Scenario: Emergency resolves when recovery criteria satisfied; alert marked RESOLVED."""
        bed_id = f"BED-RESOLVE-{int(time.time()*1000)}"
        alert_system.recovery_threshold = 4  # 4 consecutive normal readings to confirm recovery

        # 1. Trigger emergency
        alert = alert_system.evaluate_and_trigger(
            patient_id=bed_id,
            patient_name="Resolving Patient",
            location="ICU Bed 02",
            diagnosis="Ventricular Fibrillation",
            cewi_data={"cewi_score": 100, "shock_index": 2.0},
            metrics={"heart_rate": 220, "qrs_duration_ms": 160, "qtc_ms": 540},
            source_record="SYNTHETIC"
        )
        assert alert is not None
        alert_id = alert["alert_id"]
        assert alert["status"] == "ACTIVE"
        assert alert_system.get_patient_state(bed_id) == ClinicalState.EMERGENCY

        # 2. Patient starts stabilizing: Normal readings 1, 2, 3 (< recovery_threshold)
        for i in range(1, 4):
            res = alert_system.evaluate_and_trigger(
                patient_id=bed_id,
                patient_name="Resolving Patient",
                location="ICU Bed 02",
                diagnosis="Normal Sinus Rhythm",
                cewi_data={"cewi_score": 15, "shock_index": 0.6},
                metrics={"heart_rate": 74, "qrs_duration_ms": 85, "qtc_ms": 405},
                source_record="SYNTHETIC"
            )
            # Before threshold, active alert is preserved to prevent rapid chattering
            assert res is not None, f"Alert should remain active during recovery window (cycle {i})"
            assert res["status"] == "ACTIVE"

        # 3. Reading 4 reaches recovery threshold (4 >= 4) -> Episode resolves!
        res_recovered = alert_system.evaluate_and_trigger(
            patient_id=bed_id,
            patient_name="Resolving Patient",
            location="ICU Bed 02",
            diagnosis="Normal Sinus Rhythm",
            cewi_data={"cewi_score": 14, "shock_index": 0.6},
            metrics={"heart_rate": 72, "qrs_duration_ms": 85, "qtc_ms": 405},
            source_record="SYNTHETIC"
        )
        assert res_recovered is None, "Once resolved, evaluate_and_trigger returns None"
        assert alert_system.get_patient_state(bed_id) == ClinicalState.NORMAL
        assert bed_id not in alert_system.bed_active_map

        # Check alert history for RESOLVED status
        history = alert_system.get_alert_history()
        resolved_alerts = [a for a in history if a["alert_id"] == alert_id]
        assert len(resolved_alerts) == 1
        resolved_alert = resolved_alerts[0]
        assert resolved_alert["status"] == "RESOLVED"
        assert resolved_alert["state"] == ClinicalState.RESOLVED
        assert resolved_alert["resolved_by"] == "System Auto-Resolution"
        assert "recovery criteria met" in resolved_alert["resolution_note"]

    def test_04_new_emergency_after_recovery_creates_new_alert(self):
        """Scenario: A new emergency occurring after recovery creates a new alert and episode."""
        bed_id = f"BED-RECUR-{int(time.time()*1000)}"
        alert_system.recovery_threshold = 3

        # 1. First emergency
        alert1 = alert_system.evaluate_and_trigger(
            patient_id=bed_id,
            patient_name="Recurrent Patient",
            location="ICU Bed 01",
            diagnosis="Ventricular Tachycardia",
            cewi_data={"cewi_score": 92, "shock_index": 1.4},
            metrics={"heart_rate": 180, "qrs_duration_ms": 140, "qtc_ms": 500},
            source_record="SYNTHETIC"
        )
        assert alert1 is not None
        aid1 = alert1["alert_id"]
        ep1 = alert1["episode_id"]

        # 2. Fully recover (3 consecutive normal readings)
        for _ in range(3):
            alert_system.evaluate_and_trigger(
                patient_id=bed_id,
                patient_name="Recurrent Patient",
                location="ICU Bed 01",
                diagnosis="Normal Sinus Rhythm",
                cewi_data={"cewi_score": 10, "shock_index": 0.6},
                metrics={"heart_rate": 70, "qrs_duration_ms": 85, "qtc_ms": 400},
                source_record="SYNTHETIC"
            )
        assert alert_system.get_patient_state(bed_id) == ClinicalState.NORMAL
        assert bed_id not in alert_system.bed_active_map

        # Clear cooldown so new episode can trigger immediately
        alert_system.cooldown_map.clear()
        sms_service._cooldown_map.clear()

        # 3. New emergency episode begins
        alert2 = alert_system.evaluate_and_trigger(
            patient_id=bed_id,
            patient_name="Recurrent Patient",
            location="ICU Bed 01",
            diagnosis="Ventricular Fibrillation",
            cewi_data={"cewi_score": 100, "shock_index": 1.9},
            metrics={"heart_rate": 230, "qrs_duration_ms": 155, "qtc_ms": 530},
            source_record="SYNTHETIC"
        )

        assert alert2 is not None
        assert alert2["is_new_emergency"] is True
        assert alert2["alert_id"] != aid1, "New emergency must receive a new distinct alert ID"
        assert alert2["episode_id"] != ep1, "New emergency must receive a new distinct episode ID"
        assert alert_system.get_patient_state(bed_id) == ClinicalState.EMERGENCY

    def test_05_unconfigured_sms_provider_records_failed_status(self, monkeypatch):
        """Scenario: Unconfigured SMS provider records accurate FAILED status without claiming delivery."""
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("TWILIO_FROM_NUMBER", raising=False)
        sms_service.set_mock_mode(False)

        bed_id = f"BED-UNCONFIG-ALERT-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(bed_id, {
            "full_name": "Unconfigured Test Guardian",
            "phone_number": "+14155557788",
            "consent_confirmed": True,
            "is_enabled": True
        })

        alert_payload = {
            "alert_id": f"ALT-UNCONFIG-{int(time.time()*1000)}",
            "episode_id": f"EP-UNCONFIG-{int(time.time()*1000)}",
            "patient_id": bed_id,
            "priority": AlarmPriority.CRISIS,
            "trigger_condition": "Ventricular Fibrillation",
            "diagnosis": "Ventricular Fibrillation",
            "location": "Cardiac ICU"
        }

        res = sms_service.send_emergency_alert(alert_payload, patient_data_mode="LIVE")

        assert res["eligible"] is False
        assert res["status"] == "FAILED"
        assert "SMS provider not configured" in res["error"]
        assert "notification_uuid" in res

        # Verify database record reflects actual FAILED status and not delivered/sent
        notif = DatabaseManager.get_notification_by_uuid(res["notification_uuid"])
        assert notif is not None
        assert notif["status"] == "FAILED"
        assert notif["provider"] == "none"
        assert "SMS provider not configured" in notif["error_message"]

        sms_service.set_mock_mode(True)

    def test_06_historical_mitbih_playback_demo_alert_only_no_real_sms(self):
        """Scenario: MIT-BIH playback produces DEMO preview only; real SMS is strictly prevented."""
        bed_id = f"BED-MITBIH-SAFE-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(bed_id, {
            "full_name": "Demo Mode Guardian",
            "phone_number": "+91 98765 00001",
            "consent_confirmed": True,
            "is_enabled": True
        })

        alert = alert_system.evaluate_and_trigger(
            patient_id=bed_id,
            patient_name="Demo Subject",
            location="Simulation Station",
            diagnosis="Ventricular Tachycardia",
            cewi_data={"cewi_score": 92},
            metrics={"heart_rate": 185},
            source_record="200" # MIT-BIH Record 200
        )

        assert alert["source_type"] == "RECORDED_BENCHMARK"
        assert "PhysioNet MIT-BIH" in alert["source_label"]
        assert "Research telemetry indicator" in alert["disclaimer"]

        # Guardian notification evaluation under MIT-BIH playback
        with patch.object(sms_service, "_dispatch_sms_provider") as mock_dispatch:
            res = sms_service.send_emergency_alert(alert, patient_data_mode="MIT-BIH")
            # Network provider dispatch must NOT be called
            mock_dispatch.assert_not_called()

        assert res["eligible"] is True
        assert res["mode"] == "DEMO_RECORDED"
        assert "notification" in res

        notif = res["notification"]
        assert notif["message_type"] == "DEMO_PREVIEW"
        assert notif["status"] == "DEMO_RECORDED"
        assert notif["message_body"].startswith("DEMO — Recorded ECG playback")
        assert "not a live patient emergency" in notif["message_body"].lower()

    def test_07_duplicate_requests_and_restart_deduplication(self):
        """Scenario: Application restart and duplicate requests do not resend notification for same episode."""
        bed_id = f"BED-RESTART-{int(time.time()*1000)}"
        DatabaseManager.save_guardian(bed_id, {
            "full_name": "Restart Guardian",
            "phone_number": "+14155558899",
            "consent_confirmed": True,
            "is_enabled": True
        })

        # 1. Create emergency episode
        alert = alert_system.evaluate_and_trigger(
            patient_id=bed_id,
            patient_name="Persistent Patient",
            location="ICU Bed 04",
            diagnosis="Ventricular Fibrillation",
            cewi_data={"cewi_score": 100},
            metrics={"heart_rate": 240},
            source_record="SYNTHETIC"
        )
        episode_id = alert["episode_id"]

        # Send first notification
        res1 = sms_service.send_emergency_alert(alert, patient_data_mode="LIVE")
        assert res1["eligible"] is True
        alert_system.record_episode_notification(
            bed_id,
            episode_id,
            notification_uuid=res1.get("notification_uuid"),
            notification_sent=True
        )

        # 2. Duplicate request test
        res_dup = sms_service.send_emergency_alert(alert, patient_data_mode="LIVE")
        assert res_dup["eligible"] is False
        assert res_dup.get("already_sent") is True

        # 3. Simulate application restart: instantiate a new ClinicalAlertSystem instance
        new_alert_system = ClinicalAlertSystem()
        # Verify it re-hydrated the active episode from SQLite
        assert bed_id in new_alert_system.active_episodes
        ep = new_alert_system.active_episodes[bed_id]
        assert ep["episode_id"] == episode_id
        assert ep["notification_sent"] == 1
        assert new_alert_system.get_patient_state(bed_id) == ClinicalState.EMERGENCY

        # Subsequent evaluation cycle on restarted system maintains existing episode
        alert_post_restart = new_alert_system.evaluate_and_trigger(
            patient_id=bed_id,
            patient_name="Persistent Patient",
            location="ICU Bed 04",
            diagnosis="Ventricular Fibrillation",
            cewi_data={"cewi_score": 98},
            metrics={"heart_rate": 235},
            source_record="SYNTHETIC"
        )
        # Does NOT create a new emergency
        assert alert_post_restart.get("is_new_emergency") is False

        # Attempting notification again must be blocked by database idempotency
        res_after_restart = sms_service.send_emergency_alert(alert_post_restart, patient_data_mode="LIVE")
        assert res_after_restart["eligible"] is False
        assert "already exists" in res_after_restart["reason"] or res_after_restart.get("already_sent") is True

    def test_08_abnormal_prediction_not_classified_as_emergency(self):
        """Scenario: Abnormal conditions (PVC, AFIB, mild tachycardia) set state to ABNORMAL, not EMERGENCY."""
        bed_id = f"BED-ABNORM-{int(time.time()*1000)}"

        # Atrial Fibrillation with moderate CEWI
        alert_afib = alert_system.evaluate_and_trigger(
            patient_id=bed_id,
            patient_name="AFib Patient",
            location="Stepdown Unit",
            diagnosis="Atrial Fibrillation",
            cewi_data={"cewi_score": 45},
            metrics={"heart_rate": 110, "qrs_duration_ms": 90, "qtc_ms": 420},
            source_record="209"
        )

        assert alert_afib is not None
        assert alert_afib["priority"] in (AlarmPriority.WARNING, AlarmPriority.ADVISORY)
        assert alert_afib["state"] == ClinicalState.ABNORMAL
        assert alert_afib.get("is_new_emergency") is False
        assert alert_system.get_patient_state(bed_id) == ClinicalState.ABNORMAL

        # Ensure no emergency episode was opened
        assert alert_system.active_episodes.get(bed_id) is None
