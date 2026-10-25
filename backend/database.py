"""
CardioSentinel | ECG Intelligence - SQLite Database & Persistence Layer
Manages guardian contacts, phone verification, notification audit trail,
and delivery status with thread safety and parameterized queries.
"""

import sqlite3
import os
import threading
import time
import uuid
import re
import secrets
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Tuple, Union

# Base directory for database file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "cardiosentinel.db")

# Thread-local storage for SQLite connections
_local = threading.local()
_db_lock = threading.Lock()


def get_db_connection() -> sqlite3.Connection:
    """Returns a thread-local SQLite database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(DATA_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        _local.conn = conn
    return _local.conn


def normalize_phone_number(phone: str) -> Tuple[bool, str, str]:
    """
    Validates and normalizes phone number to E.164 international format.
    Accepts valid formats (e.g. +14155552671, +91 98765 43210, +44 20 7946 0991).
    Returns: (is_valid, normalized_phone, error_message)
    """
    if not phone or not isinstance(phone, str):
        return False, "", "Phone number cannot be empty."

    # Strip whitespace, hyphens, brackets, dots
    cleaned = re.sub(r"[\s\-\(\)\.]", "", phone.strip())

    if not cleaned:
        return False, "", "Phone number cannot be empty."

    # Check for leading 00 (international call prefix) and normalize to +
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]

    # Check for leading +
    if not cleaned.startswith("+"):
        # If 10 digits without +, default to +91 or require country code
        if re.match(r"^\d{10}$", cleaned):
            cleaned = "+91" + cleaned
        elif re.match(r"^\d{11,15}$", cleaned):
            cleaned = "+" + cleaned
        else:
            return False, "", "Phone number must include international country code (e.g. +1XXXXXXXXXX or +91XXXXXXXXXX)."

    # Validate E.164 standard: + followed by country code (1-3 digits) and total 7 to 15 digits
    if not re.match(r"^\+[1-9]\d{6,14}$", cleaned):
        return False, "", "Invalid international phone format. Expected E.164 format: +[country code][digits] (e.g. +919876543210 or +14155552671)."

    return True, cleaned, ""


def mask_phone_number(phone: str) -> str:
    """
    Masks phone number for clinical privacy and audit logs.
    Example: +919876543210 -> +91 ••••• ••210
             +14155552671  -> +1 ••••• ••671
    """
    if not phone:
        return "--"
    cleaned = re.sub(r"[\s\-\(\)]", "", str(phone).strip())
    if len(cleaned) < 7:
        return cleaned[:2] + "****" if len(cleaned) >= 2 else "****"

    if cleaned.startswith("+"):
        prefix = cleaned[:3]
        suffix = cleaned[-3:]
        return f"{prefix} ••••• ••{suffix}"
    else:
        prefix = cleaned[:2]
        suffix = cleaned[-3:]
        return f"{prefix} ••••• ••{suffix}"


class DatabaseManager:
    """Encapsulates all database operations for the Guardian Notification System."""

    @classmethod
    def init_db(cls):
        """Initializes database schema, executes safe migrations, and seeds records."""
        with _db_lock:
            conn = get_db_connection()
            with conn:
                # 1. Guardians table (clean schema without OTP fields, with consent_timestamp)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS guardians (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id TEXT UNIQUE NOT NULL,
                        full_name TEXT NOT NULL,
                        phone_number TEXT NOT NULL,
                        phone_normalized TEXT NOT NULL,
                        phone_masked TEXT NOT NULL,
                        relationship TEXT NOT NULL,
                        is_enabled INTEGER DEFAULT 1,
                        is_verified INTEGER DEFAULT 0,
                        consent_confirmed INTEGER DEFAULT 0,
                        consent_timestamp TEXT,
                        preferred_categories TEXT DEFAULT 'CRISIS,WARNING',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                """)

                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(guardians);")
                cols = [row["name"] for row in cursor.fetchall()]

                # Safe migration: ensure consent_timestamp column exists
                if "consent_timestamp" not in cols:
                    try:
                        conn.execute("ALTER TABLE guardians ADD COLUMN consent_timestamp TEXT;")
                    except Exception:
                        pass

                # Remove legacy OTP columns if present
                otp_legacy_cols = [
                    "verification_code",
                    "verification_code_hash",
                    "verification_salt",
                    "verification_expiry",
                    "verification_attempts",
                    "verification_last_sent"
                ]
                for col_name in otp_legacy_cols:
                    if col_name in cols:
                        try:
                            conn.execute(f"ALTER TABLE guardians DROP COLUMN {col_name};")
                        except Exception:
                            pass

                # Ensure all guardians have is_verified = 0 (no fake Verified statuses)
                conn.execute("UPDATE guardians SET is_verified = 0 WHERE is_verified != 0;")

                # 2. Guardian notifications table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS guardian_notifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        notification_uuid TEXT UNIQUE NOT NULL,
                        alert_id TEXT,
                        patient_id TEXT NOT NULL,
                        guardian_id INTEGER,
                        recipient_phone_masked TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        message_body TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        provider_message_id TEXT,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        idempotency_key TEXT UNIQUE,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (guardian_id) REFERENCES guardians(id)
                    );
                """)

                # 3. Audit logs table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action TEXT NOT NULL,
                        actor TEXT DEFAULT 'Staff Clinician',
                        patient_id TEXT,
                        details TEXT,
                        timestamp TEXT NOT NULL
                    );
                """)

                # 4. Emergency episodes table for persistent episode lifecycle and deduplication
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS emergency_episodes (
                        episode_id TEXT PRIMARY KEY,
                        patient_id TEXT NOT NULL,
                        alert_id TEXT NOT NULL,
                        state TEXT NOT NULL DEFAULT 'EMERGENCY',
                        trigger_condition TEXT NOT NULL,
                        diagnosis TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        notification_sent INTEGER DEFAULT 0,
                        notification_status TEXT,
                        notification_uuid TEXT,
                        started_at TEXT NOT NULL,
                        last_updated_at TEXT NOT NULL,
                        resolved_at TEXT,
                        explanation_json TEXT
                    );
                """)

                # Safe migration: ensure explanation_json column exists in emergency_episodes
                cursor.execute("PRAGMA table_info(emergency_episodes);")
                ep_cols = [row["name"] for row in cursor.fetchall()]
                if "explanation_json" not in ep_cols:
                    try:
                        conn.execute("ALTER TABLE emergency_episodes ADD COLUMN explanation_json TEXT;")
                    except Exception:
                        pass

                # Create helpful indices
                conn.execute("CREATE INDEX IF NOT EXISTS idx_guardians_pid ON guardians(patient_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_notif_pid ON guardian_notifications(patient_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_notif_status ON guardian_notifications(status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_notif_idemp ON guardian_notifications(idempotency_key);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_pid_state ON emergency_episodes(patient_id, state);")

            # Seed default guardians if table is empty
            cls._seed_defaults(conn)

    @classmethod
    def _seed_defaults(cls, conn: sqlite3.Connection):
        """Seeds registered contacts with is_verified=0 (Registered — Not Phone Verified)."""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS cnt FROM guardians;")
        count = cursor.fetchone()["cnt"]
        if count == 0:
            now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
            defaults = [
                {
                    "patient_id": "BED-01",
                    "full_name": "Arthur Vance",
                    "phone_number": "+91 98765 43210",
                    "relationship": "Spouse & Primary Guardian",
                    "is_enabled": 1,
                    "is_verified": 0,
                    "consent_confirmed": 1,
                    "preferred_categories": "CRISIS,WARNING",
                },
                {
                    "patient_id": "BED-02",
                    "full_name": "Clara Brody",
                    "phone_number": "+91 98450 11223",
                    "relationship": "Daughter",
                    "is_enabled": 1,
                    "is_verified": 0,
                    "consent_confirmed": 1,
                    "preferred_categories": "CRISIS,WARNING",
                },
                {
                    "patient_id": "BED-03",
                    "full_name": "Dr. Jonathan Lin",
                    "phone_number": "+91 98112 34567",
                    "relationship": "Primary Physician",
                    "is_enabled": 1,
                    "is_verified": 0,
                    "consent_confirmed": 0,
                    "preferred_categories": "CRISIS",
                },
                {
                    "patient_id": "REMOTE-01",
                    "full_name": "Margaret Pendelton",
                    "phone_number": "+91 99001 88220",
                    "relationship": "Caregiver",
                    "is_enabled": 1,
                    "is_verified": 0,
                    "consent_confirmed": 1,
                    "preferred_categories": "CRISIS,WARNING",
                }
            ]

            for d in defaults:
                _, norm, _ = normalize_phone_number(d["phone_number"])
                masked = mask_phone_number(norm)
                consent_ts = now_iso if d["consent_confirmed"] else None
                conn.execute("""
                    INSERT INTO guardians (
                        patient_id, full_name, phone_number, phone_normalized,
                        phone_masked, relationship, is_enabled, is_verified,
                        consent_confirmed, consent_timestamp, preferred_categories, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?);
                """, (
                    d["patient_id"], d["full_name"], d["phone_number"], norm,
                    masked, d["relationship"], d["is_enabled"],
                    d["consent_confirmed"], consent_ts, d["preferred_categories"], now_iso, now_iso
                ))

            # Add an initial demonstration notification record for BED-01
            conn.execute("""
                INSERT INTO guardian_notifications (
                    notification_uuid, alert_id, patient_id, guardian_id,
                    recipient_phone_masked, message_type, message_body,
                    provider, provider_message_id, status, error_message,
                    retry_count, max_retries, idempotency_key, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                f"NOTIF-{uuid.uuid4().hex[:8].upper()}",
                "ALT-INIT-001",
                "BED-01",
                1,
                "+91 ••••• ••210",
                "DEMO_PREVIEW",
                "DEMO — Recorded ECG playback: CardioSentinel detected an event during recorded ECG playback. This is a demonstration notification, not a live patient emergency.",
                "local_preview",
                "DEMO-INITIAL-001",
                "DEMO_RECORDED",
                None,
                0,
                3,
                f"IDEMP-INIT-BED-01-{int(time.time())}",
                now_iso,
                now_iso
            ))

            conn.execute("""
                INSERT INTO audit_logs (action, actor, patient_id, details, timestamp)
                VALUES ('SYSTEM_INIT', 'System', 'ALL', 'Initialized Guardian Database schema with benchmark defaults.', ?);
            """, (now_iso,))
            conn.commit()

    @classmethod
    def get_guardian(cls, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves registered guardian for a specific bed/patient."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM guardians WHERE patient_id = ?;", (patient_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        is_ver = int(row["is_verified"]) if ("is_verified" in row.keys() and row["is_verified"] is not None) else 0
        res["is_verified"] = is_ver
        res["phone_status"] = "Phone Verified" if is_ver == 1 else "Registered — Not Phone Verified"
        # Never leak security hashes or salts to callers
        res.pop("verification_code_hash", None)
        res.pop("verification_salt", None)
        res.pop("verification_code", None)
        return res

    @classmethod
    def get_all_guardians(cls) -> List[Dict[str, Any]]:
        """Retrieves all registered guardians."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM guardians ORDER BY patient_id ASC;")
        res = []
        for r in cursor.fetchall():
            d = dict(r)
            is_ver = int(r["is_verified"]) if ("is_verified" in r.keys() and r["is_verified"] is not None) else 0
            d["is_verified"] = is_ver
            d["phone_status"] = "Phone Verified" if is_ver == 1 else "Registered — Not Phone Verified"
            d.pop("verification_code_hash", None)
            d.pop("verification_salt", None)
            d.pop("verification_code", None)
            res.append(d)
        return res

    @classmethod
    def save_guardian(cls, patient_id: str, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        """
        Saves or updates guardian settings. Validates input, normalizes phone to E.164,
        enforces explicit consent with timestamp, prevents accidental duplicate submissions,
        masks phone, logs audit event, and returns updated dictionary.
        """
        full_name = str(data.get("full_name", "")).strip()
        phone_input = str(data.get("phone_number", "")).strip()
        relationship = str(data.get("relationship", "Primary Caregiver")).strip()
        is_enabled = 1 if data.get("is_enabled", True) else 0
        consent_confirmed = 1 if data.get("consent_confirmed", False) else 0
        preferred_categories = str(data.get("preferred_categories", "CRISIS,WARNING")).strip()

        if not full_name:
            return False, {}, "Guardian full name is required."

        if not phone_input:
            return False, {}, "Guardian phone number is required."

        is_valid, norm_phone, err = normalize_phone_number(phone_input)
        if not is_valid:
            return False, {}, err

        if is_enabled and not consent_confirmed:
            return False, {}, "Explicit consent confirmation is required before enabling emergency notifications."

        masked_phone = mask_phone_number(norm_phone)
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")

        with _db_lock:
            conn = get_db_connection()
            with conn:
                existing = cls.get_guardian(patient_id)
                if existing:
                    # Prevent accidental exact duplicate registration
                    if (existing.get("full_name") == full_name and
                        existing.get("phone_normalized") == norm_phone and
                        existing.get("relationship") == relationship and
                        existing.get("is_enabled") == is_enabled and
                        existing.get("consent_confirmed") == consent_confirmed and
                        existing.get("preferred_categories") == preferred_categories):
                        return True, existing, "Guardian configuration already up to date."

                    existing_ts = existing.get("consent_timestamp")
                    consent_ts = existing_ts if (consent_confirmed and existing.get("consent_confirmed") and existing_ts) else (now_iso if consent_confirmed else None)

                    conn.execute("""
                        UPDATE guardians SET
                            full_name = ?,
                            phone_number = ?,
                            phone_normalized = ?,
                            phone_masked = ?,
                            relationship = ?,
                            is_enabled = ?,
                            is_verified = 0,
                            consent_confirmed = ?,
                            consent_timestamp = ?,
                            preferred_categories = ?,
                            updated_at = ?
                        WHERE patient_id = ?;
                    """, (
                        full_name, phone_input, norm_phone, masked_phone,
                        relationship, is_enabled, consent_confirmed, consent_ts,
                        preferred_categories, now_iso, patient_id
                    ))
                    action = "UPDATE"
                    audit_msg = f"Updated guardian settings for {full_name} ({masked_phone}) for bed {patient_id}."
                else:
                    consent_ts = now_iso if consent_confirmed else None
                    conn.execute("""
                        INSERT INTO guardians (
                            patient_id, full_name, phone_number, phone_normalized,
                            phone_masked, relationship, is_enabled, is_verified,
                            consent_confirmed, consent_timestamp, preferred_categories, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?);
                    """, (
                        patient_id, full_name, phone_input, norm_phone,
                        masked_phone, relationship, is_enabled, consent_confirmed,
                        consent_ts, preferred_categories, now_iso, now_iso
                    ))
                    action = "REGISTER"
                    audit_msg = f"Registered new guardian {full_name} ({masked_phone}) for bed {patient_id}."

                conn.execute("""
                    INSERT INTO audit_logs (action, actor, patient_id, details, timestamp)
                    VALUES (?, 'Clinician / Administrator', ?, ?, ?);
                """, (action, patient_id, audit_msg, now_iso))

        updated = cls.get_guardian(patient_id)
        return True, updated or {}, "Guardian configuration saved successfully."

    @classmethod
    def remove_guardian(cls, patient_id: str) -> Tuple[bool, str]:
        """
        Explicitly removes the registered guardian for a monitored bed.
        Preserves notification history and audit trail intact.
        """
        existing = cls.get_guardian(patient_id)
        if not existing:
            return False, "No registered guardian found to remove for this bed."

        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        with _db_lock:
            conn = get_db_connection()
            with conn:
                conn.execute("DELETE FROM guardians WHERE patient_id = ?;", (patient_id,))
                conn.execute("""
                    INSERT INTO audit_logs (action, actor, patient_id, details, timestamp)
                    VALUES ('REMOVE_GUARDIAN', 'Clinician / Administrator', ?, ?, ?);
                """, (patient_id, f"Removed guardian contact {existing.get('full_name')} ({existing.get('phone_masked')}). Emergency notifications disarmed.", now_iso))

        return True, "Guardian contact removed successfully. Form has been reset."

    @classmethod
    def log_audit(cls, action: str, actor: str = "Clinician / Administrator", patient_id: Optional[str] = None, details: Optional[str] = None):
        """Records an immutable audit event in the database."""
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        with _db_lock:
            conn = get_db_connection()
            with conn:
                conn.execute("""
                    INSERT INTO audit_logs (action, actor, patient_id, details, timestamp)
                    VALUES (?, ?, ?, ?, ?);
                """, (action, actor, patient_id, details, now_iso))

    @classmethod
    def get_verification_status(cls, patient_id: str, provider_configured: bool = False) -> Dict[str, Any]:
        """
        Returns the honest verification status:
        - 'Registered — Not Phone Verified' if contact is registered (ownership not verified via challenge).
        - 'Not Registered' if no guardian contact exists.
        """
        guardian = cls.get_guardian(patient_id)
        if not guardian:
            return {
                "status": "UNREGISTERED",
                "label": "Not Registered",
                "badge_class": "badge-pill warning",
                "explanation": "No guardian contact registered for this bed."
            }

        if guardian.get("is_verified") == 1:
            return {
                "status": "VERIFIED",
                "label": "Phone Verified",
                "badge_class": "badge-pill stable",
                "explanation": "Phone ownership confirmed via carrier verification challenge.",
                "consent_timestamp": guardian.get("consent_timestamp")
            }

        expl = (
            "Phone registered and formatted to E.164 standard. "
            "Verification challenge pending provider configuration."
            if not provider_configured else
            "Phone registered and formatted to E.164 standard. Phone ownership verification challenge pending."
        )

        return {
            "status": "REGISTERED_UNVERIFIED",
            "label": "Registered — Not Phone Verified",
            "badge_class": "badge-pill warning",
            "explanation": expl,
            "consent_timestamp": guardian.get("consent_timestamp")
        }

    @classmethod
    def create_notification(
        cls,
        patient_id: str,
        message_type: str,
        message_body: str,
        status: str,
        provider: str,
        recipient_phone_masked: str,
        alert_id: Optional[str] = None,
        guardian_id: Optional[int] = None,
        provider_message_id: Optional[str] = None,
        error_message: Optional[str] = None,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a persistent record of a guardian notification attempt."""
        notif_uuid = f"NOTIF-{uuid.uuid4().hex[:8].upper()}"
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        idemp = idempotency_key or f"IDEMP-{uuid.uuid4().hex[:12].upper()}"

        with _db_lock:
            conn = get_db_connection()
            with conn:
                conn.execute("""
                    INSERT INTO guardian_notifications (
                        notification_uuid, alert_id, patient_id, guardian_id,
                        recipient_phone_masked, message_type, message_body,
                        provider, provider_message_id, status, error_message,
                        retry_count, max_retries, idempotency_key, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 3, ?, ?, ?);
                """, (
                    notif_uuid, alert_id, patient_id, guardian_id,
                    recipient_phone_masked, message_type, message_body,
                    provider, provider_message_id, status, error_message,
                    idemp, now_iso, now_iso
                ))

                conn.execute("""
                    INSERT INTO audit_logs (action, actor, patient_id, details, timestamp)
                    VALUES ('NOTIF_CREATE', 'System Notification Service', ?, ?, ?);
                """, (patient_id, f"Created {message_type} notification {notif_uuid} status={status}.", now_iso))

        return cls.get_notification_by_uuid(notif_uuid) or {}

    @classmethod
    def update_notification_status(
        cls,
        notification_uuid: str,
        status: str,
        provider_message_id: Optional[str] = None,
        error_message: Optional[str] = None,
        increment_retry: bool = False
    ) -> bool:
        """Updates delivery status and provider confirmation."""
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        with _db_lock:
            conn = get_db_connection()
            with conn:
                if increment_retry:
                    conn.execute("""
                        UPDATE guardian_notifications SET
                            status = ?,
                            provider_message_id = COALESCE(?, provider_message_id),
                            error_message = ?,
                            retry_count = retry_count + 1,
                            updated_at = ?
                        WHERE notification_uuid = ?;
                    """, (status, provider_message_id, error_message, now_iso, notification_uuid))
                else:
                    conn.execute("""
                        UPDATE guardian_notifications SET
                            status = ?,
                            provider_message_id = COALESCE(?, provider_message_id),
                            error_message = ?,
                            updated_at = ?
                        WHERE notification_uuid = ?;
                    """, (status, provider_message_id, error_message, now_iso, notification_uuid))
        return True

    @classmethod
    def get_notification_by_uuid(cls, notification_uuid: str) -> Optional[Dict[str, Any]]:
        """Retrieves single notification by UUID."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM guardian_notifications WHERE notification_uuid = ?;", (notification_uuid,))
        row = cursor.fetchone()
        return dict(row) if row else None

    @classmethod
    def get_notification_by_provider_id(cls, provider_message_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves single notification by provider message ID (e.g. Twilio MessageSid)."""
        if not provider_message_id:
            return None
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM guardian_notifications WHERE provider_message_id = ? ORDER BY id DESC LIMIT 1;", (provider_message_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    @classmethod
    def get_notifications(
        cls,
        patient_id: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieves notification records with optional filtering."""
        conn = get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM guardian_notifications WHERE 1=1"
        params: List[Any] = []

        if patient_id and patient_id != "ALL":
            query += " AND patient_id = ?"
            params.append(patient_id)

        if status_filter and status_filter != "ALL":
            if status_filter == "DEMO":
                query += " AND (status = 'DEMO_RECORDED' OR message_type = 'DEMO_PREVIEW')"
            else:
                query += " AND status = ?"
                params.append(status_filter.upper())

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]

    @classmethod
    def get_delivery_summary(cls, patient_id: Optional[str] = None) -> Dict[str, int]:
        """Calculates delivery counts (sent, delivered, failed, pending, demo)."""
        conn = get_db_connection()
        cursor = conn.cursor()

        query = "SELECT status, COUNT(*) as cnt FROM guardian_notifications"
        params = []
        if patient_id and patient_id != "ALL":
            query += " WHERE patient_id = ?"
            params.append(patient_id)
        query += " GROUP BY status"

        cursor.execute(query, params)
        counts = {"SENT": 0, "DELIVERED": 0, "FAILED": 0, "PENDING": 0, "ACCEPTED": 0, "DEMO_RECORDED": 0, "TOTAL": 0}

        for row in cursor.fetchall():
            st = str(row["status"]).upper()
            c = int(row["cnt"])
            counts["TOTAL"] += c
            if st in counts:
                counts[st] = c
            elif "DELIVER" in st:
                counts["DELIVERED"] += c
            elif "FAIL" in st:
                counts["FAILED"] += c
            elif "PEND" in st or "QUEUE" in st:
                counts["PENDING"] += c
            elif "ACCEPT" in st:
                counts["ACCEPTED"] += c

        # Dispatched total is sum of real carrier attempts (delivered + accepted + pending)
        counts["SENT"] = counts["DELIVERED"] + counts["PENDING"] + counts.get("ACCEPTED", 0)
        return counts

    @classmethod
    def check_idempotency(cls, idempotency_key: str) -> Optional[Dict[str, Any]]:
        """Checks if a notification was already queued/sent under this idempotency key."""
        if not idempotency_key:
            return None
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM guardian_notifications WHERE idempotency_key = ?;", (idempotency_key,))
        row = cursor.fetchone()
        return dict(row) if row else None

    @classmethod
    def get_audit_logs(cls, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent audit logs."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?;", (limit,))
        return [dict(r) for r in cursor.fetchall()]

    @classmethod
    def create_episode(
        cls,
        episode_id: str,
        patient_id: str,
        alert_id: Optional[str] = None,
        trigger_condition: Optional[str] = None,
        diagnosis: Optional[str] = None,
        priority: str = "HIGH PRIORITY (CRISIS / CODE BLUE)",
        state: str = "EMERGENCY",
        explanation_json: Optional[Union[str, Dict[str, Any]]] = None,
        condition: Optional[str] = None,
        smart_explanation: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Creates a persistent emergency episode record for state tracking and deduplication."""
        import json
        alert_id = alert_id or f"ALT-{episode_id}"
        trigger_condition = trigger_condition or condition or "Emergency Arrhythmia Episode"
        diagnosis = diagnosis or condition or "Cardiac Arrhythmia"
        explanation_json = explanation_json if explanation_json is not None else smart_explanation

        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        expl_str = json.dumps(explanation_json) if isinstance(explanation_json, (dict, list)) else explanation_json
        with _db_lock:
            conn = get_db_connection()
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO emergency_episodes (
                        episode_id, patient_id, alert_id, state,
                        trigger_condition, diagnosis, priority,
                        notification_sent, notification_status, notification_uuid,
                        started_at, last_updated_at, resolved_at, explanation_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, ?, NULL, ?);
                """, (
                    episode_id, patient_id, alert_id, state,
                    trigger_condition, diagnosis, priority,
                    now_iso, now_iso, expl_str
                ))
        return cls.get_episode(episode_id) or {}

    @classmethod
    def get_episode(cls, episode_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an episode by its ID."""
        import json
        if not episode_id:
            return None
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM emergency_episodes WHERE episode_id = ?;", (episode_id,))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("explanation_json"):
            try:
                d["smart_explanation"] = json.loads(d["explanation_json"])
            except Exception:
                pass
        return d

    @classmethod
    def get_active_episode(cls, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the active ongoing emergency episode for a patient, if any."""
        import json
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM emergency_episodes WHERE patient_id = ? AND state = 'EMERGENCY' ORDER BY started_at DESC LIMIT 1;",
            (patient_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("explanation_json"):
            try:
                d["smart_explanation"] = json.loads(d["explanation_json"])
            except Exception:
                pass
        return d

    @classmethod
    def get_active_episodes(cls) -> List[Dict[str, Any]]:
        """Retrieves all active emergency episodes across all patients."""
        import json
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM emergency_episodes WHERE state = 'EMERGENCY';")
        res = []
        for r in cursor.fetchall():
            d = dict(r)
            if d.get("explanation_json"):
                try:
                    d["smart_explanation"] = json.loads(d["explanation_json"])
                except Exception:
                    pass
            res.append(d)
        return res

    @classmethod
    def update_episode(
        cls,
        episode_id: str,
        trigger_condition: Optional[str] = None,
        diagnosis: Optional[str] = None,
        notification_sent: Optional[int] = None,
        notification_status: Optional[str] = None,
        notification_uuid: Optional[str] = None,
        explanation_json: Optional[Union[str, Dict[str, Any]]] = None
    ) -> bool:
        """Updates the ongoing state, condition, or notification status of an episode."""
        import json
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        updates = ["last_updated_at = ?"]
        params: List[Any] = [now_iso]

        if trigger_condition is not None:
            updates.append("trigger_condition = ?")
            params.append(trigger_condition)
        if diagnosis is not None:
            updates.append("diagnosis = ?")
            params.append(diagnosis)
        if notification_sent is not None:
            updates.append("notification_sent = ?")
            params.append(int(notification_sent))
        if notification_status is not None:
            updates.append("notification_status = ?")
            params.append(notification_status)
        if notification_uuid is not None:
            updates.append("notification_uuid = ?")
            params.append(notification_uuid)
        if explanation_json is not None:
            updates.append("explanation_json = ?")
            expl_str = json.dumps(explanation_json) if isinstance(explanation_json, (dict, list)) else explanation_json
            params.append(expl_str)

        params.append(episode_id)
        sql = f"UPDATE emergency_episodes SET {', '.join(updates)} WHERE episode_id = ?;"

        with _db_lock:
            conn = get_db_connection()
            with conn:
                conn.execute(sql, params)
        return True

    @classmethod
    def resolve_episode(cls, episode_id: str, resolved_at: Optional[str] = None) -> bool:
        """Transitions an emergency episode to RESOLVED."""
        now_iso = resolved_at or time.strftime("%Y-%m-%d %H:%M:%S")
        with _db_lock:
            conn = get_db_connection()
            with conn:
                conn.execute("""
                    UPDATE emergency_episodes SET
                        state = 'RESOLVED',
                        resolved_at = ?,
                        last_updated_at = ?
                    WHERE episode_id = ?;
                """, (now_iso, now_iso, episode_id))
        return True


# Automatically initialize database schema on module load
DatabaseManager.init_db()
