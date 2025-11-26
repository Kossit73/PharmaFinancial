"""Persistent subscription status tracking shared between UI sessions and webhooks."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from .paystack import SubscriptionStatus

LOGGER = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path(
    os.getenv("SUBSCRIPTION_STORE_PATH")
    or Path.home() / ".pharma_financial" / "subscriptions.db"
)


def _normalize_email(value: str | None) -> str:
    """Return a canonical email representation used as the primary key."""

    return (value or "").strip().lower()


def _serialise_payload(payload: Mapping[str, Any] | None) -> str | None:
    """JSON encode payloads while tolerating non-serialisable values."""

    if payload is None:
        return None
    try:
        return json.dumps(payload, default=str)
    except (TypeError, ValueError):
        return json.dumps({"repr": repr(payload)})


@dataclass
class StoredSubscriptionRecord:
    """Represents the last known subscription state stored on disk."""

    email: str
    is_active: bool
    status_message: str
    updated_at: float
    source: str | None = None
    expires_at: float | None = None
    payload: Mapping[str, Any] | None = None

    def is_expired(self) -> bool:
        """Return True when the stored record should no longer be trusted."""

        return bool(self.expires_at and time.time() >= float(self.expires_at))


class SubscriptionStore:
    """Lightweight SQLite-backed persistence for subscription states."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.path = Path(db_path or DEFAULT_STORE_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#
    def write_status(
        self,
        status: SubscriptionStatus,
        *,
        source: str | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        """Insert or update the subscription state for ``status.email``."""

        normalized = _normalize_email(status.email)
        if not normalized:
            return

        expires_at = None
        if ttl_seconds:
            expires_at = time.time() + float(ttl_seconds)

        payload = _serialise_payload(status.payload if status.payload else None)

        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO subscription_status (
                    email,
                    is_active,
                    status_message,
                    payload,
                    updated_at,
                    source,
                    ttl_seconds,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    is_active=excluded.is_active,
                    status_message=excluded.status_message,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at,
                    source=excluded.source,
                    ttl_seconds=excluded.ttl_seconds,
                    expires_at=excluded.expires_at
                """,
                (
                    normalized,
                    1 if status.is_active else 0,
                    status.message,
                    payload,
                    time.time(),
                    source,
                    ttl_seconds,
                    expires_at,
                ),
            )

    def get_status(self, email: str) -> StoredSubscriptionRecord | None:
        """Return the stored subscription record for ``email`` when present."""

        normalized = _normalize_email(email)
        if not normalized:
            return None

        with self._lock:
            row = self._conn.execute(
                """
                SELECT email, is_active, status_message, payload, updated_at, source, expires_at
                FROM subscription_status
                WHERE email = ?
                """,
                (normalized,),
            ).fetchone()

        if not row:
            return None

        payload_data = row["payload"]
        payload = None
        if isinstance(payload_data, str) and payload_data:
            try:
                payload = json.loads(payload_data)
            except json.JSONDecodeError:
                payload = {"raw": payload_data}

        return StoredSubscriptionRecord(
            email=row["email"],
            is_active=bool(row["is_active"]),
            status_message=row["status_message"] or "",
            payload=payload,
            updated_at=float(row["updated_at"] or 0.0),
            source=row["source"],
            expires_at=float(row["expires_at"]) if row["expires_at"] is not None else None,
        )

    def remove_status(self, email: str) -> None:
        """Delete any stored status for ``email``."""

        normalized = _normalize_email(email)
        if not normalized:
            return

        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM subscription_status WHERE email = ?",
                (normalized,),
            )

    def record_event(
        self,
        email: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Append a webhook event row for future auditing."""

        normalized = _normalize_email(email) or "unknown"
        encoded_payload = _serialise_payload(payload)

        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO subscription_events (email, event_type, payload, recorded_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalized, event_type, encoded_payload, time.time()),
            )

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _ensure_schema(self) -> None:
        """Create the SQLite tables when they do not yet exist."""

        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS subscription_status (
                    email TEXT PRIMARY KEY,
                    is_active INTEGER NOT NULL,
                    status_message TEXT,
                    payload TEXT,
                    updated_at REAL NOT NULL,
                    source TEXT,
                    ttl_seconds REAL,
                    expires_at REAL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS subscription_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT,
                    recorded_at REAL NOT NULL
                )
                """
            )


_GLOBAL_STORE: SubscriptionStore | None = None
_GLOBAL_LOCK = threading.Lock()


def get_subscription_store(db_path: str | Path | None = None) -> SubscriptionStore | None:
    """Return a cached store instance to share across modules."""

    global _GLOBAL_STORE
    if db_path:
        return SubscriptionStore(db_path)

    with _GLOBAL_LOCK:
        if _GLOBAL_STORE is None:
            try:
                _GLOBAL_STORE = SubscriptionStore()
            except Exception as exc:  # pragma: no cover - filesystem failures
                LOGGER.warning("Unable to initialise subscription store: %s", exc)
                _GLOBAL_STORE = None
        return _GLOBAL_STORE
