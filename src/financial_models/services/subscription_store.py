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
from typing import Any, Mapping, Optional

from .paystack import SubscriptionStatus

LOGGER = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path(
    os.getenv("SUBSCRIPTION_STORE_PATH")
    or Path.home() / ".financial_models" / "subscriptions.db"
)
DEFAULT_STORE_URL = (os.getenv("SUBSCRIPTION_STORE_URL") or "").strip() or None
DEFAULT_CACHE_URL = (os.getenv("SUBSCRIPTION_CACHE_URL") or "").strip() or None

try:  # pragma: no cover - optional dependency
    import psycopg2
    from psycopg2 import extras as psycopg2_extras
except Exception:  # pragma: no cover - allow importing without psycopg2 installed
    psycopg2 = None  # type: ignore
    psycopg2_extras = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import redis
except Exception:  # pragma: no cover - allow importing without redis installed
    redis = None  # type: ignore


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


def _parse_payload(payload_data: Any) -> Mapping[str, Any] | None:
    if isinstance(payload_data, str) and payload_data:
        try:
            return json.loads(payload_data)
        except json.JSONDecodeError:
            return {"raw": payload_data}
    return None


def _json_dump(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return json.dumps({"repr": repr(value)})


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
        payload = _parse_payload(payload_data)

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


class PostgresSubscriptionStore:
    """Postgres-backed persistence for subscription states."""

    def __init__(self, db_url: str) -> None:
        if psycopg2 is None:
            raise RuntimeError("psycopg2 is required when SUBSCRIPTION_STORE_URL is set.")
        self.db_url = db_url
        self._lock = threading.Lock()
        self._conn = psycopg2.connect(self.db_url)
        self._conn.autocommit = True
        self._ensure_schema()

    def write_status(
        self,
        status: SubscriptionStatus,
        *,
        source: str | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        normalized = _normalize_email(status.email)
        if not normalized:
            return
        expires_at = None
        if ttl_seconds:
            expires_at = time.time() + float(ttl_seconds)
        payload = _serialise_payload(status.payload if status.payload else None)
        with self._lock, self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
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
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (email) DO UPDATE SET
                        is_active=EXCLUDED.is_active,
                        status_message=EXCLUDED.status_message,
                        payload=EXCLUDED.payload,
                        updated_at=EXCLUDED.updated_at,
                        source=EXCLUDED.source,
                        ttl_seconds=EXCLUDED.ttl_seconds,
                        expires_at=EXCLUDED.expires_at
                    """,
                    (
                        normalized,
                        bool(status.is_active),
                        status.message,
                        payload,
                        time.time(),
                        source,
                        ttl_seconds,
                        expires_at,
                    ),
                )

    def get_status(self, email: str) -> StoredSubscriptionRecord | None:
        normalized = _normalize_email(email)
        if not normalized:
            return None
        with self._lock:
            with self._conn.cursor(cursor_factory=psycopg2_extras.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT email, is_active, status_message, payload, updated_at, source, expires_at
                    FROM subscription_status
                    WHERE email = %s
                    """,
                    (normalized,),
                )
                row = cur.fetchone()
        if not row:
            return None
        payload = _parse_payload(row["payload"])
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
        normalized = _normalize_email(email)
        if not normalized:
            return
        with self._lock, self._conn:
            with self._conn.cursor() as cur:
                cur.execute("DELETE FROM subscription_status WHERE email = %s", (normalized,))

    def record_event(
        self,
        email: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        normalized = _normalize_email(email) or "unknown"
        encoded_payload = _serialise_payload(payload)
        with self._lock, self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO subscription_events (email, event_type, payload, recorded_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (normalized, event_type, encoded_payload, time.time()),
                )

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        with self._lock, self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS subscription_status (
                        email TEXT PRIMARY KEY,
                        is_active BOOLEAN NOT NULL,
                        status_message TEXT,
                        payload TEXT,
                        updated_at DOUBLE PRECISION NOT NULL,
                        source TEXT,
                        ttl_seconds DOUBLE PRECISION,
                        expires_at DOUBLE PRECISION
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS subscription_events (
                        id SERIAL PRIMARY KEY,
                        email TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        payload TEXT,
                        recorded_at DOUBLE PRECISION NOT NULL
                    )
                    """
                )


class RedisSubscriptionCache:
    """Redis-backed cache for subscription states."""

    def __init__(self, cache_url: str) -> None:
        if redis is None:
            raise RuntimeError("redis is required when SUBSCRIPTION_CACHE_URL is set.")
        self.cache_url = cache_url
        self._client = redis.Redis.from_url(cache_url)

    def _key(self, email: str) -> str:
        return f"subscription_status:{email}"

    def write_status(
        self,
        record: StoredSubscriptionRecord,
        *,
        ttl_seconds: float | None = None,
    ) -> None:
        data = {
            "email": record.email,
            "is_active": record.is_active,
            "status_message": record.status_message,
            "payload": record.payload,
            "updated_at": record.updated_at,
            "source": record.source,
            "expires_at": record.expires_at,
        }
        payload = _json_dump(data)
        key = self._key(record.email)
        if ttl_seconds and ttl_seconds > 0:
            self._client.setex(key, int(ttl_seconds), payload)
        else:
            self._client.set(key, payload)

    def get_status(self, email: str) -> StoredSubscriptionRecord | None:
        normalized = _normalize_email(email)
        if not normalized:
            return None
        payload = self._client.get(self._key(normalized))
        if not payload:
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return StoredSubscriptionRecord(
            email=data.get("email", normalized),
            is_active=bool(data.get("is_active")),
            status_message=str(data.get("status_message") or ""),
            payload=data.get("payload"),
            updated_at=float(data.get("updated_at") or 0.0),
            source=data.get("source"),
            expires_at=float(data.get("expires_at")) if data.get("expires_at") is not None else None,
        )

    def remove_status(self, email: str) -> None:
        normalized = _normalize_email(email)
        if not normalized:
            return
        self._client.delete(self._key(normalized))


class CompositeSubscriptionStore:
    """Wrap a persistent store with an optional Redis cache."""

    def __init__(
        self,
        store: SubscriptionStore | PostgresSubscriptionStore,
        cache: RedisSubscriptionCache | None = None,
    ) -> None:
        self._store = store
        self._cache = cache

    def write_status(
        self,
        status: SubscriptionStatus,
        *,
        source: str | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self._store.write_status(status, source=source, ttl_seconds=ttl_seconds)
        if self._cache and status.email:
            record = self._store.get_status(status.email)
            if record:
                self._cache.write_status(record, ttl_seconds=ttl_seconds)

    def get_status(self, email: str) -> StoredSubscriptionRecord | None:
        if self._cache:
            cached = self._cache.get_status(email)
            if cached:
                return cached
        record = self._store.get_status(email)
        if record and self._cache:
            ttl_seconds = None
            if record.expires_at is not None:
                ttl_seconds = max(0.0, float(record.expires_at) - time.time())
            self._cache.write_status(record, ttl_seconds=ttl_seconds)
        return record

    def remove_status(self, email: str) -> None:
        self._store.remove_status(email)
        if self._cache:
            self._cache.remove_status(email)

    def record_event(
        self,
        email: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        self._store.record_event(email, event_type, payload)

    def close(self) -> None:
        self._store.close()


_GLOBAL_STORE: SubscriptionStore | PostgresSubscriptionStore | CompositeSubscriptionStore | None = None
_GLOBAL_LOCK = threading.Lock()


def _build_store() -> SubscriptionStore | PostgresSubscriptionStore:
    if DEFAULT_STORE_URL:
        return PostgresSubscriptionStore(DEFAULT_STORE_URL)
    return SubscriptionStore()


def _build_cache() -> RedisSubscriptionCache | None:
    if DEFAULT_CACHE_URL:
        return RedisSubscriptionCache(DEFAULT_CACHE_URL)
    return None


def get_subscription_store(
    db_path: str | Path | None = None,
) -> SubscriptionStore | PostgresSubscriptionStore | CompositeSubscriptionStore | None:
    """Return a cached store instance to share across modules."""

    global _GLOBAL_STORE
    if db_path:
        return SubscriptionStore(db_path)

    with _GLOBAL_LOCK:
        if _GLOBAL_STORE is None:
            try:
                store = _build_store()
                cache = _build_cache()
                _GLOBAL_STORE = CompositeSubscriptionStore(store, cache) if cache else store
            except Exception as exc:  # pragma: no cover - filesystem failures
                LOGGER.warning("Unable to initialise subscription store: %s", exc)
                _GLOBAL_STORE = None
        return _GLOBAL_STORE
