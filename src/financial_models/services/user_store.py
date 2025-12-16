"""SQLite-backed user registry for authentication."""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from passlib.context import CryptContext

LOGGER = logging.getLogger(__name__)

DEFAULT_USER_DB_PATH = Path(
    os.getenv("FINANCIAL_MODELS_USER_DB") or Path.home() / ".financial_models" / "users.db"
)


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@dataclass
class UserRecord:
    """Represents a stored user."""

    id: int
    email: str
    name: str | None
    provider: str
    created_at: float


class UserStore:
    """Thread-safe SQLite-backed user store."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.path = Path(db_path or DEFAULT_USER_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def create_user(self, email: str, password: str, name: str | None = None, provider: str = "local") -> UserRecord:
        normalized = (email or "").strip().lower()
        if not normalized:
            raise ValueError("Email is required.")
        if provider == "local" and not password:
            raise ValueError("Password is required for local users.")
        hashed = pwd_context.hash(password) if provider == "local" else ""
        created_at = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO users (email, password_hash, name, provider, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized, hashed, name, provider, created_at),
            )
            user_id = self._conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return UserRecord(id=int(user_id), email=normalized, name=name, provider=provider, created_at=created_at)

    def get_user(self, email: str) -> Optional[UserRecord]:
        normalized = (email or "").strip().lower()
        if not normalized:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT id, email, password_hash, name, provider, created_at FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
        if not row:
            return None
        return UserRecord(
            id=int(row["id"]),
            email=row["email"],
            name=row["name"],
            provider=row["provider"],
            created_at=float(row["created_at"]),
        )

    def verify_user(self, email: str, password: str) -> Optional[UserRecord]:
        normalized = (email or "").strip().lower()
        if not normalized:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT id, email, password_hash, name, provider, created_at FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
        if not row or not row["password_hash"]:
            return None
        if not pwd_context.verify(password, row["password_hash"]):
            return None
        return UserRecord(
            id=int(row["id"]),
            email=row["email"],
            name=row["name"],
            provider=row["provider"],
            created_at=float(row["created_at"]),
        )

    def ensure_user(self, email: str, name: str | None, provider: str) -> UserRecord:
        """Find or create a user for social login."""

        normalized = (email or "").strip().lower()
        if not normalized:
            raise ValueError("Email is required.")
        existing = self.get_user(normalized)
        if existing:
            return existing
        return self.create_user(email=normalized, password="social-login", name=name, provider=provider)

    def update_user(self, email: str, *, name: str | None = None, password: str | None = None) -> UserRecord:
        """Update a user's display name and/or password."""

        normalized = (email or "").strip().lower()
        if not normalized:
            raise ValueError("Email is required.")
        hashed = None
        if password:
            hashed = pwd_context.hash(password)
        with self._lock, self._conn:
            if hashed is not None:
                self._conn.execute(
                    "UPDATE users SET name = COALESCE(?, name), password_hash = ? WHERE email = ?",
                    (name, hashed, normalized),
                )
            else:
                self._conn.execute(
                    "UPDATE users SET name = COALESCE(?, name) WHERE email = ?",
                    (name, normalized),
                )
            row = self._conn.execute(
                "SELECT id, email, password_hash, name, provider, created_at FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
        if not row:
            raise ValueError("User not found.")
        return UserRecord(
            id=int(row["id"]),
            email=row["email"],
            name=row["name"],
            provider=row["provider"],
            created_at=float(row["created_at"]),
        )

    def delete_user(self, email: str) -> None:
        normalized = (email or "").strip().lower()
        if not normalized:
            return
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM users WHERE email = ?", (normalized,))

    def list_users(self) -> list[UserRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, email, name, provider, created_at FROM users ORDER BY created_at ASC"
            ).fetchall()
        records: list[UserRecord] = []
        for row in rows:
            records.append(
                UserRecord(
                    id=int(row["id"]),
                    email=row["email"],
                    name=row["name"],
                    provider=row["provider"],
                    created_at=float(row["created_at"]),
                )
            )
        return records

    def _ensure_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT,
                    name TEXT,
                    provider TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )


_GLOBAL_STORE: UserStore | None = None
_GLOBAL_LOCK = threading.Lock()


def get_user_store(db_path: str | Path | None = None) -> UserStore:
    global _GLOBAL_STORE
    if db_path:
        return UserStore(db_path)
    with _GLOBAL_LOCK:
        if _GLOBAL_STORE is None:
            _GLOBAL_STORE = UserStore()
        return _GLOBAL_STORE
