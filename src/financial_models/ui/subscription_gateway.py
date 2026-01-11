"""Gateway for reading/writing subscription records."""
from __future__ import annotations

import os
from typing import Mapping, Optional

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from ..services.subscription_store import StoredSubscriptionRecord, get_subscription_store
from ..services.paystack import SubscriptionStatus


class SubscriptionGateway:
    """Wraps subscription persistence behind the HTTP API when configured."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or os.getenv("FINANCIAL_MODELS_API_URL") or "").strip()
        self.timeout = timeout
        if self.use_api:
            if requests is None:
                raise RuntimeError(
                    "The 'requests' package is required when FINANCIAL_MODELS_API_URL is set."
                )
            self.session = requests.Session()
        else:
            self.session = None

    @property
    def use_api(self) -> bool:
        return bool(self.base_url)

    def get_record(self, email: str) -> StoredSubscriptionRecord | None:
        normalized = (email or "").strip().lower()
        if not normalized:
            return None
        if self.use_api:
            response = self.session.get(  # type: ignore[union-attr]
                f"{self.base_url.rstrip('/')}/subscriptions/status",
                params={"email": normalized},
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return StoredSubscriptionRecord(
                email=data["email"],
                is_active=bool(data["is_active"]),
                status_message=data.get("status_message", ""),
                updated_at=float(data.get("updated_at", 0.0)),
                source=data.get("source"),
                expires_at=data.get("expires_at"),
                payload=data.get("payload"),
            )

        store = get_subscription_store()
        if not store:
            return None
        return store.get_status(normalized)

    def remove_record(self, email: str) -> None:
        normalized = (email or "").strip().lower()
        if not normalized:
            return
        if self.use_api:
            response = self.session.delete(  # type: ignore[union-attr]
                f"{self.base_url.rstrip('/')}/subscriptions/status",
                params={"email": normalized},
                timeout=self.timeout,
            )
            if response.status_code not in (200, 204, 404):
                response.raise_for_status()
            return
        store = get_subscription_store()
        if store:
            store.remove_status(normalized)

    def write_status(
        self,
        status: SubscriptionStatus,
        *,
        source: str,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        normalized = (status.email or "").strip().lower()
        if not normalized:
            return
        if self.use_api:
            payload: dict[str, object] = {
                "email": normalized,
                "is_active": bool(status.is_active),
                "status_message": status.message or "",
                "payload": status.payload or None,
                "source": source,
            }
            if ttl_seconds is not None:
                payload["ttl_seconds"] = ttl_seconds
            response = self.session.post(  # type: ignore[union-attr]
                f"{self.base_url.rstrip('/')}/subscriptions/status",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return

        store = get_subscription_store()
        if store:
            store.write_status(status, source=source, ttl_seconds=ttl_seconds)
