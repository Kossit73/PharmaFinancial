"""FastAPI application for processing Paystack webhook events."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
from typing import Any, Mapping, MutableMapping

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI, HTTPException, Request
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import uvicorn
except Exception:  # pragma: no cover
    uvicorn = None  # type: ignore

from .services.paystack import SubscriptionStatus
from .services.subscription_store import SubscriptionStore, get_subscription_store

LOGGER = logging.getLogger(__name__)

PAYSTACK_SIGNATURE_HEADER = "x-paystack-signature"
REVOCATION_EVENTS = {
    "subscription.disable",
    "subscription.not_renew",
    "invoice.payment_failed",
    "charge.failed",
}
ACTIVATION_EVENTS = {
    "subscription.create",
    "subscription.enable",
    "charge.success",
    "invoice.payment_success",
}
DEFAULT_ACTIVE_TTL_SECONDS = float(os.getenv("SUBSCRIPTION_ACTIVE_TTL_SECONDS", 10 * 60))


def _normalize_email(value: str | None) -> str:
    return (value or "").strip().lower()


class PaystackWebhookProcessor:
    """Encapsulates webhook parsing and subscription persistence logic."""

    def __init__(self, store: SubscriptionStore | None, signing_secret: str | None) -> None:
        self.store = store
        self.signing_secret = (signing_secret or "").strip()

    def handle(self, body: bytes, headers: Mapping[str, str]) -> Mapping[str, Any]:
        payload = self._parse_payload(body, headers)
        event_name = str(payload.get("event") or "").strip().lower()
        if not event_name:
            return {"status": "ignored", "message": "Missing event name"}

        data = payload.get("data") if isinstance(payload, Mapping) else {}
        email = self._extract_email(data if isinstance(data, Mapping) else None)

        store = self.store
        if not store:
            raise HTTPException(status_code=503, detail="Subscription store unavailable")

        status = None
        if email:
            status = self._record_status(store, event_name, email, data if isinstance(data, Mapping) else None)
        else:
            LOGGER.info("Webhook %s missing email; recording event only", event_name)
            store.record_event("unknown", event_name, data if isinstance(data, Mapping) else None)
        return {"status": "ok", "event": event_name, "email": email, "state": status}

    # ------------------------------------------------------------------ helpers
    def _parse_payload(self, body: bytes, headers: Mapping[str, str]) -> Mapping[str, Any]:
        signature = headers.get(PAYSTACK_SIGNATURE_HEADER, "")
        if self.signing_secret and not signature:
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        if self.signing_secret:
            digest = hmac.new(self.signing_secret.encode("utf-8"), body, hashlib.sha512).hexdigest()
            if not hmac.compare_digest(digest, signature.strip()):
                raise HTTPException(status_code=401, detail="Invalid webhook signature")
        try:
            return json.loads(body.decode("utf-8"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    def _record_status(
        self,
        store: SubscriptionStore,
        event_name: str,
        email: str,
        payload: Mapping[str, Any] | None,
    ) -> str:
        normalized = _normalize_email(email)
        status_payload = payload if isinstance(payload, Mapping) else None
        state = "ignored"

        if event_name in REVOCATION_EVENTS:
            status = SubscriptionStatus(
                email=normalized,
                is_active=False,
                message=f"Revoked via {event_name}",
                payload=status_payload,
            )
            store.write_status(status, source=f"webhook:{event_name}", ttl_seconds=None)
            state = "revoked"
            LOGGER.info("Revoked subscription for %s via %s", normalized, event_name)
        elif event_name in ACTIVATION_EVENTS:
            status = SubscriptionStatus(
                email=normalized,
                is_active=True,
                message=f"Activated via {event_name}",
                payload=status_payload,
            )
            store.write_status(
                status,
                source=f"webhook:{event_name}",
                ttl_seconds=DEFAULT_ACTIVE_TTL_SECONDS,
            )
            state = "active"
            LOGGER.info("Activated subscription for %s via %s", normalized, event_name)
        else:
            LOGGER.debug("Ignoring webhook %s for %s; not actionable", event_name, normalized)

        store.record_event(normalized or "unknown", event_name, status_payload)
        return state

    def _extract_email(self, payload: Mapping[str, Any] | None) -> str:
        if not isinstance(payload, Mapping):
            return ""
        email = payload.get("email") or ""
        customer = payload.get("customer")
        if not email and isinstance(customer, Mapping):
            email = customer.get("email") or ""
        elif not email and isinstance(customer, str):
            email = customer
        metadata = payload.get("metadata")
        if not email and isinstance(metadata, Mapping):
            email = metadata.get("email") or metadata.get("user_email") or ""
        return _normalize_email(str(email))


def create_webhook_app(
    *,
    store: SubscriptionStore | None = None,
    signing_secret: str | None = None,
) -> FastAPI:
    """Create a FastAPI instance that processes Paystack webhooks."""

    if FastAPI is None:  # pragma: no cover - dependency missing
        raise RuntimeError("FastAPI is not installed; install optional dependencies to use the webhook server.")

    store = store or get_subscription_store()
    processor = PaystackWebhookProcessor(store, signing_secret or os.getenv("PAYSTACK_WEBHOOK_SECRET"))
    app = FastAPI(
        title="Paystack Webhook Receiver",
        version="1.0.0",
        description="Validates Paystack signatures and updates the shared subscription store.",
    )

    @app.post("/paystack")
    async def handle_webhook(request: Request) -> MutableMapping[str, Any]:
        body = await request.body()
        headers = {k.lower(): v for k, v in request.headers.items()}
        return processor.handle(body, headers)

    @app.get("/health")
    def healthcheck() -> Mapping[str, str]:
        return {"status": "ok"}

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paystack webhook receiver.")
    parser.add_argument("--host", default=os.getenv("PAYSTACK_WEBHOOK_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PAYSTACK_WEBHOOK_PORT", "8080")))
    parser.add_argument(
        "--db",
        default=os.getenv("SUBSCRIPTION_STORE_PATH"),
        help="Path to the subscription SQLite store (defaults to ~/.pharma_financial/subscriptions.db).",
    )
    parser.add_argument(
        "--secret",
        default=os.getenv("PAYSTACK_WEBHOOK_SECRET") or os.getenv("PAYSTACK_SECRET_KEY"),
        help="Webhook signing secret. Defaults to PAYSTACK_WEBHOOK_SECRET or PAYSTACK_SECRET_KEY.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point used by ``python -m pharma_financial.webhook``."""

    if FastAPI is None or uvicorn is None:
        raise SystemExit("FastAPI and uvicorn are required to run the webhook server.")

    args = _parse_args()
    store = get_subscription_store(args.db)
    app = create_webhook_app(store=store, signing_secret=args.secret)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
