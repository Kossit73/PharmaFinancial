"""Minimal HTTP server used to process Paystack webhook events."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from .paystack import SubscriptionStatus
from .subscription_store import SubscriptionStore

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


class PaystackWebhookHandler(BaseHTTPRequestHandler):
    """HTTP handler that validates and records Paystack webhook events."""

    subscription_store: SubscriptionStore | None = None
    signing_secret: str | None = None

    def do_POST(self) -> None:  # noqa: D401 - HTTP verb handler
        if self.path not in ("/", "/paystack"):
            self._send(HTTPStatus.NOT_FOUND, {"status": "error", "message": "Not found"})
            return

        LOGGER.info("Incoming Paystack webhook: path=%s headers=%s", self.path, dict(self.headers))
        length = int(self.headers.get("content-length") or 0)
        raw_body = self.rfile.read(length)
        LOGGER.debug("Webhook raw payload: %s", raw_body)

        if not self._valid_signature(raw_body):
            LOGGER.warning("Rejecting Paystack webhook due to invalid signature.")
            self._send(
                HTTPStatus.UNAUTHORIZED,
                {"status": "error", "message": "Invalid webhook signature"},
            )
            return

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except ValueError:
            self._send(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "Invalid JSON payload"})
            return

        LOGGER.info("Decoded webhook payload: %s", payload)

        event_name = str(payload.get("event") or "").strip().lower()
        data = payload.get("data") if isinstance(payload, Mapping) else {}
        email = self._extract_email(data)

        if not event_name:
            self._send(HTTPStatus.OK, {"status": "ignored", "message": "Missing event name"})
            return

        store = self.subscription_store
        if not store:
            LOGGER.warning("Subscription store not configured; webhook ignored.")
            self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"status": "error", "message": "Store not configured"})
            return

        status = None
        if email:
            status = self._record_status(store, event_name, email, data)
        else:
            LOGGER.info("Webhook %s missing email; recording event only", event_name)
            store.record_event("unknown", event_name, data if isinstance(data, Mapping) else None)

        self._send(
            HTTPStatus.OK,
            {
                "status": "ok",
                "event": event_name,
                "email": email,
                "state": status,
            },
        )

    # ------------------------------------------------------------------#
    # Helpers
    # ------------------------------------------------------------------#
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

    def _valid_signature(self, raw_body: bytes) -> bool:
        secret = (self.signing_secret or "").strip()
        signature = self.headers.get(PAYSTACK_SIGNATURE_HEADER, "").strip()
        if not secret:
            return True  # Developers may disable signature checks for local testing
        if not signature:
            return False
        digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha512).hexdigest()
        return hmac.compare_digest(digest, signature)

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

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - integrate with logging
        LOGGER.info("Webhook: " + fmt, *args)

    def _send(self, status_code: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _build_handler(store: SubscriptionStore, secret: str | None) -> type[PaystackWebhookHandler]:
    class _Handler(PaystackWebhookHandler):
        subscription_store = store
        signing_secret = secret

    return _Handler


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
        help="HMAC secret used to verify Paystack webhook signatures.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - command-line utility
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    store = SubscriptionStore(args.db)
    handler = _build_handler(store, args.secret)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    LOGGER.info("Listening for Paystack webhooks on %s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down webhook server.")
    finally:
        store.close()
        server.server_close()


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
