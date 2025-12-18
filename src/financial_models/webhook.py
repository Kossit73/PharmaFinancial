"""Backward-compatible shim for the webhook server.

The webhook implementation now lives under ``financial_models.services.webhook``.
This module re-exports its public API so existing entry points keep working.
"""

from __future__ import annotations

from financial_models.services.webhook import (  # noqa: F401
    ACTIVATION_EVENTS,
    DEFAULT_ACTIVE_TTL_SECONDS,
    PAYSTACK_SIGNATURE_HEADER,
    REVOCATION_EVENTS,
    PaystackWebhookProcessor,
    create_webhook_app,
    main,
)

__all__ = [
    "ACTIVATION_EVENTS",
    "DEFAULT_ACTIVE_TTL_SECONDS",
    "PAYSTACK_SIGNATURE_HEADER",
    "PaystackWebhookProcessor",
    "REVOCATION_EVENTS",
    "create_webhook_app",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    main()
