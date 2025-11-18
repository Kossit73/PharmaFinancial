"""Helper utilities for interacting with Paystack's subscription APIs."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Mapping, MutableMapping

import requests
from requests import Response, Session

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.paystack.co"


class PaystackError(RuntimeError):
    """Base error raised for Paystack integration failures."""


class PaystackNotFound(PaystackError):
    """Raised when a Paystack resource such as a customer cannot be located."""


@dataclass
class SubscriptionStatus:
    """Represents the subscription state of an email address."""

    email: str
    is_active: bool
    message: str = ""
    payload: Mapping[str, Any] | None = None


class PaystackClient:
    """Simple wrapper around the Paystack REST API."""

    def __init__(
        self,
        *,
        secret_key: str | None = None,
        base_url: str | None = None,
        plan_code: str | None = None,
        session: Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("PAYSTACK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.secret_key = secret_key or os.getenv("PAYSTACK_SECRET_KEY")
        self.plan_code = plan_code or os.getenv("PAYSTACK_PLAN_CODE")
        self.session = session or requests.Session()
        self.timeout = timeout

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#

    def has_active_subscription(self, email: str) -> SubscriptionStatus:
        """Return the subscription status for an email address."""

        email = (email or "").strip().lower()
        if not email:
            return SubscriptionStatus(email=email, is_active=False, message="Email address is required.")

        customer = self._fetch_customer(email)
        if not customer:
            LOGGER.info("Paystack customer lookup failed for %s", email)
            return SubscriptionStatus(
                email=email, is_active=False, message="No Paystack customer was found for this email."
            )

        code = (
            str(customer.get("customer_code"))
            or str(customer.get("code"))
            or str(customer.get("id") or "")
        ).strip()
        if not code:
            return SubscriptionStatus(
                email=email, is_active=False, message="Customer record is missing an identifier."
            )

        candidate_subs: list[Mapping[str, Any]] = []
        inline_subs = customer.get("subscriptions")
        if isinstance(inline_subs, list):
            candidate_subs.extend(inline_subs)

        if not candidate_subs:
            subscriptions = self._request(
                "GET",
                "/subscription",
                params={"customer": code, "status": "active", "perPage": 50},
            )
            if isinstance(subscriptions, list):
                candidate_subs.extend(subscriptions)

        LOGGER.debug("Paystack subscription payload for %s: %s", email, candidate_subs)
        active_payload: Mapping[str, Any] | None = None
        for item in candidate_subs:
            status = str(item.get("status", "")).lower()
            if status == "active":
                active_payload = item
                break

        if active_payload:
            LOGGER.info("Active Paystack subscription detected for %s", email)
            return SubscriptionStatus(
                email=email,
                is_active=True,
                message="Active Paystack subscription detected.",
                payload=active_payload,
            )

        LOGGER.info("No active Paystack subscriptions for %s; payload count=%s", email, len(candidate_subs))
        return SubscriptionStatus(
            email=email,
            is_active=False,
            message="No active Paystack subscriptions were found for this email.",
        )

    def create_subscription_checkout(
        self,
        email: str,
        *,
        amount_kobo: int | None = None,
        plan_code: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Initialise a Paystack transaction and return the checkout URL."""

        email = (email or "").strip().lower()
        if not email:
            raise PaystackError("Email address is required to initialise a Paystack transaction.")

        payload: MutableMapping[str, Any] = {"email": email}
        plan = plan_code or self.plan_code
        if plan:
            payload["plan"] = plan

        if amount_kobo:
            payload["amount"] = int(amount_kobo)

        if "amount" not in payload and "plan" not in payload:
            raise PaystackError("Either a plan code or an amount must be supplied to start a subscription.")

        if metadata:
            payload["metadata"] = metadata

        data = self._request("POST", "/transaction/initialize", json=payload)
        if not isinstance(data, Mapping):
            raise PaystackError("Unexpected response received while creating Paystack transaction.")

        checkout_url = str(data.get("authorization_url") or "").strip()
        if not checkout_url:
            raise PaystackError("Paystack response did not include an authorization URL.")
        return checkout_url

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#

    def _fetch_customer(self, email: str) -> Mapping[str, Any] | None:
        """Fetch a Paystack customer by email."""

        try:
            customer = self._request("GET", f"/customer/{email}")
            LOGGER.debug("Paystack customer response for %s: %s", email, customer)
            return customer
        except PaystackNotFound:
            LOGGER.info("Paystack customer not found: %s", email)
            return None
        except PaystackError as exc:
            LOGGER.warning("Paystack customer lookup failed: %s", exc)
            return None

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Issue an HTTP request to the Paystack API and decode the response."""

        if not self.secret_key:
            raise PaystackError(
                "PAYSTACK_SECRET_KEY is not configured. Set it to enable subscription checks."
            )

        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = kwargs.pop("headers", {})
        headers.setdefault("Authorization", f"Bearer {self.secret_key}")
        headers.setdefault("Content-Type", "application/json")

        try:
            LOGGER.debug(
                "Paystack request: %s %s headers=%s kwargs=%s",
                method.upper(),
                url,
                {k: v for k, v in headers.items() if k.lower() != "authorization"},
                {k: v for k, v in kwargs.items()},
            )
            response = self.session.request(
                method.upper(), url, headers=headers, timeout=self.timeout, **kwargs
            )
            LOGGER.debug("Paystack %s %s -> status=%s", method.upper(), url, response.status_code)
        except requests.RequestException as exc:  # pragma: no cover - network handling
            raise PaystackError("Unable to communicate with Paystack.") from exc

        return self._parse_response(response)

    def _parse_response(self, response: Response) -> Any:
        """Decode the JSON payload for a Paystack response."""

        status_code = response.status_code
        try:
            payload = response.json()
            LOGGER.debug(
                "Paystack response payload (status=%s): %s",
                status_code,
                payload,
            )
        except ValueError as exc:
            raise PaystackError("Paystack response could not be decoded as JSON.") from exc

        status_flag = bool(payload.get("status", status_code < 400))
        message = str(payload.get("message") or response.reason or "Paystack error").strip()

        if status_code == 404:
            raise PaystackNotFound(message)

        if status_code >= 400 or not status_flag:
            raise PaystackError(message)

        return payload.get("data")
