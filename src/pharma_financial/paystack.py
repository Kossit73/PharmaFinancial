"""Helper utilities for interacting with Paystack's subscription APIs."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Mapping, MutableMapping

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.paystack.co"


class PaystackError(RuntimeError):
    """Base error raised for Paystack integration failures."""


class PaystackAuthError(PaystackError):
    """Raised when Paystack authentication fails."""


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
        default_amount_kobo: int | None = None,
        callback_url: str | None = None,
        cancel_action_url: str | None = None,
        default_metadata: Mapping[str, Any] | None = None,
        session: Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("PAYSTACK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.secret_key = secret_key or os.getenv("PAYSTACK_SECRET_KEY")
        self.plan_code = plan_code or os.getenv("PAYSTACK_PLAN_CODE")
        self.default_amount_kobo = self._normalize_amount(
            default_amount_kobo or os.getenv("PAYSTACK_PLAN_AMOUNT_KOBO")
        )
        self.callback_url = (callback_url or os.getenv("PAYSTACK_CALLBACK_URL") or "").strip() or None
        cancel_source = cancel_action_url or os.getenv("PAYSTACK_CANCEL_ACTION_URL")
        self.cancel_action_url = (cancel_source or self.callback_url or "").strip() or None
        self.default_metadata: dict[str, Any] = {}
        if default_metadata:
            self.default_metadata.update(dict(default_metadata))
        if session is None:
            self.session = self._build_session()
            self._owns_session = True
        else:
            self.session = session
            self._owns_session = False
        self.timeout = timeout
        self._plan_amount_cache: dict[str, int] = {}

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#

    def has_active_subscription(self, email: str) -> SubscriptionStatus:
        """Return the subscription status for an email address."""

        return self.has_active_subscription_for_email(email)

    def create_subscription_checkout(
        self,
        email: str,
        *,
        amount_kobo: int | None = None,
        plan_code: str | None = None,
        callback_url: str | None = None,
        cancel_action_url: str | None = None,
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

        amount_value = self._resolve_amount(amount_kobo, plan)
        if amount_value:
            payload["amount"] = amount_value

        resolved_callback = (callback_url or self.callback_url or "").strip()
        if resolved_callback:
            payload["callback_url"] = resolved_callback

        metadata_payload: dict[str, Any] = {}
        if self.default_metadata:
            metadata_payload.update(self.default_metadata)
        if metadata:
            metadata_payload.update(dict(metadata))

        resolved_cancel_source = cancel_action_url or self.cancel_action_url or resolved_callback
        resolved_cancel = (resolved_cancel_source or "").strip()
        if resolved_cancel:
            metadata_payload.setdefault("cancel_action", resolved_cancel)

        if "amount" not in payload and "plan" not in payload:
            raise PaystackError("Either a plan code or an amount must be supplied to start a subscription.")

        if metadata_payload:
            payload["metadata"] = metadata_payload

        data = self._request("POST", "/transaction/initialize", json=payload)
        if not isinstance(data, Mapping):
            raise PaystackError("Unexpected response received while creating Paystack transaction.")

        checkout_url = str(data.get("authorization_url") or "").strip()
        if not checkout_url:
            raise PaystackError("Paystack response did not include an authorization URL.")
        return checkout_url

    def get_customer_by_email(self, email: str) -> Mapping[str, Any] | None:
        """Return the first Paystack customer record that matches ``email``."""

        normalized = (email or "").strip().lower()
        if not normalized:
            return None

        try:
            data = self._request("GET", "/customer", params={"email": normalized})
        except PaystackError as exc:
            LOGGER.warning("Paystack customer search failed for %s: %s", normalized, exc)
            return None

        if isinstance(data, list):
            for item in data:
                if str(item.get("email", "")).strip().lower() == normalized:
                    return item
            if data:
                return data[0]
            return None

        if isinstance(data, Mapping):
            return data

        return None

    def get_subscriptions_for_customer(self, customer_identifier: str) -> list[Mapping[str, Any]]:
        """Return subscriptions associated with a Paystack customer identifier."""

        code = (customer_identifier or "").strip()
        if not code:
            return []

        per_page = 50
        page = 1
        subscriptions: list[Mapping[str, Any]] = []

        while True:
            try:
                chunk = self._request(
                    "GET",
                    "/subscription",
                    params={"customer": code, "perPage": per_page, "page": page},
                )
            except PaystackNotFound:
                break
            except PaystackError as exc:
                LOGGER.warning("Could not fetch subscriptions for customer %s: %s", code, exc)
                break

            batch = self._normalize_subscription_payload(chunk)
            if not batch:
                break
            subscriptions.extend(batch)

            if len(batch) < per_page:
                break
            page += 1

        return subscriptions

    def _normalize_subscription_payload(self, payload: Any) -> list[Mapping[str, Any]]:
        """Convert subscription API responses to a consistent list."""

        if payload is None:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, Mapping):
            for key in ("data", "subscriptions", "items"):
                maybe = payload.get(key)
                if isinstance(maybe, list):
                    return maybe
            return [payload]
        return []

    def has_active_subscription_for_email(self, email: str) -> SubscriptionStatus:
        """Return the subscription status for ``email`` using helper lookups."""

        normalized = (email or "").strip().lower()
        if not normalized:
            return SubscriptionStatus(email=normalized, is_active=False, message="Email required")

        customer = self.get_customer_by_email(normalized)
        if not customer:
            return SubscriptionStatus(email=normalized, is_active=False, message="No Paystack customer found")

        customer_id = str(customer.get("id") or "")
        customer_code = str(
            customer.get("customer_code") or customer.get("code") or customer.get("id") or ""
        ).strip()
        if not customer_code:
            return SubscriptionStatus(email=normalized, is_active=False, message="Customer has no identifier")

        subscriptions: list[Mapping[str, Any]] = []

        detail = self.get_customer_detail(customer_code)
        if detail:
            inline = detail.get("subscriptions")
            if isinstance(inline, list):
                subscriptions.extend(inline)

        if not subscriptions and customer_id and customer_id != customer_code:
            detail = self.get_customer_detail(customer_id)
            if detail:
                inline = detail.get("subscriptions")
                if isinstance(inline, list):
                    subscriptions.extend(inline)

        if not subscriptions and customer_code:
            subscriptions = self.get_subscriptions_for_customer(customer_code)
        if (
            not subscriptions
            and customer_id
            and customer_id != customer_code
        ):
            subscriptions = self.get_subscriptions_for_customer(customer_id)

        LOGGER.debug("Found %s subscriptions for Paystack customer %s", len(subscriptions), customer_code)

        for sub in subscriptions:
            status = str(sub.get("status", "")).strip().lower()
            if status == "active":
                LOGGER.info("Active Paystack subscription detected for %s", normalized)
                return SubscriptionStatus(
                    email=normalized, is_active=True, message="Active subscription found", payload=sub
                )

        try:
            transactions = self._request("GET", "/transaction", params={"customer": customer_code, "perPage": 50})
        except PaystackError as exc:
            LOGGER.debug("Unable to fetch Paystack transactions for %s: %s", customer_code, exc)
            transactions = None

        if isinstance(transactions, list):
            for tx in transactions:
                if str(tx.get("status", "")).strip().lower() == "success":
                    LOGGER.info("Successful Paystack transaction detected for %s", normalized)
                    return SubscriptionStatus(
                        email=normalized,
                        is_active=True,
                        message="Successful transaction found (non-subscription)",
                        payload=tx,
                    )

        LOGGER.info("No active Paystack subscriptions or transactions for %s", normalized)
        return SubscriptionStatus(
            email=normalized,
            is_active=False,
            message="No active subscription or recent successful payment found",
        )

    def get_customer_detail(self, customer_identifier: str) -> Mapping[str, Any] | None:
        """Return the detailed Paystack customer record for ``customer_identifier``."""

        identifier = (customer_identifier or "").strip()
        if not identifier:
            return None

        try:
            detail = self._request("GET", f"/customer/{identifier}")
        except PaystackNotFound:
            return None
        except PaystackError as exc:
            LOGGER.warning("Paystack customer detail lookup failed for %s: %s", identifier, exc)
            return None

        if isinstance(detail, Mapping):
            return detail
        return None

    def verify_transaction(self, reference: str) -> Mapping[str, Any] | None:
        """Verify a Paystack transaction reference string."""

        ref = (reference or "").strip()
        if not ref:
            return None

        try:
            data = self._request("GET", f"/transaction/verify/{ref}")
        except PaystackNotFound:
            return None
        except PaystackError as exc:
            LOGGER.warning("Paystack transaction verification failed: %s", exc)
            return None

        if isinstance(data, Mapping):
            return data
        return None

    def close(self) -> None:
        """Close the underlying HTTP session if this client owns it."""

        if getattr(self, "_owns_session", False) and self.session:
            self.session.close()

    def __enter__(self) -> "PaystackClient":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        self.close()

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    @staticmethod
    def _normalize_amount(value: Any) -> int | None:
        """Return a positive integer representation of ``value`` when possible."""

        if value is None:
            return None
        try:
            amount = int(value)
        except (TypeError, ValueError):
            return None
        if amount <= 0:
            return None
        return amount

    def _resolve_amount(self, amount_kobo: int | None, plan_code: str | None) -> int | None:
        """Return the checkout amount in kobo for the request."""

        explicit = self._normalize_amount(amount_kobo)
        if explicit:
            return explicit

        if self.default_amount_kobo:
            return self.default_amount_kobo

        plan = (plan_code or "").strip()
        if plan:
            inferred = self._plan_amount(plan)
            if inferred:
                return inferred
        return None

    def _plan_amount(self, plan_code: str) -> int | None:
        """Retrieve (and cache) the configured Paystack plan amount."""

        cached = self._plan_amount_cache.get(plan_code)
        if cached:
            return cached

        try:
            plan = self._request("GET", f"/plan/{plan_code}")
        except PaystackError as exc:
            LOGGER.warning("Unable to retrieve Paystack plan %s: %s", plan_code, exc)
            return None

        amount = None
        if isinstance(plan, Mapping):
            amount = plan.get("amount")

        normalized = self._normalize_amount(amount)
        if not normalized:
            LOGGER.warning(
                "Paystack plan %s does not include a valid amount field. Payload: %s",
                plan_code,
                plan,
            )
            return None

        self._plan_amount_cache[plan_code] = normalized
        return normalized

    def _build_session(self) -> Session:
        """Return a requests session configured with retry behaviour."""

        session = requests.Session()
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST", "PUT", "PATCH", "DELETE"),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

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
        errors = payload.get("errors")
        if errors:
            if isinstance(errors, list):
                detail = "; ".join(str(item) for item in errors if item)
            elif isinstance(errors, Mapping):
                detail = "; ".join(f"{key}: {value}" for key, value in errors.items())
            else:
                detail = str(errors)
            if detail:
                message = f"{message}: {detail}"

        if status_code == 404:
            raise PaystackNotFound(message)

        if status_code == 401:
            raise PaystackAuthError(message or "Paystack authentication failed.")

        if status_code >= 400 or not status_flag:
            raise PaystackError(message)

        if "data" in payload:
            data = payload.get("data")
        else:
            data = payload
        return data
