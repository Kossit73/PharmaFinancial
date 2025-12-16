"""FastAPI server exposing the Pharmaceuticals financial model."""
from __future__ import annotations

import logging
import math
import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence, Type

from fastapi import Depends, FastAPI, Header, HTTPException, status

from ..core.inputs import ModelInputs, load_inputs, parse_inputs
from ..core.model import FinancialModel
from ..core.table import Table
from ..services.paystack import PaystackClient, PaystackError, SubscriptionStatus
from ..services.subscription_store import StoredSubscriptionRecord, get_subscription_store
from .schemas import (
    AIInsightsPayload,
    ModelRunRequest,
    ModelRunResponse,
    PharmaModelRunRequest,
    PharmaValidationRequest,
    ScenarioToolResultPayload,
    SubscriptionCheckRequest,
    SubscriptionCheckResponse,
    SubscriptionStatusRecord,
    SubscriptionStatusUpsert,
    TablePayload,
    ValidationRequest,
    ValidationResponse,
)


def _clean_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _table_payload(table: Table | None) -> TablePayload | None:
    if table is None:
        return None
    sanitized = {key: [_clean_value(v) for v in values] for key, values in table.as_dict().items()}
    return TablePayload(index_name=table.index_name, index=list(table.index), data=sanitized)


def _ai_payload(insights) -> AIInsightsPayload | None:
    if insights is None:
        return None
    return AIInsightsPayload(
        enabled=bool(insights.enabled),
        generative_summary=insights.generative_summary,
        metadata=insights.metadata,
        ml_forecast=_table_payload(insights.ml_forecast),
    )

@dataclass
class ModelSpec:
    """Describes a registered model and how to execute it."""

    name: str
    load_inputs: Callable[[], ModelInputs]
    parse_inputs: Callable[[Mapping[str, Any]], ModelInputs]
    model_factory: Callable[[ModelInputs], Any] = FinancialModel
    run_request_model: Type[ModelRunRequest] = ModelRunRequest
    validate_request_model: Type[ValidationRequest] = ValidationRequest


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "pharma": ModelSpec(
        name="Pharmaceuticals",
        load_inputs=load_inputs,
        parse_inputs=parse_inputs,
        model_factory=FinancialModel,
        run_request_model=PharmaModelRunRequest,
        validate_request_model=PharmaValidationRequest,
    )
}


def _resolve_inputs(payload: Dict[str, Any] | None, spec: ModelSpec) -> ModelInputs:
    if payload:
        return spec.parse_inputs(payload)
    return spec.load_inputs()


def _run_model(inputs: ModelInputs, spec: ModelSpec) -> ModelRunResponse:
    model = spec.model_factory(inputs)
    outputs = model.run()
    return ModelRunResponse(
        summary_metrics=_table_payload(outputs.summary_metrics),
        income_statement=_table_payload(outputs.income_statement),
        balance_sheet=_table_payload(outputs.balance_sheet),
        cash_flow=_table_payload(outputs.cash_flow),
        goal_seek=_table_payload(outputs.goal_seek),
        break_even=_table_payload(outputs.break_even),
        payback=_table_payload(outputs.payback),
        discounted_payback=_table_payload(outputs.discounted_payback),
        monte_carlo=_table_payload(outputs.monte_carlo),
        scenario_results={name: _table_payload(table) for name, table in outputs.scenario_results.items()},
        sensitivity_results={name: _table_payload(table) for name, table in outputs.sensitivity_results.items()},
        scenario_tool_results={
            name: ScenarioToolResultPayload(rows=result.rows, interpretation=result.interpretation)
            for name, result in outputs.scenario_tool_results.items()
        },
        ai_insights=_ai_payload(outputs.ai_insights),
        risk_factor_diagnostics=_table_payload(outputs.risk_factor_diagnostics),
    )


def _record_payload(record: StoredSubscriptionRecord) -> SubscriptionStatusRecord:
    return SubscriptionStatusRecord(
        email=record.email,
        is_active=record.is_active,
        status_message=record.status_message,
        updated_at=record.updated_at,
        source=record.source,
        expires_at=record.expires_at,
        payload=record.payload,
    )


LOGGER = logging.getLogger(__name__)

API_TOKEN_ENV = "PHARMA_FINANCIAL_API_TOKEN"
API_TOKEN_HEADER = "X-API-Key"
GOOGLE_AUDIENCE_ENV = "PHARMA_FINANCIAL_GOOGLE_AUDIENCE"
GOOGLE_VALID_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}

try:  # pragma: no cover - optional dependency
    from google.oauth2 import id_token as google_id_token  # type: ignore
    from google.auth.transport import requests as google_requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    google_id_token = None  # type: ignore
    google_requests = None  # type: ignore


@dataclass
class AuthContext:
    """Represents the authenticated caller."""

    method: str
    subject: str | None = None
    email: str | None = None
    claims: Mapping[str, Any] | None = None


def _expected_api_token() -> str | None:
    token = os.getenv(API_TOKEN_ENV, "").strip()
    return token or None


def _google_audiences() -> list[str]:
    value = os.getenv(GOOGLE_AUDIENCE_ENV, "")
    audiences = [item.strip() for item in value.split(",") if item.strip()]
    return audiences


def _verify_google_token(token: str, audiences: Sequence[str]) -> Mapping[str, Any]:
    """Validate a Google ID token for one of the configured audiences."""

    if google_id_token is None or google_requests is None:  # pragma: no cover - configuration guard
        raise RuntimeError("Google authentication requires the 'google-auth' package.")

    request = google_requests.Request()
    errors: MutableMapping[str, Exception] = {}
    for audience in audiences:
        try:
            payload = google_id_token.verify_oauth2_token(token, request, audience)
        except Exception as exc:  # pragma: no cover - upstream validation details unsuitable for unit tests
            errors[audience] = exc
            continue
        issuer = str(payload.get("iss") or "")
        if issuer not in GOOGLE_VALID_ISSUERS:
            errors[audience] = ValueError("Unexpected token issuer.")
            continue
        return payload
    if errors:
        LOGGER.debug("Google token verification failed: %s", errors)
    raise ValueError("Unable to verify Google ID token for the configured audience(s).")


def require_authorization(
    x_api_key: str | None = Header(default=None, alias=API_TOKEN_HEADER),
    authorization: str | None = Header(default=None, convert_underscores=False),
) -> AuthContext | None:
    """Enforce API token or Google social auth when configured."""

    expected = _expected_api_token()
    google_audiences = _google_audiences()

    if expected is None and not google_audiences:
        return None  # auth disabled

    if expected is not None and x_api_key:
        if secrets.compare_digest(x_api_key.strip(), expected):
            return AuthContext(method="api_token")

    if google_audiences and authorization:
        scheme, _, token = authorization.strip().partition(" ")
        token = token.strip()
        if scheme.lower() == "bearer" and token:
            try:
                payload = _verify_google_token(token, google_audiences)
            except RuntimeError as exc:  # pragma: no cover - dependency guard
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            except ValueError:
                pass
            else:
                return AuthContext(
                    method="google",
                    subject=str(payload.get("sub") or ""),
                    email=payload.get("email"),
                    claims=payload,
                )

    detail = "Unauthorized request."
    if expected is not None and not x_api_key:
        detail = "Missing API token."
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


@lru_cache(maxsize=1)
def get_paystack_client() -> PaystackClient:
    """Create (and cache) a Paystack client based on env vars."""

    secret_key = os.getenv("PAYSTACK_SECRET_KEY")
    plan_code = os.getenv("PAYSTACK_PLAN_CODE")
    default_amount = os.getenv("PAYSTACK_PLAN_AMOUNT_KOBO")
    amount = int(default_amount) if default_amount and default_amount.isdigit() else None
    callback_url = os.getenv("PAYSTACK_CALLBACK_URL")
    cancel_url = os.getenv("PAYSTACK_CANCEL_ACTION_URL")
    return PaystackClient(
        secret_key=secret_key,
        plan_code=plan_code,
        default_amount_kobo=amount,
        callback_url=callback_url,
        cancel_action_url=cancel_url,
        fetch_plan_amount=True,
    )


def create_app() -> FastAPI:
    """Instantiate the FastAPI application."""

    app = FastAPI(
        title="Pharmaceuticals Financial Model API",
        version="1.0.0",
        description="HTTP interface for running the Pharmaceuticals financial engine.",
    )

    @app.get("/health")
    def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    def _register_model_routes(model_type: str, spec: ModelSpec) -> None:
        run_request_model = spec.run_request_model
        validate_request_model = spec.validate_request_model

        @app.post(f"/model/{model_type}/run", response_model=ModelRunResponse)
        def run_model_versioned(
            request: run_request_model, _: AuthContext | None = Depends(require_authorization)
        ) -> ModelRunResponse:
            try:
                inputs = _resolve_inputs(dict(request.inputs) if request.inputs is not None else None, spec)
            except Exception as exc:  # pragma: no cover - validation handled explicitly in /inputs/{model_type}/validate
                raise HTTPException(status_code=400, detail=f"Invalid inputs: {exc}") from exc
            return _run_model(inputs, spec)

        @app.post(f"/inputs/{model_type}/validate", response_model=ValidationResponse)
        def validate_inputs_versioned(
            request: validate_request_model,
            _: AuthContext | None = Depends(require_authorization),
        ) -> ValidationResponse:
            try:
                spec.parse_inputs(dict(request.inputs))
            except Exception as exc:
                return ValidationResponse(valid=False, message=str(exc))
            return ValidationResponse(valid=True, message="Inputs parsed successfully.")

    for model_type, spec in MODEL_REGISTRY.items():
        _register_model_routes(model_type, spec)

    @app.post("/subscriptions/check", response_model=SubscriptionCheckResponse)
    def check_subscription(
        request: SubscriptionCheckRequest,
        client: PaystackClient = Depends(get_paystack_client),
        _: AuthContext | None = Depends(require_authorization),
    ) -> SubscriptionCheckResponse:
        try:
            status = client.has_active_subscription(request.email)
        except PaystackError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if not isinstance(status, SubscriptionStatus):
            raise HTTPException(status_code=500, detail="Unexpected Paystack response.")
        return SubscriptionCheckResponse(
            email=status.email,
            is_active=status.is_active,
            message=status.message,
            payload=status.payload,
        )

    @app.get("/subscriptions/status", response_model=SubscriptionStatusRecord)
    def get_subscription_status(
        email: str,
        _: AuthContext | None = Depends(require_authorization),
    ) -> SubscriptionStatusRecord:
        store = get_subscription_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Subscription store unavailable.")
        record = store.get_status(email)
        if record is None:
            raise HTTPException(status_code=404, detail="Subscription not found.")
        if record.is_expired():
            store.remove_status(email)
            raise HTTPException(status_code=404, detail="Subscription not found.")
        return _record_payload(record)

    @app.post("/subscriptions/status", response_model=SubscriptionStatusRecord)
    def upsert_subscription_status(
        request: SubscriptionStatusUpsert,
        _: AuthContext | None = Depends(require_authorization),
    ) -> SubscriptionStatusRecord:
        store = get_subscription_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Subscription store unavailable.")
        status = SubscriptionStatus(
            email=request.email,
            is_active=request.is_active,
            message=request.status_message,
            payload=request.payload,
        )
        store.write_status(status, source=request.source or "api", ttl_seconds=request.ttl_seconds)
        record = store.get_status(request.email)
        if record is None:
            raise HTTPException(status_code=500, detail="Unable to persist subscription.")
        return _record_payload(record)

    @app.delete("/subscriptions/status", status_code=204)
    def delete_subscription_status(
        email: str,
        _: AuthContext | None = Depends(require_authorization),
    ) -> None:
        store = get_subscription_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Subscription store unavailable.")
        store.remove_status(email)
        return None

    return app


app = create_app()

__all__ = ["app", "create_app", "get_paystack_client"]
