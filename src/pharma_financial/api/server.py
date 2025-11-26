"""FastAPI server exposing the Pharmaceuticals financial model."""
from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException

from ..core.inputs import ModelInputs, load_inputs, parse_inputs
from ..core.model import FinancialModel
from ..core.table import Table
from ..services.paystack import PaystackClient, PaystackError, SubscriptionStatus
from ..services.subscription_store import StoredSubscriptionRecord, get_subscription_store
from .schemas import (
    AIInsightsPayload,
    ModelRunRequest,
    ModelRunResponse,
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


def _run_model(inputs: ModelInputs) -> ModelRunResponse:
    model = FinancialModel(inputs)
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


def _resolve_inputs(payload: Dict[str, Any] | None) -> ModelInputs:
    if payload:
        return parse_inputs(payload)
    return load_inputs()


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

    @app.post("/model/run", response_model=ModelRunResponse)
    def run_model(request: ModelRunRequest) -> ModelRunResponse:
        try:
            inputs = _resolve_inputs(dict(request.inputs) if request.inputs is not None else None)
        except Exception as exc:  # pragma: no cover - validation handled explicitly in /inputs/validate
            raise HTTPException(status_code=400, detail=f"Invalid inputs: {exc}") from exc
        return _run_model(inputs)

    @app.post("/inputs/validate", response_model=ValidationResponse)
    def validate_inputs(request: ValidationRequest) -> ValidationResponse:
        try:
            parse_inputs(dict(request.inputs))
        except Exception as exc:
            return ValidationResponse(valid=False, message=str(exc))
        return ValidationResponse(valid=True, message="Inputs parsed successfully.")

    @app.post("/subscriptions/check", response_model=SubscriptionCheckResponse)
    def check_subscription(
        request: SubscriptionCheckRequest,
        client: PaystackClient = Depends(get_paystack_client),
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
    def get_subscription_status(email: str) -> SubscriptionStatusRecord:
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
    def upsert_subscription_status(request: SubscriptionStatusUpsert) -> SubscriptionStatusRecord:
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
    def delete_subscription_status(email: str) -> None:
        store = get_subscription_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Subscription store unavailable.")
        store.remove_status(email)
        return None

    return app


app = create_app()

__all__ = ["app", "create_app", "get_paystack_client"]
