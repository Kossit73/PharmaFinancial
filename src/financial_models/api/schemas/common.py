"""Pydantic schemas shared across API endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, EmailStr, Field


class TablePayload(BaseModel):
    """JSON-friendly representation of a Table."""

    index_name: str = Field(default="Year", description="Name of the index column.")
    index: List[Any] = Field(default_factory=list, description="Index labels.")
    data: Dict[str, List[Any]] = Field(default_factory=dict, description="Columnar data.")


class ScenarioToolResultPayload(BaseModel):
    """Represents a scenario tool output."""

    rows: List[Mapping[str, Any]]
    interpretation: str


class AIInsightsPayload(BaseModel):
    """Serialised AI insights."""

    enabled: bool
    generative_summary: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None
    ml_forecast: Optional[TablePayload] = None


class ModelRunRequest(BaseModel):
    """Request body for /model/{model_type}/run."""

    inputs: Optional[Mapping[str, Any]] = Field(
        default=None,
        description="Full modelling payload. When omitted the default inputs are used.",
    )


class ModelRunResponse(BaseModel):
    """Response payload returned by /model/{model_type}/run."""

    summary_metrics: TablePayload
    income_statement: TablePayload
    balance_sheet: TablePayload
    cash_flow: TablePayload
    goal_seek: TablePayload
    break_even: TablePayload
    payback: TablePayload
    discounted_payback: TablePayload
    monte_carlo: TablePayload
    scenario_results: Dict[str, TablePayload]
    sensitivity_results: Dict[str, TablePayload]
    scenario_tool_results: Dict[str, ScenarioToolResultPayload]
    ai_insights: Optional[AIInsightsPayload] = None
    risk_factor_diagnostics: Optional[TablePayload] = None


class ValidationRequest(BaseModel):
    """Request body for /inputs/{model_type}/validate."""

    inputs: Mapping[str, Any]


class ValidationResponse(BaseModel):
    """Response returned by /inputs/{model_type}/validate."""

    valid: bool
    message: str


class SubscriptionCheckRequest(BaseModel):
    """Request body for /subscriptions/check."""

    email: EmailStr


class SubscriptionCheckResponse(BaseModel):
    """Response returned by /subscriptions/check."""

    email: EmailStr
    is_active: bool
    message: str
    payload: Optional[Mapping[str, Any]] = None
    cached: bool = False
    cached_at: Optional[float] = None


class SubscriptionStatusRecord(BaseModel):
    """Represents a stored subscription record."""

    email: EmailStr
    is_active: bool
    status_message: str
    updated_at: float
    source: Optional[str] = None
    expires_at: Optional[float] = None
    payload: Optional[Mapping[str, Any]] = None


class SubscriptionStatusUpsert(BaseModel):
    """Request body to persist subscription records."""

    email: EmailStr
    is_active: bool
    status_message: str = ""
    payload: Optional[Mapping[str, Any]] = None
    source: Optional[str] = None
    ttl_seconds: Optional[float] = None


class SubscriptionCheckoutRequest(BaseModel):
    """Request body for generating a Paystack checkout link."""

    email: EmailStr
    metadata: Optional[Mapping[str, Any]] = None
    scenario: Optional[str] = Field(default=None, description="Optional scenario label to include in metadata.")


class SubscriptionCheckoutResponse(BaseModel):
    """Response payload containing a checkout URL."""

    email: EmailStr
    checkout_url: str


class SubscriptionVerifyRequest(BaseModel):
    """Request body for verifying a Paystack transaction reference."""

    reference: str


class SubscriptionVerifyResponse(BaseModel):
    """Response payload after verifying a Paystack transaction."""

    email: Optional[EmailStr] = None
    is_active: bool
    message: str
    payload: Optional[Mapping[str, Any]] = None


class AuthUpdateRequest(BaseModel):
    """Request body for updating the current user."""

    name: Optional[str] = Field(default=None, description="Optional display name.")
    password: Optional[str] = Field(default=None, description="Optional new password (local accounts only).")
