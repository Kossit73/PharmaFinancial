from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field

from ..common import TablePayload
from .inputs import BroilerInputsPayload


class BroilerModelRunRequest(BaseModel):
    """Request body for /model/broiler_chicken/run."""

    inputs: Optional[BroilerInputsPayload] = Field(default=None, description="Optional assumption overrides.")


class BroilerValidationRequest(BaseModel):
    """Request body for /inputs/broiler_chicken/validate."""

    inputs: Optional[BroilerInputsPayload] = Field(default=None)


class BroilerModelRunResponse(BaseModel):
    """Response payload for broiler chicken model run."""

    assumptions_schedule: TablePayload
    income_statement: TablePayload
    balance_sheet: TablePayload
    cash_flow_statement: TablePayload
    cashflows: TablePayload
    revenue_summary: TablePayload
    valuation: Dict[str, Optional[float]]
    advanced_analytics: Dict[str, TablePayload]


__all__ = [
    "BroilerInputsPayload",
    "BroilerModelRunRequest",
    "BroilerModelRunResponse",
    "BroilerValidationRequest",
]
