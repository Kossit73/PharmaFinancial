from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field

from ..common import TablePayload
from .inputs import CassavaInputsPayload


class CassavaModelRunRequest(BaseModel):
    """Request body for /model/cassava_ethanol/run."""

    inputs: Optional[CassavaInputsPayload] = Field(
        default=None, description="Optional scenario payload. Defaults to FARM_ONLY inputs."
    )


class CassavaValidationRequest(BaseModel):
    """Request body for /inputs/cassava_ethanol/validate."""

    inputs: Optional[CassavaInputsPayload] = Field(default=None)


class CassavaModelRunResponse(BaseModel):
    """Response payload for cassava ethanol model run."""

    income_statement_monthly: TablePayload
    income_statement_annual: TablePayload
    balance_sheet_monthly: TablePayload
    balance_sheet_annual: TablePayload
    cash_flow_monthly: TablePayload
    cash_flow_annual: TablePayload
    break_even: TablePayload
    payback: TablePayload
    metrics: Dict[str, Optional[float]]
    scenario: str


__all__ = [
    "CassavaInputsPayload",
    "CassavaModelRunRequest",
    "CassavaModelRunResponse",
    "CassavaValidationRequest",
]
