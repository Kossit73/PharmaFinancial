from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field

from ..common import TablePayload
from .inputs import GoatInputsPayload


class GoatAnalysisPayload(BaseModel):
    title: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    tables: Dict[str, TablePayload]


class GoatModelRunRequest(BaseModel):
    """Request body for /model/goat/run."""

    inputs: Optional[GoatInputsPayload] = Field(
        default=None, description="Full goat farming modelling payload. When omitted defaults are used."
    )


class GoatValidationRequest(BaseModel):
    """Request body for /inputs/goat/validate."""

    inputs: GoatInputsPayload


class GoatModelRunResponse(BaseModel):
    """Response payload for goat model run."""

    schedule: TablePayload
    scenario: TablePayload
    performance: TablePayload
    cash_flow: TablePayload
    position: TablePayload
    kpis: TablePayload
    break_even: TablePayload
    advanced: Dict[str, GoatAnalysisPayload]
    valuation_summary: Dict[str, Optional[float]]


__all__ = [
    "GoatAnalysisPayload",
    "GoatInputsPayload",
    "GoatModelRunRequest",
    "GoatModelRunResponse",
    "GoatValidationRequest",
]
