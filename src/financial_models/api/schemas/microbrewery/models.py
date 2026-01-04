from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field

from ..common import TablePayload
from .inputs import MicrobreweryInputsPayload


class MicrobreweryModelRunRequest(BaseModel):
    """Request body for /model/microbrewery/run."""

    inputs: Optional[MicrobreweryInputsPayload] = Field(
        default=None, description="Full microbrewery payload. When omitted the defaults are used."
    )


class MicrobreweryValidationRequest(BaseModel):
    """Request body for /inputs/microbrewery/validate."""

    inputs: MicrobreweryInputsPayload


class MicrobreweryModelRunResponse(BaseModel):
    """Response payload for microbrewery model run."""

    monthly: TablePayload
    annual: TablePayload
    prices: TablePayload
    debt_schedules: Dict[str, TablePayload]
    valuation: Dict[str, float]


__all__ = [
    "MicrobreweryInputsPayload",
    "MicrobreweryModelRunRequest",
    "MicrobreweryModelRunResponse",
    "MicrobreweryValidationRequest",
]
