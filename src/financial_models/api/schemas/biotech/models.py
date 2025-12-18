from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field

from ..common import AIInsightsPayload, TablePayload
from .inputs import BiotechInputsPayload


class BiotechModelRunRequest(BaseModel):
    """Request body for /model/biotech/run."""

    inputs: Optional[BiotechInputsPayload] = Field(
        default=None, description="Full biotech modelling payload. When omitted the defaults are used."
    )


class BiotechValidationRequest(BaseModel):
    """Request body for /inputs/biotech/validate."""

    inputs: BiotechInputsPayload


class BiotechModelRunResponse(BaseModel):
    """Response payload for biotech model run."""

    rnpv: float
    consolidated: TablePayload
    dcf_table: TablePayload
    per_product: Dict[str, TablePayload]
    per_product_prob: Dict[str, TablePayload]
    ai_insights: Optional[AIInsightsPayload] = None


__all__ = [
    "BiotechInputsPayload",
    "BiotechModelRunRequest",
    "BiotechModelRunResponse",
    "BiotechValidationRequest",
]
