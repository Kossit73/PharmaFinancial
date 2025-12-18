from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, EmailStr, Field

from ..common import TablePayload


class BiotechInputsPayload(BaseModel):
    """Biotech model inputs payload."""

    model_config: Mapping[str, Any]
    products: List[Mapping[str, Any]]


class BiotechModelRunRequest(BaseModel):
    """Request body for /model/biotech/run."""

    inputs: Optional[Mapping[str, Any]] = Field(
        default=None, description="Full biotech modelling payload. When omitted the defaults are used."
    )


class BiotechValidationRequest(BaseModel):
    """Request body for /inputs/biotech/validate."""

    inputs: Mapping[str, Any]


class BiotechModelRunResponse(BaseModel):
    """Response payload for biotech model run."""

    rnpv: float
    consolidated: TablePayload
    dcf_table: TablePayload
    per_product: Dict[str, TablePayload]
    per_product_prob: Dict[str, TablePayload]

