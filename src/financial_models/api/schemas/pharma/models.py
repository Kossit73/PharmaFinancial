"""Pydantic schemas for the pharma model."""
from __future__ import annotations

from typing import Optional

from pydantic import Field

from ..common import ModelRunRequest, ValidationRequest
from .inputs import PharmaInputsPayload


class PharmaModelRunRequest(ModelRunRequest):
    """Model run request specific to the pharmaceuticals engine."""

    inputs: Optional[PharmaInputsPayload] = Field(
        default=None,
        description="Full pharmaceuticals modelling payload. When omitted the default inputs are used.",
    )


class PharmaValidationRequest(ValidationRequest):
    """Validation request specific to the pharmaceuticals engine."""

    inputs: PharmaInputsPayload = Field(..., description="Full pharmaceuticals modelling payload.")


__all__ = [
    "PharmaInputsPayload",
    "PharmaModelRunRequest",
    "PharmaValidationRequest",
]
