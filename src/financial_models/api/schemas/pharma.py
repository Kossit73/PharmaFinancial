"""Schemas specific to the pharmaceuticals model."""
from __future__ import annotations

from .common import ModelRunRequest, ValidationRequest


class PharmaModelRunRequest(ModelRunRequest):
    """Model run request specific to the pharmaceuticals engine."""

    pass


class PharmaValidationRequest(ValidationRequest):
    """Validation request specific to the pharmaceuticals engine."""

    pass
