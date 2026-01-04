from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class BroilerInputsPayload(BaseModel):
    """Flat mapping of broiler model assumptions. Keys map to fields in `broiler_chicken.assumptions.Assumptions`."""

    class Config:
        extra = "allow"


__all__ = ["BroilerInputsPayload"]
