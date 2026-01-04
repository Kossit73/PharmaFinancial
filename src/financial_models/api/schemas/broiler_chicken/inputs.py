from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict


class BroilerInputsPayload(BaseModel):
    """Flat mapping of broiler model assumptions. Keys map to fields in `broiler_chicken.assumptions.Assumptions`."""

    model_config = ConfigDict(extra="allow")


__all__ = ["BroilerInputsPayload"]
