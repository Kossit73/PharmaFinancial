from __future__ import annotations

from typing import Any, List, Mapping

from pydantic import BaseModel, ConfigDict, Field


class BiotechInputsPayload(BaseModel):
    """Biotech model inputs payload."""

    model_config = ConfigDict(populate_by_name=True)

    config: Mapping[str, Any] = Field(..., alias="model_config")
    products: List[Mapping[str, Any]]


__all__ = ["BiotechInputsPayload"]
