from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CassavaInputsPayload(BaseModel):
    """Inputs for the cassava ethanol model."""

    scenario: Optional[str] = Field(default=None, description="Scenario name: FARM_ONLY, BUY_ONLY, or HYBRID.")


__all__ = ["CassavaInputsPayload"]
