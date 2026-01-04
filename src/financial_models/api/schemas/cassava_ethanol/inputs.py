from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator


class CassavaInputsPayload(BaseModel):
    """Inputs for the cassava ethanol model."""

    model_config = ConfigDict(extra="forbid")

    scenario: Optional[str] = Field(default=None, description="Scenario name: FARM_ONLY, BUY_ONLY, or HYBRID.")

    @field_validator("scenario")
    def _validate_scenario(cls, value: str | None) -> str | None:
        if value is None:
            return value
        scenario = value.strip().upper()
        if scenario not in {"FARM_ONLY", "BUY_ONLY", "HYBRID"}:
            raise ValueError("scenario must be FARM_ONLY, BUY_ONLY, or HYBRID.")
        return scenario


__all__ = ["CassavaInputsPayload"]
