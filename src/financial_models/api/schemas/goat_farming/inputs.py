from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field


class GoatScenarioPayload(BaseModel):
    milk_price_pct: Optional[float] = Field(default=None, description="Shock applied to milk price (e.g. 0.05 = +5%).")
    feed_cost_pct: Optional[float] = Field(default=None, description="Shock applied to feed cost (e.g. 0.1 = +10%).")


class GoatInputsPayload(BaseModel):
    schedule: List[Mapping[str, Any]] = Field(description="Financial schedule rows including a Period column.")
    period_column: Optional[str] = Field(default="Period", description="Column containing period labels.")
    valuation_inputs: Optional[Dict[str, float]] = Field(default=None, description="Optional valuation assumptions (e.g. WACC, NPV).")
    supplementary_tables: Optional[Mapping[str, List[Mapping[str, Any]]]] = Field(
        default=None, description="Optional supplementary tables (e.g. capex schedule)."
    )
    scenario: Optional[GoatScenarioPayload] = Field(default=None, description="Optional scenario shocks.")


__all__ = ["GoatInputsPayload", "GoatScenarioPayload"]
