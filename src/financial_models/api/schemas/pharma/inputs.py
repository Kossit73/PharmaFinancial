"""Pydantic schema for the pharmaceuticals model payload."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, Field
from pydantic import ConfigDict


class PharmaInputsPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    """Top-level payload accepted by the pharmaceuticals model."""

    years: Sequence[int] = Field(..., description="Projection years.")
    production_estimate: Optional[Mapping[str, Sequence[float]]] = None
    production_capacity: Optional[Mapping[str, Sequence[float]]] = None
    pricing: Optional[Mapping[str, Any]] = None
    revenue_growth_rate: Optional[Sequence[float]] = None
    costs: Optional[Mapping[str, Any]] = None
    labor: Optional[Mapping[str, Any]] = None
    depreciation: Optional[Mapping[str, Any]] = None
    distributor_commission: Optional[Mapping[str, Any]] = None
    capital_expenditure: Optional[Mapping[str, Any]] = None
    financing: Optional[Mapping[str, Any]] = None
    working_capital: Optional[Mapping[str, Any]] = None
    tax: Optional[Mapping[str, Any]] = None
    risk: Optional[Mapping[str, Any]] = None
    scenarios: Optional[Mapping[str, Any]] = None
    scenario_tools: Optional[Mapping[str, Any]] = None
    sensitivity: Optional[Mapping[str, Any]] = None
    monte_carlo: Optional[Mapping[str, Any]] = None
    goal_seek: Optional[Mapping[str, Any]] = None
    ai: Optional[Mapping[str, Any]] = None
