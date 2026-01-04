"""Pydantic schema for the pharmaceuticals model payload."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class PharmaInputsPayload(BaseModel):
    """Top-level payload accepted by the pharmaceuticals model."""

    model_config = ConfigDict(extra="allow")

    years: Sequence[int] = Field(..., description="Projection years.")
    production_estimate: Mapping[str, Sequence[float]] = Field(
        ..., description="Product → annual production units."
    )
    unit_costs: Mapping[str, Mapping[str, float]] = Field(
        ..., description="Product → cost components (production, price, freight)."
    )
    markup: Mapping[str, float] = Field(..., description="Product → markup percentage (0-1).")
    total_production_units: Optional[Mapping[str, float]] = Field(
        default=None, description="Optional product → total units; defaults to sum of production_estimate."
    )
    production_capacity: Optional[Mapping[str, float]] = Field(
        default=None, description="Optional product → max capacity units per year."
    )
    inflation_rate: Optional[float] = Field(
        default=None, description="Default inflation rate applied when a series is not provided."
    )
    inflation_series: Optional[Sequence[float]] = Field(
        default=None, description="Per-year inflation factors matching the projection horizon."
    )
    raw_material_cost: Mapping[str, Any] = Field(
        ..., description="Raw material costs (per_unit and optional annual schedule)."
    )
    utility_costs: Mapping[str, Any] = Field(
        ..., description="Utility schedule per year (electricity/water/steam) or scalar fallbacks."
    )
    labor: Mapping[str, Any] = Field(
        ..., description="Direct/indirect labour cost assumptions."
    )
    fixed_variable_costs: Optional[Mapping[str, Any]] = Field(
        default=None, description="Optional fixed/variable overrides per product."
    )
    break_even: Optional[Mapping[str, Any]] = Field(
        default=None, description="Optional break-even assumptions per product."
    )
    depreciation: Mapping[str, Any] = Field(
        ..., description="Depreciation rows with asset type, acquisition, method, and life."
    )
    distributor_commission: Optional[Mapping[str, Any]] = Field(
        default=None, description="Commission rows with rate, payment days, and revenue share."
    )
    capital_expenditure: Mapping[str, Any] = Field(
        ..., description="Initial/contingency/project reserve and annual additions."
    )
    financing: Mapping[str, Any] = Field(
        ..., description="Financing structure (discount rate, dividend payout, senior/revolver/overdraft schedules)."
    )
    working_capital: Mapping[str, Any] = Field(
        ..., description="Working capital days and calendar days per projection year."
    )
    tax: Mapping[str, Any] = Field(
        ..., description="Tax rate, timing adjustment, and optional schedule per year."
    )
    risk: Mapping[str, Sequence[float]] = Field(
        ..., description="Risk schedules (e.g., inherent/climate/political) aligned to years."
    )
    scenarios: Mapping[str, Mapping[str, Sequence[float]]] = Field(
        ..., description="Inflation/interest scenarios keyed by name."
    )
    scenario_tools: Optional[Mapping[str, Sequence[str]]] = Field(
        default=None, description="Scenario tools to compute (e.g., decision_tree, stress_testing)."
    )
    sensitivity: Mapping[str, Any] = Field(
        ..., description="Sensitivity variables and shock ranges."
    )
    monte_carlo: Mapping[str, Any] = Field(
        ..., description="Monte Carlo settings (iterations, revenue_growth_range, variables, metrics, optional seed)."
    )
    goal_seek: Optional[Mapping[str, Any]] = Field(
        default=None, description="Goal seek configuration (metric, target, source, optional year)."
    )
    ai: Optional[Mapping[str, Any]] = Field(
        default=None, description="Optional AI settings (provider, model, forecast horizon, features)."
    )

    @field_validator("years")
    @classmethod
    def _ensure_years_non_empty(cls, value: Sequence[int]) -> Sequence[int]:
        if not list(value):
            raise ValueError("years must contain at least one projection year.")
        return value

    @model_validator(mode="after")
    def _validate_horizon_alignment(self) -> "PharmaInputsPayload":
        horizon = len(self.years)
        if horizon and self.production_estimate:
            for name, schedule in self.production_estimate.items():
                seq = list(schedule)
                if seq and len(seq) != horizon:
                    raise ValueError(
                        f"production_estimate for '{name}' must have {horizon} values (got {len(seq)})."
                    )
        if self.inflation_series is not None:
            series = list(self.inflation_series)
            if series and len(series) != horizon:
                raise ValueError(
                    f"inflation_series must have {horizon} values (got {len(series)})."
                )
        if self.risk:
            for name, schedule in self.risk.items():
                seq = list(schedule)
                if seq and len(seq) != horizon:
                    raise ValueError(
                        f"risk schedule '{name}' must have {horizon} values (got {len(seq)})."
                    )
        return self
