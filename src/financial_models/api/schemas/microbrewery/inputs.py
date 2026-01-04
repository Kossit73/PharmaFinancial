from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import BaseModel, Field


class MicrobreweryConfigPayload(BaseModel):
    start_date: Optional[str] = Field(default=None, description="Model start date (YYYY-MM-DD).")
    months: Optional[int] = Field(default=None, description="Number of projection months.")
    pricing_cost_basis_month: Optional[int] = Field(default=None, description="Month index used for base pricing.")
    price_inflation_annual: Optional[float] = Field(default=None, description="Annual price inflation rate.")
    cost_inflation_annual: Optional[float] = Field(default=None, description="Annual cost inflation rate.")
    tax_rate: Optional[float] = Field(default=None, description="Corporate tax rate.")
    days_receivables: Optional[float] = Field(default=None, description="Days sales outstanding.")
    days_inventory: Optional[float] = Field(default=None, description="Days inventory outstanding.")
    days_payables: Optional[float] = Field(default=None, description="Days payables outstanding.")
    other_current_assets_pct_revenue: Optional[float] = Field(default=None, description="Other current assets % of revenue.")
    other_current_liabilities_pct_direct_costs: Optional[float] = Field(
        default=None, description="Other current liabilities % of direct costs."
    )
    wacc_annual: Optional[float] = Field(default=None, description="Annual WACC used for valuation.")
    exit_month: Optional[int] = Field(default=None, description="Month index used for exit valuation.")
    exit_ev_ebitda_multiple: Optional[float] = Field(default=None, description="Exit EV/EBITDA multiple.")
    initial_cash: Optional[float] = Field(default=None, description="Opening cash balance.")


class MicrobreweryDividendPolicyPayload(BaseModel):
    enabled: Optional[bool] = Field(default=None, description="Whether dividends are paid.")
    model: Optional[str] = Field(default=None, description="cash_sweep or share_of_profits.")
    start_month: Optional[int] = Field(default=None, description="Month index to begin dividends.")
    minimum_cash_position: Optional[float] = Field(default=None, description="Minimum cash when sweeping.")
    payout_ratio: Optional[float] = Field(default=None, description="Payout ratio for share_of_profits.")


class MicrobreweryCapexItemPayload(BaseModel):
    name: str
    amount: float
    capex_month: int = Field(default=0)
    depreciation_years: float = Field(default=0.0)


class MicrobreweryDebtFacilityPayload(BaseModel):
    name: str
    principal: float
    annual_interest_rate: float
    draw_month: int = Field(default=0)
    grace_months: int = Field(default=0)
    term_months: int = Field(default=0)
    repayment_type: str = Field(default="linear")
    specified_principal_payments: Optional[Dict[int, float]] = Field(default=None)


class MicrobrewerySalesPlanEntry(BaseModel):
    date: str = Field(description="Month start date (YYYY-MM-DD).")
    sku_id: int
    channel: str
    units: float


class MicrobreweryInputsPayload(BaseModel):
    config: Optional[MicrobreweryConfigPayload] = Field(default=None, description="Global modelling parameters.")
    dividend_policy: Optional[MicrobreweryDividendPolicyPayload] = Field(default=None, description="Dividend settings.")
    skus: List[Mapping[str, Any]]
    channels: List[Mapping[str, Any]]
    sales_plan: List[MicrobrewerySalesPlanEntry]
    opex_fixed_monthly: Optional[Union[float, int, Mapping[str, Any], List[Mapping[str, Any]]]] = Field(
        default=None, description="Fixed monthly operating costs."
    )
    other_income_monthly: Optional[Union[float, int, Mapping[str, Any], List[Mapping[str, Any]]]] = Field(
        default=None, description="Other monthly income."
    )
    capex_items: Optional[List[MicrobreweryCapexItemPayload]] = Field(default=None, description="CAPEX schedule.")
    debt_facilities: Optional[List[MicrobreweryDebtFacilityPayload]] = Field(
        default=None, description="Debt facility terms."
    )
    equity_injections: Optional[Mapping[str, float]] = Field(default=None, description="Month index to equity amount.")


__all__ = [
    "MicrobreweryInputsPayload",
    "MicrobreweryConfigPayload",
    "MicrobreweryDividendPolicyPayload",
    "MicrobreweryCapexItemPayload",
    "MicrobreweryDebtFacilityPayload",
    "MicrobrewerySalesPlanEntry",
]
