from __future__ import annotations

from typing import Any, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BiotechConfigPayload(BaseModel):
    """Core configuration for the biotech model."""

    model_config = ConfigDict(extra="allow")

    first_year: int = Field(..., description="Projection start year.")
    n_years: int = Field(..., description="Projection horizon length.")
    discount_rate: float = Field(..., description="Discount rate for DCF.")
    currency: Optional[str] = Field(default=None, description="Currency code (e.g., USD).")
    tax_rate: Optional[float] = Field(default=None, description="Corporate tax rate (0-1).")
    sales_tax_rate: Optional[float] = Field(default=None, description="Sales tax/VAT rate (0-1).")
    working_capital_pct_sales: Optional[float] = Field(
        default=None, description="Working capital percentage of sales."
    )
    ev_ebitda_multiple: Optional[float] = Field(
        default=None, description="Exit multiple assumption for valuation."
    )
    sales_ramp_factors: Optional[List[float]] = Field(
        default=None, description="Sales ramp factors applied post-launch."
    )

    @field_validator("n_years")
    @classmethod
    def _positive_horizon(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n_years must be positive.")
        return value


class BiotechProductPayload(BaseModel):
    """Drug candidate inputs."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Product name.")
    stage: Optional[str] = Field(default=None, description="Clinical stage (e.g., Phase II).")
    success_prob: float = Field(..., description="Probability of success (0-1).")
    include_in_consolidation: bool = Field(
        default=True, description="Whether to include the product in consolidated outputs."
    )
    time_to_market: int = Field(..., description="Years until launch.")
    patent_years: int = Field(..., description="Patent life in years.")
    preexisting_market: bool = Field(default=False, description="Whether the market already exists.")
    patent_revenue_target: float = Field(..., description="Peak revenue during patent protection.")
    post_patent_revenue_target: float = Field(
        ..., description="Peak revenue after patent expiry."
    )
    market_growth_patent: float = Field(..., description="Annual growth during patent period.")
    market_growth_post: float = Field(..., description="Annual growth post patent.")
    cogs_patent: float = Field(..., description="COGS ratio during patent period.")
    cogs_post: float = Field(..., description="COGS ratio post patent.")
    sales_marketing_pct: float = Field(..., description="Sales & marketing cost ratio.")
    gna_pct: float = Field(..., description="G&A cost ratio.")
    royalty_pct: float = Field(..., description="Royalty percentage.")
    rd_remaining_pre_launch: float = Field(..., description="Remaining R&D spend before launch.")
    rd_annual_post_launch: float = Field(..., description="Annual R&D spend after launch.")
    capex_remaining_pre_launch: float = Field(..., description="Remaining capex before launch.")
    capex_annual_post_launch: float = Field(..., description="Annual capex after launch.")
    rd_capitalization_ratio: float = Field(..., description="Portion of R&D capitalized.")
    rd_amort_years: int = Field(..., description="Years to amortize capitalized R&D.")
    capex_dep_years: int = Field(..., description="Capex depreciation years.")

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("product name must be provided.")
        return text

    @field_validator("success_prob")
    @classmethod
    def _probability_range(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("success_prob must be between 0 and 1.")
        return float(value)


class BiotechInputsPayload(BaseModel):
    """Biotech model inputs payload."""

    model_config = ConfigDict(populate_by_name=True)

    config: BiotechConfigPayload = Field(..., alias="model_config")
    products: List[BiotechProductPayload]

    @model_validator(mode="after")
    def _ensure_products(self) -> "BiotechInputsPayload":
        if not self.products:
            raise ValueError("At least one product must be provided.")
        return self


__all__ = [
    "BiotechInputsPayload",
    "BiotechConfigPayload",
    "BiotechProductPayload",
]
