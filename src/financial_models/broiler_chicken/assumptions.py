"""Core assumptions and helpers for the broiler chicken financial model."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class Assumptions:
    farm_name: str = "Baseline Broiler Farm"
    cycles_per_year: int = 6
    production_start_year: int = 2025
    production_horizon_years: int = 10
    birds_per_cycle: int = 20000
    mortality_rate: float = 0.05
    final_weight_kg: float = 2.5
    live_price_per_kg: float = 1.85
    eggs_price_per_dozen: float = 1.9
    manure_price_per_ton: float = 45.0
    live_bird_price_per_head: float = 1.5
    byproduct_price_per_kg: float = 0.35
    chick_cost: float = 0.55
    feed_conversion_ratio: float = 1.65
    feed_cost_per_kg: float = 0.42
    processing_cost_per_bird: float = 0.18
    vaccination_cost_per_bird: float = 0.06
    litter_disposal_per_cycle: float = 3200.0
    propane_per_cycle: float = 4100.0
    electricity_per_cycle: float = 1800.0
    labor_per_cycle: float = 9500.0
    maintenance_per_cycle: float = 2400.0
    management_fee_per_cycle: float = 3500.0
    insurance_per_cycle: float = 1200.0
    overhead_per_cycle: float = 2700.0
    capex_housing: float = 950000.0
    capex_equipment: float = 280000.0
    working_capital: float = 60000.0
    discount_rate: float = 0.1
    price_growth: float = 0.02
    cost_inflation: float = 0.015
    tax_rate: float = 0.24
    debt_ratio: float = 0.55
    debt_interest_rate: float = 0.055
    debt_term_years: int = 7
    depreciation_years: int = 15
    maintenance_capex_annual: float = 25000.0


ASSUMPTION_SCHEDULE_LAYOUT = [
    ("Production", "Farm name", "farm_name"),
    ("Production", "Cycles per year", "cycles_per_year"),
    ("Production", "Production start year", "production_start_year"),
    ("Production", "Production horizon (years)", "production_horizon_years"),
    ("Production", "Birds placed per cycle", "birds_per_cycle"),
    ("Production", "Mortality rate", "mortality_rate"),
    ("Production", "Final weight (kg)", "final_weight_kg"),
    ("Production", "Feed conversion ratio", "feed_conversion_ratio"),
    ("Production", "Live price per kg", "live_price_per_kg"),
    ("Production", "Eggs price per dozen", "eggs_price_per_dozen"),
    ("Production", "Manure price per ton", "manure_price_per_ton"),
    ("Production", "Live bird price per head", "live_bird_price_per_head"),
    ("Production", "By-product price per kg", "byproduct_price_per_kg"),
    ("Production", "Annual price growth", "price_growth"),
    ("Operating costs", "Feed cost per kg", "feed_cost_per_kg"),
    ("Operating costs", "Chick cost per bird", "chick_cost"),
    ("Operating costs", "Processing cost per bird", "processing_cost_per_bird"),
    ("Operating costs", "Vaccination cost per bird", "vaccination_cost_per_bird"),
    ("Operating costs", "Litter & disposal per cycle", "litter_disposal_per_cycle"),
    ("Operating costs", "Propane per cycle", "propane_per_cycle"),
    ("Operating costs", "Electricity per cycle", "electricity_per_cycle"),
    ("Operating costs", "Labor per cycle", "labor_per_cycle"),
    ("Operating costs", "Maintenance per cycle", "maintenance_per_cycle"),
    ("Operating costs", "Management fee per cycle", "management_fee_per_cycle"),
    ("Operating costs", "Insurance per cycle", "insurance_per_cycle"),
    ("Operating costs", "Overhead per cycle", "overhead_per_cycle"),
    ("Capital structure", "Housing capex", "capex_housing"),
    ("Capital structure", "Equipment capex", "capex_equipment"),
    ("Capital structure", "Maintenance capex (annual)", "maintenance_capex_annual"),
    ("Capital structure", "Working capital", "working_capital"),
    ("Capital structure", "Depreciation years", "depreciation_years"),
    ("Financing", "Debt ratio", "debt_ratio"),
    ("Financing", "Debt interest rate", "debt_interest_rate"),
    ("Financing", "Debt term (years)", "debt_term_years"),
    ("Financing", "Discount rate", "discount_rate"),
    ("Financing", "Cost inflation", "cost_inflation"),
    ("Financing", "Tax rate", "tax_rate"),
]


REVENUE_CATEGORIES = [
    "Broiler Revenue",
    "Eggs Revenue",
    "Poultry Manure Revenue",
    "Live Birds Revenue",
    "By-Product (feathers, offal, livers) Revenue",
]


def build_assumptions_schedule(assumptions: Assumptions) -> List[Dict[str, Any]]:
    """Return a tabular schedule summarising model assumptions grouped by schedule."""

    raw = asdict(assumptions)
    start_year = int(raw.get("production_start_year", 0) or 0)
    horizon_years = int(raw.get("production_horizon_years", 0) or 0)
    if horizon_years <= 0:
        horizon_years = 1
    end_year = start_year + horizon_years - 1 if start_year else horizon_years
    raw["production_end_year"] = end_year
    schedule_rows: List[Dict[str, Any]] = []
    for schedule, label, key in ASSUMPTION_SCHEDULE_LAYOUT:
        schedule_rows.append(
            {
                "schedule": schedule,
                "item": label,
                "value": raw.get(key),
            }
        )
    schedule_rows.append(
        {
            "schedule": "Production",
            "item": "Production end year",
            "value": raw.get("production_end_year"),
        }
    )
    return schedule_rows
