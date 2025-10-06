"""Utilities for loading and validating model input assumptions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional
import json


@dataclass
class ProductParameters:
    name: str
    production_cost: float
    selling_price: float
    freight_cost: float
    markup: Optional[float] = None


@dataclass
class UtilitySchedule:
    electricity_per_day: float
    water_per_day: float
    steam_per_hour: float
    operating_days: List[int]
    operating_hours: List[int]


@dataclass
class DepreciationItem:
    asset: str
    value: float
    useful_life: Optional[int]

    @property
    def annual_depreciation(self) -> float:
        if not self.useful_life or self.useful_life <= 0:
            return 0.0
        return self.value / self.useful_life


@dataclass
class FinancingParameters:
    initial_investment: float
    discount_rate: float
    senior_debt_interest: float
    revolver_interest: float
    cash_interest: float
    dividend_payout: float
    senior_debt_schedule: Mapping[int, float]
    revolver_initial: float
    share_capital: float


@dataclass
class WorkingCapitalDays:
    accounts_receivable: List[int]
    inventory: List[int]
    prepaid_expenses: List[int]
    other_assets: List[int]
    accounts_payable: List[int]
    other_liabilities: List[int]


@dataclass
class MonteCarloParameters:
    iterations: int
    revenue_growth_range: Iterable[float]
    metrics: List[str] = field(default_factory=lambda: ["NPV"])


@dataclass
class SensitivityParameters:
    variables: Mapping[str, Iterable[float]]


@dataclass
class ModelInputs:
    years: List[int]
    production_estimate: Mapping[str, List[float]]
    unit_costs: Mapping[str, ProductParameters]
    markup: Mapping[str, float]
    total_production_units: Mapping[str, float]
    production_capacity: Mapping[str, float]
    inflation_series: List[float]
    raw_material_cost_per_unit: float
    utility_schedule: UtilitySchedule
    direct_labor_costs: Mapping[str, float]
    indirect_labor_costs: Mapping[str, float]
    depreciation_items: List[DepreciationItem]
    capital_expenditure: Mapping[str, float]
    financing: FinancingParameters
    working_capital_days: WorkingCapitalDays
    tax_rate: float
    tax_rates: List[float]
    tax_timing_adjustment: float
    risk_schedule: Mapping[str, List[float]]
    scenarios: Mapping[str, Mapping[str, List[float]]]
    sensitivity: SensitivityParameters
    monte_carlo: MonteCarloParameters

    @property
    def products(self) -> List[str]:
        return list(self.production_estimate.keys())


def _parse_product_parameters(data: Mapping[str, Mapping[str, float]],
                              markup: Mapping[str, float]) -> Dict[str, ProductParameters]:
    return {
        name: ProductParameters(
            name=name,
            production_cost=values["production"],
            selling_price=values["price"],
            freight_cost=values.get("freight", 0.0),
            markup=markup.get(name),
        )
        for name, values in data.items()
    }


def _parse_depreciation(data: Mapping[str, Mapping[str, Optional[float]]]) -> List[DepreciationItem]:
    items: List[DepreciationItem] = []
    for asset, values in data.items():
        items.append(
            DepreciationItem(
                asset=asset,
                value=float(values.get("value", 0.0)),
                useful_life=values.get("life") if values.get("life") is not None else None,
            )
        )
    return items


def _parse_working_capital(days: Mapping[str, List[int]]) -> WorkingCapitalDays:
    return WorkingCapitalDays(
        accounts_receivable=days["accounts_receivable"],
        inventory=days["inventory"],
        prepaid_expenses=days["prepaid_expenses"],
        other_assets=days["other_assets"],
        accounts_payable=days["accounts_payable"],
        other_liabilities=days["other_liabilities"],
    )


def _parse_financing(financing: Mapping[str, object]) -> FinancingParameters:
    return FinancingParameters(
        initial_investment=float(financing["initial_investment"]),
        discount_rate=float(financing["discount_rate"]),
        senior_debt_interest=float(financing["senior_debt_interest"]),
        revolver_interest=float(financing["revolver_interest"]),
        cash_interest=float(financing["cash_interest"]),
        dividend_payout=float(financing["dividend_payout"]),
        senior_debt_schedule={int(year): float(value) for year, value in financing["senior_debt_schedule"].items()},
        revolver_initial=float(financing["revolver_initial"]),
        share_capital=float(financing["share_capital"]),
    )


def _parse_sensitivity(data: Mapping[str, Iterable[float]]) -> SensitivityParameters:
    return SensitivityParameters(variables=data)


def _coerce_schedule(values: Iterable[float], length: int) -> List[float]:
    """Normalise a schedule to match the projection horizon length."""
    sequence = [float(value) for value in values]
    if not sequence:
        return [0.0 for _ in range(length)]
    if len(sequence) >= length:
        return sequence[:length]
    padding = [sequence[-1] for _ in range(length - len(sequence))]
    return sequence + padding


def parse_inputs(raw: Mapping[str, object]) -> ModelInputs:
    """Parse a mapping of raw inputs into :class:`ModelInputs`."""
    years = [int(year) for year in raw["years"]]
    unit_costs = _parse_product_parameters(raw["unit_costs"], raw["markup"])
    depreciation_items = _parse_depreciation(raw["depreciation"])
    working_capital = _parse_working_capital(raw["working_capital"]["days"])
    financing = _parse_financing(raw["financing"])
    sensitivity = _parse_sensitivity(raw["sensitivity"]["variables"])

    utility = UtilitySchedule(
        electricity_per_day=float(raw["utility_costs"]["electricity_per_day"]),
        water_per_day=float(raw["utility_costs"]["water_per_day"]),
        steam_per_hour=float(raw["utility_costs"]["steam_per_hour"]),
        operating_days=[int(x) for x in raw["utility_costs"]["days"]],
        operating_hours=[int(x) for x in raw["utility_costs"]["hours"]],
    )

    monte_source = raw["monte_carlo"]
    monte_carlo = MonteCarloParameters(
        iterations=int(monte_source["iterations"]),
        revenue_growth_range=monte_source["revenue_growth_range"],
        metrics=list(monte_source.get("metrics", ["NPV"])),
    )

    tax_data = raw["tax"]
    schedule = tax_data.get("schedule") or []
    tax_schedule = _coerce_schedule(schedule, len(years)) if schedule else [
        float(tax_data.get("rate", 0.0)) for _ in years
    ]

    production_estimate = raw["production_estimate"]

    total_units_raw = raw.get("total_production_units", {})
    capacity_raw = raw.get("production_capacity", {})

    total_units: Dict[str, float] = {}
    capacity: Dict[str, float] = {}
    for name in unit_costs:
        estimate = production_estimate.get(name, [])
        estimate_total = sum(float(value) for value in estimate)
        total_units[name] = float(total_units_raw.get(name, estimate_total)) or 0.0
        capacity[name] = float(capacity_raw.get(name, 0.0))

    risk_data = raw.get("risk", {})
    risk_schedule = {
        name: _coerce_schedule(values, len(years))
        for name, values in risk_data.items()
    }

    if not risk_schedule:
        risk_schedule = {"inherent": [0.0 for _ in years]}

    return ModelInputs(
        years=years,
        production_estimate=raw["production_estimate"],
        unit_costs=unit_costs,
        markup=raw["markup"],
        total_production_units=total_units,
        production_capacity=capacity,
        inflation_series=raw["inflation_series"],
        raw_material_cost_per_unit=float(raw["raw_material_cost"]["per_unit"]),
        utility_schedule=utility,
        direct_labor_costs=raw["labor"]["direct"],
        indirect_labor_costs=raw["labor"]["indirect"],
        depreciation_items=depreciation_items,
        capital_expenditure=raw["capital_expenditure"],
        financing=financing,
        working_capital_days=working_capital,
        tax_rate=float(tax_data.get("rate", 0.0)),
        tax_rates=tax_schedule,
        tax_timing_adjustment=float(tax_data.get("timing_adjustment", 0.0)),
        risk_schedule=risk_schedule,
        scenarios=raw["scenarios"],
        sensitivity=sensitivity,
        monte_carlo=monte_carlo,
    )


def load_inputs(path: Optional[Path] = None) -> ModelInputs:
    """Load model assumptions from JSON."""
    if path is None:
        path = Path(__file__).resolve().parent / "data" / "default_inputs.json"
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return parse_inputs(raw)


__all__ = [
    "ModelInputs",
    "ProductParameters",
    "UtilitySchedule",
    "DepreciationItem",
    "FinancingParameters",
    "WorkingCapitalDays",
    "MonteCarloParameters",
    "SensitivityParameters",
    "parse_inputs",
    "load_inputs",
]
