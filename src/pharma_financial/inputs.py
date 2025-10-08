"""Utilities for loading and validating model input assumptions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
import json


@dataclass
class ProductParameters:
    name: str
    production_cost: float
    selling_price: float
    freight_cost: float
    markup: Optional[float] = None


@dataclass
class BreakEvenRow:
    product: str
    fixed_cost: float
    selling_price: float
    variable_cost: float
    target_profit: float = 0.0
    expected_volume: float = 0.0


@dataclass
class DistributorCommissionRow:
    year: int
    product: str
    rate: float
    payment_days: int = 0
    revenue_share: float = 1.0


@dataclass
class UtilitySchedule:
    electricity_per_day: List[float]
    electricity_rate: List[float]
    electricity_days: List[int]
    water_per_day: List[float]
    water_rate: List[float]
    water_days: List[int]
    steam_per_hour: List[float]
    steam_rate: List[float]
    steam_days: List[int]
    steam_hours: List[int]

    def annual_totals(self) -> List[float]:
        totals: List[float] = []
        length = len(self.electricity_per_day)
        for idx in range(length):
            electricity = (
                self.electricity_per_day[idx]
                * self.electricity_rate[idx]
                * self.electricity_days[idx]
            )
            water = (
                self.water_per_day[idx]
                * self.water_rate[idx]
                * self.water_days[idx]
            )
            steam = (
                self.steam_per_hour[idx]
                * self.steam_rate[idx]
                * self.steam_days[idx]
                * self.steam_hours[idx]
            )
            totals.append(electricity + water + steam)
        return totals


@dataclass
class DepreciationRow:
    asset_type: str
    year: int
    acquisition: float
    depreciation_rate: float
    asset_life: Optional[int] = None
    method: str = "straight_line"
    opening_net_book: float = 0.0
    opening_cumulative: float = 0.0
    override_net_book: bool = False
    override_cumulative: bool = False


@dataclass
class FinancingParameters:
    initial_investment: float
    discount_rate: float
    senior_debt_interest: float
    revolver_interest: float
    cash_interest: float
    dividend_payout: float
    share_capital: float
    senior_debt_entries: List["DebtEntry"]
    revolver_entries: List["DebtEntry"]
    overdraft_entries: List["DebtEntry"]


@dataclass
class DebtEntry:
    year: int
    amount: float
    outstanding: float
    duration: int = 1

    def first_payment(self, rate: float) -> float:
        """Return the first scheduled payment for the debt entry.

        The payment is calculated from the current outstanding balance and the
        configured interest rate.  When the duration is one period or the
        computed payment would exceed the remaining outstanding balance, the
        method returns the full outstanding amount so that the liability is
        extinguished within the configured life span.
        """

        principal = max(float(self.amount), float(self.outstanding))
        outstanding = min(float(self.outstanding), principal)
        cumulative_interest = principal - outstanding
        current_outstanding = max(principal - cumulative_interest, 0.0)
        if current_outstanding <= 0:
            return 0.0

        duration = max(int(self.duration or 0), 1)
        base_payment = current_outstanding * float(rate)
        principal_share = current_outstanding / duration if duration > 0 else current_outstanding
        payment = max(base_payment, principal_share)
        if payment > current_outstanding:
            payment = current_outstanding
        return payment


@dataclass
class WorkingCapitalDays:
    accounts_receivable: List[int]
    inventory: List[int]
    prepaid_expenses: List[int]
    other_assets: List[int]
    accounts_payable: List[int]
    other_liabilities: List[int]
    calendar_days: List[int]


@dataclass
class MonteCarloParameters:
    iterations: int
    revenue_growth_range: Iterable[float]
    metrics: List[str] = field(default_factory=lambda: ["NPV"])
    variables: List[str] = field(default_factory=lambda: ["revenue_growth"])


@dataclass
class SensitivityParameters:
    variables: Mapping[str, Iterable[float]]


@dataclass
class GoalSeekParameters:
    metric: str
    target: float
    source: str = "income_statement"
    year: Optional[int] = None


@dataclass
class AIParameters:
    enabled: bool = False
    provider: str = "OpenAI"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    forecast_horizon: int = 3
    ml_methods: List[str] = field(default_factory=lambda: ["linear_regression"])
    generative_features: List[str] = field(default_factory=lambda: ["summary"])


@dataclass
class ModelInputs:
    years: List[int]
    production_estimate: Mapping[str, List[float]]
    unit_costs: Mapping[str, ProductParameters]
    markup: Mapping[str, float]
    total_production_units: Mapping[str, float]
    production_capacity: Mapping[str, float]
    break_even_rows: List[BreakEvenRow]
    fixed_cost_overrides: Mapping[str, float]
    variable_cost_overrides: Mapping[str, float]
    inflation_series: List[float]
    raw_material_cost_per_unit: float
    utility_schedule: UtilitySchedule
    direct_labor_costs: Mapping[str, float]
    indirect_labor_costs: Mapping[str, float]
    depreciation_schedule: List[DepreciationRow]
    distributor_commission: List[DistributorCommissionRow]
    capital_expenditure: Mapping[str, float]
    financing: FinancingParameters
    working_capital_days: WorkingCapitalDays
    tax_rate: float
    tax_rates: List[float]
    tax_timing_adjustment: float
    risk_schedule: Mapping[str, List[float]]
    scenarios: Mapping[str, Mapping[str, List[float]]]
    scenario_tools: Mapping[str, List[str]]
    sensitivity: SensitivityParameters
    monte_carlo: MonteCarloParameters
    goal_seek: Optional[GoalSeekParameters]
    ai: AIParameters

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


def _parse_depreciation_schedule(
    data: object, years: Iterable[int]
) -> List[DepreciationRow]:
    rows: List[DepreciationRow] = []

    if isinstance(data, Mapping):
        raw_rows = data.get("rows")
    else:
        raw_rows = None

    if isinstance(raw_rows, Iterable):
        for item in raw_rows:
            if not isinstance(item, Mapping):
                continue
            asset = str(item.get("asset_type") or item.get("asset") or "").strip()
            if not asset:
                continue
            year_value = item.get("year")
            try:
                year = int(year_value)
            except (TypeError, ValueError):
                continue
            acquisition = float(
                item.get("acquisition", item.get("asset_cost", 0.0)) or 0.0
            )
            rate = float(
                item.get("depreciation_rate", item.get("rate", 0.0)) or 0.0
            )
            life_value = item.get("asset_life")
            try:
                asset_life = int(life_value) if life_value not in (None, "") else None
            except (TypeError, ValueError):
                asset_life = None

            method_value = str(
                item.get("method", "straight_line") or "straight_line"
            ).strip().lower()
            if method_value not in {"straight_line", "reducing_balance"}:
                method_value = "straight_line"
            opening_net_book = float(item.get("opening_net_book", 0.0) or 0.0)
            opening_cumulative = float(item.get("opening_cumulative", 0.0) or 0.0)
            has_opening_nb = "opening_net_book" in item
            has_opening_cum = "opening_cumulative" in item
            override_net_flag = item.get("override_net_book")
            override_cum_flag = item.get("override_cumulative")
            rows.append(
                DepreciationRow(
                    asset_type=asset,
                    year=year,
                    acquisition=acquisition,
                    depreciation_rate=rate,
                    asset_life=asset_life,
                    method=method_value,
                    opening_net_book=opening_net_book,
                    opening_cumulative=opening_cumulative,
                    override_net_book=bool(
                        has_opening_nb if override_net_flag is None else override_net_flag
                    ),
                    override_cumulative=bool(
                        has_opening_cum if override_cum_flag is None else override_cum_flag
                    ),
                )
            )

    if rows:
        rows.sort(key=lambda row: (row.asset_type, row.year))
        return rows

    # Backwards compatibility for the legacy straight-line mapping format.
    if not isinstance(data, Mapping):
        return rows

    sequence_years = list(years)
    fallback: List[DepreciationRow] = []
    for asset, values in data.items():
        if not isinstance(values, Mapping):
            continue
        base_value = float(values.get("value", 0.0) or 0.0)
        life_value = values.get("life")
        useful_life: Optional[int]
        try:
            useful_life = int(life_value) if life_value not in (None, "") else None
        except (TypeError, ValueError):
            useful_life = None

        previous_net_book = 0.0
        previous_cumulative = 0.0

        for index, year in enumerate(sequence_years):
            acquisition = base_value if index == 0 else 0.0

            if useful_life and useful_life > 0 and index < useful_life:
                annual_depreciation = base_value / useful_life
            else:
                annual_depreciation = 0.0

            total_asset_cost = acquisition + previous_net_book
            rate = annual_depreciation / total_asset_cost if total_asset_cost else 0.0
            cumulative = previous_cumulative + annual_depreciation
            net_book = total_asset_cost - cumulative
            if net_book < 0 and annual_depreciation > 0:
                net_book = 0.0
                cumulative = total_asset_cost

            fallback.append(
                DepreciationRow(
                    asset_type=str(asset),
                    year=int(year),
                    acquisition=acquisition,
                    depreciation_rate=rate,
                    asset_life=useful_life,
                    method="straight_line",
                    opening_net_book=previous_net_book,
                    opening_cumulative=previous_cumulative,
                    override_net_book=index == 0,
                    override_cumulative=index == 0,
                )
            )

            previous_net_book = net_book
            previous_cumulative = cumulative

    fallback.sort(key=lambda row: (row.asset_type, row.year))
    return fallback


def _coerce_percentage(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric > 1.0:
        return numeric / 100.0
    if numeric < 0.0:
        return 0.0
    return numeric


def _parse_distributor_commission(
    data: object,
    years: Sequence[int],
    unit_costs: Mapping[str, ProductParameters],
) -> List[DistributorCommissionRow]:
    rows: list[DistributorCommissionRow] = []

    if isinstance(data, Mapping):
        raw_rows = data.get("rows")
        if raw_rows is None and all(
            key in data for key in ("year", "product", "rate")
        ):
            raw_rows = [data]
    else:
        raw_rows = data if isinstance(data, Iterable) else []

    if isinstance(raw_rows, Iterable):
        for item in raw_rows:
            if not isinstance(item, Mapping):
                continue
            product = str(item.get("product", "")).strip()
            if not product:
                continue
            year_value = item.get("year")
            try:
                year = int(year_value)
            except (TypeError, ValueError):
                continue
            rate = _coerce_percentage(item.get("rate", item.get("percentage", 0.0)))
            payment_days_value = item.get("payment_days", item.get("days", 0))
            try:
                payment_days = int(payment_days_value)
            except (TypeError, ValueError):
                payment_days = 0
            revenue_share = _coerce_percentage(
                item.get("revenue_share", item.get("share", 1.0))
            )
            rows.append(
                DistributorCommissionRow(
                    year=year,
                    product=product,
                    rate=rate,
                    payment_days=payment_days,
                    revenue_share=revenue_share if revenue_share > 0 else 1.0,
                )
            )

    if rows:
        rows.sort(key=lambda row: (row.year, row.product.lower()))
        return rows

    return []


def _parse_ai(data: object) -> AIParameters:
    if not isinstance(data, Mapping):
        data = {}

    enabled = bool(data.get("enabled", False))
    provider = str(data.get("provider", "OpenAI") or "OpenAI")
    model = str(data.get("model", "gpt-4") or "gpt-4")

    horizon_value = data.get("forecast_horizon", 3)
    try:
        forecast_horizon = int(float(horizon_value))
    except (TypeError, ValueError):
        forecast_horizon = 3
    forecast_horizon = max(forecast_horizon, 0)

    def _clean_list(values: object, default: list[str]) -> list[str]:
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            cleaned = [str(value).strip() for value in values if str(value).strip()]
        else:
            cleaned = [str(values).strip()] if values else []
        return cleaned or default

    ml_methods = _clean_list(data.get("ml_methods", ["linear_regression"]), ["linear_regression"])
    generative_features = _clean_list(data.get("generative_features", ["summary"]), ["summary"])

    api_key_value = data.get("api_key")
    if isinstance(api_key_value, (str, bytes)):
        api_key = api_key_value.strip() or None
    else:
        api_key = None

    return AIParameters(
        enabled=enabled,
        provider=provider,
        model=model,
        api_key=api_key,
        forecast_horizon=forecast_horizon,
        ml_methods=ml_methods,
        generative_features=generative_features,
    )


def _parse_working_capital(
    data: Mapping[str, object], years: Sequence[int]
) -> WorkingCapitalDays:
    length = len(list(years))

    if isinstance(data, Mapping) and isinstance(data.get("days"), Mapping):
        days_mapping = data.get("days", {})  # type: ignore[assignment]
    else:
        days_mapping = data

    if not isinstance(days_mapping, Mapping):
        raise TypeError("working capital days must be provided as a mapping")

    def _extract(name: str) -> List[int]:
        values = days_mapping.get(name, [])  # type: ignore[assignment]
        if isinstance(values, Mapping):
            values = list(values.values())
        return _coerce_int_schedule(values, length)

    calendar_source: Iterable[float] | None = None
    if isinstance(data, Mapping):
        calendar_candidate = (
            data.get("calendar_days")
            or data.get("calendar")
            or data.get("days_in_year")
        )
        if isinstance(calendar_candidate, Iterable) and not isinstance(calendar_candidate, (str, bytes)):
            calendar_source = [float(value) for value in calendar_candidate]

    if calendar_source is not None:
        calendar_days = _coerce_int_schedule(calendar_source, length)
    else:
        calendar_days = [366 if year % 4 == 0 else 365 for year in years]

    return WorkingCapitalDays(
        accounts_receivable=_extract("accounts_receivable"),
        inventory=_extract("inventory"),
        prepaid_expenses=_extract("prepaid_expenses"),
        other_assets=_extract("other_assets"),
        accounts_payable=_extract("accounts_payable"),
        other_liabilities=_extract("other_liabilities"),
        calendar_days=calendar_days,
    )


def _parse_debt_entries(data: object) -> List[DebtEntry]:
    entries: List[DebtEntry] = []

    if data is None:
        iterable: Iterable[Mapping[str, object]] = []
    elif isinstance(data, Mapping):
        iterable = data.values()  # type: ignore[assignment]
    else:
        iterable = data  # type: ignore[assignment]

    for item in iterable:
        if item is None:
            continue
        year_value = item.get("year") if isinstance(item, Mapping) else None
        if year_value is None:
            continue
        try:
            year = int(year_value)
        except (TypeError, ValueError):
            continue
        amount = float(item.get("amount", 0.0)) if isinstance(item, Mapping) else 0.0
        outstanding = float(item.get("outstanding", amount)) if isinstance(item, Mapping) else amount
        duration_value = item.get("duration") if isinstance(item, Mapping) else None
        try:
            duration = int(duration_value) if duration_value not in (None, "") else 1
        except (TypeError, ValueError):
            duration = 1
        duration = max(1, duration)
        entries.append(
            DebtEntry(year=year, amount=amount, outstanding=outstanding, duration=duration)
        )
    entries.sort(key=lambda entry: entry.year)
    return entries


def _parse_financing(financing: Mapping[str, object]) -> FinancingParameters:
    return FinancingParameters(
        initial_investment=float(financing["initial_investment"]),
        discount_rate=float(financing["discount_rate"]),
        senior_debt_interest=float(financing["senior_debt_interest"]),
        revolver_interest=float(financing["revolver_interest"]),
        cash_interest=float(financing["cash_interest"]),
        dividend_payout=float(financing["dividend_payout"]),
        share_capital=float(financing["share_capital"]),
        senior_debt_entries=_parse_debt_entries(financing.get("senior_debt", [])),
        revolver_entries=_parse_debt_entries(financing.get("revolver", [])),
        overdraft_entries=_parse_debt_entries(financing.get("overdraft", [])),
    )


def _parse_sensitivity(data: Mapping[str, Iterable[float]]) -> SensitivityParameters:
    return SensitivityParameters(variables=data)


def _parse_break_even_rows(
    raw: object,
    unit_costs: Mapping[str, ProductParameters],
    total_units: Mapping[str, float],
    fixed_overrides: Optional[Mapping[str, float]] = None,
    variable_overrides: Optional[Mapping[str, float]] = None,
) -> List[BreakEvenRow]:
    rows: List[BreakEvenRow] = []

    fixed_overrides = fixed_overrides or {}
    variable_overrides = variable_overrides or {}

    if isinstance(raw, Mapping):
        candidate_rows = raw.get("rows", raw.get("data", []))
    else:
        candidate_rows = raw or []

    if not isinstance(candidate_rows, Iterable):
        return rows

    for entry in candidate_rows:
        if not isinstance(entry, Mapping):
            continue

        product = str(entry.get("product") or entry.get("Product") or "").strip()
        if not product:
            continue

        params = unit_costs.get(product)
        default_price = params.selling_price if params else 0.0
        default_volume = float(total_units.get(product, 0.0))

        if "fixed_cost" in entry or "Fixed Cost" in entry:
            fixed_cost = float(entry.get("fixed_cost", entry.get("Fixed Cost", 0.0)) or 0.0)
        else:
            fixed_cost = float(fixed_overrides.get(product, 0.0) or 0.0)

        selling_price = float(
            entry.get("selling_price", entry.get("Selling Price", default_price)) or default_price
        )

        if "variable_cost" in entry or "Variable Cost" in entry:
            variable_cost = float(
                entry.get("variable_cost", entry.get("Variable Cost", 0.0)) or 0.0
            )
        else:
            variable_cost = float(variable_overrides.get(product, 0.0) or 0.0)

        target_profit = float(entry.get("target_profit", entry.get("Target Profit", 0.0)))
        expected_volume = float(entry.get("expected_volume", entry.get("Expected Volume", default_volume)))

        rows.append(
            BreakEvenRow(
                product=product,
                fixed_cost=fixed_cost,
                selling_price=selling_price,
                variable_cost=variable_cost,
                target_profit=target_profit,
                expected_volume=expected_volume,
            )
        )

    return rows


def _parse_fixed_variable_costs(raw: object) -> tuple[Dict[str, float], Dict[str, float]]:
    fixed: Dict[str, float] = {}
    variable: Dict[str, float] = {}

    if isinstance(raw, Mapping):
        entries = raw.get("rows", raw.get("data", []))
    else:
        entries = raw or []

    if not isinstance(entries, Iterable):
        return fixed, variable

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        product = str(entry.get("product") or entry.get("Product") or "").strip()
        if not product:
            continue

        has_fixed_key = "fixed_cost" in entry or "Fixed Cost" in entry
        fixed_value = entry.get("fixed_cost") if "fixed_cost" in entry else entry.get("Fixed Cost")

        if has_fixed_key:
            try:
                fixed[product] = float(fixed_value or 0.0)
            except (TypeError, ValueError):
                fixed[product] = 0.0

        has_variable_key = "variable_cost" in entry or "Variable Cost" in entry
        variable_value = (
            entry.get("variable_cost") if "variable_cost" in entry else entry.get("Variable Cost")
        )

        if has_variable_key:
            try:
                variable[product] = float(variable_value or 0.0)
            except (TypeError, ValueError):
                variable[product] = 0.0

        if product not in variable:
            variable[product] = 0.0

    return fixed, variable


def _coerce_schedule(values: Iterable[float], length: int) -> List[float]:
    """Normalise a schedule to match the projection horizon length."""
    sequence = [float(value) for value in values]
    if not sequence:
        return [0.0 for _ in range(length)]
    if len(sequence) >= length:
        return sequence[:length]
    padding = [sequence[-1] for _ in range(length - len(sequence))]
    return sequence + padding


def _coerce_int_schedule(values: Iterable[float], length: int) -> List[int]:
    return [int(round(value)) for value in _coerce_schedule(values, length)]


def parse_inputs(raw: Mapping[str, object]) -> ModelInputs:
    """Parse a mapping of raw inputs into :class:`ModelInputs`."""
    years = [int(year) for year in raw["years"]]
    unit_costs = _parse_product_parameters(raw["unit_costs"], raw["markup"])
    depreciation_schedule = _parse_depreciation_schedule(raw.get("depreciation", {}), years)
    commission_rows = _parse_distributor_commission(
        raw.get("distributor_commission"), years, unit_costs
    )
    working_capital = _parse_working_capital(raw["working_capital"], years)
    financing = _parse_financing(raw["financing"])
    sensitivity = _parse_sensitivity(raw["sensitivity"]["variables"])

    utility_source = raw["utility_costs"]
    years_length = len(years)
    utility_rows = list(utility_source.get("years", [])) if isinstance(utility_source, Mapping) else []

    if utility_rows:
        def _floats(key: str) -> List[float]:
            return _coerce_schedule(
                [float(row.get(key, 0.0) or 0.0) for row in utility_rows],
                years_length,
            )

        def _ints(key: str) -> List[int]:
            return _coerce_int_schedule(
                [float(row.get(key, 0.0) or 0.0) for row in utility_rows],
                years_length,
            )

        utility = UtilitySchedule(
            electricity_per_day=_floats("electricity_per_day"),
            electricity_rate=_floats("electricity_rate"),
            electricity_days=_ints("electricity_days"),
            water_per_day=_floats("water_per_day"),
            water_rate=_floats("water_rate"),
            water_days=_ints("water_days"),
            steam_per_hour=_floats("steam_per_hour"),
            steam_rate=_floats("steam_rate"),
            steam_days=_ints("steam_days"),
            steam_hours=_ints("steam_hours"),
        )
    else:
        days = utility_source.get("days", []) if isinstance(utility_source, Mapping) else []
        hours = utility_source.get("hours", []) if isinstance(utility_source, Mapping) else []
        electricity_per_day = float(utility_source.get("electricity_per_day", 0.0)) if isinstance(utility_source, Mapping) else 0.0
        water_per_day = float(utility_source.get("water_per_day", 0.0)) if isinstance(utility_source, Mapping) else 0.0
        steam_per_hour = float(utility_source.get("steam_per_hour", 0.0)) if isinstance(utility_source, Mapping) else 0.0

        electricity_days = _coerce_int_schedule(days, years_length)
        water_days = electricity_days
        steam_hours = _coerce_int_schedule(hours, years_length)
        steam_days = _coerce_int_schedule([1 for _ in range(len(steam_hours))], years_length)

        utility = UtilitySchedule(
            electricity_per_day=[electricity_per_day for _ in range(years_length)],
            electricity_rate=[1.0 for _ in range(years_length)],
            electricity_days=electricity_days,
            water_per_day=[water_per_day for _ in range(years_length)],
            water_rate=[1.0 for _ in range(years_length)],
            water_days=water_days,
            steam_per_hour=[steam_per_hour for _ in range(years_length)],
            steam_rate=[1.0 for _ in range(years_length)],
            steam_days=steam_days,
            steam_hours=steam_hours,
        )

    monte_source = raw["monte_carlo"]
    monte_carlo = MonteCarloParameters(
        iterations=int(monte_source["iterations"]),
        revenue_growth_range=monte_source["revenue_growth_range"],
        metrics=list(monte_source.get("metrics", ["NPV"])),
        variables=[
            str(value)
            for value in monte_source.get("variables", ["revenue_growth"])
            if str(value)
        ],
    )

    tax_data = raw["tax"]
    schedule = tax_data.get("schedule") or []
    tax_schedule = _coerce_schedule(schedule, len(years)) if schedule else [
        float(tax_data.get("rate", 0.0)) for _ in years
    ]
    inflation_series = _coerce_schedule(raw.get("inflation_series", []), len(years))

    production_source = raw.get("production_estimate", {})
    production_estimate: Dict[str, List[float]] = {}
    if isinstance(production_source, Mapping):
        for name, values in production_source.items():
            if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
                sequence = [float(value) for value in values]
            else:
                sequence = [float(values) if values is not None else 0.0]
            production_estimate[str(name)] = _coerce_schedule(sequence, years_length)
    else:
        production_source = {}


    total_units_raw = raw.get("total_production_units", {})
    capacity_raw = raw.get("production_capacity", {})

    total_units: Dict[str, float] = {}
    capacity: Dict[str, float] = {}
    for name in unit_costs:
        estimate = production_estimate.get(name, [])
        estimate_total = sum(float(value) for value in estimate)
        total_units[name] = float(total_units_raw.get(name, estimate_total)) or 0.0
        capacity[name] = float(capacity_raw.get(name, 0.0))

    fixed_overrides, variable_overrides = _parse_fixed_variable_costs(
        raw.get("fixed_variable_costs")
    )
    for product in raw["production_estimate"].keys():
        variable_overrides.setdefault(product, 0.0)

    break_even_rows = _parse_break_even_rows(
        raw.get("break_even"),
        unit_costs,
        total_units,
        fixed_overrides=fixed_overrides,
        variable_overrides=variable_overrides,
    )

    risk_data = raw.get("risk", {})
    risk_schedule = {
        name: _coerce_schedule(values, len(years))
        for name, values in risk_data.items()
    }

    if not risk_schedule:
        risk_schedule = {"inherent": [0.0 for _ in years]}

    scenario_tools_raw = raw.get("scenario_tools", {})
    scenario_tools: Dict[str, List[str]] = {}
    if isinstance(scenario_tools_raw, Mapping):
        for key, values in scenario_tools_raw.items():
            key_name = str(key).strip()
            if not key_name:
                continue
            cleaned: List[str] = []
            if isinstance(values, Iterable):
                for value in values:
                    text = str(value).strip()
                    if text:
                        cleaned.append(text)
            scenario_tools[key_name] = cleaned
    else:
        scenario_tools = {}

    goal_seek = None
    goal_data = raw.get("goal_seek")
    if isinstance(goal_data, Mapping):
        metric = str(goal_data.get("metric", "Net Income"))
        target = float(goal_data.get("target", 0.0))
        source = str(goal_data.get("source", "income_statement"))
        year_value = goal_data.get("year")
        try:
            year = int(year_value) if year_value is not None else None
        except (TypeError, ValueError):
            year = None
        goal_seek = GoalSeekParameters(metric=metric, target=target, source=source, year=year)

    ai_params = _parse_ai(raw.get("ai", {}))

    if not production_estimate:
        production_estimate = {
            str(name): _coerce_schedule([], len(years)) for name in unit_costs.keys()
        }

    return ModelInputs(
        years=years,
        production_estimate=production_estimate,
        unit_costs=unit_costs,
        markup=raw["markup"],
        total_production_units=total_units,
        production_capacity=capacity,
        break_even_rows=break_even_rows,
        fixed_cost_overrides=fixed_overrides,
        variable_cost_overrides=variable_overrides,
        inflation_series=inflation_series,
        raw_material_cost_per_unit=float(raw["raw_material_cost"]["per_unit"]),
        utility_schedule=utility,
        direct_labor_costs=raw["labor"]["direct"],
        indirect_labor_costs=raw["labor"]["indirect"],
        depreciation_schedule=depreciation_schedule,
        distributor_commission=commission_rows,
        capital_expenditure=raw["capital_expenditure"],
        financing=financing,
        working_capital_days=working_capital,
        tax_rate=float(tax_data.get("rate", 0.0)),
        tax_rates=tax_schedule,
        tax_timing_adjustment=float(tax_data.get("timing_adjustment", 0.0)),
        risk_schedule=risk_schedule,
        scenarios=raw["scenarios"],
        scenario_tools=scenario_tools,
        sensitivity=sensitivity,
        monte_carlo=monte_carlo,
        goal_seek=goal_seek,
        ai=ai_params,
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
    "BreakEvenRow",
    "UtilitySchedule",
    "DepreciationRow",
    "FinancingParameters",
    "DebtEntry",
    "WorkingCapitalDays",
    "DistributorCommissionRow",
    "MonteCarloParameters",
    "SensitivityParameters",
    "GoalSeekParameters",
    "AIParameters",
    "parse_inputs",
    "load_inputs",
]
