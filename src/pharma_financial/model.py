"""Core financial model built without third-party scientific dependencies."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .ai import AIInsights, GenerativeAdvisor, MachineLearningAdvisor
from .debt import amortise_entries
from .inputs import (
    AssumptionEvidence,
    BankabilityParameters,
    BreakEvenRow,
    DebtEntry,
    DownsideCaseParameters,
    LaborModelParameters,
    LaborRoleParameters,
    ModelInputs,
    ProductParameters,
    WorkingCapitalDays,
)
from .table import Table, build_table


CASH_FLOW_NET_COLUMN = "Net Cash Flow for the Period"
CASH_FLOW_BEGIN_COLUMN = "Cash and Cash Equivalents at the Beginning of the Period"
CASH_FLOW_END_COLUMN = "Cash and Cash Equivalents at the End of the Period"


Number = float | int


@dataclass
class FinancialOutputs:
    income_statement: Table
    balance_sheet: Table
    cash_flow: Table
    summary_metrics: Table
    goal_seek: Table
    break_even: Table
    payback: Table
    discounted_payback: Table
    scenario_results: Dict[str, Table]
    sensitivity_results: Dict[str, Table]
    monte_carlo: Table
    scenario_tool_results: Mapping[str, "ScenarioToolResult"]
    risk_factor_diagnostics: Optional[Table] = None
    ai_insights: Optional[AIInsights] = None
    commercial_diagnostics: Optional[Table] = None
    bankability_gate: Optional[Table] = None
    data_quality_exceptions: Optional[List[Mapping[str, object]]] = None
    evidence_register: Optional[List[Mapping[str, object]]] = None
    sources_and_uses: Optional[Table] = None
    liquidity_bridge: Optional[Table] = None
    covenant_headroom: Optional[Table] = None
    downside_case_summary: Optional[Table] = None


@dataclass
class ScenarioToolResult:
    rows: List[Mapping[str, Any]]
    interpretation: str


def _cumulative(values: Iterable[Number]) -> List[float]:
    total = 0.0
    result: List[float] = []
    for value in values:
        total += float(value)
        result.append(total)
    return result


def _difference(values: Iterable[Number]) -> List[float]:
    result: List[float] = []
    previous: float | None = None
    for value in values:
        if previous is None:
            result.append(float(value))
        else:
            result.append(float(value) - previous)
        previous = float(value)
    return result


def _shift(values: List[float], fill_value: float = 0.0) -> List[float]:
    return [fill_value] + values[:-1]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("nan")
    return numerator / denominator


def _is_finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def _average(values: Iterable[Number]) -> float:
    cleaned: List[float] = []
    for value in values:
        numeric = float(value)
        if _is_finite(numeric):
            cleaned.append(numeric)
    if not cleaned:
        return float("nan")
    return sum(cleaned) / len(cleaned)


def _rolling_cagr(values: Sequence[Number], window: int) -> float:
    if window < 2:
        return float("nan")
    series = [float(value) for value in values]
    if len(series) < window:
        return float("nan")
    cagr_values: List[float] = []
    for idx in range(len(series) - window + 1):
        start = series[idx]
        end = series[idx + window - 1]
        if start <= 0 or end <= 0:
            continue
        cagr = (end / start) ** (1 / (window - 1)) - 1
        if _is_finite(cagr):
            cagr_values.append(cagr)
    return _average(cagr_values)


def _weighted_average(values: Iterable[Number], weights: Iterable[Number]) -> float:
    total_weight = 0.0
    total_value = 0.0
    for value, weight in zip(values, weights):
        numeric = float(value)
        weight_value = float(weight)
        if not _is_finite(numeric) or not _is_finite(weight_value):
            continue
        if weight_value <= 0:
            continue
        total_weight += weight_value
        total_value += numeric * weight_value
    if total_weight <= 0:
        return float("nan")
    return total_value / total_weight


def _score_range(value: float, low: float, high: float, *, inverse: bool = False) -> float:
    if not _is_finite(value) or high == low:
        return float("nan")
    scaled = (value - low) / (high - low)
    if inverse:
        scaled = 1.0 - scaled
    return max(0.0, min(1.0, scaled))


def _weighted_score(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total_weight = 0.0
    total_score = 0.0
    for key, score in scores.items():
        if not _is_finite(score):
            continue
        weight = float(weights.get(key, 0.0))
        if weight <= 0:
            continue
        total_weight += weight
        total_score += weight * score
    if total_weight <= 0:
        return float("nan")
    return total_score / total_weight


def _value_for_year(table: Table, column: str, year: Optional[int]) -> float:
    if column not in table.data:
        return float("nan")
    values = table.column(column)
    if not values:
        return float("nan")
    if year is None:
        return values[-1]
    try:
        position = table.index.index(year)
    except ValueError:
        position = len(values) - 1
    position = max(0, min(position, len(values) - 1))
    return values[position]


class FinancialModel:
    """Implements the Pharmaceuticals financial engine."""

    def __init__(self, inputs: ModelInputs):
        self.inputs = inputs
        self.years = inputs.years
        self.products = inputs.products
        self._inflation = self._build_inflation_factors(inputs.inflation_series)
        self._production_cache: Dict[str, List[float]] | None = None
        self._unit_prices_cache: Dict[str, float] | None = None
        self._unit_costs_cache: Dict[str, float] | None = None
        self._price_adjustments_cache: Dict[str, List[float]] | None = None
        self._variable_costs_cache: Dict[str, float] | None = None
        self._freight_costs_cache: Dict[str, float] | None = None
        self._total_units_cache: List[float] | None = None
        self._risk_factors_cache: List[float] | None = None
        self._risk_cost_factors_cache: List[float] | None = None
        self._depreciation_cache: "tuple[list[dict], dict[int, float], dict[int, float]] | None" = None
        self._senior_interest_cache: List[float] | None = None
        self._senior_outstanding_cache: List[float] | None = None
        self._revolver_interest_cache: List[float] | None = None
        self._revolver_outstanding_cache: List[float] | None = None
        self._overdraft_interest_cache: List[float] | None = None
        self._overdraft_outstanding_cache: List[float] | None = None
        self._commission_cache: dict[int, dict[str, tuple[float, float, int]]] | None = None
        self._distributor_receivable_cache: List[float] | None = None
        self._distributor_share_cache: List[float] | None = None
        self._interest_schedule_cache: List[float] | None = None
        self._tax_schedule_cache: List[float] | None = None
        self._revenue_schedule_cache: Table | None = None
        self._cost_structure_cache: Table | None = None
        self._income_statement_cache: Table | None = None
        self._cash_flow_cache: Table | None = None
        self._balance_sheet_cache: Table | None = None
        self._working_capital_cache: Table | None = None
        self._summary_metrics_cache: Table | None = None
        self._irr_result: "IRRResult | None" = None
        self._risk_factor_diagnostics_cache: Table | None = None
        self._calendar_days_cache: List[float] | None = None
        self._inflation_array_cache: np.ndarray | None = None
        self._risk_revenue_array_cache: np.ndarray | None = None
        self._risk_cost_array_cache: np.ndarray | None = None
        self._utility_arrays_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._labor_kpi_cache: Dict[str, float] | None = None
        self._monte_carlo_cache: Table | None = None
        self._commercial_diagnostics_cache: Table | None = None
        self._bankability_gate_cache: Table | None = None
        self._data_quality_exceptions_cache: List[Mapping[str, object]] | None = None
        self._evidence_register_cache: List[Mapping[str, object]] | None = None
        self._sources_and_uses_cache: Table | None = None
        self._liquidity_bridge_cache: Table | None = None
        self._covenant_headroom_cache: Table | None = None
        self._downside_case_summary_cache: Table | None = None

    # ------------------------------------------------------------------ core
    def _build_inflation_factors(self, series: Iterable[Number]) -> List[float]:
        factors: List[float] = []
        running = 1.0
        for rate in series:
            running *= 1.0 + float(rate)
            factors.append(running)
        return factors

    def _pad_series(
        self,
        series: Sequence[Number],
        length: int,
        *,
        fill: float,
    ) -> List[float]:
        if not series:
            return [float(fill) for _ in range(length)]
        values = [float(value) for value in series]
        if len(values) < length:
            values.extend([values[-1] for _ in range(length - len(values))])
        return values[:length]

    def _invalidate_statement_caches(self) -> None:
        self._revenue_schedule_cache = None
        self._cost_structure_cache = None
        self._income_statement_cache = None
        self._cash_flow_cache = None
        self._balance_sheet_cache = None
        self._working_capital_cache = None
        self._summary_metrics_cache = None
        self._irr_result = None
        self._risk_factors_cache = None
        self._risk_cost_factors_cache = None
        self._risk_factor_diagnostics_cache = None
        self._distributor_receivable_cache = None
        self._distributor_share_cache = None
        self._interest_schedule_cache = None
        self._tax_schedule_cache = None
        self._calendar_days_cache = None
        self._inflation_array_cache = None
        self._risk_revenue_array_cache = None
        self._risk_cost_array_cache = None
        self._utility_arrays_cache = None
        self._labor_kpi_cache = None
        self._monte_carlo_cache = None
        self._commercial_diagnostics_cache = None
        self._bankability_gate_cache = None
        self._data_quality_exceptions_cache = None
        self._evidence_register_cache = None
        self._sources_and_uses_cache = None
        self._liquidity_bridge_cache = None
        self._covenant_headroom_cache = None
        self._downside_case_summary_cache = None

    def _inflation_array(self) -> np.ndarray:
        if self._inflation_array_cache is None:
            inflation = self._pad_series(self._inflation, len(self.years), fill=1.0)
            self._inflation_array_cache = np.array(inflation, dtype=float)
        return self._inflation_array_cache

    def _risk_revenue_array(self) -> np.ndarray:
        if self._risk_revenue_array_cache is None:
            factors = self._pad_series(self._risk_factors(), len(self.years), fill=1.0)
            self._risk_revenue_array_cache = np.array(factors, dtype=float)
        return self._risk_revenue_array_cache

    def _risk_cost_array(self) -> np.ndarray:
        if self._risk_cost_array_cache is None:
            factors = self._pad_series(self._risk_cost_factors(), len(self.years), fill=1.0)
            self._risk_cost_array_cache = np.array(factors, dtype=float)
        return self._risk_cost_array_cache

    def _utility_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._utility_arrays_cache is None:
            year_count = len(self.years)
            utility = self.inputs.utility_schedule
            electricity = (
                np.array(self._pad_series(utility.electricity_per_day, year_count, fill=0.0), dtype=float)
                * np.array(self._pad_series(utility.electricity_rate, year_count, fill=0.0), dtype=float)
                * np.array(self._pad_series(utility.electricity_days, year_count, fill=0.0), dtype=float)
            )
            water = (
                np.array(self._pad_series(utility.water_per_day, year_count, fill=0.0), dtype=float)
                * np.array(self._pad_series(utility.water_rate, year_count, fill=0.0), dtype=float)
                * np.array(self._pad_series(utility.water_days, year_count, fill=0.0), dtype=float)
            )
            steam = (
                np.array(self._pad_series(utility.steam_per_hour, year_count, fill=0.0), dtype=float)
                * np.array(self._pad_series(utility.steam_rate, year_count, fill=0.0), dtype=float)
                * np.array(self._pad_series(utility.steam_days, year_count, fill=0.0), dtype=float)
                * np.array(self._pad_series(utility.steam_hours, year_count, fill=0.0), dtype=float)
            )
            self._utility_arrays_cache = (electricity, water, steam)
        return self._utility_arrays_cache

    def _production(self) -> Dict[str, List[float]]:
        if self._production_cache is not None:
            return self._production_cache
        production: Dict[str, List[float]] = {}
        year_count = len(self.years)
        for product in self.products:
            values = self.inputs.production_estimate.get(product, [])
            production[product] = self._pad_series(values, year_count, fill=0.0)
        self._production_cache = production
        return production

    def _unit_prices(self) -> Dict[str, float]:
        if self._unit_prices_cache is not None:
            return self._unit_prices_cache
        self._unit_prices_cache = {
            name: params.selling_price for name, params in self.inputs.unit_costs.items()
        }
        return self._unit_prices_cache

    def _unit_costs(self) -> Dict[str, float]:
        if self._unit_costs_cache is not None:
            return self._unit_costs_cache
        self._unit_costs_cache = {
            name: params.production_cost for name, params in self.inputs.unit_costs.items()
        }
        return self._unit_costs_cache

    def _price_adjustments(self) -> Dict[str, List[float]]:
        if self._price_adjustments_cache is not None:
            return self._price_adjustments_cache
        adjustments: Dict[str, List[float]] = {}
        year_count = len(self.years)
        for product in self.products:
            series = list(self.inputs.price_adjustments.get(product, []))
            if not series:
                adjustments[product] = [1.0 for _ in range(year_count)]
                continue
            adjustments[product] = self._pad_series(series, year_count, fill=float(series[-1]))
        self._price_adjustments_cache = adjustments
        return adjustments

    def _commission_parameters(self) -> dict[int, dict[str, tuple[float, float, int]]]:
        if self._commission_cache is not None:
            return self._commission_cache

        fallback_rates: dict[str, float] = {}
        for name in self.inputs.unit_costs.keys():
            # Default distributor commission rate is 5%
            fallback_rates[name] = 0.05

        schedule: dict[int, dict[str, tuple[float, float, int]]] = {}
        for idx, year in enumerate(self.years):
            per_product: dict[str, tuple[float, float, int]] = {}
            for product in self.products:
                base_rate = fallback_rates.get(product, 0.05)
                per_product[product] = (base_rate, 1.0, 0)
            schedule[int(year)] = per_product

        for row in self.inputs.distributor_commission:
            year = int(row.year)
            if year not in schedule:
                continue
            product = row.product
            if product not in schedule[year]:
                continue
            rate = max(float(row.rate), 0.0)
            share = float(row.revenue_share) if row.revenue_share > 0 else 1.0
            payment_days = max(int(row.payment_days), 0)
            schedule[year][product] = (rate, share, payment_days)

        self._commission_cache = schedule
        return schedule

    def _freight_costs(self) -> Dict[str, float]:
        if self._freight_costs_cache is not None:
            return self._freight_costs_cache
        self._freight_costs_cache = {
            name: params.freight_cost for name, params in self.inputs.unit_costs.items()
        }
        return self._freight_costs_cache

    def _variable_costs(self) -> Dict[str, float]:
        if self._variable_costs_cache is not None:
            return self._variable_costs_cache
        overrides = getattr(self.inputs, "variable_cost_overrides", {})
        raw_per_unit = max(float(getattr(self.inputs, "raw_material_cost_per_unit", 0.0) or 0.0), 0.0)
        factors = getattr(self.inputs, "raw_material_factors", {}) or {}

        resolved: Dict[str, float] = {}
        for product in self.products:
            override_value = overrides.get(product) if isinstance(overrides, Mapping) else None
            if override_value is not None:
                try:
                    resolved[product] = max(float(override_value), 0.0)
                    continue
                except (TypeError, ValueError):
                    pass
            factor = 1.0
            if isinstance(factors, Mapping):
                try:
                    factor = max(float(factors.get(product, 1.0) or 1.0), 0.0)
                except (TypeError, ValueError):
                    factor = 1.0
            resolved[product] = raw_per_unit * factor

        self._variable_costs_cache = resolved
        return self._variable_costs_cache

    def _per_product_raw_material_series(self) -> Dict[str, np.ndarray]:
        production = self._production()
        variable_lookup = self._variable_costs()
        inflation = self._inflation_array()
        risk = self._risk_cost_array()

        series: Dict[str, np.ndarray] = {}
        for product in self.products:
            units = np.array(production.get(product, [0.0 for _ in self.years]), dtype=float)
            variable_cost = variable_lookup.get(product, 0.0)
            series[product] = units * variable_cost * inflation * risk
        return series

    def _sum_series_map(self, series_by_key: Mapping[str, np.ndarray]) -> np.ndarray:
        total = np.zeros(len(self.years), dtype=float)
        for series in series_by_key.values():
            total += np.array(series, dtype=float)
        return total

    def _scale_series_map(
        self,
        series_by_key: Mapping[str, np.ndarray],
        scale: float,
        risk_series: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        return {
            key: np.array(series, dtype=float) * float(scale) * risk_series
            for key, series in series_by_key.items()
        }

    def _total_units(self) -> Dict[str, float]:
        return {name: float(value) for name, value in self.inputs.total_production_units.items()}

    def _total_units_by_year(self) -> List[float]:
        if self._total_units_cache is not None:
            return self._total_units_cache
        year_count = len(self.years)
        totals = np.zeros(year_count, dtype=float)
        for values in self._production().values():
            totals += np.array(values, dtype=float)
        self._total_units_cache = totals.tolist()
        return self._total_units_cache

    def _shift_series(self, values: Sequence[Number], periods: int, *, fill: float = 0.0) -> List[float]:
        if periods <= 0:
            return [float(value) for value in values]
        shifted = [float(fill) for _ in range(periods)] + [float(value) for value in values]
        return shifted[: len(values)]

    def _scale_capex_mapping(
        self,
        mapping: Mapping[str, object],
        multiplier: float,
    ) -> Dict[str, object]:
        scaled: Dict[str, object] = dict(mapping)
        for key in ("initial", "contingency", "project_reserve"):
            try:
                scaled[key] = float(mapping.get(key, 0.0)) * multiplier
            except (TypeError, ValueError, AttributeError):
                scaled[key] = 0.0
        additions = mapping.get("annual_additions", {})
        if isinstance(additions, Mapping):
            scaled["annual_additions"] = {
                str(year): float(value) * multiplier for year, value in additions.items()
            }
        return scaled

    def _adjust_working_capital_days(
        self,
        days: WorkingCapitalDays,
        *,
        receivable_delta: int = 0,
        inventory_delta: int = 0,
        payable_delta: int = 0,
    ) -> WorkingCapitalDays:
        def _shift(values: Sequence[int], delta: int) -> List[int]:
            return [max(int(value) + delta, 0) for value in values]

        return WorkingCapitalDays(
            accounts_receivable=_shift(days.accounts_receivable, receivable_delta),
            inventory=_shift(days.inventory, inventory_delta),
            prepaid_expenses=list(days.prepaid_expenses),
            other_assets=list(days.other_assets),
            accounts_payable=_shift(days.accounts_payable, payable_delta),
            other_liabilities=list(days.other_liabilities),
            calendar_days=list(days.calendar_days),
        )

    def _inputs_for_downside_case(self, case: DownsideCaseParameters) -> ModelInputs:
        scenario_inputs = copy.deepcopy(self.inputs)
        if case.approval_delay_years > 0 or abs(case.volume_multiplier - 1.0) > 1e-9:
            updated_production: Dict[str, List[float]] = {}
            for product, series in scenario_inputs.production_estimate.items():
                scaled = [float(value) * case.volume_multiplier for value in series]
                updated_production[product] = self._shift_series(
                    scaled,
                    case.approval_delay_years,
                    fill=0.0,
                )
            scenario_inputs.production_estimate = updated_production

        if abs(case.price_multiplier - 1.0) > 1e-9:
            for params in scenario_inputs.unit_costs.values():
                params.selling_price *= case.price_multiplier

        if abs(case.raw_material_multiplier - 1.0) > 1e-9:
            scenario_inputs.raw_material_cost_per_unit *= case.raw_material_multiplier
            scenario_inputs.variable_cost_overrides = {
                product: float(value) * case.raw_material_multiplier
                for product, value in scenario_inputs.variable_cost_overrides.items()
            }

        if abs(case.direct_labor_multiplier - 1.0) > 1e-9:
            scenario_inputs.direct_labor_costs = {
                role: float(value) * case.direct_labor_multiplier
                for role, value in scenario_inputs.direct_labor_costs.items()
            }
            if scenario_inputs.labor_model is not None:
                for role in scenario_inputs.labor_model.roles:
                    if role.labor_type == "direct":
                        role.base_salary *= case.direct_labor_multiplier
                scenario_inputs.labor_model.contractor_rate = [
                    float(value) * case.direct_labor_multiplier
                    for value in scenario_inputs.labor_model.contractor_rate
                ]

        if abs(case.overhead_multiplier - 1.0) > 1e-9:
            scenario_inputs.indirect_labor_costs = {
                role: float(value) * case.overhead_multiplier
                for role, value in scenario_inputs.indirect_labor_costs.items()
            }
            if scenario_inputs.labor_model is not None:
                for role in scenario_inputs.labor_model.roles:
                    if role.labor_type != "direct":
                        role.base_salary *= case.overhead_multiplier
            utility = scenario_inputs.utility_schedule
            utility.electricity_rate = [
                float(value) * case.overhead_multiplier for value in utility.electricity_rate
            ]
            utility.water_rate = [
                float(value) * case.overhead_multiplier for value in utility.water_rate
            ]
            utility.steam_rate = [
                float(value) * case.overhead_multiplier for value in utility.steam_rate
            ]

        if (
            case.receivable_days_delta
            or case.inventory_days_delta
            or case.payable_days_delta
        ):
            scenario_inputs.working_capital_days = self._adjust_working_capital_days(
                scenario_inputs.working_capital_days,
                receivable_delta=case.receivable_days_delta,
                inventory_delta=case.inventory_days_delta,
                payable_delta=case.payable_days_delta,
            )

        if abs(case.capex_multiplier - 1.0) > 1e-9:
            scenario_inputs.capital_expenditure = self._scale_capex_mapping(
                scenario_inputs.capital_expenditure,
                case.capex_multiplier,
            )
            for row in scenario_inputs.depreciation_schedule:
                row.acquisition *= case.capex_multiplier

        return scenario_inputs

    def _summary_metric_value(self, name: str, table: Optional[Table] = None) -> float:
        source = table or self.summary_metrics()
        if name in source.index:
            position = source.index.index(name)
            return float(source.data["Value"][position])
        return float("nan")

    def _evidence_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for entry in self.inputs.assumption_evidence:
            completed = sum(
                1
                for value in (entry.source, entry.owner, entry.benchmark_year, entry.rationale)
                if str(value).strip()
            )
            required_fields = 4
            coverage = completed / required_fields if required_fields else 1.0
            missing_fields = [
                label
                for label, value in (
                    ("source", entry.source),
                    ("owner", entry.owner),
                    ("benchmark_year", entry.benchmark_year),
                    ("rationale", entry.rationale),
                )
                if not str(value).strip()
            ]
            status = "Ready" if coverage >= 1.0 else "Partial" if coverage > 0 else "Missing"
            rows.append(
                {
                    "Assumption": entry.assumption,
                    "Category": entry.category,
                    "Value Reference": entry.value_reference,
                    "Source": entry.source,
                    "Owner": entry.owner,
                    "Benchmark Year": entry.benchmark_year,
                    "Rationale": entry.rationale,
                    "Coverage Ratio": coverage,
                    "Missing Fields": ", ".join(missing_fields),
                    "Status": status,
                    "Required": "Yes" if entry.required else "No",
                }
            )
        return rows

    def _evidence_coverage_ratio(self) -> float:
        rows = [row for row in self._evidence_rows() if row.get("Required") == "Yes"]
        if not rows:
            return 0.0
        return float(sum(float(row["Coverage Ratio"]) for row in rows) / len(rows))

    # -------------------------------------------------------------- schedules
    def revenue_schedule(self) -> Table:
        """Build the revenue schedule using the production ramp."""

        if self._revenue_schedule_cache is not None:
            return self._revenue_schedule_cache

        prices = self._unit_prices()
        price_adjustments = self._price_adjustments()
        production = self._production()
        commission_params = self._commission_parameters()
        year_count = len(self.years)
        risk_array = self._risk_revenue_array()
        inflation_array = self._inflation_array()

        gross_totals_array = np.zeros(year_count, dtype=float)
        columns: MutableMapping[str, List[float]] = {}

        for product in self.products:
            units = np.array(production.get(product, [0.0 for _ in range(year_count)]), dtype=float)
            price = float(prices.get(product, 0.0))
            adjustment = np.array(price_adjustments.get(product, [1.0 for _ in range(year_count)]), dtype=float)
            gross_values = units * price * adjustment * inflation_array * risk_array
            columns[product] = gross_values.tolist()
            gross_totals_array += gross_values

        gross_totals = gross_totals_array.tolist()
        commission: List[float] = []
        net_revenue: List[float] = []
        distributor_receivable_weighted: List[float] = [0.0 for _ in self.years]
        distributor_share_by_year: List[float] = []

        for idx, year in enumerate(self.years):
            gross_year = gross_totals[idx]
            commission_year = 0.0
            year_rates = commission_params.get(int(year), {})
            distributor_gross = 0.0
            for product in self.products:
                gross_value = columns[product][idx]
                rate, share, payment_days = year_rates.get(product, (0.0, 1.0, 0))
                effective_share = max(share, 0.0)
                commission_rate = max(rate, 0.0)
                commission_amount = gross_value * effective_share * commission_rate
                distributor_portion = gross_value * effective_share - commission_amount
                distributor_receivable_weighted[idx] += distributor_portion * max(payment_days, 0)
                commission_year += commission_amount
                distributor_gross += gross_value * effective_share
            commission.append(commission_year)
            net_revenue.append(gross_year - commission_year)
            if gross_year:
                distributor_share_by_year.append(distributor_gross / gross_year)
            else:
                distributor_share_by_year.append(0.0)

        self._distributor_receivable_cache = distributor_receivable_weighted
        self._distributor_share_cache = distributor_share_by_year

        columns["Gross Revenue"] = gross_totals
        columns["Distributors Commission"] = commission
        columns["Net Revenue"] = net_revenue
        table = build_table(self.years, columns)
        self._revenue_schedule_cache = table
        return table

    def _labor_headcount_for_role(
        self,
        role: LaborRoleParameters,
        labor: LaborModelParameters,
        total_units: np.ndarray,
        year_index: int,
    ) -> float:
        planned = float(role.planned_headcount[year_index])
        if role.behavior != "variable":
            return max(planned, 0.0)

        productivity = max(float(role.productivity_target[year_index]), 1e-9)
        required_hours = float(total_units[year_index]) / productivity
        base_hours = max(float(labor.operating_hours_per_shift[year_index]), 1.0)
        shifts = max(float(labor.shifts[year_index]), 1.0)
        utilization = max(float(labor.utilization[year_index]), 1e-6)
        absenteeism = min(max(float(labor.absenteeism[year_index]), 0.0), 0.95)
        capacity_hours = base_hours * shifts * utilization * (1.0 - absenteeism)
        raw_fte = required_hours / max(capacity_hours, 1e-9)

        delay_quarters = max(int(labor.hiring_delay_quarters[year_index]), 0)
        lag_factor = min(delay_quarters / 4.0, 1.0)
        previous = planned
        if year_index > 0:
            previous = float(role.planned_headcount[year_index - 1])
        lagged_fte = previous + (raw_fte - previous) * (1.0 - lag_factor)
        return max(math.ceil(max(lagged_fte, 0.0)), 0.0)

    def _labor_series(self, total_units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labor = self.inputs.labor_model
        year_count = len(self.years)
        if labor is None or not labor.roles:
            self._labor_kpi_cache = None
            return np.zeros(year_count, dtype=float), np.zeros(year_count, dtype=float)

        direct = np.zeros(year_count, dtype=float)
        indirect = np.zeros(year_count, dtype=float)
        total_hours = np.zeros(year_count, dtype=float)
        fixed_cost = np.zeros(year_count, dtype=float)
        variable_cost = np.zeros(year_count, dtype=float)

        for role in labor.roles:
            salary = float(role.base_salary)
            for idx in range(year_count):
                escalation = (
                    float(labor.wage_escalation_direct[idx])
                    if role.labor_type == "direct"
                    else float(labor.wage_escalation_indirect[idx])
                )
                if idx > 0:
                    salary *= 1.0 + escalation
                benefits = max(float(role.benefits_rate[idx]), 0.0)
                burden = max(float(role.burden_rate[idx]), 0.0)
                overtime = max(float(role.overtime_rate[idx]), 0.0)
                overtime_cap = max(float(labor.overtime_cap[idx]), 0.0)
                effective_salary = salary * (1.0 + benefits + burden) * (1.0 + min(overtime, overtime_cap))
                headcount = self._labor_headcount_for_role(role, labor, total_units, idx)
                role_cost = headcount * effective_salary

                if role.labor_type == "direct":
                    direct[idx] += role_cost
                else:
                    indirect[idx] += role_cost

                if role.behavior == "fixed":
                    fixed_cost[idx] += role_cost
                else:
                    variable_cost[idx] += role_cost

                hours_per_fte = max(float(labor.operating_hours_per_shift[idx]) * max(float(labor.shifts[idx]), 1.0), 1.0)
                total_hours[idx] += headcount * hours_per_fte

        contractor = np.array(labor.contractor_hours, dtype=float) * np.array(labor.contractor_rate, dtype=float)
        step_cost = np.array(labor.transition_training_cost, dtype=float) + np.array(labor.supervision_increment, dtype=float) + np.array(labor.shift_allowance, dtype=float)
        for idx in range(1, year_count):
            if float(labor.shifts[idx]) <= float(labor.shifts[idx - 1]):
                step_cost[idx] = 0.0

        direct += contractor + step_cost
        total_labor = direct + indirect

        produced_units = np.maximum(total_units, 1e-9)
        labor_cost_per_unit = total_labor / produced_units
        units_per_labor_hour = produced_units / np.maximum(total_hours, 1e-9)

        self._labor_kpi_cache = {
            "Average Labor Cost per Unit": float(np.mean(labor_cost_per_unit)),
            "Average Units per Labor Hour": float(np.mean(units_per_labor_hour)),
            "Average Fixed Labor Share": float(np.mean(fixed_cost / np.maximum(fixed_cost + variable_cost, 1e-9))),
        }
        return direct, indirect

    def cost_structure(self) -> Table:
        if self._cost_structure_cache is not None:
            return self._cost_structure_cache

        production = self._production()
        year_count = len(self.years)
        total_units = self._total_units_by_year()

        risk_array = self._risk_cost_array()
        inflation_array = self._inflation_array()

        per_product_raw_materials = self._per_product_raw_material_series()
        raw_material_cost_array = self._sum_series_map(per_product_raw_materials)
        raw_material_cost = raw_material_cost_array.tolist()

        electricity, water, steam = self._utility_arrays()
        utilities = (electricity + water + steam).tolist()

        total_units_array = np.array(total_units, dtype=float)
        if self.inputs.labor_model is not None:
            direct_array, indirect_array = self._labor_series(total_units_array)
            direct_labor = (direct_array * inflation_array * risk_array).tolist()
            indirect_labor = (indirect_array * inflation_array * risk_array).tolist()
        else:
            base_direct = sum(self.inputs.direct_labor_costs.values())
            baseline_units = total_units[0] or 1.0
            direct_labor = (
                base_direct
                * (total_units_array / baseline_units)
                * inflation_array
                * risk_array
            ).tolist()

            base_indirect = sum(self.inputs.indirect_labor_costs.values())
            indirect_labor = (base_indirect * inflation_array * risk_array).tolist()

        utility_cost_share = self.inputs.utility_cost_of_sales_share
        utility_cost_of_sales = [value * utility_cost_share for value in utilities]
        utility_general_admin = [value - cos_share for value, cos_share in zip(utilities, utility_cost_of_sales)]

        cost_of_sales = [
            raw + util_share + direct
            for raw, util_share, direct in zip(
                raw_material_cost, utility_cost_of_sales, direct_labor
            )
        ]
        general_admin = [
            base + util_share
            for base, util_share in zip(indirect_labor, utility_general_admin)
        ]
        total_expenses = [cos + ga for cos, ga in zip(cost_of_sales, general_admin)]

        columns: Dict[str, List[float]] = {
            "Raw Materials": raw_material_cost,
            "Utilities": utilities,
            "Direct Labor": direct_labor,
            "Cost of Sales": cost_of_sales,
            "General & Admin": general_admin,
            "Total Expenses": total_expenses,
        }
        for product in self.products:
            columns[f"Raw Materials - {product}"] = per_product_raw_materials.get(
                product,
                np.zeros(year_count, dtype=float),
            ).tolist()

        table = build_table(self.years, columns)
        self._cost_structure_cache = table
        return table

    def _depreciation_rollforward(self) -> "tuple[list[dict], dict[int, float], dict[int, float]]":
        if self._depreciation_cache is not None:
            return self._depreciation_cache

        per_year_depreciation: dict[int, float] = {int(year): 0.0 for year in self.years}
        per_year_net_book: dict[int, float] = {int(year): 0.0 for year in self.years}
        details: list[dict] = []

        year_index = {year: idx for idx, year in enumerate(self.years)}
        enumerated_schedule = list(enumerate(self.inputs.depreciation_schedule))
        enumerated_schedule.sort(key=lambda item: (item[1].asset_type, item[1].year, item[0]))

        for _, row in enumerated_schedule:
            asset = row.asset_type
            method = (row.method or "straight_line").lower()
            if method not in {"straight_line", "reducing_balance"}:
                method = "straight_line"

            start_position = year_index.get(row.year)
            if start_position is None:
                continue

            configured_life = row.asset_life if row.asset_life not in (None, 0) else None
            if configured_life is not None and configured_life < 0:
                configured_life = None

            available_years = len(self.years) - start_position
            if available_years <= 0:
                continue

            if configured_life is None:
                life_span = available_years
            else:
                life_span = min(configured_life, available_years)
            if life_span <= 0:
                continue

            previous_net_book = float(row.opening_net_book or 0.0)
            previous_cumulative = float(row.opening_cumulative or 0.0)

            for offset in range(life_span):
                year_idx = start_position + offset
                year = self.years[year_idx]
                acquisition_amount = row.acquisition if offset == 0 else 0.0
                opening_net_book = previous_net_book
                opening_cumulative = previous_cumulative
                total_asset_cost = (
                    opening_net_book + opening_cumulative + acquisition_amount
                )
                allowable = max(total_asset_cost - opening_cumulative, 0.0)

                if configured_life is not None and method == "straight_line":
                    remaining_periods = max(configured_life - offset, 1)
                    total_depreciation = allowable / remaining_periods if remaining_periods else allowable
                else:
                    if method == "reducing_balance":
                        depreciation_base = opening_net_book + (acquisition_amount * 0.5)
                    else:
                        depreciation_base = total_asset_cost
                    total_depreciation = depreciation_base * row.depreciation_rate
                    if configured_life is not None and offset >= configured_life - 1:
                        total_depreciation = allowable
                    elif total_depreciation > allowable:
                        total_depreciation = allowable

                cumulative_depreciation = opening_cumulative + total_depreciation
                net_book_value = max(total_asset_cost - cumulative_depreciation, 0.0)

                per_year_depreciation[year] = (
                    per_year_depreciation.get(year, 0.0) + total_depreciation
                )
                per_year_net_book[year] = per_year_net_book.get(year, 0.0) + net_book_value

                details.append(
                    {
                        "asset_type": asset,
                        "year": year,
                        "acquisition_year": row.year,
                        "acquisition": acquisition_amount,
                        "opening_net_book": opening_net_book,
                        "opening_cumulative": opening_cumulative,
                        "total_asset_cost": total_asset_cost,
                        "configured_rate": row.depreciation_rate,
                        "depreciation_rate": (
                            total_depreciation / total_asset_cost if total_asset_cost else 0.0
                        ),
                        "total_depreciation": total_depreciation,
                        "cumulative_depreciation": cumulative_depreciation,
                        "net_book_value": net_book_value,
                        "method": method,
                        "asset_life": configured_life,
                        "life_year_index": offset,
                        "life_span": life_span,
                    }
                )

                previous_net_book = net_book_value
                previous_cumulative = cumulative_depreciation

        self._depreciation_cache = (details, per_year_depreciation, per_year_net_book)
        return self._depreciation_cache

    def depreciation_schedule(self) -> List[float]:
        _, per_year, _ = self._depreciation_rollforward()
        return [per_year.get(year, 0.0) for year in self.years]

    # ----------------------------------------------------------- main outputs
    def income_statement(self) -> Table:
        if self._income_statement_cache is not None:
            return self._income_statement_cache

        revenue = self.revenue_schedule()
        costs = self.cost_structure()
        depreciation = self.depreciation_schedule()

        gross_revenue = revenue.column("Gross Revenue")
        distributors_commission = revenue.column("Distributors Commission")
        net_revenue = revenue.column("Net Revenue")

        cost_of_sales = costs.column("Cost of Sales")
        general_admin = costs.column("General & Admin")

        net_revenue_array = np.array(net_revenue, dtype=float)
        cost_of_sales_array = np.array(cost_of_sales, dtype=float)
        general_admin_array = np.array(general_admin, dtype=float)
        depreciation_array = np.array(depreciation, dtype=float)

        gross_profit = (net_revenue_array - cost_of_sales_array).tolist()
        ebitda_array = net_revenue_array - cost_of_sales_array - general_admin_array
        ebitda = ebitda_array.tolist()
        ebit_array = ebitda_array - depreciation_array
        ebit = ebit_array.tolist()
        interest = self._interest_schedule()
        interest_array = np.array(interest, dtype=float)
        ebt = (ebit_array - interest_array).tolist()
        taxes: List[float] = []
        net_income: List[float] = []
        nol_balance = 0.0
        apply_nol = bool(self.inputs.tax_loss_carryforward)
        nol_limit = float(self.inputs.tax_loss_limit)
        for idx, (value, rate) in enumerate(zip(ebt, self._tax_schedule())):
            if value <= 0:
                tax = 0.0
                base_net = value
                if apply_nol:
                    nol_balance += abs(value)
            else:
                taxable_income = value
                if apply_nol and nol_balance > 0:
                    max_offset = value * nol_limit
                    offset = min(nol_balance, max_offset)
                    taxable_income = max(value - offset, 0.0)
                    nol_balance -= offset
                tax = taxable_income * rate
                base_net = value - tax

            taxes.append(tax)
            net_income.append(base_net)

        gross_profit_margin = [_safe_ratio(gp, gr) for gp, gr in zip(gross_profit, gross_revenue)]
        ebitda_margin = [_safe_ratio(e, r) for e, r in zip(ebitda, net_revenue)]
        ebit_margin = [_safe_ratio(e, r) for e, r in zip(ebit, net_revenue)]
        roe = [_safe_ratio(n, self.inputs.financing.share_capital) for n in net_income]

        columns: MutableMapping[str, List[float]] = {
            "Gross Revenue": gross_revenue,
            "Distributors Commission": distributors_commission,
            "Net Revenue": net_revenue,
            "Cost of Sales": cost_of_sales,
            "Gross Profit": gross_profit,
            "General & Admin": general_admin,
            "EBITDA": ebitda,
            "Total Depreciation Expense": depreciation,
            "EBIT": ebit,
            "Interest": interest,
            "EBT": ebt,
            "Taxes": taxes,
            "Net Income": net_income,
            "Gross Profit Margin": gross_profit_margin,
            "EBITDA Margin": ebitda_margin,
            "EBIT Margin": ebit_margin,
            "Return on Equity": roe,
        }

        table = build_table(self.years, columns)
        self._income_statement_cache = table
        return table

    def _compute_amortisation(
        self, entries: List[DebtEntry], rate: float
    ) -> tuple[List[float], List[float]]:
        length = len(self.years)
        interest_schedule, _, outstanding_schedule, _ = amortise_entries(
            entries, rate, self.years
        )
        if len(interest_schedule) < length:
            interest_schedule.extend([0.0] * (length - len(interest_schedule)))
        if len(outstanding_schedule) < length:
            outstanding_schedule.extend([0.0] * (length - len(outstanding_schedule)))
        return interest_schedule[:length], outstanding_schedule[:length]

    def _senior_debt_schedules(self) -> tuple[List[float], List[float]]:
        if (
            self._senior_interest_cache is not None
            and self._senior_outstanding_cache is not None
        ):
            return self._senior_interest_cache, self._senior_outstanding_cache

        interest_schedule, outstanding_schedule = self._compute_amortisation(
            self.inputs.financing.senior_debt_entries,
            float(self.inputs.financing.senior_debt_interest),
        )

        self._senior_interest_cache = interest_schedule
        self._senior_outstanding_cache = outstanding_schedule
        return interest_schedule, outstanding_schedule

    def _revolver_schedules(self) -> tuple[List[float], List[float]]:
        if (
            self._revolver_interest_cache is not None
            and self._revolver_outstanding_cache is not None
        ):
            return self._revolver_interest_cache, self._revolver_outstanding_cache

        interest_schedule, outstanding_schedule = self._compute_amortisation(
            self.inputs.financing.revolver_entries,
            float(self.inputs.financing.revolver_interest),
        )

        self._revolver_interest_cache = interest_schedule
        self._revolver_outstanding_cache = outstanding_schedule
        return interest_schedule, outstanding_schedule

    def _overdraft_schedules(self) -> tuple[List[float], List[float]]:
        if (
            self._overdraft_interest_cache is not None
            and self._overdraft_outstanding_cache is not None
        ):
            return self._overdraft_interest_cache, self._overdraft_outstanding_cache

        interest_schedule, outstanding_schedule = self._compute_amortisation(
            self.inputs.financing.overdraft_entries,
            float(self.inputs.financing.cash_interest),
        )

        self._overdraft_interest_cache = interest_schedule
        self._overdraft_outstanding_cache = outstanding_schedule
        return interest_schedule, outstanding_schedule

    def _debt_service_schedule(self) -> List[float]:
        length = len(self.years)
        if length == 0:
            return []

        schedules: List[List[float]] = []
        financing = self.inputs.financing
        debt_sources = [
            (financing.senior_debt_entries, financing.senior_debt_interest),
            (financing.revolver_entries, financing.revolver_interest),
            (financing.overdraft_entries, financing.cash_interest),
        ]
        for entries, rate in debt_sources:
            if not entries:
                continue
            interest_schedule, principal_schedule, _outstanding, _ = amortise_entries(
                entries, float(rate), self.years
            )
            if len(interest_schedule) < length:
                interest_schedule.extend([0.0] * (length - len(interest_schedule)))
            if len(principal_schedule) < length:
                principal_schedule.extend([0.0] * (length - len(principal_schedule)))
            schedules.append(
                [
                    interest_schedule[idx] + principal_schedule[idx]
                    for idx in range(length)
                ]
            )

        if not schedules:
            return [0.0 for _ in range(length)]

        total = np.sum(np.array(schedules, dtype=float), axis=0).tolist()
        return total

    def _interest_schedule(self) -> List[float]:
        if self._interest_schedule_cache is not None:
            return self._interest_schedule_cache
        financing = self.inputs.financing
        senior_interest, _ = self._senior_debt_schedules()
        revolver_interest, _ = self._revolver_schedules()
        overdraft_interest, _ = self._overdraft_schedules()

        interest: List[float] = []
        for idx in range(len(self.years)):
            total_interest = (
                senior_interest[idx]
                + revolver_interest[idx]
                + overdraft_interest[idx]
            )
            interest.append(total_interest)
        self._interest_schedule_cache = interest
        return interest

    def _tax_schedule(self) -> List[float]:
        if self._tax_schedule_cache is not None:
            return self._tax_schedule_cache
        if getattr(self.inputs, "tax_rates", None):
            self._tax_schedule_cache = list(self.inputs.tax_rates)
        else:
            self._tax_schedule_cache = [self.inputs.tax_rate for _ in self.years]
        return self._tax_schedule_cache

    def _risk_factors(self) -> List[float]:
        if self._risk_factors_cache is not None:
            return self._risk_factors_cache
        self._risk_factor_schedules()
        return self._risk_factors_cache or [1.0 for _ in self.years]

    def _risk_cost_factors(self) -> List[float]:
        if self._risk_cost_factors_cache is not None:
            return self._risk_cost_factors_cache
        self._risk_factor_schedules()
        return self._risk_cost_factors_cache or [1.0 for _ in self.years]

    def _risk_factor_schedules(self) -> None:
        if self._risk_factors_cache is not None and self._risk_cost_factors_cache is not None:
            return

        schedule = getattr(self.inputs, "risk_schedule", {})
        weights = getattr(self.inputs, "risk_weights", {}) or {}

        revenue_factors: List[float] = []
        cost_factors: List[float] = []
        diagnostics_columns: Dict[str, List[float]] = {}

        if schedule:
            for name in schedule.keys():
                diagnostics_columns[f"{name} Revenue Factor"] = []
                diagnostics_columns[f"{name} Cost Factor"] = []

        for idx in range(len(self.years)):
            revenue_factor = 1.0
            cost_factor = 1.0
            for name, values in schedule.items():
                if not values:
                    continue
                series = list(values)
                rate = series[idx] if idx < len(series) else series[-1]
                if not 0.0 <= float(rate) <= 1.0:
                    raise ValueError(
                        f"Risk schedule '{name}' has value {rate} outside the [0, 1] range"
                    )
                weight = weights.get(name, {})
                revenue_weight = float(weight.get("revenue", 1.0)) if weight else 1.0
                cost_weight = float(weight.get("costs", 1.0)) if weight else 1.0
                revenue_component = max(0.0, 1.0 - float(rate) * revenue_weight)
                cost_component = max(0.0, 1.0 - float(rate) * cost_weight)
                diagnostics_columns.setdefault(f"{name} Revenue Factor", []).append(revenue_component)
                diagnostics_columns.setdefault(f"{name} Cost Factor", []).append(cost_component)
                revenue_factor *= revenue_component
                cost_factor *= cost_component
            revenue_factors.append(revenue_factor)
            cost_factors.append(cost_factor)

        if not revenue_factors:
            revenue_factors = [1.0 for _ in self.years]
        if not cost_factors:
            cost_factors = [1.0 for _ in self.years]

        diagnostics_columns["Revenue Factor"] = revenue_factors
        diagnostics_columns["Cost Factor"] = cost_factors
        diagnostics_columns["Combined Factor"] = revenue_factors
        diagnostics_columns["Combined Shock (%)"] = [
            (1.0 - value) * 100.0 for value in revenue_factors
        ]
        self._risk_factor_diagnostics_cache = build_table(self.years, diagnostics_columns)
        self._risk_factors_cache = revenue_factors
        self._risk_cost_factors_cache = cost_factors

    def risk_factor_diagnostics(self) -> Table:
        if self._risk_factor_diagnostics_cache is None:
            self._risk_factors()
        if self._risk_factor_diagnostics_cache is None:
            defaults = {
                "Revenue Factor": [1.0 for _ in self.years],
                "Cost Factor": [1.0 for _ in self.years],
                "Combined Factor": [1.0 for _ in self.years],
                "Combined Shock (%)": [0.0 for _ in self.years],
            }
            self._risk_factor_diagnostics_cache = build_table(self.years, defaults)
        return self._risk_factor_diagnostics_cache

    def cash_flow_statement(self) -> Table:
        if self._cash_flow_cache is not None:
            return self._cash_flow_cache

        income = self.income_statement()
        depreciation = self.depreciation_schedule()
        working_balances = self._working_capital_balances()

        net_income = income.column("Net Income")
        tax_expense = income.column("Taxes")
        interest_expense = income.column("Interest")
        depreciation_expense = list(depreciation)

        net_income_array = np.array(net_income, dtype=float)
        tax_expense_array = np.array(tax_expense, dtype=float)
        interest_expense_array = np.array(interest_expense, dtype=float)
        depreciation_array = np.array(depreciation_expense, dtype=float)

        operating_profit = (
            net_income_array + tax_expense_array + interest_expense_array
        ).tolist()

        inventory_change = _difference(working_balances.column("Inventory"))
        receivable_change = _difference(working_balances.column("Accounts Receivable"))
        payable_change = _difference(working_balances.column("Accounts Payable"))
        prepaid_change = _difference(working_balances.column("Prepaid Expenses"))
        other_asset_change = _difference(working_balances.column("Other Assets"))
        other_liability_change = _difference(working_balances.column("Other Liabilities"))

        inventory_adjustment = -np.array(inventory_change, dtype=float)
        receivable_adjustment = -np.array(receivable_change, dtype=float)
        payable_adjustment = np.array(payable_change, dtype=float)
        prepaid_adjustment = -np.array(prepaid_change, dtype=float)
        other_asset_adjustment = -np.array(other_asset_change, dtype=float)
        other_liability_adjustment = np.array(other_liability_change, dtype=float)

        cash_flow_from_operations = (
            np.array(operating_profit, dtype=float)
            + depreciation_array
            + inventory_adjustment
            + receivable_adjustment
            + payable_adjustment
            + prepaid_adjustment
            + other_asset_adjustment
            + other_liability_adjustment
        ).tolist()

        interest_paid = -interest_expense_array
        taxes_paid = -tax_expense_array

        net_cash_from_operations = (
            np.array(cash_flow_from_operations, dtype=float) + interest_paid + taxes_paid
        ).tolist()

        capex = self._capex_series()
        capital_expenditure = -np.array(capex, dtype=float)
        net_cash_from_investing = capital_expenditure.tolist()

        financing_components = self._financing_cash_flow_components()
        if financing_components:
            financing_arrays = [
                np.array(series, dtype=float) for series in financing_components.values()
            ]
            net_cash_from_financing = np.sum(financing_arrays, axis=0).tolist()
        else:
            net_cash_from_financing = [0.0 for _ in self.years]

        net_cash_flow = (
            np.array(net_cash_from_operations, dtype=float)
            + np.array(net_cash_from_investing, dtype=float)
            + np.array(net_cash_from_financing, dtype=float)
        ).tolist()

        beginning_cash = _shift(_cumulative(net_cash_flow), fill_value=0.0)
        ending_cash = [begin + change for begin, change in zip(beginning_cash, net_cash_flow)]

        columns: Dict[str, List[float]] = {
            "Cash Flow from Operations": cash_flow_from_operations,
            "Net Cash Generated from Operating Activities": net_cash_from_operations,
            "Net Cash Used in Investing Activities": net_cash_from_investing,
            "Net Cash Used in Financing Activities": net_cash_from_financing,
            CASH_FLOW_NET_COLUMN: net_cash_flow,
            CASH_FLOW_BEGIN_COLUMN: beginning_cash,
            CASH_FLOW_END_COLUMN: ending_cash,
            "Net Increase/Decrease in Cash": net_cash_flow,
        }

        table = build_table(self.years, columns)
        self._cash_flow_cache = table
        return table

    def balance_sheet(self) -> Table:
        if self._balance_sheet_cache is not None:
            return self._balance_sheet_cache

        cash_flow = self.cash_flow_statement()
        working_capital = self._working_capital_balances()
        net_ppe = self._net_ppe_schedule()
        accounts_payable = working_capital.column("Accounts Payable")
        other_liabilities = working_capital.column("Other Liabilities")

        _, senior_outstanding = self._senior_debt_schedules()
        _, revolver_outstanding = self._revolver_schedules()
        _, overdraft_outstanding = self._overdraft_schedules()

        cash_array = np.array(cash_flow.column(CASH_FLOW_END_COLUMN), dtype=float)
        ar_array = np.array(working_capital.column("Accounts Receivable"), dtype=float)
        inventory_array = np.array(working_capital.column("Inventory"), dtype=float)
        prepaid_array = np.array(working_capital.column("Prepaid Expenses"), dtype=float)
        other_assets_array = np.array(working_capital.column("Other Assets"), dtype=float)
        net_ppe_array = np.array(net_ppe, dtype=float)

        total_current_assets = (
            cash_array + ar_array + inventory_array + prepaid_array + other_assets_array
        )
        total_assets = total_current_assets + net_ppe_array

        ap_array = np.array(accounts_payable, dtype=float)
        other_liabilities_array = np.array(other_liabilities, dtype=float)
        senior_array = np.array(senior_outstanding, dtype=float)
        revolver_array = np.array(revolver_outstanding, dtype=float)
        overdraft_array = np.array(overdraft_outstanding, dtype=float)

        total_liabilities = (
            ap_array + other_liabilities_array + senior_array + revolver_array + overdraft_array
        )
        shareholders_equity = total_assets - total_liabilities
        total_liabilities_equity = total_liabilities + shareholders_equity

        table = build_table(
            self.years,
            {
                "Cash": cash_array.tolist(),
                "Accounts Receivable": ar_array.tolist(),
                "Inventory": inventory_array.tolist(),
                "Prepaid Expenses": prepaid_array.tolist(),
                "Other Assets": other_assets_array.tolist(),
                "Net PP&E": net_ppe_array.tolist(),
                "Total Assets": total_assets.tolist(),
                "Accounts Payable": accounts_payable,
                "Other Liabilities": other_liabilities,
                "Overdraft": overdraft_outstanding,
                "Total Liabilities": total_liabilities.tolist(),
                "Shareholders' Equity": shareholders_equity.tolist(),
                "Total Liabilities & Equity": total_liabilities_equity.tolist(),
            },
        )
        self._balance_sheet_cache = table
        return table

    # -------------------------------------------------------------- schedules
    def _capex_series(self) -> List[float]:
        capex = [0.0 for _ in self.years]
        if not capex:
            return capex

        config = self.inputs.capital_expenditure or {}
        year_index = {year: idx for idx, year in enumerate(self.years)}

        first_year_total = 0.0
        for key, value in config.items():
            if key == "annual_additions":
                continue
            try:
                first_year_total += float(value)
            except (TypeError, ValueError):
                continue
        capex[0] += first_year_total

        additions = config.get("annual_additions", {})
        if isinstance(additions, Mapping):
            for year_value, addition in additions.items():
                try:
                    year = int(year_value)
                except (TypeError, ValueError):
                    continue
                try:
                    index = year_index.get(year)
                    if index is None:
                        continue
                    capex[index] += float(addition)
                except (TypeError, ValueError):
                    continue

        for row in self.inputs.depreciation_schedule:
            try:
                acquisition = float(row.acquisition or 0.0)
            except (TypeError, ValueError):
                acquisition = 0.0
            if not acquisition:
                continue
            idx = year_index.get(row.year)
            if idx is None:
                continue
            capex[idx] += acquisition

        return capex

    def _calendar_days(self) -> List[float]:
        if self._calendar_days_cache is not None:
            return list(self._calendar_days_cache)

        days = list(getattr(self.inputs.working_capital_days, "calendar_days", []) or [])
        if not days:
            days = [366 if year % 4 == 0 else 365 for year in self.years]

        if len(days) < len(self.years):
            fill = days[-1] if days else 365
            days = days + [fill for _ in range(len(self.years) - len(days))]

        self._calendar_days_cache = [float(value) for value in days[: len(self.years)]]
        return list(self._calendar_days_cache)

    def _distributor_receivable_balances(
        self, *, weighted: Optional[Sequence[float]] = None
    ) -> List[float]:
        if weighted is None:
            if self._distributor_receivable_cache is None:
                self.revenue_schedule()
            weighted_series = list(self._distributor_receivable_cache or [0.0 for _ in self.years])
        else:
            weighted_series = [float(value) for value in weighted]
            if len(weighted_series) < len(self.years):
                weighted_series = weighted_series + [0.0] * (len(self.years) - len(weighted_series))
            else:
                weighted_series = weighted_series[: len(self.years)]

        days_in_year = self._calendar_days()
        balances: List[float] = []
        for weighted_value, day_count in zip(weighted_series, days_in_year):
            if day_count:
                balances.append(weighted_value / day_count)
            else:
                balances.append(0.0)
        return balances

    def _working_capital_series_from_inputs(
        self,
        revenue: Sequence[float],
        cost_of_sales: Sequence[float],
        *,
        distributor_weighted: Optional[Sequence[float]] = None,
        distributor_share: Optional[Sequence[float]] = None,
        days: Optional[WorkingCapitalDays] = None,
    ) -> Dict[str, List[float]]:
        days = days or self.inputs.working_capital_days

        def _pad(series: Sequence[float]) -> List[float]:
            values = [float(value) for value in series]
            if len(values) < len(self.years):
                fill = values[-1] if values else 0.0
                values = values + [fill for _ in range(len(self.years) - len(values))]
            return values[: len(self.years)]

        revenue_series = _pad(revenue)
        cost_series = _pad(cost_of_sales)

        days_in_year = [float(value) for value in self._calendar_days()]

        def _expand(series: Iterable[int]) -> List[float]:
            values = [float(value) for value in series]
            expanded: List[float] = []
            carry = 0.0
            for index in range(len(self.years)):
                if index < len(values):
                    carry = values[index]
                expanded.append(carry)
            return expanded

        def _calc(series: List[float], base: List[float]) -> List[float]:
            calculated: List[float] = []
            for value, denominator, day in zip(base, days_in_year, series):
                if abs(denominator) < 1e-12:
                    calculated.append(0.0)
                else:
                    calculated.append(value / denominator * day)
            return calculated

        ar_days = _expand(days.accounts_receivable)
        inventory_days = _expand(days.inventory)
        prepaid_days = _expand(days.prepaid_expenses)
        other_asset_days = _expand(days.other_assets)
        ap_days = _expand(days.accounts_payable)
        other_liability_days = _expand(days.other_liabilities)

        if distributor_share is None:
            distributor_share_series = [0.0 for _ in self.years]
        else:
            distributor_share_series = [float(value) for value in distributor_share]
            if len(distributor_share_series) < len(self.years):
                fill = distributor_share_series[-1] if distributor_share_series else 0.0
                distributor_share_series = distributor_share_series + [
                    fill for _ in range(len(self.years) - len(distributor_share_series))
                ]
            distributor_share_series = distributor_share_series[: len(self.years)]

        direct_share_series = [
            max(0.0, min(1.0, 1.0 - share)) for share in distributor_share_series
        ]

        ar_base = _calc(
            ar_days,
            [revenue_value * share for revenue_value, share in zip(revenue_series, direct_share_series)],
        )
        distributor_receivables = self._distributor_receivable_balances(weighted=distributor_weighted)
        ar = [base + dist for base, dist in zip(ar_base, distributor_receivables)]
        inventory = _calc(inventory_days, cost_series)
        prepaid = _calc(prepaid_days, cost_series)
        other_assets = _calc(other_asset_days, cost_series)
        ap = _calc(ap_days, cost_series)
        other_liabilities = _calc(other_liability_days, cost_series)

        net_working_capital = [
            a + inv + pre + other - pay - other_liab
            for a, inv, pre, other, pay, other_liab in zip(ar, inventory, prepaid, other_assets, ap, other_liabilities)
        ]

        return {
            "Days in Year": days_in_year,
            "Accounts Receivable Days": ar_days,
            "Accounts Receivable (Base)": ar_base,
            "Distributor Receivables": distributor_receivables,
            "Accounts Receivable": ar,
            "Inventory Days": inventory_days,
            "Inventory": inventory,
            "Prepaid Expenses Days": prepaid_days,
            "Prepaid Expenses": prepaid,
            "Other Assets Days": other_asset_days,
            "Other Assets": other_assets,
            "Accounts Payable Days": ap_days,
            "Accounts Payable": ap,
            "Other Liabilities Days": other_liability_days,
            "Other Liabilities": other_liabilities,
            "Net Working Capital": net_working_capital,
        }

    def _working_capital_balances_from_series(
        self,
        revenue: Sequence[float],
        cost_of_sales: Sequence[float],
        *,
        distributor_weighted: Optional[Sequence[float]] = None,
        distributor_share: Optional[Sequence[float]] = None,
    ) -> Table:
        columns = self._working_capital_series_from_inputs(
            revenue,
            cost_of_sales,
            distributor_weighted=distributor_weighted,
            distributor_share=distributor_share,
        )
        return build_table(self.years, columns)

    def _working_capital_balances(self) -> Table:
        if self._working_capital_cache is not None:
            return self._working_capital_cache

        revenue = self.revenue_schedule().column("Net Revenue")
        cost_of_sales = self.cost_structure().column("Cost of Sales")
        weighted = self._distributor_receivable_cache or [0.0 for _ in self.years]
        distributor_share = self._distributor_share_cache or [0.0 for _ in self.years]
        table = self._working_capital_balances_from_series(
            revenue,
            cost_of_sales,
            distributor_weighted=weighted,
            distributor_share=distributor_share,
        )
        self._working_capital_cache = table
        return table

    def _working_capital_changes(self) -> List[float]:
        balances = self._working_capital_balances().column("Net Working Capital")
        return _difference(balances)

    def working_capital_schedule(self) -> Table:
        """Expose working-capital balances alongside year-over-year changes."""

        balances = self._working_capital_balances()
        changes = self._working_capital_changes()
        return balances.with_columns(**{"Change in Net Working Capital": changes})

    def inventory_schedule(self) -> Table:
        """Reconcile inventory inputs to the balance-sheet values."""

        cost_structure = self.cost_structure()
        working_capital = self._working_capital_balances()

        cost_of_sales = cost_structure.column("Cost of Sales")
        balance_inventory = working_capital.column("Inventory")
        inventory_days_source = list(self.inputs.working_capital_days.inventory)
        days_in_year = self._calendar_days()

        per_product_purchased = self._per_product_raw_material_series()
        per_product_inventory: Dict[str, List[float]] = {product: [] for product in self.products}

        inventory_days: List[float] = []
        calculated_inventory: List[float] = []
        variance: List[float] = []
        turnover: List[float] = []

        for idx, cost in enumerate(cost_of_sales):
            if inventory_days_source:
                if idx < len(inventory_days_source):
                    inventory_day_value = float(inventory_days_source[idx])
                else:
                    inventory_day_value = float(inventory_days_source[-1])
            else:
                inventory_day_value = 0.0

            day_length = float(days_in_year[idx]) if days_in_year[idx] else 0.0
            inventory_days.append(inventory_day_value)

            calculated = cost / day_length * inventory_day_value if day_length else 0.0
            calculated_inventory.append(calculated)

            for product in self.products:
                purchased = per_product_purchased.get(product, np.zeros(len(self.years), dtype=float))
                per_product_cost = float(purchased[idx]) if idx < len(purchased) else 0.0
                per_product_value = per_product_cost / day_length * inventory_day_value if day_length else 0.0
                per_product_inventory[product].append(per_product_value)

            actual_inventory = balance_inventory[idx] if idx < len(balance_inventory) else calculated
            difference = calculated - actual_inventory
            if abs(difference) < 1e-6:
                difference = 0.0
            variance.append(difference)

            if actual_inventory:
                turnover.append(cost / actual_inventory)
            else:
                turnover.append(float("nan"))

        columns: Dict[str, List[float]] = {
            "Cost of Sales": cost_of_sales,
            "Days in Year": days_in_year,
            "Inventory Days": inventory_days,
            "Calculated Inventory": calculated_inventory,
            "Balance Sheet Inventory": balance_inventory,
            "Variance": variance,
            "Inventory Turnover": turnover,
        }
        for product in self.products:
            columns[f"Material Purchased - {product}"] = per_product_purchased.get(
                product,
                np.zeros(len(self.years), dtype=float),
            ).tolist()
            columns[f"Inventory - {product}"] = per_product_inventory.get(product, [0.0 for _ in self.years])

        return build_table(self.years, columns)

    def _net_ppe_schedule(self) -> List[float]:
        if self.inputs.depreciation_schedule:
            _, _, per_year_net = self._depreciation_rollforward()
            return [per_year_net.get(year, 0.0) for year in self.years]

        capex = self._capex_series()
        depreciation = self.depreciation_schedule()
        cumulative_capex = _cumulative(capex)
        cumulative_depreciation = _cumulative(depreciation)
        return [cap - dep for cap, dep in zip(cumulative_capex, cumulative_depreciation)]

    def _instrument_values(self, entries: List[DebtEntry], attribute: str) -> List[float]:
        per_year: Dict[int, float] = {}
        for entry in entries:
            value = float(getattr(entry, attribute))
            per_year[entry.year] = per_year.get(entry.year, 0.0) + value
        return [per_year.get(year, 0.0) for year in self.years]

    def _liability_balance(self) -> List[float]:
        financing = self.inputs.financing
        _, senior = self._senior_debt_schedules()
        _, revolver = self._revolver_schedules()
        _, overdraft = self._overdraft_schedules()
        return [senior[idx] + revolver[idx] + overdraft[idx] for idx in range(len(self.years))]

    def _dividend_payments(self) -> List[float]:
        net_income = self.income_statement().column("Net Income")
        payout = self.inputs.financing.dividend_payout
        return [-max(ni, 0.0) * payout for ni in net_income]

    def _financing_cash_flow_components(self) -> Dict[str, List[float]]:
        financing = self.inputs.financing
        _, senior_outstanding = self._senior_debt_schedules()
        senior_changes = _difference(senior_outstanding)
        _, revolver_outstanding = self._revolver_schedules()
        revolver_changes = _difference(revolver_outstanding)
        _, overdraft_outstanding = self._overdraft_schedules()
        overdraft_changes = _difference(overdraft_outstanding)

        debt_movements = [
            senior_changes[idx] + revolver_changes[idx] + overdraft_changes[idx]
            for idx in range(len(self.years))
        ]

        share_issuance = [0.0 for _ in self.years]
        if financing.share_capital:
            share_issuance[0] = float(financing.share_capital)

        initial_investment = [0.0 for _ in self.years]
        if financing.initial_investment:
            initial_investment[0] = -float(financing.initial_investment)

        dividends_paid = self._dividend_payments()

        return {
            "Debt Drawdown/(Repayment)": debt_movements,
            "Share Capital Raised": share_issuance,
            "Initial Investment": initial_investment,
            "Dividends Paid": dividends_paid,
        }

    def _equity_schedule(self, cash_flow: Table, income: Table) -> List[float]:
        financing = self.inputs.financing
        net_income = income.column("Net Income")
        dividends = [max(ni, 0.0) * financing.dividend_payout for ni in net_income]
        retained = _cumulative([ni - div for ni, div in zip(net_income, dividends)])
        return [financing.share_capital + value for value in retained]

    # ---------------------------------------------------- analysis & metrics
    def evidence_register(self) -> List[Mapping[str, object]]:
        if self._evidence_register_cache is not None:
            return self._evidence_register_cache

        rows = self._evidence_rows()
        self._evidence_register_cache = rows
        return self._evidence_register_cache

    def data_quality_exceptions(self) -> List[Mapping[str, object]]:
        if self._data_quality_exceptions_cache is not None:
            return self._data_quality_exceptions_cache

        issues: List[Mapping[str, object]] = []
        financing = self.inputs.financing

        def _add_issue(area: str, severity: str, issue: str, action: str) -> None:
            issues.append(
                {
                    "Area": area,
                    "Severity": severity,
                    "Issue": issue,
                    "Recommended Action": action,
                }
            )

        if abs(float(financing.discount_rate) - 1.0) < 1e-9:
            _add_issue(
                "Funding",
                "Critical",
                "Discount rate is still set to the placeholder value 1.0.",
                "Replace with an evidence-backed hurdle rate or WACC.",
            )
        for label, value in (
            ("Senior debt interest", financing.senior_debt_interest),
            ("Revolver interest", financing.revolver_interest),
            ("Cash interest", financing.cash_interest),
            ("Dividend payout", financing.dividend_payout),
            ("Share capital", financing.share_capital),
        ):
            if abs(float(value) - 1.0) < 1e-9:
                _add_issue(
                    "Funding",
                    "High",
                    f"{label} is still set to the placeholder value 1.0.",
                    "Replace placeholder funding assumptions with sourced financing terms.",
                )

        capex = self.inputs.capital_expenditure
        for key in ("initial", "contingency", "project_reserve"):
            try:
                value = float(capex.get(key, 0.0)) if isinstance(capex, Mapping) else 0.0
            except (TypeError, ValueError):
                value = 0.0
            if abs(value - 1.0) < 1e-9:
                _add_issue(
                    "Capex",
                    "High",
                    f"Capital expenditure field '{key}' is still set to the placeholder value 1.0.",
                    "Replace with a quoted or benchmarked capex number.",
                )

        if abs(float(self.inputs.tax_rate)) < 1e-9 and not any(abs(float(value)) > 1e-9 for value in self.inputs.tax_rates):
            _add_issue(
                "Tax",
                "High",
                "Tax assumptions are zero across the model horizon.",
                "Set the applicable tax regime and carryforward treatment.",
            )

        dep_placeholders = [
            row.asset_type
            for row in self.inputs.depreciation_schedule
            if abs(float(row.depreciation_rate) - 1.0) < 1e-9 or abs(float(row.acquisition) - 1.0) < 1e-9
        ]
        if dep_placeholders:
            _add_issue(
                "Fixed Assets",
                "High",
                f"Depreciation schedule contains placeholder rows for {', '.join(dep_placeholders[:4])}.",
                "Replace placeholder asset costs, useful lives, and depreciation rates.",
            )

        utility = self.inputs.utility_schedule
        utility_samples = (
            list(utility.electricity_rate)
            + list(utility.water_rate)
            + list(utility.steam_rate)
        )
        if utility_samples and sum(1 for value in utility_samples if abs(float(value) - 1.0) < 1e-9) / len(utility_samples) > 0.5:
            _add_issue(
                "Utilities",
                "Medium",
                "Utility schedule still relies heavily on placeholder rate assumptions.",
                "Replace utility rates and usage with plant-level estimates or supplier quotes.",
            )

        for row in self._evidence_rows():
            if row["Required"] == "Yes" and float(row["Coverage Ratio"]) < 1.0:
                _add_issue(
                    "Evidence",
                    "Medium",
                    f"{row['Assumption']} is missing: {row['Missing Fields'] or 'required metadata'}.",
                    "Complete the evidence register before circulating the model to investors.",
                )

        if not issues:
            issues.append(
                {
                    "Area": "Validation",
                    "Severity": "Info",
                    "Issue": "No structural data-quality exceptions were detected.",
                    "Recommended Action": "Continue validating economics and investor downside cases.",
                }
            )

        self._data_quality_exceptions_cache = issues
        return self._data_quality_exceptions_cache

    def commercial_diagnostics(self) -> Table:
        if self._commercial_diagnostics_cache is not None:
            return self._commercial_diagnostics_cache

        production = self._production()
        prices = self._unit_prices()
        production_costs = self._unit_costs()
        freight_costs = self._freight_costs()
        raw_material_costs = self._variable_costs()

        rows: List[Mapping[str, float | str]] = []
        for product in self.products:
            volume_series = [float(value) for value in production.get(product, [])]
            capacity = float(self.inputs.production_capacity.get(product, 0.0) or 0.0)
            peak_units = max(volume_series) if volume_series else 0.0
            first_units = volume_series[0] if volume_series else 0.0
            realized_capacity = peak_units / capacity if capacity > 0 else float("nan")
            selling_price = float(prices.get(product, 0.0))
            production_cost = float(production_costs.get(product, 0.0))
            freight_cost = float(freight_costs.get(product, 0.0))
            raw_cost = float(raw_material_costs.get(product, 0.0))
            contribution = selling_price - production_cost - freight_cost - raw_cost
            rows.append(
                {
                    "Product": product,
                    "Year 1 Units": first_units,
                    "Peak Units": peak_units,
                    "Peak Capacity Utilisation": realized_capacity,
                    "Selling Price": selling_price,
                    "Production Cost": production_cost,
                    "Freight Cost": freight_cost,
                    "Raw Material Cost per Unit": raw_cost,
                    "Estimated Contribution per Unit": contribution,
                }
            )

        self._commercial_diagnostics_cache = build_table(
            [str(row["Product"]) for row in rows],
            {
                "Year 1 Units": [float(row["Year 1 Units"]) for row in rows],
                "Peak Units": [float(row["Peak Units"]) for row in rows],
                "Peak Capacity Utilisation": [float(row["Peak Capacity Utilisation"]) for row in rows],
                "Selling Price": [float(row["Selling Price"]) for row in rows],
                "Production Cost": [float(row["Production Cost"]) for row in rows],
                "Freight Cost": [float(row["Freight Cost"]) for row in rows],
                "Raw Material Cost per Unit": [float(row["Raw Material Cost per Unit"]) for row in rows],
                "Estimated Contribution per Unit": [float(row["Estimated Contribution per Unit"]) for row in rows],
            },
            index_name="Product",
        )
        return self._commercial_diagnostics_cache

    def sources_and_uses(self) -> Table:
        if self._sources_and_uses_cache is not None:
            return self._sources_and_uses_cache

        financing = self.inputs.financing
        capex_total = sum(self._capex_series())
        sources = [
            ("Source", "Share Capital", float(financing.share_capital or 0.0)),
            (
                "Source",
                "Senior Debt",
                float(sum(max(float(entry.amount), 0.0) for entry in financing.senior_debt_entries)),
            ),
            (
                "Source",
                "Revolver",
                float(sum(max(float(entry.amount), 0.0) for entry in financing.revolver_entries)),
            ),
            (
                "Source",
                "Overdraft",
                float(sum(max(float(entry.amount), 0.0) for entry in financing.overdraft_entries)),
            ),
        ]
        uses = [
            ("Use", "Initial Investment", abs(float(financing.initial_investment or 0.0))),
            ("Use", "Capital Expenditure", float(capex_total)),
        ]
        all_rows = sources + uses
        total_sources = sum(amount for row_type, _, amount in all_rows if row_type == "Source")
        total_uses = sum(amount for row_type, _, amount in all_rows if row_type == "Use")
        all_rows.extend(
            [
                ("Summary", "Total Sources", total_sources),
                ("Summary", "Total Uses", total_uses),
                ("Summary", "Funding Gap", total_sources - total_uses),
            ]
        )
        self._sources_and_uses_cache = build_table(
            [label for _, label, _ in all_rows],
            {
                "Amount": [
                    amount if row_type == "Source" else -amount if row_type == "Use" else amount
                    for row_type, _, amount in all_rows
                ],
            },
            index_name="Line Item",
        )
        return self._sources_and_uses_cache

    def liquidity_bridge(self) -> Table:
        if self._liquidity_bridge_cache is not None:
            return self._liquidity_bridge_cache

        cash_flow = self.cash_flow_statement()
        min_buffer = float(self.inputs.bankability.min_cash_buffer)
        ending_cash = cash_flow.column(CASH_FLOW_END_COLUMN)
        headroom = [float(value) - min_buffer for value in ending_cash]
        self._liquidity_bridge_cache = build_table(
            self.years,
            {
                "Net Cash Generated from Operating Activities": cash_flow.column("Net Cash Generated from Operating Activities"),
                "Net Cash Used in Investing Activities": cash_flow.column("Net Cash Used in Investing Activities"),
                "Net Cash Used in Financing Activities": cash_flow.column("Net Cash Used in Financing Activities"),
                CASH_FLOW_NET_COLUMN: cash_flow.column(CASH_FLOW_NET_COLUMN),
                CASH_FLOW_END_COLUMN: ending_cash,
                "Minimum Cash Buffer": [min_buffer for _ in self.years],
                "Cash Buffer Headroom": headroom,
            },
        )
        return self._liquidity_bridge_cache

    def covenant_headroom(self) -> Table:
        if self._covenant_headroom_cache is not None:
            return self._covenant_headroom_cache

        debt_service = self._debt_service_schedule()
        operating_cash = self.cash_flow_statement().column("Net Cash Generated from Operating Activities")
        min_dscr = float(self.inputs.bankability.min_dscr)
        min_buffer = float(self.inputs.bankability.min_cash_buffer)
        ending_cash = self.cash_flow_statement().column(CASH_FLOW_END_COLUMN)
        dscr_values = [
            _safe_ratio(operating_value, service)
            for operating_value, service in zip(operating_cash, debt_service)
        ]
        self._covenant_headroom_cache = build_table(
            self.years,
            {
                "Debt Service": debt_service,
                "Operating Cash Flow": operating_cash,
                "DSCR": dscr_values,
                "Minimum DSCR": [min_dscr for _ in self.years],
                "DSCR Headroom": [float(value) - min_dscr for value in dscr_values],
                "Ending Cash": ending_cash,
                "Minimum Cash Buffer": [min_buffer for _ in self.years],
                "Cash Buffer Headroom": [float(value) - min_buffer for value in ending_cash],
            },
        )
        return self._covenant_headroom_cache

    def downside_case_summary(self) -> Table:
        if self._downside_case_summary_cache is not None:
            return self._downside_case_summary_cache

        rows: List[Mapping[str, float | str]] = []
        for case in self.inputs.downside_cases:
            scenario_model = FinancialModel(self._inputs_for_downside_case(case))
            summary = scenario_model.summary_metrics()
            ending_cash = scenario_model.cash_flow_statement().column(CASH_FLOW_END_COLUMN)
            rows.append(
                {
                    "Case": case.name,
                    "NPV": scenario_model._summary_metric_value("NPV", summary),
                    "IRR": scenario_model._summary_metric_value("IRR", summary),
                    "Average Debt Service Coverage": scenario_model._summary_metric_value(
                        "Average Debt Service Coverage",
                        summary,
                    ),
                    "Investor Viability Score": scenario_model._summary_metric_value(
                        "Investor Viability Score",
                        summary,
                    ),
                    "Investor Gate Pass Ratio": scenario_model._summary_metric_value(
                        "Investor Gate Pass Ratio",
                        summary,
                    ),
                    "Minimum Ending Cash": min(float(value) for value in ending_cash) if ending_cash else float("nan"),
                }
            )

        self._downside_case_summary_cache = build_table(
            [str(row["Case"]) for row in rows],
            {
                "NPV": [float(row["NPV"]) for row in rows],
                "IRR": [float(row["IRR"]) for row in rows],
                "Average Debt Service Coverage": [
                    float(row["Average Debt Service Coverage"]) for row in rows
                ],
                "Investor Viability Score": [float(row["Investor Viability Score"]) for row in rows],
                "Investor Gate Pass Ratio": [float(row["Investor Gate Pass Ratio"]) for row in rows],
                "Minimum Ending Cash": [float(row["Minimum Ending Cash"]) for row in rows],
            },
            index_name="Case",
        )
        return self._downside_case_summary_cache

    def scenario_analysis(self) -> Dict[str, Table]:
        results: Dict[str, Table] = {}
        base_inflation = list(self.inputs.inflation_series)
        base_discount = float(self.inputs.financing.discount_rate)
        for name, scenario in self.inputs.scenarios.items():
            inflation_override = scenario.get("inflation", base_inflation)
            inflation_series = [float(value) for value in inflation_override]
            interest_values = scenario.get("interest", [base_discount])
            try:
                discount_rate = float(interest_values[0]) if interest_values else base_discount
            except (TypeError, ValueError, IndexError):
                discount_rate = base_discount

            scenario_inputs = copy.deepcopy(self.inputs)
            scenario_inputs.inflation_series = inflation_series
            scenario_inputs.financing.discount_rate = discount_rate
            scenario_model = FinancialModel(scenario_inputs)
            income = scenario_model.income_statement()
            results[name] = income.select(["Net Revenue", "EBITDA", "EBIT", "Net Income"])
        for case in self.inputs.downside_cases:
            scenario_model = FinancialModel(self._inputs_for_downside_case(case))
            income = scenario_model.income_statement()
            results[f"Downside: {case.name}"] = income.select(
                ["Net Revenue", "EBITDA", "EBIT", "Net Income"]
            )
        return results

    def scenario_toolkit(self, scenarios: Mapping[str, Table]) -> Dict[str, ScenarioToolResult]:
        configured = getattr(self.inputs, "scenario_tools", {}) or {}
        if not scenarios or not configured:
            return {}

        scenario_names = list(scenarios.keys())
        if not scenario_names:
            return {}

        base_name = None
        for name in scenario_names:
            if name.lower() == "base":
                base_name = name
                break
        if base_name is None:
            base_name = scenario_names[0]

        def _final_values(variable: str) -> Dict[str, float]:
            values: Dict[str, float] = {}
            for scenario_name, table in scenarios.items():
                if variable not in table.data:
                    continue
                series = table.column(variable)
                if not series:
                    continue
                values[scenario_name] = float(series[-1])
            return values

        def _series_for(variable: str, scenario_name: str) -> List[float]:
            table = scenarios.get(scenario_name)
            if table is None or variable not in table.data:
                return []
            return table.column(variable)

        results: Dict[str, ScenarioToolResult] = {}

        # Decision tree analysis
        decision_variables = configured.get("decision_tree", [])
        if decision_variables:
            decision_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            for variable in decision_variables:
                finals = _final_values(variable)
                if not finals:
                    continue
                best_scenario = max(finals, key=finals.get)
                worst_scenario = min(finals, key=finals.get)
                best_value = finals[best_scenario]
                worst_value = finals[worst_scenario]
                expected = sum(finals.values()) / len(finals)
                decision_rows.append(
                    {
                        "Variable": variable,
                        "Best Scenario": best_scenario,
                        "Best Value": best_value,
                        "Worst Scenario": worst_scenario,
                        "Worst Value": worst_value,
                        "Expected Value": expected,
                    }
                )
                narratives.append(
                    f"{variable} peaks under {best_scenario} and softens the most under {worst_scenario}."
                )
            if decision_rows:
                interpretation = " ".join(narratives) or "Decision tree insights derived from configured scenarios."
                results["decision_tree"] = ScenarioToolResult(rows=decision_rows, interpretation=interpretation)

        # Stress testing
        stress_variables = configured.get("stress_testing", [])
        if stress_variables:
            stress_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            for variable in stress_variables:
                finals = _final_values(variable)
                if not finals:
                    continue
                upside = max(finals.values())
                downside = min(finals.values())
                base_value = finals.get(base_name, sum(finals.values()) / len(finals))
                stress_range = upside - downside
                stress_rows.append(
                    {
                        "Variable": variable,
                        "Base": base_value,
                        "Downside": downside,
                        "Upside": upside,
                        "Stress Range": stress_range,
                    }
                )
                narratives.append(
                    f"{variable} endures a swing of {stress_range:,.2f} across configured stress scenarios."
                )
            if stress_rows:
                results["stress_testing"] = ScenarioToolResult(
                    rows=stress_rows,
                    interpretation=" ".join(narratives) or "Stress testing compares upside and downside spans.",
                )

        # Backtesting
        backtesting_variables = configured.get("backtesting", [])
        if backtesting_variables:
            back_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            for variable in backtesting_variables:
                base_series = _series_for(variable, base_name)
                if not base_series:
                    continue
                comparisons: List[float] = []
                final_errors: List[float] = []
                for scenario_name, table in scenarios.items():
                    if scenario_name == base_name or variable not in table.data:
                        continue
                    series = _series_for(variable, scenario_name)
                    if len(series) != len(base_series):
                        continue
                    errors = [abs(a - b) for a, b in zip(series, base_series)]
                    if not errors:
                        continue
                    comparisons.append(sum(errors) / len(errors))
                    final_errors.append(abs(series[-1] - base_series[-1]))
                if not comparisons:
                    continue
                average_error = sum(comparisons) / len(comparisons)
                max_error = max(final_errors) if final_errors else 0.0
                back_rows.append(
                    {
                        "Variable": variable,
                        "Reference Scenario": base_name,
                        "Mean Absolute Error": average_error,
                        "Worst Final Deviation": max_error,
                    }
                )
                narratives.append(
                    f"{variable} deviates on average by {average_error:,.2f} from the {base_name} path."
                )
            if back_rows:
                results["backtesting"] = ScenarioToolResult(
                    rows=back_rows,
                    interpretation=" ".join(narratives) or "Backtesting compares alternative scenarios to the reference path.",
                )

        # Walk-forward testing
        walk_forward_variables = configured.get("walk_forward", configured.get("walk_forward_testing", []))
        if not isinstance(walk_forward_variables, list):
            walk_forward_variables = list(walk_forward_variables)  # type: ignore[arg-type]
        if walk_forward_variables:
            walk_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            for variable in walk_forward_variables:
                base_series = _series_for(variable, base_name)
                if len(base_series) < 2:
                    continue
                growth_rates = []
                for idx in range(1, len(base_series)):
                    previous = base_series[idx - 1]
                    current = base_series[idx]
                    growth_rates.append(_safe_ratio(current - previous, previous))
                if not growth_rates:
                    continue
                average_growth = sum(growth_rates) / len(growth_rates)
                variance = sum((rate - average_growth) ** 2 for rate in growth_rates) / len(growth_rates)
                volatility = variance ** 0.5
                walk_rows.append(
                    {
                        "Variable": variable,
                        "Average Growth": average_growth,
                        "Volatility": volatility,
                    }
                )
                narratives.append(
                    f"{variable} grows on average {average_growth:.2%} with volatility of {volatility:.2%}."
                )
            if walk_rows:
                results["walk_forward"] = ScenarioToolResult(
                    rows=walk_rows,
                    interpretation=" ".join(narratives) or "Walk-forward analysis summarises stability in the reference scenario.",
                )

        # Driver-based modelling
        driver_variables = configured.get("driver_based", configured.get("driver_based_modeling", []))
        if not isinstance(driver_variables, list):
            driver_variables = list(driver_variables)  # type: ignore[arg-type]
        if driver_variables:
            driver_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            revenue_series = _series_for("Net Revenue", base_name)
            for variable in driver_variables:
                variable_series = _series_for(variable, base_name)
                if not revenue_series or not variable_series:
                    continue
                ratios = [
                    _safe_ratio(var, rev)
                    for var, rev in zip(variable_series, revenue_series)
                    if abs(rev) > 1e-12
                ]
                if not ratios:
                    continue
                final_ratio = ratios[-1]
                average_ratio = sum(ratios) / len(ratios)
                driver_rows.append(
                    {
                        "Variable": variable,
                        "Average Contribution": average_ratio,
                        "Latest Contribution": final_ratio,
                    }
                )
                narratives.append(
                    f"{variable} contributes {final_ratio:.2%} of revenue in the latest projection."
                )
            if driver_rows:
                results["driver_based"] = ScenarioToolResult(
                    rows=driver_rows,
                    interpretation=" ".join(narratives) or "Driver-based modelling links value drivers to revenue.",
                )

        # Real options analysis
        roa_variables = configured.get("real_options", configured.get("real_options_analysis", []))
        if not isinstance(roa_variables, list):
            roa_variables = list(roa_variables)  # type: ignore[arg-type]
        if roa_variables:
            roa_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            for variable in roa_variables:
                finals = _final_values(variable)
                if not finals:
                    continue
                base_value = finals.get(base_name)
                if base_value is None:
                    base_value = sum(finals.values()) / len(finals)
                upside = max(finals.values())
                option_value = max(upside - base_value, 0.0)
                roa_rows.append(
                    {
                        "Variable": variable,
                        "Reference": base_value,
                        "Best Case": upside,
                        "Option Value": option_value,
                    }
                )
                narratives.append(
                    f"{variable} offers an upside optionality of {option_value:,.2f} relative to the reference path."
                )
            if roa_rows:
                results["real_options"] = ScenarioToolResult(
                    rows=roa_rows,
                    interpretation=" ".join(narratives) or "Real options quantify strategic upside over the reference scenario.",
                )

        return results

    def sensitivity_analysis(self) -> Dict[str, Table]:
        results: Dict[str, Table] = {}
        for variable, adjustments in self.inputs.sensitivity.variables.items():
            multipliers = []
            npvs = []
            irrs = []
            for multiplier in adjustments:
                scenario_inputs = copy.deepcopy(self.inputs)
                if variable in {"tablet_price", "selling_price", "price"}:
                    for params in scenario_inputs.unit_costs.values():
                        params.selling_price *= multiplier
                elif variable in {"volume", "production"}:
                    scenario_inputs.production_estimate = {
                        product: [float(value) * multiplier for value in values]
                        for product, values in scenario_inputs.production_estimate.items()
                    }
                elif variable == "raw_material_cost":
                    scaled = {
                        product: value * multiplier
                        for product, value in scenario_inputs.variable_cost_overrides.items()
                    }
                    scenario_inputs.variable_cost_overrides = scaled
                    scenario_inputs.raw_material_cost_per_unit *= multiplier
                    scenario_inputs.raw_material_factors = {
                        product: float(value)
                        for product, value in scenario_inputs.raw_material_factors.items()
                    }
                elif variable == "discount_rate":
                    scenario_inputs.financing.discount_rate = multiplier
                elif variable == "receivable_days":
                    scenario_inputs.working_capital_days.accounts_receivable = [
                        max(int(round(value * multiplier)), 0)
                        for value in scenario_inputs.working_capital_days.accounts_receivable
                    ]
                elif variable == "inventory_days":
                    scenario_inputs.working_capital_days.inventory = [
                        max(int(round(value * multiplier)), 0)
                        for value in scenario_inputs.working_capital_days.inventory
                    ]
                elif variable == "capex":
                    scenario_inputs.capital_expenditure = self._scale_capex_mapping(
                        scenario_inputs.capital_expenditure,
                        multiplier,
                    )
                    for row in scenario_inputs.depreciation_schedule:
                        row.acquisition *= multiplier
                elif variable in {
                    "wage_direct",
                    "wage_indirect",
                    "absenteeism",
                    "overtime_cap",
                    "hiring_delay",
                } and scenario_inputs.labor_model is not None:
                    labor_model = scenario_inputs.labor_model
                    if variable == "wage_direct":
                        labor_model.wage_escalation_direct = [value * multiplier for value in labor_model.wage_escalation_direct]
                    elif variable == "wage_indirect":
                        labor_model.wage_escalation_indirect = [value * multiplier for value in labor_model.wage_escalation_indirect]
                    elif variable == "absenteeism":
                        labor_model.absenteeism = [min(max(value * multiplier, 0.0), 0.95) for value in labor_model.absenteeism]
                    elif variable == "overtime_cap":
                        labor_model.overtime_cap = [max(value * multiplier, 0.0) for value in labor_model.overtime_cap]
                    elif variable == "hiring_delay":
                        labor_model.hiring_delay_quarters = [
                            max(int(round(value * multiplier)), 0)
                            for value in labor_model.hiring_delay_quarters
                        ]

                scenario_model = FinancialModel(scenario_inputs)
                metrics = scenario_model.summary_metrics()
                multipliers.append(multiplier)
                npvs.append(metrics.column("Value")[0])
                irrs.append(metrics.column("Value")[1])
            index = list(range(1, len(multipliers) + 1))
            results[variable] = build_table(index, {"Multiplier": multipliers, "NPV": npvs, "IRR": irrs}, index_name="Case")
        return results

    def monte_carlo_simulation(self) -> Table:
        if self._monte_carlo_cache is not None:
            return self._monte_carlo_cache

        params = self.inputs.monte_carlo
        iterations = params.iterations
        bounds = [float(value) for value in params.revenue_growth_range]
        if len(bounds) >= 2:
            default_low, default_high = sorted((bounds[0], bounds[1]))
        elif bounds:
            span = abs(bounds[0])
            default_low, default_high = -span, span
        else:
            default_low, default_high = -0.05, 0.05

        def _build_baseline(model: "FinancialModel") -> Dict[str, Any]:
            revenue_table = model.revenue_schedule()
            base_revenue = np.array(revenue_table.column("Net Revenue"), dtype=float)
            costs = model.cost_structure()
            raw_materials = np.array(costs.column("Raw Materials"), dtype=float)
            per_product_raw_materials = {
                product: np.array(costs.column(f"Raw Materials - {product}"), dtype=float)
                for product in model.products
                if f"Raw Materials - {product}" in costs.data
            }
            utilities = np.array(costs.column("Utilities"), dtype=float)
            direct_labor = np.array(costs.column("Direct Labor"), dtype=float)
            indirect_labor = np.array(costs.column("General & Admin"), dtype=float)
            depreciation = np.array(model.depreciation_schedule(), dtype=float)
            interest = np.array(model._interest_schedule(), dtype=float)
            tax_schedule = np.array(model._tax_schedule(), dtype=float)
            discount_rate = model.inputs.financing.discount_rate
            capex = np.array(model._capex_series(), dtype=float)
            financing_components = model._financing_cash_flow_components()
            other_financing = {
                name: list(series)
                for name, series in financing_components.items()
                if name != "Dividends Paid"
            }
            base_weighted = np.array(model._distributor_receivable_cache or [0.0 for _ in model.years], dtype=float)
            base_distributor_share = np.array(model._distributor_share_cache or [0.0 for _ in model.years], dtype=float)
            return {
                "base_revenue": base_revenue,
                "raw_materials": raw_materials,
                "per_product_raw_materials": per_product_raw_materials,
                "utilities": utilities,
                "direct_labor": direct_labor,
                "indirect_labor": indirect_labor,
                "depreciation": depreciation,
                "interest": interest,
                "tax_schedule": tax_schedule,
                "discount_rate": discount_rate,
                "capex": capex,
                "other_financing": other_financing,
                "base_weighted": base_weighted,
                "base_distributor_share": base_distributor_share,
            }

        base_baseline = _build_baseline(self)
        baselines = {"base": base_baseline}
        scenario_weights = {name: weight for name, weight in (params.scenario_weights or {}).items() if weight > 0}
        if scenario_weights:
            for name, scenario in self.inputs.scenarios.items():
                if name not in scenario_weights:
                    continue
                scenario_inputs = copy.deepcopy(self.inputs)
                inflation_override = scenario.get("inflation", scenario_inputs.inflation_series)
                scenario_inputs.inflation_series = [float(value) for value in inflation_override]
                interest_values = scenario.get("interest", [scenario_inputs.financing.discount_rate])
                try:
                    scenario_inputs.financing.discount_rate = (
                        float(interest_values[0]) if interest_values else scenario_inputs.financing.discount_rate
                    )
                except (TypeError, ValueError, IndexError):
                    pass
                scenario_model = FinancialModel(scenario_inputs)
                baselines[name] = _build_baseline(scenario_model)
            if "base" not in scenario_weights:
                baselines.pop("base", None)
        else:
            scenario_weights = {"base": 1.0}

        weighted_scenarios = [
            (name, weight)
            for name, weight in scenario_weights.items()
            if weight > 0 and name in baselines
        ]
        if not weighted_scenarios:
            weighted_scenarios = [("base", 1.0)]

        total_weight = sum(weight for _, weight in weighted_scenarios)
        cumulative_weights: List[tuple[str, float]] = []
        running = 0.0
        for name, weight in weighted_scenarios:
            running += weight
            cumulative_weights.append((name, running))

        deterministic_share = params.deterministic_share

        import random

        rng = random.Random()
        if params.seed is not None:
            rng.seed(params.seed)
        fallback_distribution = (params.distribution or "uniform").lower()
        distribution_overrides = params.distributions or {}

        def _distribution_type(variable: str) -> str:
            override = distribution_overrides.get(variable, {})
            dist_name = override.get("type") or override.get("distribution")
            return str(dist_name or fallback_distribution).strip().lower()

        def _distribution_bounds(variable: str, override: Mapping[str, Any]) -> tuple[float, float]:
            range_override = override.get("range") if isinstance(override, Mapping) else None
            low = override.get("low")
            high = override.get("high")
            if low is None and high is None and isinstance(range_override, Iterable):
                values = [float(value) for value in range_override]
                if len(values) >= 2:
                    low, high = values[0], values[1]
            if low is None or high is None:
                low, high = (default_low, default_high) if variable == "revenue_growth" else (-0.05, 0.05)
            return float(low), float(high)

        def _sample_value(
            variable: str,
            *,
            shock: Optional[float] = None,
        ) -> float:
            override = distribution_overrides.get(variable, {})
            distribution = _distribution_type(variable)
            low, high = _distribution_bounds(variable, override)
            min_value = override.get("min", low)
            max_value = override.get("max", high)
            if distribution == "normal":
                mean = float(override.get("mean", (low + high) / 2))
                std = float(override.get("std", (high - low) / 6 if high != low else max(abs(high), 1.0) / 6))
                z = shock if shock is not None else rng.gauss(0.0, 1.0)
                value = mean + std * z
            elif distribution in {"lognormal", "log-normal", "log_normal"}:
                mu = float(override.get("mu", 0.0))
                sigma = float(override.get("sigma", 0.25))
                z = shock if shock is not None else rng.gauss(0.0, 1.0)
                offset = float(override.get("offset", 1.0))
                value = math.exp(mu + sigma * z) - offset
            elif distribution in {"triangular", "triangle"}:
                mode = float(override.get("mode", (low + high) / 2))
                value = rng.triangular(low, high, mode)
            else:
                value = rng.uniform(low, high)
            return max(min(value, float(max_value)), float(min_value))

        def _build_cholesky(variable_order: List[str]) -> List[List[float]]:
            correlations = params.correlations or {}
            size = len(variable_order)
            matrix = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
            for i, var_i in enumerate(variable_order):
                for j, var_j in enumerate(variable_order):
                    if i == j:
                        continue
                    value = correlations.get(var_i, {}).get(var_j)
                    if value is None:
                        value = correlations.get(var_j, {}).get(var_i)
                    if value is None:
                        continue
                    try:
                        corr = float(value)
                    except (TypeError, ValueError):
                        continue
                    matrix[i][j] = max(min(corr, 1.0), -1.0)

            cholesky = [[0.0 for _ in range(size)] for _ in range(size)]
            for i in range(size):
                for j in range(i + 1):
                    total = sum(cholesky[i][k] * cholesky[j][k] for k in range(j))
                    if i == j:
                        value = matrix[i][i] - total
                        cholesky[i][j] = value ** 0.5 if value > 0 else 0.0
                    else:
                        denominator = cholesky[j][j]
                        cholesky[i][j] = (matrix[i][j] - total) / denominator if denominator else 0.0
            return cholesky

        def _correlated_shocks(
            variable_order: List[str],
            cholesky: List[List[float]],
        ) -> Dict[str, float]:
            size = len(variable_order)
            if size == 0:
                return {}
            normals = [rng.gauss(0.0, 1.0) for _ in range(size)]
            correlated = [
                sum(cholesky[i][k] * normals[k] for k in range(i + 1))
                for i in range(size)
            ]
            return dict(zip(variable_order, correlated))

        metric_names = [metric.strip() for metric in params.metrics]
        allowed_metrics = {
            "NPV",
            "Average Net Income",
            "Average EBITDA",
            "Average Cash Flow",
        }
        metrics_to_track = [metric for metric in metric_names if metric in allowed_metrics]
        if "NPV" not in metrics_to_track:
            metrics_to_track.insert(0, "NPV")

        variable_codes = [
            value
            for value in getattr(params, "variables", ["revenue_growth"])
            if value
        ]
        if not variable_codes:
            variable_codes = ["revenue_growth"]
        if "revenue_growth" not in variable_codes:
            variable_codes.insert(0, "revenue_growth")

        results: Dict[str, List[float]] = {metric: [] for metric in metrics_to_track}
        use_correlated = bool(params.correlations)
        correlation_variables = [
            code
            for code in variable_codes
            if code in (params.correlations or {})
            and _distribution_type(code) in {"normal", "lognormal", "log-normal", "log_normal"}
        ]
        if use_correlated and not correlation_variables:
            use_correlated = False
        cholesky = _build_cholesky(correlation_variables) if use_correlated else []

        for _ in range(iterations):
            scenario_roll = rng.random() * total_weight
            scenario_name = weighted_scenarios[-1][0]
            for name, cumulative in cumulative_weights:
                if scenario_roll <= cumulative:
                    scenario_name = name
                    break
            baseline = baselines.get(scenario_name, base_baseline)
            base_revenue = baseline["base_revenue"]
            raw_materials = baseline["raw_materials"]
            per_product_raw_materials = baseline.get("per_product_raw_materials", {})
            utilities = baseline["utilities"]
            direct_labor = baseline["direct_labor"]
            indirect_labor = baseline["indirect_labor"]
            depreciation = baseline["depreciation"]
            interest = baseline["interest"]
            tax_schedule = baseline["tax_schedule"]
            discount_rate = baseline["discount_rate"]
            capex = baseline["capex"]
            other_financing = baseline["other_financing"]
            base_weighted = baseline["base_weighted"]
            base_distributor_share = baseline["base_distributor_share"]
            base_revenue_safe = np.where(np.abs(base_revenue) > 1e-9, base_revenue, 1.0)
            capital_expenditure = (-capex).tolist()

            deterministic = rng.random() < deterministic_share
            shocks = (
                _correlated_shocks(correlation_variables, cholesky)
                if use_correlated
                else {}
            )
            if "revenue_growth" in variable_codes:
                if deterministic:
                    growth_rates = np.zeros(len(self.years), dtype=float)
                else:
                    revenue_shock = shocks.get("revenue_growth")
                    growth_rates = np.array(
                        [_sample_value("revenue_growth", shock=revenue_shock) for _ in self.years],
                        dtype=float,
                    )
            else:
                growth_rates = np.zeros(len(self.years), dtype=float)

            price_factor = 1.0
            if "selling_price" in variable_codes and not deterministic:
                price_factor += _sample_value("selling_price", shock=shocks.get("selling_price"))
            if "price" in variable_codes and not deterministic:
                price_factor += _sample_value("price", shock=shocks.get("price"))
            price_factor = max(price_factor, 0.0)

            raw_factor = 1.0
            if "raw_material_cost" in variable_codes and not deterministic:
                raw_factor += _sample_value("raw_material_cost", shock=shocks.get("raw_material_cost"))
            labor_factor = 1.0
            if "labor_cost" in variable_codes and not deterministic:
                labor_factor += _sample_value("labor_cost", shock=shocks.get("labor_cost"))
            if "wage_direct" in variable_codes and not deterministic:
                labor_factor += _sample_value("wage_direct", shock=shocks.get("wage_direct"))
            if "wage_indirect" in variable_codes and not deterministic:
                labor_factor += _sample_value("wage_indirect", shock=shocks.get("wage_indirect"))
            if "absenteeism" in variable_codes and not deterministic:
                labor_factor += max(_sample_value("absenteeism", shock=shocks.get("absenteeism")), -0.5)
            if "overtime_cap" in variable_codes and not deterministic:
                labor_factor += _sample_value("overtime_cap", shock=shocks.get("overtime_cap"))
            if "hiring_delay" in variable_codes and not deterministic:
                labor_factor += _sample_value("hiring_delay", shock=shocks.get("hiring_delay")) * 0.25
            labor_factor = max(labor_factor, 0.0)
            utility_factor = 1.0
            if "utility_cost" in variable_codes and not deterministic:
                utility_factor += _sample_value("utility_cost", shock=shocks.get("utility_cost"))
            interest_factor = 1.0
            if "senior_debt" in variable_codes and not deterministic:
                interest_factor += _sample_value("senior_debt", shock=shocks.get("senior_debt"))
            tax_factor = 1.0
            if "tax_rate" in variable_codes and not deterministic:
                tax_factor += _sample_value("tax_rate", shock=shocks.get("tax_rate"))
            if tax_factor < 0:
                tax_factor = 0.0

            approval_delay_years = 0
            if "approval_delay" in variable_codes and not deterministic:
                approval_delay_years = max(
                    int(round(_sample_value("approval_delay", shock=shocks.get("approval_delay")))),
                    0,
                )

            receivable_days_delta = 0
            if "receivable_days" in variable_codes and not deterministic:
                receivable_days_delta = int(
                    round(_sample_value("receivable_days", shock=shocks.get("receivable_days")))
                )

            inventory_days_delta = 0
            if "inventory_days" in variable_codes and not deterministic:
                inventory_days_delta = int(
                    round(_sample_value("inventory_days", shock=shocks.get("inventory_days")))
                )

            payable_days_delta = 0
            if "payable_days" in variable_codes and not deterministic:
                payable_days_delta = int(
                    round(_sample_value("payable_days", shock=shocks.get("payable_days")))
                )

            capex_factor = 1.0
            if "capex" in variable_codes and not deterministic:
                capex_factor += _sample_value("capex", shock=shocks.get("capex"))
            capex_factor = max(capex_factor, 0.0)

            risk_adjustment = 1.0
            if "other" in variable_codes and not deterministic:
                risk_adjustment = max(0.0, 1.0 - _sample_value("other", shock=shocks.get("other")))
            risk_series = (
                np.full(len(self.years), risk_adjustment, dtype=float)
                if "other" in variable_codes
                else np.ones(len(self.years), dtype=float)
            )

            simulated_revenue = base_revenue * (1.0 + growth_rates) * price_factor * risk_series
            if approval_delay_years > 0:
                simulated_revenue = np.array(
                    self._shift_series(simulated_revenue.tolist(), approval_delay_years, fill=0.0),
                    dtype=float,
                )

            if per_product_raw_materials:
                per_product_raw_series = self._scale_series_map(
                    per_product_raw_materials,
                    raw_factor,
                    risk_series,
                )
                raw_series = self._sum_series_map(per_product_raw_series)
            else:
                raw_series = raw_materials * raw_factor * risk_series
            utility_series = np.maximum(0.0, utilities * utility_factor)
            direct_series = direct_labor * labor_factor * risk_series
            indirect_series = indirect_labor * labor_factor * risk_series

            utility_cost_share = utility_series * self.inputs.utility_cost_of_sales_share
            utility_admin_share = utility_series - utility_cost_share
            simulated_cost_of_sales = raw_series + utility_cost_share + direct_series
            general_admin_series = indirect_series + utility_admin_share

            gross_profit = simulated_revenue - simulated_cost_of_sales
            ebitda = gross_profit - general_admin_series
            ebit = ebitda - depreciation

            interest_series = (
                interest * interest_factor
                if "senior_debt" in variable_codes
                else interest.copy()
            )

            ebt = ebit - interest_series
            effective_tax = (
                np.clip(tax_schedule * tax_factor, 0.0, 1.0)
                if "tax_rate" in variable_codes
                else tax_schedule.copy()
            )
            taxes: List[float] = []
            net_income: List[float] = []
            nol_balance = 0.0
            apply_nol = bool(self.inputs.tax_loss_carryforward)
            nol_limit = float(self.inputs.tax_loss_limit)
            for idx in range(len(self.years)):
                taxable = float(ebt[idx])
                if taxable <= 0:
                    tax = 0.0
                    if apply_nol:
                        nol_balance += abs(taxable)
                else:
                    if apply_nol and nol_balance > 0:
                        max_offset = taxable * nol_limit
                        offset = min(nol_balance, max_offset)
                        taxable = max(taxable - offset, 0.0)
                        nol_balance -= offset
                    tax = taxable * float(effective_tax[idx])
                taxes.append(tax)
                net_income.append(ebt[idx] - tax)

            ratio = simulated_revenue / base_revenue_safe
            weighted_adjusted = base_weighted * ratio
            working_capital_days = self._adjust_working_capital_days(
                self.inputs.working_capital_days,
                receivable_delta=receivable_days_delta,
                inventory_delta=inventory_days_delta,
                payable_delta=payable_days_delta,
            )

            working_balances = self._working_capital_series_from_inputs(
                simulated_revenue,
                simulated_cost_of_sales,
                distributor_weighted=weighted_adjusted.tolist(),
                distributor_share=base_distributor_share.tolist(),
                days=working_capital_days,
            )

            inventory_change = _difference(working_balances["Inventory"])
            receivable_change = _difference(working_balances["Accounts Receivable"])
            payable_change = _difference(working_balances["Accounts Payable"])
            prepaid_change = _difference(working_balances["Prepaid Expenses"])
            other_asset_change = _difference(working_balances["Other Assets"])
            other_liability_change = _difference(working_balances["Other Liabilities"])

            operating_profit = [
                net_income[idx] + taxes[idx] + float(interest_series[idx])
                for idx in range(len(self.years))
            ]
            inventory_adjustment = [-value for value in inventory_change]
            receivable_adjustment = [-value for value in receivable_change]
            payable_adjustment = [value for value in payable_change]
            prepaid_adjustment = [-value for value in prepaid_change]
            other_asset_adjustment = [-value for value in other_asset_change]
            other_liability_adjustment = [value for value in other_liability_change]

            cash_flow_from_operations = [
                operating_profit[idx]
                + depreciation[idx]
                + inventory_adjustment[idx]
                + receivable_adjustment[idx]
                + payable_adjustment[idx]
                + prepaid_adjustment[idx]
                + other_asset_adjustment[idx]
                + other_liability_adjustment[idx]
                for idx in range(len(self.years))
            ]

            interest_paid = [-float(value) for value in interest_series]
            taxes_paid = [-value for value in taxes]
            net_cash_from_operations = [
                cash_flow_from_operations[idx]
                + interest_paid[idx]
                + taxes_paid[idx]
                for idx in range(len(self.years))
            ]

            if capex_factor != 1.0:
                capital_expenditure = [value * capex_factor for value in capital_expenditure]
            net_cash_from_investing = list(capital_expenditure)

            dividends = [-max(ni, 0.0) * self.inputs.financing.dividend_payout for ni in net_income]
            net_cash_from_financing = [
                dividends[idx]
                + sum(series[idx] for series in other_financing.values())
                for idx in range(len(self.years))
            ]

            net_cash_flow = [
                net_cash_from_operations[idx]
                + net_cash_from_investing[idx]
                + net_cash_from_financing[idx]
                for idx in range(len(self.years))
            ]

            discounted = [
                cf / (1 + discount_rate) ** idx
                for idx, cf in enumerate(net_cash_flow)
            ]
            npv = sum(discounted)

            averages = {
                "Average Net Income": sum(net_income) / len(net_income),
                "Average EBITDA": sum(ebitda) / len(ebitda),
                "Average Cash Flow": sum(net_cash_flow) / len(net_cash_flow),
            }

            for metric in metrics_to_track:
                if metric == "NPV":
                    results[metric].append(npv)
                else:
                    results[metric].append(averages[metric])

        self._monte_carlo_cache = build_table(range(1, iterations + 1), results, index_name="Iteration")
        return self._monte_carlo_cache

    def _assumption_data_quality_score(self) -> float:
        checks: List[tuple[bool, float]] = []
        financing = self.inputs.financing
        checks.append((0.0 < float(financing.discount_rate) < 0.5, 15.0))
        checks.append((0.0 <= float(financing.senior_debt_interest) < 0.5, 10.0))
        checks.append((0.0 <= float(financing.revolver_interest) < 0.6, 8.0))
        checks.append((0.0 <= float(financing.cash_interest) < 0.5, 8.0))
        checks.append((float(financing.senior_debt_interest) >= float(financing.cash_interest), 8.0))

        placeholder_hits = 0
        placeholder_total = 0
        for value in (
            financing.discount_rate,
            financing.senior_debt_interest,
            financing.revolver_interest,
            financing.cash_interest,
            financing.dividend_payout,
            financing.share_capital,
        ):
            placeholder_total += 1
            if abs(float(value) - 1.0) < 1e-9:
                placeholder_hits += 1
        dep_rates = [float(row.depreciation_rate) for row in self.inputs.depreciation_schedule]
        if dep_rates:
            placeholder_total += len(dep_rates)
            placeholder_hits += sum(1 for value in dep_rates if abs(value - 1.0) < 1e-9)
        placeholder_ratio = (placeholder_hits / placeholder_total) if placeholder_total else 0.0
        checks.append((placeholder_ratio <= 0.25, 25.0))

        capex_mapping = self.inputs.capital_expenditure
        if isinstance(capex_mapping, Mapping):
            capex_values = [
                float(capex_mapping.get("initial", 0.0) or 0.0),
                float(capex_mapping.get("contingency", 0.0) or 0.0),
                float(capex_mapping.get("project_reserve", 0.0) or 0.0),
            ]
            checks.append((all(abs(value - 1.0) >= 1e-9 for value in capex_values), 10.0))

        utility = self.inputs.utility_schedule
        utility_rates = (
            list(utility.electricity_rate)
            + list(utility.water_rate)
            + list(utility.steam_rate)
        )
        if utility_rates:
            placeholder_rate_ratio = sum(
                1 for value in utility_rates if abs(float(value) - 1.0) < 1e-9
            ) / len(utility_rates)
            checks.append((placeholder_rate_ratio <= 0.5, 8.0))

        tax_configured = abs(float(self.inputs.tax_rate)) > 1e-9 or any(
            abs(float(value)) > 1e-9 for value in self.inputs.tax_rates
        )
        checks.append((tax_configured, 8.0))

        labor = self.inputs.labor_model
        if labor is not None and labor.roles:
            metadata_completion = sum(
                1
                for role in labor.roles
                if role.source.strip() and role.owner.strip() and role.benchmark_year.strip()
            ) / max(len(labor.roles), 1)
            checks.append((metadata_completion >= 0.8, 10.0))

        checks.append(
            (
                self._evidence_coverage_ratio()
                >= float(self.inputs.bankability.min_evidence_coverage),
                16.0,
            )
        )

        score = 100.0
        for passed, penalty in checks:
            if not passed:
                score -= penalty
        return max(0.0, min(score, 100.0))

    def _bankability_checks(
        self,
        *,
        npv_value: float,
        irr_value: float,
        discounted_payback_years: float,
        average_dscr: float,
        cash_end: Sequence[Number],
        quality_score: float,
        evidence_coverage: float,
        viability_score: float,
    ) -> tuple[List[Dict[str, float | str]], Dict[str, float]]:
        thresholds = self.inputs.bankability
        irr_hurdle = max(float(self.inputs.financing.discount_rate), float(thresholds.min_irr))
        min_cash = min(float(value) for value in cash_end) if cash_end else float("nan")
        rows: List[Dict[str, float | str]] = [
            {
                "Gate": "NPV Positive",
                "Actual": float(npv_value),
                "Threshold": 0.0,
                "Comparator": ">",
                "Passed": 1.0 if npv_value > 0.0 else 0.0,
            },
            {
                "Gate": "IRR Hurdle",
                "Actual": float(irr_value),
                "Threshold": float(irr_hurdle),
                "Comparator": ">=",
                "Passed": 1.0 if _is_finite(irr_value) and irr_value >= irr_hurdle else 0.0,
            },
            {
                "Gate": "Average DSCR",
                "Actual": float(average_dscr),
                "Threshold": float(thresholds.min_dscr),
                "Comparator": ">=",
                "Passed": 1.0
                if _is_finite(average_dscr) and average_dscr >= float(thresholds.min_dscr)
                else 0.0,
            },
            {
                "Gate": "Discounted Payback",
                "Actual": float(discounted_payback_years),
                "Threshold": float(thresholds.max_discounted_payback),
                "Comparator": "<=",
                "Passed": 1.0
                if _is_finite(discounted_payback_years)
                and discounted_payback_years <= float(thresholds.max_discounted_payback)
                else 0.0,
            },
            {
                "Gate": "Minimum Ending Cash",
                "Actual": float(min_cash),
                "Threshold": float(thresholds.min_cash_buffer),
                "Comparator": ">=",
                "Passed": 1.0
                if _is_finite(min_cash) and min_cash >= float(thresholds.min_cash_buffer)
                else 0.0,
            },
            {
                "Gate": "Assumption Quality",
                "Actual": float(quality_score),
                "Threshold": float(thresholds.min_assumption_quality),
                "Comparator": ">=",
                "Passed": 1.0
                if quality_score >= float(thresholds.min_assumption_quality)
                else 0.0,
            },
            {
                "Gate": "Evidence Coverage",
                "Actual": float(evidence_coverage),
                "Threshold": float(thresholds.min_evidence_coverage),
                "Comparator": ">=",
                "Passed": 1.0
                if evidence_coverage >= float(thresholds.min_evidence_coverage)
                else 0.0,
            },
            {
                "Gate": "Viability Score",
                "Actual": float(viability_score),
                "Threshold": float(thresholds.min_viability_score),
                "Comparator": ">=",
                "Passed": 1.0
                if viability_score >= float(thresholds.min_viability_score)
                else 0.0,
            },
        ]
        return rows, {"IRR Hurdle": float(irr_hurdle), "Minimum Ending Cash": float(min_cash)}

    def _investor_gate_metrics(
        self,
        *,
        npv_value: float,
        irr_value: float,
        discounted_payback_years: float,
        average_dscr: float,
        cash_end: Sequence[Number],
        quality_score: float,
        evidence_coverage: float,
        viability_score: float,
    ) -> Dict[str, float]:
        rows, metadata = self._bankability_checks(
            npv_value=npv_value,
            irr_value=irr_value,
            discounted_payback_years=discounted_payback_years,
            average_dscr=average_dscr,
            cash_end=cash_end,
            quality_score=quality_score,
            evidence_coverage=evidence_coverage,
            viability_score=viability_score,
        )
        passed = sum(int(float(row["Passed"])) for row in rows)
        ratio = passed / len(rows) if rows else float("nan")
        return {
            "Investor Gate Pass Count": float(passed),
            "Investor Gate Pass Ratio": ratio,
            "Investor Gate Status": 1.0 if passed == len(rows) else 0.0,
            "IRR Hurdle": float(metadata["IRR Hurdle"]),
            "Minimum Ending Cash": float(metadata["Minimum Ending Cash"]),
        }

    def _investor_risk_metrics(self, irr_hurdle: float) -> Dict[str, float]:
        monte = self.monte_carlo_simulation()
        if not monte.index:
            return {}

        def _clean(values: Sequence[Number]) -> List[float]:
            cleaned: List[float] = []
            for value in values:
                numeric = float(value)
                if _is_finite(numeric):
                    cleaned.append(numeric)
            return cleaned

        result: Dict[str, float] = {}
        if "NPV" in monte.data:
            npv_values = _clean(monte.column("NPV"))
            if npv_values:
                npv_array = np.array(npv_values, dtype=float)
                result["Probability NPV < 0"] = float(np.mean(npv_array < 0.0))
                result["NPV P10"] = float(np.percentile(npv_array, 10))
                result["NPV P50"] = float(np.percentile(npv_array, 50))
                result["NPV P90"] = float(np.percentile(npv_array, 90))
        if "IRR" in monte.data:
            irr_values = _clean(monte.column("IRR"))
            if irr_values:
                irr_array = np.array(irr_values, dtype=float)
                result["Probability IRR < Hurdle"] = float(np.mean(irr_array < irr_hurdle))
                result["IRR P10"] = float(np.percentile(irr_array, 10))
                result["IRR P50"] = float(np.percentile(irr_array, 50))
                result["IRR P90"] = float(np.percentile(irr_array, 90))
        return result

    def bankability_gate(self) -> Table:
        if self._bankability_gate_cache is not None:
            return self._bankability_gate_cache

        summary = self.summary_metrics()
        cash_end = self.cash_flow_statement().column(CASH_FLOW_END_COLUMN)
        rows, _ = self._bankability_checks(
            npv_value=self._summary_metric_value("NPV", summary),
            irr_value=self._summary_metric_value("IRR", summary),
            discounted_payback_years=self._summary_metric_value("Discounted Payback", summary),
            average_dscr=self._summary_metric_value("Average Debt Service Coverage", summary),
            cash_end=cash_end,
            quality_score=self._summary_metric_value("Assumption Data Quality Score", summary),
            evidence_coverage=self._summary_metric_value("Evidence Coverage Ratio", summary),
            viability_score=self._summary_metric_value("Investor Viability Score", summary),
        )
        self._bankability_gate_cache = build_table(
            [str(row["Gate"]) for row in rows],
            {
                "Actual": [float(row["Actual"]) for row in rows],
                "Threshold": [float(row["Threshold"]) for row in rows],
                "Passed": [float(row["Passed"]) for row in rows],
            },
            index_name="Gate",
        )
        return self._bankability_gate_cache

    def summary_metrics(self) -> Table:
        if self._summary_metrics_cache is not None:
            return self._summary_metrics_cache

        cash_flow_table = self.cash_flow_statement()
        cash_flow = cash_flow_table.column(CASH_FLOW_NET_COLUMN)
        income = self.income_statement()
        net_revenue = income.column("Net Revenue")
        discount_rate = self.inputs.financing.discount_rate
        discounted = [cf / (1 + discount_rate) ** idx for idx, cf in enumerate(cash_flow)]
        npv_value = sum(discounted)
        irr_result = npf_irr(cash_flow)
        irr_value = irr_result.value
        payback_years = self._payback_period(cash_flow)
        discounted_payback_years = self._payback_period(discounted)

        gross_margin_values = income.column("Gross Profit Margin")
        ebitda_margin_values = income.column("EBITDA Margin")
        net_income = income.column("Net Income")
        net_margin_values = [
            _safe_ratio(value, revenue) for value, revenue in zip(net_income, net_revenue)
        ]
        weighted_gross_margin = _weighted_average(gross_margin_values, net_revenue)
        weighted_ebitda_margin = _weighted_average(ebitda_margin_values, net_revenue)
        weighted_net_margin = _weighted_average(net_margin_values, net_revenue)
        if not _is_finite(weighted_gross_margin):
            weighted_gross_margin = _average(gross_margin_values)
        if not _is_finite(weighted_ebitda_margin):
            weighted_ebitda_margin = _average(ebitda_margin_values)
        if not _is_finite(weighted_net_margin):
            weighted_net_margin = _average(net_margin_values)

        operating_cash = cash_flow_table.column("Net Cash Generated from Operating Activities")
        operating_cash_margin = _weighted_average(
            [_safe_ratio(value, revenue) for value, revenue in zip(operating_cash, net_revenue)],
            net_revenue,
        )
        if not _is_finite(operating_cash_margin):
            operating_cash_margin = _average(
                [_safe_ratio(value, revenue) for value, revenue in zip(operating_cash, net_revenue)]
            )

        revenue_cagr = float("nan")
        if len(net_revenue) > 1 and net_revenue[0] > 0 and net_revenue[-1] > 0:
            revenue_cagr = (net_revenue[-1] / net_revenue[0]) ** (1 / (len(net_revenue) - 1)) - 1
        mid_period_cagr = float("nan")
        if len(net_revenue) > 3:
            mid_index = len(net_revenue) // 2
            start = net_revenue[mid_index - 1]
            end = net_revenue[-1]
            span = len(net_revenue) - mid_index
            if span > 0 and start > 0 and end > 0:
                mid_period_cagr = (end / start) ** (1 / span) - 1

        rolling_cagr = _rolling_cagr(net_revenue, 3)

        debt_service = self._debt_service_schedule()
        dscr_values = [
            _safe_ratio(operating_cash_value, service)
            for operating_cash_value, service in zip(operating_cash, debt_service)
        ]
        average_dscr = _average(dscr_values)

        initial_investment = float(self.inputs.financing.initial_investment or 0.0)
        total_capex = sum(self._capex_series())
        total_investment = abs(initial_investment) + total_capex
        profitability_index = float("nan")
        if total_investment > 0:
            profitability_index = (npv_value + total_investment) / total_investment

        viability_config = self.inputs.viability

        def _config_for(
            name: str,
            default_low: float,
            default_high: float,
            inverse: bool = False,
        ) -> tuple[float, float, bool]:
            config = viability_config.metrics.get(name)
            if config is None:
                return default_low, default_high, inverse
            return config.low, config.high, config.inverse

        irr_low, irr_high, irr_inverse = _config_for("IRR", 0.12, 0.25)
        pi_low, pi_high, pi_inverse = _config_for("Profitability Index", 1.1, 1.6)
        pay_low, pay_high, pay_inverse = _config_for("Payback", 3.0, 8.0, True)
        ebitda_low, ebitda_high, ebitda_inverse = _config_for("EBITDA Margin", 0.15, 0.35)
        cagr_low, cagr_high, cagr_inverse = _config_for("Revenue CAGR", 0.05, 0.2)
        dscr_low, dscr_high, dscr_inverse = _config_for("DSCR", 1.2, 2.0)

        viability_scores = {
            "IRR": _score_range(irr_value, irr_low, irr_high, inverse=irr_inverse),
            "Profitability Index": _score_range(profitability_index, pi_low, pi_high, inverse=pi_inverse),
            "Payback": _score_range(payback_years, pay_low, pay_high, inverse=pay_inverse),
            "EBITDA Margin": _score_range(weighted_ebitda_margin, ebitda_low, ebitda_high, inverse=ebitda_inverse),
            "Revenue CAGR": _score_range(revenue_cagr, cagr_low, cagr_high, inverse=cagr_inverse),
            "DSCR": _score_range(average_dscr, dscr_low, dscr_high, inverse=dscr_inverse),
        }
        viability_score = _weighted_score(viability_scores, viability_config.weights) * 100

        metric_names = [
            "NPV",
            "IRR",
            "Payback Period",
            "Discounted Payback",
            "Profitability Index",
            "Revenue CAGR",
            "Mid-period Revenue CAGR",
            "Rolling Revenue CAGR (3Y Avg)",
            "Weighted Average Gross Margin",
            "Weighted Average EBITDA Margin",
            "Weighted Average Net Margin",
            "Weighted Average Operating Cash Flow Margin",
            "Average Debt Service Coverage",
            "Investor Viability Score",
        ]
        metric_values = [
            npv_value,
            irr_value,
            payback_years,
            discounted_payback_years,
            profitability_index,
            revenue_cagr,
            mid_period_cagr,
            rolling_cagr,
            weighted_gross_margin,
            weighted_ebitda_margin,
            weighted_net_margin,
            operating_cash_margin,
            average_dscr,
            viability_score,
        ]

        labor_kpis = self._labor_kpi_cache or {}
        for label in (
            "Average Labor Cost per Unit",
            "Average Units per Labor Hour",
            "Average Fixed Labor Share",
        ):
            if label in labor_kpis:
                metric_names.append(label)
                metric_values.append(float(labor_kpis[label]))

        cash_end = cash_flow_table.column(CASH_FLOW_END_COLUMN)
        quality_score = self._assumption_data_quality_score()
        evidence_coverage = self._evidence_coverage_ratio()
        gate_metrics = self._investor_gate_metrics(
            npv_value=npv_value,
            irr_value=irr_value,
            discounted_payback_years=discounted_payback_years,
            average_dscr=average_dscr,
            cash_end=cash_end,
            quality_score=quality_score,
            evidence_coverage=evidence_coverage,
            viability_score=viability_score,
        )
        risk_metrics = self._investor_risk_metrics(
            gate_metrics.get("IRR Hurdle", float(self.inputs.financing.discount_rate))
        )

        for label, value in gate_metrics.items():
            if label == "IRR Hurdle":
                continue
            metric_names.append(label)
            metric_values.append(float(value))

        metric_names.append("Evidence Coverage Ratio")
        metric_values.append(float(evidence_coverage))
        metric_names.append("Assumption Data Quality Score")
        metric_values.append(float(quality_score))

        for label in (
            "Probability NPV < 0",
            "Probability IRR < Hurdle",
            "NPV P10",
            "NPV P50",
            "NPV P90",
            "IRR P10",
            "IRR P50",
            "IRR P90",
        ):
            if label in risk_metrics:
                metric_names.append(label)
                metric_values.append(float(risk_metrics[label]))

        table = build_table(
            metric_names,
            {"Value": metric_values},
            index_name="Metric",
        )
        self._summary_metrics_cache = table
        self._irr_result = irr_result
        return table

    def irr_diagnostics(self) -> Optional["IRRResult"]:
        if self._irr_result is None:
            self.summary_metrics()
        return self._irr_result

    def ai_enhancements(
        self,
        *,
        income: Table,
        summary: Table,
        cash_flow: Table,
    ) -> Optional[AIInsights]:
        config = getattr(self.inputs, "ai", None)
        if config is None:
            return None

        revenues: Sequence[float] = []
        if "Net Revenue" in income.data:
            revenues = income.column("Net Revenue")

        ml_table: Optional[Table] = None
        advisor: Optional[MachineLearningAdvisor] = None
        if revenues and config.forecast_horizon > 0:
            advisor = MachineLearningAdvisor(config)
            ml_table = advisor.revenue_forecast(self.years, revenues)

        generative = GenerativeAdvisor(config)
        summary_text = generative.summarise(
            summary=summary,
            income=income,
            cash_flow=cash_flow,
            ml_table=ml_table,
        )

        metadata = dict(generative.metadata)
        if advisor is not None:
            metadata["ml_diagnostics"] = {
                name: dict(values) for name, values in advisor.diagnostics.items()
            }
            metadata["ml_backtest"] = {
                name: dict(values) for name, values in advisor.backtest_diagnostics.items()
            }
            metadata["ml_explainability"] = {
                name: dict(values) for name, values in advisor.explainability.items()
            }
            metadata["ml_audit_log"] = dict(advisor.audit_log)
        metadata["ai_config"] = {
            "enabled": config.enabled,
            "provider": config.provider,
            "model": config.model,
            "forecast_horizon": config.forecast_horizon,
            "ml_methods": list(config.ml_methods),
            "generative_features": list(config.generative_features),
            "regularization": config.regularization,
            "min_forecast": config.min_forecast,
            "max_forecast_multiplier": config.max_forecast_multiplier,
            "seasonality_period": config.seasonality_period,
        }
        metadata["risk_factor_overview"] = self.risk_factor_diagnostics().as_dict()
        irr_info = self.irr_diagnostics()
        if irr_info is not None:
            metadata["irr_diagnostics"] = {
                "value": irr_info.value,
                "solutions": list(irr_info.solutions),
                "iterations": irr_info.iterations,
                "method": irr_info.method,
                "converged": irr_info.converged,
                "tolerance": irr_info.tolerance,
                "message": irr_info.message,
            }

        return AIInsights(
            ml_forecast=ml_table,
            generative_summary=summary_text,
            enabled=config.enabled,
            metadata=metadata,
        )

    def goal_seek_metrics(
        self,
        summary: Optional[Table] = None,
        income: Optional[Table] = None,
        cash_flow: Optional[Table] = None,
    ) -> Table:
        config = getattr(self.inputs, "goal_seek", None)
        if config is None:
            return build_table([], {"Target": []}, index_name="Metric")

        source = (config.source or "income_statement").lower()
        metric_name = config.metric

        actual = float("nan")
        if source == "summary":
            summary_table = summary or self.summary_metrics()
            if metric_name in summary_table.index:
                position = summary_table.index.index(metric_name)
                actual = summary_table.data["Value"][position]
        elif source == "cash_flow":
            cash_table = cash_flow or self.cash_flow_statement()
            actual = _value_for_year(cash_table, metric_name, config.year)
        else:
            income_table = income or self.income_statement()
            actual = _value_for_year(income_table, metric_name, config.year)

        gap = config.target - actual
        multiplier = float("nan")
        if abs(actual) > 1e-9:
            multiplier = config.target / actual

        return build_table(
            [metric_name],
            {
                "Target": [config.target],
                "Actual": [actual],
                "Gap": [gap],
                "Required Multiplier": [multiplier],
            },
            index_name="Metric",
        )

    def break_even_analysis(self) -> Table:
        configured: Dict[str, BreakEvenRow] = {
            row.product: row for row in self.inputs.break_even_rows
        }

        price_lookup = self._unit_prices()
        variable_lookup = self._variable_costs()
        fixed_overrides = getattr(self.inputs, "fixed_cost_overrides", {})
        total_units_lookup = self._total_units()

        product_order: List[str] = []
        seen: Dict[str, None] = {}
        for name in list(self.products) + list(configured.keys()):
            if name not in seen:
                seen[name] = None
                product_order.append(name)

        columns: Dict[str, List[float]] = {
            "Fixed Cost": [],
            "Variable Cost per Unit": [],
            "Selling Price": [],
            "Contribution Margin": [],
            "Contribution Margin Ratio": [],
            "Target Profit": [],
            "Break-even Units": [],
            "Break-even Revenue": [],
            "Expected Volume": [],
            "Margin of Safety (Units)": [],
            "Margin of Safety (%)": [],
        }

        for product in product_order:
            params: ProductParameters | None = self.inputs.unit_costs.get(product)
            override = configured.get(product)

            selling_price = (
                override.selling_price
                if override is not None
                else price_lookup.get(product, params.selling_price if params else 0.0)
            )
            variable_cost = (
                override.variable_cost
                if override is not None
                else variable_lookup.get(product, 0.0)
            )
            if override is not None:
                fixed_cost = override.fixed_cost
            else:
                fixed_cost = float(fixed_overrides.get(product, 0.0) or 0.0)
            target_profit = override.target_profit if override is not None else 0.0
            expected_volume = (
                override.expected_volume
                if override is not None
                else total_units_lookup.get(product, 0.0)
            )

            contribution = selling_price - variable_cost
            ratio = _safe_ratio(contribution, selling_price)

            if contribution <= 0:
                break_even_units = float("nan")
                break_even_revenue = float("nan")
            else:
                break_even_units = (fixed_cost + target_profit) / contribution
                break_even_revenue = break_even_units * selling_price

            if expected_volume > 0 and break_even_units == break_even_units:
                margin_of_safety_units = expected_volume - break_even_units
                margin_of_safety_pct = _safe_ratio(margin_of_safety_units, expected_volume)
            else:
                margin_of_safety_units = float("nan")
                margin_of_safety_pct = float("nan")

            columns["Fixed Cost"].append(fixed_cost)
            columns["Variable Cost per Unit"].append(variable_cost)
            columns["Selling Price"].append(selling_price)
            columns["Contribution Margin"].append(contribution)
            columns["Contribution Margin Ratio"].append(ratio)
            columns["Target Profit"].append(target_profit)
            columns["Break-even Units"].append(break_even_units)
            columns["Break-even Revenue"].append(break_even_revenue)
            columns["Expected Volume"].append(expected_volume)
            columns["Margin of Safety (Units)"].append(margin_of_safety_units)
            columns["Margin of Safety (%)"].append(margin_of_safety_pct)

        return build_table(product_order, columns, index_name="Product")

    def _payback_period(self, cash_flows: Iterable[Number]) -> float:
        cumulative = _cumulative(cash_flows)
        start_year = float(self.years[0]) if self.years else 0.0
        for idx, value in enumerate(cumulative):
            if value >= 0:
                if idx == 0:
                    return 0.0
                previous = cumulative[idx - 1]
                if value == previous:
                    return float(self.years[idx] - start_year)
                year_before = self.years[idx - 1]
                year_after = self.years[idx]
                step = year_after - year_before
                fraction = (0.0 - previous) / (value - previous)
                return float((year_before - start_year) + step * fraction)
        return float("nan")

    def payback_schedule(self) -> Table:
        cash_flows = self.cash_flow_statement().column(CASH_FLOW_NET_COLUMN)
        cumulative = _cumulative(cash_flows)
        return build_table(self.years, {"Cash Flow": cash_flows, "Cumulative": cumulative})

    def discounted_payback_schedule(self) -> Table:
        discount_rate = self.inputs.financing.discount_rate
        cash_flows = self.cash_flow_statement().column(CASH_FLOW_NET_COLUMN)
        discounted = [cf / (1 + discount_rate) ** idx for idx, cf in enumerate(cash_flows)]
        cumulative = _cumulative(discounted)
        return build_table(self.years, {"Discounted Cash Flow": discounted, "Cumulative": cumulative})

    def run(self) -> FinancialOutputs:
        core = self.run_core()
        scenarios = self.scenario_analysis()
        scenario_tools = self.scenario_toolkit(scenarios)
        sensitivity = self.sensitivity_analysis()
        monte_carlo = self.monte_carlo_simulation()
        ai_insights = self.ai_enhancements(
            income=core.income_statement,
            summary=core.summary_metrics,
            cash_flow=core.cash_flow,
        )
        return FinancialOutputs(
            income_statement=core.income_statement,
            balance_sheet=core.balance_sheet,
            cash_flow=core.cash_flow,
            summary_metrics=core.summary_metrics,
            goal_seek=core.goal_seek,
            break_even=core.break_even,
            payback=core.payback,
            discounted_payback=core.discounted_payback,
            scenario_results=scenarios,
            sensitivity_results=sensitivity,
            monte_carlo=monte_carlo,
            scenario_tool_results=scenario_tools,
            risk_factor_diagnostics=core.risk_factor_diagnostics,
            ai_insights=ai_insights,
            commercial_diagnostics=core.commercial_diagnostics,
            bankability_gate=core.bankability_gate,
            data_quality_exceptions=core.data_quality_exceptions,
            evidence_register=core.evidence_register,
            sources_and_uses=core.sources_and_uses,
            liquidity_bridge=core.liquidity_bridge,
            covenant_headroom=core.covenant_headroom,
            downside_case_summary=core.downside_case_summary,
        )

    def run_core(self) -> FinancialOutputs:
        income = self.income_statement()
        balance = self.balance_sheet()
        cash_flow = self.cash_flow_statement()
        summary = self.summary_metrics()
        goal_seek = self.goal_seek_metrics(summary=summary, income=income, cash_flow=cash_flow)
        break_even = self.break_even_analysis()
        payback = self.payback_schedule()
        discounted_payback = self.discounted_payback_schedule()
        commercial_diagnostics = self.commercial_diagnostics()
        bankability_gate = self.bankability_gate()
        data_quality_exceptions = self.data_quality_exceptions()
        evidence_register = self.evidence_register()
        sources_and_uses = self.sources_and_uses()
        liquidity_bridge = self.liquidity_bridge()
        covenant_headroom = self.covenant_headroom()
        downside_case_summary = self.downside_case_summary()
        empty_monte_carlo = build_table([], {"NPV": []}, index_name="Iteration")
        return FinancialOutputs(
            income_statement=income,
            balance_sheet=balance,
            cash_flow=cash_flow,
            summary_metrics=summary,
            goal_seek=goal_seek,
            break_even=break_even,
            payback=payback,
            discounted_payback=discounted_payback,
            scenario_results={},
            sensitivity_results={},
            monte_carlo=empty_monte_carlo,
            scenario_tool_results={},
            risk_factor_diagnostics=self.risk_factor_diagnostics(),
            ai_insights=None,
            commercial_diagnostics=commercial_diagnostics,
            bankability_gate=bankability_gate,
            data_quality_exceptions=data_quality_exceptions,
            evidence_register=evidence_register,
            sources_and_uses=sources_and_uses,
            liquidity_bridge=liquidity_bridge,
            covenant_headroom=covenant_headroom,
            downside_case_summary=downside_case_summary,
        )


@dataclass
class IRRResult:
    value: float
    iterations: int
    method: str
    converged: bool
    message: str = ""
    solutions: List[float] = field(default_factory=list)
    tolerance: float = 1e-8

    def __float__(self) -> float:
        return float(self.value)


def npf_irr(
    cashflows: Iterable[Number],
    *,
    guess: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    bracket_min: float = -0.9,
    bracket_max: float = 10.0,
    bracket_steps: int = 200,
) -> IRRResult:
    values = [float(value) for value in cashflows]
    if len(values) < 2:
        return IRRResult(
            value=float("nan"),
            iterations=0,
            method="insufficient_data",
            converged=False,
            message="At least two cash flow periods are required to compute IRR.",
            tolerance=tolerance,
        )
    if not any(value > 0 for value in values) or not any(value < 0 for value in values):
        return IRRResult(
            value=float("nan"),
            iterations=0,
            method="no_sign_change",
            converged=False,
            message="Cash flows do not change sign, so IRR is undefined.",
            tolerance=tolerance,
        )

    def _npv(rate: float) -> float:
        total = 0.0
        for idx, value in enumerate(values):
            denominator = (1 + rate) ** idx
            if abs(denominator) < 1e-18:
                return float("inf")
            total += value / denominator
        return total

    def _derivative(rate: float) -> float:
        total = 0.0
        for idx, value in enumerate(values):
            if idx == 0:
                continue
            denominator = (1 + rate) ** (idx + 1)
            if abs(denominator) < 1e-18:
                return 0.0
            total += -idx * value / denominator
        return total

    def _sign_changes(series: Iterable[float]) -> int:
        signs = [value for value in series if value != 0.0]
        if not signs:
            return 0
        changes = 0
        previous = signs[0] > 0
        for value in signs[1:]:
            current = value > 0
            if current != previous:
                changes += 1
            previous = current
        return changes

    def _bisect(lower: float, upper: float) -> tuple[float, int, bool]:
        lower_value = _npv(lower)
        upper_value = _npv(upper)
        if lower_value == 0.0:
            return lower, 0, True
        if upper_value == 0.0:
            return upper, 0, True
        if lower_value * upper_value > 0:
            return (lower + upper) / 2, 0, False
        for iteration in range(1, max_iterations + 1):
            midpoint = (lower + upper) / 2
            mid_value = _npv(midpoint)
            if abs(mid_value) < tolerance or abs(upper - lower) < tolerance:
                return midpoint, iteration, True
            if lower_value * mid_value < 0:
                upper = midpoint
                upper_value = mid_value
            else:
                lower = midpoint
                lower_value = mid_value
        return (lower + upper) / 2, max_iterations, False

    def _scan_brackets(low: float, high: float, steps: int) -> List[tuple[float, float]]:
        if steps < 2:
            return []
        step_size = (high - low) / steps
        brackets: List[tuple[float, float]] = []
        previous_rate = low
        previous_value = _npv(previous_rate)
        for idx in range(1, steps + 1):
            current_rate = low + step_size * idx
            current_value = _npv(current_rate)
            if previous_value == 0.0:
                brackets.append((previous_rate, previous_rate))
            elif current_value == 0.0:
                brackets.append((current_rate, current_rate))
            elif previous_value * current_value < 0:
                brackets.append((previous_rate, current_rate))
            previous_rate = current_rate
            previous_value = current_value
        return brackets

    sign_change_count = _sign_changes(values)
    newton_solution: Optional[float] = None
    newton_iterations = 0
    newton_converged = False

    for iteration in range(1, max_iterations + 1):
        guess = max(guess, -0.999999)
        value = _npv(guess)
        derivative = _derivative(guess)
        if abs(derivative) < 1e-12:
            break
        next_guess = guess - value / derivative
        if abs(next_guess - guess) < tolerance:
            newton_solution = next_guess
            newton_iterations = iteration
            newton_converged = True
            break
        guess = next_guess

    bracket_low = bracket_min
    bracket_high = bracket_max
    brackets = _scan_brackets(bracket_low, bracket_high, bracket_steps)
    if not brackets:
        for scale in (2.0, 5.0, 10.0):
            brackets = _scan_brackets(bracket_low, bracket_high * scale, bracket_steps)
            if brackets:
                break

    solutions: List[float] = []
    best_bisect_iterations = 0
    bisect_converged = False
    for lower, upper in brackets:
        root, iterations_used, converged = _bisect(lower, upper)
        if converged:
            bisect_converged = True
            best_bisect_iterations = max(best_bisect_iterations, iterations_used)
        if not math.isfinite(root):
            continue
        if not solutions or all(abs(root - existing) > tolerance for existing in solutions):
            solutions.append(root)

    if newton_converged and newton_solution is not None and math.isfinite(newton_solution):
        if all(abs(newton_solution - existing) > tolerance for existing in solutions):
            solutions.insert(0, newton_solution)

    solutions = sorted(solutions)
    chosen_value = float("nan")
    chosen_method = "none"
    iterations_used = 0
    converged = False
    if newton_converged and newton_solution is not None:
        chosen_value = newton_solution
        chosen_method = "newton"
        iterations_used = newton_iterations
        converged = True
    elif solutions:
        positive_solutions = [value for value in solutions if value > -1.0]
        chosen_value = positive_solutions[0] if positive_solutions else solutions[0]
        chosen_method = "bisection"
        iterations_used = best_bisect_iterations
        converged = bisect_converged

    message_parts = []
    if sign_change_count > 1 and len(solutions) > 1:
        message_parts.append("Multiple IRR solutions detected due to multiple cash flow sign changes.")
    if not converged:
        message_parts.append("IRR solver did not converge within the specified tolerance.")

    return IRRResult(
        value=chosen_value,
        iterations=iterations_used,
        method=chosen_method,
        converged=converged,
        message=" ".join(message_parts),
        solutions=solutions,
        tolerance=tolerance,
    )


__all__ = [
    "FinancialModel",
    "FinancialOutputs",
    "ScenarioToolResult",
    "AIInsights",
    "npf_irr",
    "IRRResult",
]
