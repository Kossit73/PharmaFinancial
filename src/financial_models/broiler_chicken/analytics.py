"""Advanced analytics calculations for the broiler chicken model."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

from .assumptions import Assumptions, REVENUE_CATEGORIES
from .config import (
    load_custom_simulation_definitions,
    load_monte_carlo_distributions,
)
from .financing import (
    CashFlowRow,
    IncomeStatementRow,
    BalanceSheetRow,
    discounted_cash_flow,
    npv,
    irr,
)
from .production import AnnualSummary, compute_cycles, annual_summary, _to_float


@dataclass(frozen=True)
class AnalyticsPlan:
    """Control which advanced analytics blocks are computed."""

    include_what_if: bool = True
    include_monte_carlo: bool = True
    include_break_even: bool = True
    include_goal_seek: bool = True
    include_predictive: bool = True
    include_scenario_planning: bool = True
    include_custom_simulations: bool = True

    @classmethod
    def full(cls) -> "AnalyticsPlan":
        """Return a plan that computes every analytics section."""

        return cls()

    @classmethod
    def summary(cls) -> "AnalyticsPlan":
        """Return a lightweight plan that skips expensive computations."""

        return cls(
            include_what_if=False,
            include_monte_carlo=False,
            include_goal_seek=False,
            include_predictive=False,
            include_scenario_planning=False,
            include_custom_simulations=False,
        )


def _calculate_payback_period(cashflows: Iterable[CashFlowRow]) -> float:
    cumulative = 0.0
    previous = 0.0
    payback = float("nan")
    for row in cashflows:
        cumulative += row.free_cash_flow
        if row.year == 0:
            previous = cumulative
            continue
        if cumulative >= 0 and payback != payback:
            delta = cumulative - previous
            if delta != 0:
                fraction = (0 - previous) / delta
                payback = (row.year - 1) + max(0.0, min(1.0, fraction))
            else:
                payback = float(row.year)
            break
        previous = cumulative
    return payback


def _compute_dscr_series(
    cashflows: Iterable[CashFlowRow],
) -> Tuple[List[Dict[str, float]], float, float]:
    rows: List[Dict[str, float]] = []
    values: List[float] = []
    for row in cashflows:
        if row.year == 0 or not row.debt_service:
            continue
        cash_available = row.operating_cash_flow + row.interest_expense
        try:
            dscr = cash_available / row.debt_service
        except ZeroDivisionError:
            dscr = float("nan")
        rows.append({"year": row.year, "dscr": dscr})
        if dscr == dscr:
            values.append(dscr)
    average = sum(values) / len(values) if values else float("nan")
    minimum = min(values) if values else float("nan")
    return rows, average, minimum


def _percentile(data: List[float], percentile: float) -> float:
    if not data:
        return float("nan")
    if percentile <= 0:
        return float(sorted(data)[0])
    if percentile >= 1:
        return float(sorted(data)[-1])
    ordered = sorted(data)
    k = (len(ordered) - 1) * percentile
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ordered[int(k)])
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return float(d0 + d1)


def _evaluate_case_metrics(
    assumptions: Assumptions,
    include_details: bool = False,
) -> Dict[str, Any]:
    cycles = compute_cycles(assumptions)
    annual = annual_summary(assumptions, cycles)
    cashflows, _ = discounted_cash_flow(assumptions, annual)
    valuation_cashflows = [row.free_cash_flow for row in cashflows]
    dscr_rows, avg_dscr, min_dscr = _compute_dscr_series(cashflows)
    summary = {
        "npv": npv(assumptions.discount_rate, valuation_cashflows),
        "irr": irr(valuation_cashflows),
        "payback": _calculate_payback_period(cashflows),
        "avg_dscr": avg_dscr,
        "min_dscr": min_dscr,
        "terminal_cash": cashflows[-1].cumulative_cash if cashflows else float("nan"),
    }
    result: Dict[str, Any] = {"metrics": summary, "dscr_rows": dscr_rows}
    if include_details:
        result["cashflows"] = cashflows
        result["annual"] = annual
    return result


def perform_what_if_analysis(
    assumptions: Assumptions, base_metrics: Dict[str, float]
) -> List[Dict[str, Any]]:
    scenarios = [
        {
            "name": "Baseline",
            "description": "Current assumptions",
            "changes": {},
        },
        {
            "name": "Live price +10%",
            "description": "Increase live bird price per kg by 10%",
            "changes": {
                "live_price_per_kg": assumptions.live_price_per_kg * 1.10
            },
        },
        {
            "name": "Live price -10%",
            "description": "Reduce live bird price per kg by 10%",
            "changes": {
                "live_price_per_kg": assumptions.live_price_per_kg * 0.90
            },
        },
        {
            "name": "Feed cost +10%",
            "description": "Increase feed cost per kg by 10%",
            "changes": {
                "feed_cost_per_kg": assumptions.feed_cost_per_kg * 1.10
            },
        },
        {
            "name": "Feed cost -10%",
            "description": "Reduce feed cost per kg by 10%",
            "changes": {
                "feed_cost_per_kg": assumptions.feed_cost_per_kg * 0.90
            },
        },
        {
            "name": "Mortality +2pp",
            "description": "Increase mortality rate by 2 percentage points",
            "changes": {
                "mortality_rate": max(0.0, assumptions.mortality_rate + 0.02)
            },
        },
        {
            "name": "Mortality -2pp",
            "description": "Decrease mortality rate by 2 percentage points",
            "changes": {
                "mortality_rate": max(0.0, assumptions.mortality_rate - 0.02)
            },
        },
    ]

    results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        if scenario["name"] == "Baseline":
            metrics = base_metrics
        else:
            mutated = replace(assumptions, **scenario["changes"])
            metrics = _evaluate_case_metrics(mutated)["metrics"]
        entry = {
            "Scenario": scenario["name"],
            "Description": scenario["description"],
            "NPV": metrics["npv"],
            "NPV Δ": metrics["npv"] - base_metrics["npv"],
            "IRR": metrics["irr"],
            "Avg DSCR": metrics["avg_dscr"],
            "Min DSCR": metrics["min_dscr"],
            "Payback": metrics["payback"],
        }
        results.append(entry)
    return results


def run_custom_simulations(
    assumptions: Assumptions,
    base_metrics: Dict[str, float],
    definitions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    processed: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    base_entry = {
        "Scenario": "Baseline",
        "Description": "Unmodified assumptions",
        "Parameter": "-",
        "Change type": "-",
        "Change value": 0.0,
        "NPV": base_metrics.get("npv", float("nan")),
        "NPV Δ": 0.0,
        "IRR": base_metrics.get("irr", float("nan")),
        "Avg DSCR": base_metrics.get("avg_dscr", float("nan")),
        "Min DSCR": base_metrics.get("min_dscr", float("nan")),
        "Payback": base_metrics.get("payback", float("nan")),
    }
    results.append(base_entry)

    for definition in definitions:
        if not isinstance(definition, dict):
            continue

        original = dict(definition)
        name = str(definition.get("Scenario") or definition.get("name") or "").strip()
        parameter = str(
            definition.get("Parameter") or definition.get("parameter") or ""
        ).strip()
        change_type_raw = str(
            definition.get("Change type")
            or definition.get("change_type")
            or "percent"
        ).strip().lower()
        change_value_raw = definition.get("Change value") or definition.get("change_value")
        description = definition.get("Description") or definition.get("description") or ""

        if not name:
            original["error"] = "Scenario name is required"
            invalid.append(original)
            continue
        if not parameter or not hasattr(assumptions, parameter):
            original["error"] = "Unknown or missing parameter"
            invalid.append(original)
            continue

        try:
            change_value = float(change_value_raw)
        except (TypeError, ValueError):
            original["error"] = "Change value must be numeric"
            invalid.append(original)
            continue

        current_value = getattr(assumptions, parameter)
        if current_value is None:
            original["error"] = "Parameter has no base value"
            invalid.append(original)
            continue

        if change_type_raw in {"percent", "percentage", "%"}:
            mutated_value = current_value * (1.0 + change_value / 100.0)
            applied_type = "percent"
        elif change_type_raw in {"absolute", "delta"}:
            mutated_value = current_value + change_value
            applied_type = "absolute"
        elif change_type_raw in {"target", "value"}:
            mutated_value = change_value
            applied_type = "target"
        else:
            original["error"] = "Unsupported change type"
            invalid.append(original)
            continue

        mutated = replace(assumptions, **{parameter: mutated_value})
        metrics = _evaluate_case_metrics(mutated)["metrics"]
        entry = {
            "Scenario": name,
            "Description": description,
            "Parameter": parameter,
            "Change type": applied_type,
            "Change value": change_value,
            "NPV": metrics["npv"],
            "NPV Δ": metrics["npv"] - base_metrics.get("npv", float("nan")),
            "IRR": metrics["irr"],
            "Avg DSCR": metrics["avg_dscr"],
            "Min DSCR": metrics["min_dscr"],
            "Payback": metrics["payback"],
        }
        results.append(entry)
        processed.append(
            {
                "Scenario": name,
                "Description": description,
                "Parameter": parameter,
                "Change type": applied_type,
                "Change value": change_value,
            }
        )

    delta_summary = [
        {
            "Scenario": row["Scenario"],
            "NPV": row["NPV"],
            "NPV Δ": row["NPV Δ"],
            "IRR": row["IRR"],
        }
        for row in results
        if row.get("Scenario") and row["Scenario"] != "Baseline"
    ]

    return {
        "definitions": processed,
        "results": results,
        "invalid": invalid,
        "delta_summary": delta_summary,
    }


def _draw_samples(
    rng_np: Optional["np.random.Generator"],
    rng_py: random.Random,
    parameter: str,
    base_value: float,
    spec: Dict[str, Any],
    iterations: int,
) -> Optional[Sequence[float]]:
    distribution = str(spec.get("distribution", "normal")).lower()
    mode = str(spec.get("mode", "multiplicative")).lower()

    use_numpy = rng_np is not None and np is not None

    if distribution == "normal":
        mean = float(
            spec.get("mean", 1.0 if mode in {"multiplicative", "relative"} else 0.0)
        )
        std = float(spec.get("std", 0.05))
        if use_numpy:
            draws = rng_np.normal(mean, std, size=iterations)  # type: ignore[attr-defined]
        else:
            draws = [rng_py.gauss(mean, std) for _ in range(iterations)]
    elif distribution == "lognormal":
        mean = float(spec.get("mean", 0.0))
        sigma = float(spec.get("sigma", 0.1))
        if use_numpy:
            draws = rng_np.lognormal(mean, sigma, size=iterations)  # type: ignore[attr-defined]
        else:
            draws = [rng_py.lognormvariate(mean, sigma) for _ in range(iterations)]
    elif distribution == "triangular":
        low = spec.get("low")
        high = spec.get("high")
        mode_value = spec.get("mode_value", spec.get("mode_point", spec.get("mode_param")))
        if mode_value is None and isinstance(spec.get("mode"), (int, float)):
            mode_value = spec.get("mode")
        if low is None or high is None:
            return None
        if mode_value is None:
            mode_value = (float(low) + float(high)) / 2.0
        if use_numpy:
            draws = rng_np.triangular(  # type: ignore[attr-defined]
                float(low), float(mode_value), float(high), size=iterations
            )
        else:
            draws = [
                rng_py.triangular(float(low), float(high), float(mode_value))
                for _ in range(iterations)
            ]
    elif distribution == "uniform":
        low = spec.get("low", spec.get("min"))
        high = spec.get("high", spec.get("max"))
        if low is None or high is None:
            return None
        if use_numpy:
            draws = rng_np.uniform(float(low), float(high), size=iterations)  # type: ignore[attr-defined]
        else:
            draws = [rng_py.uniform(float(low), float(high)) for _ in range(iterations)]
    else:
        return None

    if mode in {"multiplicative", "relative"}:
        if use_numpy:
            values = base_value * draws  # type: ignore[operator]
        else:
            values = [base_value * float(val) for val in draws]  # type: ignore[call-overload]
    elif mode in {"additive", "delta"}:
        if use_numpy:
            values = base_value + draws  # type: ignore[operator]
        else:
            values = [base_value + float(val) for val in draws]  # type: ignore[call-overload]
    elif mode in {"absolute", "target"}:
        values = draws
    else:
        return None

    bounds = spec.get("bounds")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        lower, upper = bounds
        if use_numpy:
            if lower is not None:
                values = np.maximum(values, float(lower))  # type: ignore[arg-type]
            if upper is not None:
                values = np.minimum(values, float(upper))  # type: ignore[arg-type]
        else:
            clipped: List[float] = []
            for val in values:  # type: ignore[assignment]
                v = float(val)
                if lower is not None:
                    v = max(float(lower), v)
                if upper is not None:
                    v = min(float(upper), v)
                clipped.append(v)
            values = clipped

    if use_numpy:
        return np.asarray(values, dtype=float)  # type: ignore[arg-type]
    return [float(val) for val in values]  # type: ignore[return-value]


def run_monte_carlo_analysis(
    assumptions: Assumptions,
    iterations: int = 200,
    distributions: Optional[List[Dict[str, Any]]] = None,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    if iterations <= 0:
        return {"summary": {"iterations": 0}, "samples": [], "settings": {"iterations": 0}}

    if distributions is None:
        distributions = load_monte_carlo_distributions()

    rng_np = np.random.default_rng(seed) if np is not None else None
    rng_py = random.Random(seed)
    param_samples: Dict[str, Sequence[float]] = {}

    for spec in distributions:
        if not isinstance(spec, dict):
            continue
        parameter = spec.get("parameter")
        if not parameter or not hasattr(assumptions, parameter):
            continue
        base_value = getattr(assumptions, parameter)
        if not isinstance(base_value, (int, float)):
            continue
        values = _draw_samples(rng_np, rng_py, parameter, float(base_value), spec, iterations)
        if values is None:
            continue
        param_samples[parameter] = values

    npv_results: List[float] = []
    irr_results: List[float] = []
    min_dscr_results: List[float] = []
    samples: List[Dict[str, float]] = []

    for idx in range(iterations):
        overrides = {param: values[idx] for param, values in param_samples.items()}
        varied = replace(assumptions, **overrides)
        evaluation = _evaluate_case_metrics(varied)["metrics"]
        npv_results.append(evaluation["npv"])
        irr_results.append(evaluation["irr"])
        min_dscr_results.append(evaluation["min_dscr"])
        sample_entry = {"iteration": idx + 1, **overrides, "npv": evaluation["npv"], "irr": evaluation["irr"], "min_dscr": evaluation["min_dscr"]}
        samples.append(sample_entry)

    summary = {
        "iterations": iterations,
        "mean_npv": float(sum(npv_results) / iterations) if iterations else float("nan"),
        "p5_npv": _percentile(npv_results, 0.05),
        "p95_npv": _percentile(npv_results, 0.95),
        "probability_negative_npv": float(
            sum(1 for value in npv_results if value < 0) / iterations
        )
        if iterations
        else float("nan"),
        "mean_irr": float(sum(irr_results) / iterations) if iterations else float("nan"),
        "mean_min_dscr": float(sum(min_dscr_results) / iterations) if iterations else float("nan"),
    }

    return {
        "summary": summary,
        "samples": samples,
        "settings": {"iterations": iterations, "seed": seed, "distributions": distributions},
    }


def break_even_analysis(
    annual: AnnualSummary,
    revenue_summary: Dict[str, List[Dict[str, Any]]],
    revenue_schedules: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    totals: Dict[str, float] = {}
    for row in revenue_summary.get("by_category", []):
        category = row.get("Category")
        revenue = row.get("Revenue")
        if category is None or revenue is None:
            continue
        totals[category] = totals.get(category, 0.0) + float(revenue)

    total_revenue = sum(totals.values())
    direct_costs: Dict[str, float] = {
        "Broiler Revenue": float(
            annual.feed_cost
            + annual.chick_cost
            + annual.processing_cost
            + annual.health_cost
        )
    }
    direct_pool = sum(direct_costs.values())
    shared_cost_pool = max(0.0, float(annual.total_cost) - direct_pool)

    results: List[Dict[str, Any]] = []
    for category in REVENUE_CATEGORIES:
        schedule = revenue_schedules.get(category, [])
        revenue = totals.get(category, 0.0)
        units = 0.0
        price_values: List[float] = []
        for row in schedule:
            unit_value = row.get("Units")
            unit_price = row.get("Unit price")
            unit_float = _to_float(unit_value)
            price_float = _to_float(unit_price)
            if unit_float is not None:
                units += unit_float
            if price_float is not None:
                price_values.append(price_float)

        if units > 0 and revenue:
            avg_price = revenue / units
        elif price_values:
            avg_price = sum(price_values) / len(price_values)
        else:
            avg_price = float("nan")

        direct_cost = direct_costs.get(category, 0.0)
        shared_cost = (
            shared_cost_pool * (revenue / total_revenue)
            if total_revenue > 0 and revenue > 0
            else 0.0
        )
        variable_cost_per_unit = (
            direct_cost / units if units > 0 and direct_cost > 0 else 0.0
        )
        contribution_margin = (
            avg_price - variable_cost_per_unit
            if avg_price == avg_price
            else float("nan")
        )
        break_even_units = (
            shared_cost / contribution_margin
            if contribution_margin and contribution_margin > 0
            else float("nan")
        )
        break_even_price = (
            (direct_cost + shared_cost) / units if units > 0 else float("nan")
        )

        results.append(
            {
                "Category": category,
                "Annual revenue": revenue,
                "Direct cost": direct_cost,
                "Shared cost": shared_cost,
                "Total cost": direct_cost + shared_cost,
                "Variable cost per unit": variable_cost_per_unit if units > 0 else float("nan"),
                "Contribution margin": contribution_margin,
                "Break-even units": break_even_units,
                "Break-even price": break_even_price,
            }
        )

    return results


def goal_seek_live_price(assumptions: Assumptions) -> Dict[str, Any]:
    target_npv = 0.0
    low, high = 0.5, 5.0
    tolerance = 1e-4
    best_price = assumptions.live_price_per_kg
    best_npv = float("nan")

    for _ in range(40):
        mid = (low + high) / 2
        mutated = replace(assumptions, live_price_per_kg=mid)
        metrics = _evaluate_case_metrics(mutated)["metrics"]
        best_price = mid
        best_npv = metrics["npv"]
        if abs(best_npv - target_npv) < tolerance:
            break
        if best_npv > target_npv:
            high = mid
        else:
            low = mid

    return {
        "Target NPV": target_npv,
        "Implied live price per kg": best_price,
        "Resulting NPV": best_npv,
    }


def build_predictive_analytics(
    cashflows: List[CashFlowRow], income_statement: List[IncomeStatementRow]
) -> Dict[str, Any]:
    historical = [row for row in cashflows if row.year > 0]
    if len(historical) < 2:
        return {
            "automated_forecast": [],
            "time_series": {"method": "AR(1)", "forecast": []},
            "risk_anomalies": {"mean_growth": float("nan"), "std_growth": float("nan"), "observations": []},
            "ml_methods": [],
        }

    revenue_growth: List[float] = []
    for prev, curr in zip(historical, historical[1:]):
        if prev.revenue:
            revenue_growth.append((curr.revenue - prev.revenue) / prev.revenue)
    mean_growth = sum(revenue_growth) / len(revenue_growth) if revenue_growth else 0.0
    std_dev = (
        math.sqrt(
            sum((value - mean_growth) ** 2 for value in revenue_growth) / len(revenue_growth)
        )
        if revenue_growth
        else 0.0
    )

    forecasts = []
    last_revenue = historical[-1].revenue
    for idx in range(1, 4):
        last_revenue *= 1 + mean_growth
        forecasts.append({"Year": historical[-1].year + idx, "Revenue forecast": last_revenue})

    ar1_coeff = revenue_growth[-1] if revenue_growth else 0.0
    arima_forecast = []
    last_growth = revenue_growth[-1] if revenue_growth else 0.0
    arima_revenue = historical[-1].revenue
    for idx in range(1, 4):
        last_growth = mean_growth + ar1_coeff * (last_growth - mean_growth)
        arima_revenue *= 1 + last_growth
        arima_forecast.append(arima_revenue)

    anomalies: List[Dict[str, Any]] = []
    for growth, row in zip(revenue_growth, historical[1:]):
        if std_dev and abs(growth - mean_growth) > 2 * std_dev:
            anomalies.append({
                "Year": row.year,
                "Observed growth": growth,
                "Flag": "Potential anomaly",
            })

    ml_methods = [
        {
            "Method": "Linear regression",
            "Description": "Trend line based on revenue history",
        },
        {
            "Method": "AR(1)",
            "Description": "Autoregressive model using last-period growth",
        },
    ]

    historical_years = [row.year for row in historical]
    time_series_analysis = {
        "method": "AR(1)",
        "forecast": [
            {
                "Year": (historical_years[-1] if historical_years else 0) + idx + 1,
                "Revenue forecast": value,
            }
            for idx, value in enumerate(arima_forecast)
        ],
    }

    risk_detection = {
        "mean_growth": mean_growth,
        "std_growth": std_dev,
        "observations": anomalies,
    }

    return {
        "automated_forecast": forecasts,
        "time_series": time_series_analysis,
        "risk_anomalies": risk_detection,
        "ml_methods": ml_methods,
    }


def scenario_planning(
    assumptions: Assumptions,
    base_summary: Dict[str, float],
    base_revenue: float,
) -> List[Dict[str, Any]]:
    scenarios = [
        {
            "name": "Baseline",
            "changes": {},
            "description": "Current assumption set",
        },
        {
            "name": "Downside",
            "changes": {
                "live_price_per_kg": assumptions.live_price_per_kg * 0.92,
                "feed_cost_per_kg": assumptions.feed_cost_per_kg * 1.08,
                "mortality_rate": min(0.4, assumptions.mortality_rate + 0.03),
            },
            "description": "Price compression with higher feed and mortality",
        },
        {
            "name": "Upside",
            "changes": {
                "live_price_per_kg": assumptions.live_price_per_kg * 1.08,
                "feed_cost_per_kg": assumptions.feed_cost_per_kg * 0.95,
                "mortality_rate": max(0.0, assumptions.mortality_rate - 0.02),
            },
            "description": "Pricing tailwinds and efficiency gains",
        },
        {
            "name": "Expansion",
            "changes": {
                "cycles_per_year": assumptions.cycles_per_year + 1,
                "birds_per_cycle": int(assumptions.birds_per_cycle * 1.05),
            },
            "description": "Add capacity through an extra cycle and larger placements",
        },
    ]

    results: List[Dict[str, Any]] = [
        {
            "Scenario": "Baseline",
            "Description": "Current assumption set",
            "Revenue": base_revenue,
            "NPV": base_summary["npv"],
            "IRR": base_summary["irr"],
            "Avg DSCR": base_summary["avg_dscr"],
            "Payback": base_summary["payback"],
            "NPV Δ": 0.0,
        }
    ]

    for scenario in scenarios[1:]:
        mutated = replace(assumptions, **scenario["changes"])
        evaluation = _evaluate_case_metrics(mutated, include_details=True)
        metrics = evaluation["metrics"]
        annual = evaluation.get("annual")
        revenue = annual.revenue if annual else float("nan")
        results.append(
            {
                "Scenario": scenario["name"],
                "Description": scenario["description"],
                "Revenue": revenue,
                "NPV": metrics["npv"],
                "IRR": metrics["irr"],
                "Avg DSCR": metrics["avg_dscr"],
                "Payback": metrics["payback"],
                "NPV Δ": metrics["npv"] - base_summary["npv"],
            }
        )
    return results


def _safe_div(numerator: float, denominator: float) -> float:
    try:
        if denominator is None:
            return float("nan")
        denom = float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if denom == 0 or math.isnan(denom):
        return float("nan")
    try:
        return float(numerator) / denom
    except (TypeError, ValueError):
        return float("nan")


def _nanmean(values: Iterable[float]) -> float:
    data = [float(v) for v in values if isinstance(v, (int, float)) and v == v]
    if not data:
        return float("nan")
    return sum(data) / len(data)


def compute_advanced_analytics(
    assumptions: Assumptions,
    cashflows: List[CashFlowRow],
    income_statement: List[IncomeStatementRow],
    balance_sheet: List[BalanceSheetRow],
    revenue_summary: Dict[str, List[Dict[str, Any]]],
    revenue_schedules: Dict[str, List[Dict[str, Any]]],
    annual: AnnualSummary,
    custom_simulation_definitions: Optional[List[Dict[str, Any]]] = None,
    monte_carlo_distributions: Optional[List[Dict[str, Any]]] = None,
    plan: Optional[AnalyticsPlan] = None,
) -> Dict[str, Any]:
    plan = plan or AnalyticsPlan.full()
    if custom_simulation_definitions is None:
        custom_simulation_definitions = (
            load_custom_simulation_definitions()
            if plan.include_custom_simulations
            else []
        )
    if monte_carlo_distributions is None:
        monte_carlo_distributions = (
            load_monte_carlo_distributions()
            if plan.include_monte_carlo
            else []
        )

    metrics: List[Dict[str, Any]] = []

    if income_statement:
        avg_ebitda_margin = sum(row.ebitda_margin for row in income_statement) / len(
            income_statement
        )
        avg_net_margin = sum(row.net_margin for row in income_statement) / len(
            income_statement
        )
    else:
        avg_ebitda_margin = 0.0
        avg_net_margin = 0.0

    valuation_cashflows = [row.free_cash_flow for row in cashflows]
    dscr_rows, avg_dscr, min_dscr = _compute_dscr_series(cashflows)
    payback = _calculate_payback_period(cashflows)
    base_metrics = {
        "npv": npv(assumptions.discount_rate, valuation_cashflows),
        "irr": irr(valuation_cashflows),
        "avg_dscr": avg_dscr,
        "min_dscr": min_dscr,
        "payback": payback,
    }

    income_by_year = {row.year: row for row in income_statement}
    balance_by_year = {row.year: row for row in balance_sheet}

    returns_rows: List[Dict[str, float]] = []
    coverage_rows: List[Dict[str, float]] = []
    leverage_rows: List[Dict[str, float]] = []

    for row in cashflows:
        if row.year == 0:
            continue

        income_row = income_by_year.get(row.year)
        balance_row = balance_by_year.get(row.year)

        if income_row and balance_row:
            roa = _safe_div(income_row.net_income, balance_row.total_assets)
            roe = _safe_div(income_row.net_income, balance_row.equity)
            invested_capital = (
                (balance_row.debt or 0.0)
                + (balance_row.equity or 0.0)
                - (balance_row.cash or 0.0)
            )
            nopat = (income_row.net_income or 0.0) + (income_row.interest or 0.0)
            roic = _safe_div(nopat, invested_capital)
            returns_rows.append(
                {
                    "year": row.year,
                    "return_on_assets": roa,
                    "return_on_equity": roe,
                    "return_on_invested_capital": roic,
                }
            )

            debt_to_equity = balance_row.debt_to_equity
            debt_ratio = _safe_div(balance_row.debt, balance_row.total_assets)
            leverage_rows.append(
                {
                    "year": row.year,
                    "debt_to_equity": debt_to_equity,
                    "debt_ratio": debt_ratio,
                    "ending_debt": row.ending_debt,
                }
            )

        interest_coverage = float("nan")
        if income_row:
            interest_coverage = _safe_div(income_row.ebit, income_row.interest)

        fcf_to_debt_service = _safe_div(row.free_cash_flow, row.debt_service)
        maintenance_cov = _safe_div(row.operating_cash_flow, row.maintenance_capex)
        opening_debt = (row.ending_debt or 0.0) + (row.principal_payment or 0.0)
        paydown_velocity = _safe_div(row.principal_payment, opening_debt)
        coverage_rows.append(
            {
                "year": row.year,
                "interest_coverage": interest_coverage,
                "fcf_to_debt_service": fcf_to_debt_service,
                "maintenance_capex_coverage": maintenance_cov,
                "debt_paydown_velocity": paydown_velocity,
            }
        )

    metrics.extend(
        [
            {"metric": "Average EBITDA margin", "value": avg_ebitda_margin},
            {"metric": "Average net margin", "value": avg_net_margin},
            {"metric": "Average DSCR", "value": avg_dscr},
            {"metric": "Minimum DSCR", "value": min_dscr},
            {"metric": "Payback period (years)", "value": payback},
            {
                "metric": "Average return on assets",
                "value": _nanmean(r["return_on_assets"] for r in returns_rows),
            },
            {
                "metric": "Average return on equity",
                "value": _nanmean(r["return_on_equity"] for r in returns_rows),
            },
            {
                "metric": "Average return on invested capital",
                "value": _nanmean(
                    r["return_on_invested_capital"] for r in returns_rows
                ),
            },
            {
                "metric": "Average interest coverage",
                "value": _nanmean(r["interest_coverage"] for r in coverage_rows),
            },
            {
                "metric": "Average free cash flow to debt service",
                "value": _nanmean(r["fcf_to_debt_service"] for r in coverage_rows),
            },
            {
                "metric": "Average maintenance capex coverage",
                "value": _nanmean(
                    r["maintenance_capex_coverage"] for r in coverage_rows
                ),
            },
            {
                "metric": "Average debt paydown velocity",
                "value": _nanmean(r["debt_paydown_velocity"] for r in coverage_rows),
            },
            {"metric": "Base case NPV", "value": base_metrics["npv"]},
            {"metric": "Base case IRR", "value": base_metrics["irr"]},
        ]
    )

    trend_rows = [
        {
            "year": row.year,
            "revenue": row.revenue,
            "ebitda": row.ebitda,
            "net_income": row.net_income,
            "free_cash_flow": row.free_cash_flow,
            "cumulative_cash": row.cumulative_cash,
        }
        for row in cashflows
        if row.year > 0
    ]

    what_if_rows = (
        perform_what_if_analysis(assumptions, base_metrics)
        if plan.include_what_if
        else []
    )
    monte_carlo_result = (
        run_monte_carlo_analysis(
            assumptions,
            distributions=monte_carlo_distributions,
        )
        if plan.include_monte_carlo
        else {
            "summary": {"iterations": 0},
            "samples": [],
            "settings": {
                "iterations": 0,
                "seed": None,
                "distributions": monte_carlo_distributions or [],
            },
        }
    )
    break_even_rows = (
        break_even_analysis(annual, revenue_summary, revenue_schedules)
        if plan.include_break_even
        else []
    )
    goal_seek_result = (
        goal_seek_live_price(assumptions) if plan.include_goal_seek else {}
    )
    predictive_payload = (
        build_predictive_analytics(cashflows, income_statement)
        if plan.include_predictive
        else {}
    )
    scenario_rows = (
        scenario_planning(
            assumptions,
            base_metrics,
            annual.revenue,
        )
        if plan.include_scenario_planning
        else []
    )
    custom_payload = (
        run_custom_simulations(
            assumptions, base_metrics, custom_simulation_definitions
        )
        if plan.include_custom_simulations
        else {
            "definitions": [],
            "results": [],
            "invalid": [],
            "delta_summary": [],
        }
    )
    if not plan.include_custom_simulations:
        custom_payload["definitions"] = custom_simulation_definitions or []

    return {
        "metrics": metrics,
        "dscr": dscr_rows,
        "trend": trend_rows,
        "returns": returns_rows,
        "coverage": coverage_rows,
        "leverage": leverage_rows,
        "what_if": what_if_rows,
        "monte_carlo": monte_carlo_result,
        "break_even": break_even_rows,
        "goal_seek": goal_seek_result,
        "predictive": predictive_payload,
        "scenario_planning": scenario_rows,
        "custom_simulations": custom_payload,
        "custom_simulation_definitions": custom_simulation_definitions,
        "base_metrics": base_metrics,
        "plan": {
            "include_what_if": plan.include_what_if,
            "include_monte_carlo": plan.include_monte_carlo,
            "include_break_even": plan.include_break_even,
            "include_goal_seek": plan.include_goal_seek,
            "include_predictive": plan.include_predictive,
            "include_scenario_planning": plan.include_scenario_planning,
            "include_custom_simulations": plan.include_custom_simulations,
        },
    }
