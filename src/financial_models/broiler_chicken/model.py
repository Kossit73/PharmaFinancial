"""High-level orchestration helpers for the broiler model."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .assumptions import Assumptions, build_assumptions_schedule
from .production import (
    AnnualSummary,
    CycleResults,
    build_revenue_schedules,
    compute_cycles,
    annual_summary,
    summarise_revenue_totals,
)
from .financing import (
    CashFlowRow,
    IncomeStatementRow,
    BalanceSheetRow,
    CashFlowStatementRow,
    build_financial_statements,
    discounted_cash_flow,
    npv,
    irr,
)
from .analytics import AnalyticsPlan, compute_advanced_analytics
from .config import (
    load_custom_simulation_definitions,
    load_monte_carlo_distributions,
)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]):
    rows = list(rows)
    if not rows:
        return
    fieldnames: List[str] = list({}.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="") as fh:
        import csv

        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any):
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)


def generate_model_outputs(
    assumptions: Assumptions,
    *,
    custom_simulation_path: Optional[Path] = None,
    monte_carlo_config_path: Optional[Path] = None,
    analytics_plan: Optional[AnalyticsPlan] = None,
) -> Dict[str, Any]:
    plan = analytics_plan or AnalyticsPlan.full()
    custom_simulations = (
        load_custom_simulation_definitions(custom_simulation_path)
        if plan.include_custom_simulations or custom_simulation_path
        else []
    )
    monte_carlo_distributions = (
        load_monte_carlo_distributions(monte_carlo_config_path)
        if plan.include_monte_carlo or monte_carlo_config_path
        else []
    )
    assumption_schedule = build_assumptions_schedule(assumptions)
    cycles = compute_cycles(assumptions)
    annual = annual_summary(assumptions, cycles)
    cashflows, loan_schedule = discounted_cash_flow(assumptions, annual)
    revenue_schedules = build_revenue_schedules(assumptions, cycles)
    revenue_summary = summarise_revenue_totals(
        revenue_schedules,
        assumptions.cycles_per_year,
        assumptions.production_horizon_years,
        assumptions.production_start_year,
    )
    timeline = {
        "start_year": assumptions.production_start_year,
        "end_year": assumptions.production_start_year
        + max(assumptions.production_horizon_years - 1, 0),
        "projection_years": max(assumptions.production_horizon_years, 1),
    }
    financials = build_financial_statements(assumptions, cashflows, loan_schedule)
    advanced = compute_advanced_analytics(
        assumptions,
        cashflows,
        financials["income_statement"],
        financials["balance_sheet"],
        revenue_summary,
        revenue_schedules,
        annual,
        custom_simulation_definitions=custom_simulations,
        monte_carlo_distributions=monte_carlo_distributions,
        plan=plan,
    )

    valuation_cashflows = [row.free_cash_flow for row in cashflows]
    discount_rate = assumptions.discount_rate
    model_npv = npv(discount_rate, valuation_cashflows)
    model_irr = irr(valuation_cashflows)

    terminal_row = cashflows[-1] if cashflows else None
    valuation = {
        "discount_rate": discount_rate,
        "npv": model_npv,
        "irr": model_irr,
        "initial_investment": cashflows[0].free_cash_flow,
        "terminal_year": terminal_row.year if terminal_row else None,
        "terminal_calendar_year": terminal_row.calendar_year if terminal_row else None,
    }

    return {
        "assumptions": assumptions,
        "assumptions_schedule": assumption_schedule,
        "cycles": cycles,
        "annual": annual,
        "cashflows": cashflows,
        "revenue_schedules": revenue_schedules,
        "revenue_summary": revenue_summary,
        "valuation": valuation,
        "timeline": timeline,
        "financial_statements": financials,
        "advanced_analytics": advanced,
    }


def load_assumptions_from_file(path: Path) -> Assumptions:
    if not path.exists():
        raise FileNotFoundError(f"Assumptions file not found: {path}")
    text = path.read_text()
    if not text.strip():
        return Assumptions()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YAML assumptions provided but PyYAML is not installed"
            ) from exc
    if not isinstance(data, dict):
        raise ValueError("Assumptions file must decode to an object")
    return Assumptions(**data)


def apply_overrides(assumptions: Assumptions, overrides: Dict[str, Any]) -> Assumptions:
    current = asdict(assumptions)
    updates = {}
    for key, value in overrides.items():
        if key not in current:
            raise KeyError(f"Unknown assumption field: {key}")
        updates[key] = _coerce_value(value, current[key])
    return replace(assumptions, **updates)


def _coerce_value(value: Any, reference: Any) -> Any:
    if isinstance(reference, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(float(value))
    if isinstance(reference, float):
        return float(value)
    return value


def parse_overrides(raw_pairs: Iterable[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for pair in raw_pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be in key=value format: {pair}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


__all__ = [
    "Assumptions",
    "AnnualSummary",
    "CycleResults",
    "CashFlowRow",
    "IncomeStatementRow",
    "BalanceSheetRow",
    "CashFlowStatementRow",
    "generate_model_outputs",
    "write_csv",
    "write_json",
    "load_assumptions_from_file",
    "apply_overrides",
    "parse_overrides",
]
