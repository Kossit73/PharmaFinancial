"""Input loader and parser for the goat farming model."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import pandas as pd

from .goat_model import GoatModel, InputSchedule

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "default_schedule.json"


@dataclass
class GoatModelParameters:
    """Container for parsed goat model inputs."""

    schedule: InputSchedule
    scenario: MutableMapping[str, float]


def _build_default_schedule() -> pd.DataFrame:
    """Create a representative 12-month schedule."""

    periods = pd.date_range("2024-01-31", periods=12, freq="ME")
    revenue = pd.Series(100_000 + (periods.month - 1) * 5_000, index=periods)
    cogs = revenue * 0.45
    gross_profit = revenue - cogs
    variable_expenses = revenue * 0.12
    direct_wages = revenue * 0.08
    fixed_expenses = pd.Series(10_000.0, index=periods)
    admin_wages = pd.Series(3_000.0, index=periods)

    ebitda = gross_profit - variable_expenses - direct_wages - fixed_expenses - admin_wages
    depreciation = pd.Series(2_000.0, index=periods)
    ebit = ebitda - depreciation
    interest = pd.Series(500.0, index=periods)
    npbt = ebit - interest
    tax = npbt * 0.25
    npat = npbt - tax

    cfo = ebitda - 1_000
    capex = pd.Series(5_000.0, index=periods)
    cfi = -capex
    cff = pd.Series(2_000.0, index=periods)
    net_cash = cfo + cfi + cff

    opening_cash = pd.Series(50_000.0, index=periods)
    opening_cash = opening_cash.cumsum().shift(1).fillna(50_000.0)
    closing_cash = opening_cash + net_cash

    current_assets = closing_cash + 20_000.0
    non_current_assets = pd.Series(100_000.0, index=periods)
    current_liabilities = pd.Series(15_000.0, index=periods)
    non_current_liabilities = pd.Series(50_000.0, index=periods)
    equity = current_assets + non_current_assets - current_liabilities - non_current_liabilities

    return pd.DataFrame(
        {
            "Period": periods,
            "Revenue": revenue,
            "COGS": cogs,
            "Gross Margin": gross_profit,
            "Variable Expenses": variable_expenses,
            "Direct Wages": direct_wages,
            "Fixed Expenses": fixed_expenses,
            "Admin Wages": admin_wages,
            "EBITDA": ebitda,
            "Depreciation & Amortization": depreciation,
            "EBIT": ebit,
            "Interest Expense": interest,
            "NPBT": npbt,
            "Tax Expense": tax,
            "NPAT": npat,
            "CFO": cfo,
            "CFI": cfi,
            "CFF": cff,
            "Net Cash Flow": net_cash,
            "Capex": capex,
            "Opening Cash Balance": opening_cash,
            "Closing Cash Balance": closing_cash,
            "Cash and Cash Equivalents": closing_cash,
            "Current Assets": current_assets,
            "Non-current Assets": non_current_assets,
            "Current Liabilities": current_liabilities,
            "Non-current Liabilities": non_current_liabilities,
            "Equity": equity,
        }
    )


def _default_payload() -> dict[str, Any]:
    """Return a JSON-serialisable default payload."""

    schedule = _build_default_schedule()
    return {
        "period_column": "Period",
        "valuation_inputs": {"WACC": 0.12, "NPV": 750_000.0, "Terminal Value": 1_000_000.0},
        "scenario": {"milk_price_pct": 0.0, "feed_cost_pct": 0.0},
        "schedule": schedule.to_dict(orient="records"),
    }


def _to_frame(records: Any, period_column: str) -> pd.DataFrame:
    if not isinstance(records, list):
        raise ValueError("schedule must be a list of record objects.")
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("schedule cannot be empty.")
    if period_column not in df.columns:
        raise ValueError(f"Missing required period column '{period_column}'.")
    return df


def parse_inputs(payload: Mapping[str, Any]) -> GoatModelParameters:
    """Parse API/CLI payload into the goat model structures."""

    if not isinstance(payload, Mapping):
        raise ValueError("Payload must be a mapping.")

    period_column = str(payload.get("period_column") or "Period")
    valuation_inputs = dict(payload.get("valuation_inputs") or {})
    scenario = dict(payload.get("scenario") or {})

    schedule_records = payload.get("schedule")
    if schedule_records is None:
        raise ValueError("schedule is required.")
    schedule_df = _to_frame(schedule_records, period_column=period_column)

    supplementary_payload = payload.get("supplementary_tables") or {}
    supplementary_tables = {}
    if isinstance(supplementary_payload, Mapping):
        for name, table_records in supplementary_payload.items():
            if table_records is None:
                continue
            if isinstance(table_records, list):
                supplementary_tables[name] = pd.DataFrame(table_records)
            elif isinstance(table_records, Mapping):
                supplementary_tables[name] = pd.DataFrame([table_records])

    input_schedule = InputSchedule.from_frame(
        schedule_df,
        period_col=period_column,
        valuation_inputs=valuation_inputs,
        supplementary_tables=supplementary_tables,
    )
    return GoatModelParameters(schedule=input_schedule, scenario=scenario)


def load_inputs(path: Path | None = None) -> GoatModelParameters:
    """Load inputs from JSON or return bundled defaults."""

    if path is None:
        return parse_inputs(_default_payload())

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_inputs(payload)


def write_default_inputs(path: Path = DEFAULT_DATA_PATH) -> None:
    payload = _default_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
