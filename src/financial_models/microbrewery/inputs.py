"""Input loaders and validators for the microbrewery model."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .model import (
    CapexItem,
    DebtFacility,
    DividendPolicy,
    MicrobreweryFinancialModel,
    ModelConfig,
    ModelInputs,
    phase_growth_series,
)

DEFAULT_INPUTS_PATH = Path(__file__).resolve().parent / "data" / "default_inputs.json"


@dataclass
class MicrobreweryModelParameters:
    """Container holding the full parameter set used by the engine."""

    config: ModelConfig
    dividend_policy: DividendPolicy
    inputs: ModelInputs


def _load_json(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _ensure_dataframe(records: Iterable[Mapping[str, Any]], required: Sequence[str], name: str) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")
    return df


def _timeline(config: ModelConfig) -> pd.DatetimeIndex:
    return pd.date_range(config.start_date, periods=config.months, freq="MS")


def _coerce_month_index(value: Any, idx: pd.DatetimeIndex) -> int:
    if value is None:
        return -1
    try:
        month_idx = int(value)
        if 0 <= month_idx < len(idx):
            return month_idx
    except Exception:
        pass
    try:
        ts = pd.to_datetime(value)
        pos = idx.get_loc(ts, method="nearest") if ts in idx else idx.get_indexer([ts], method="nearest")[0]
        if pos >= 0:
            return int(pos)
    except Exception:
        pass
    return -1


def _series_from_payload(value: Any, idx: pd.DatetimeIndex, name: str) -> float | pd.Series:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    series = pd.Series(0.0, index=idx, name=name)
    if isinstance(value, Mapping):
        for key, amount in value.items():
            pos = _coerce_month_index(key, idx)
            if pos >= 0:
                series.iloc[pos] = float(amount)
        return series
    if isinstance(value, Sequence):
        for item in value:
            if not isinstance(item, Mapping):
                continue
            pos = _coerce_month_index(item.get("month") or item.get("date"), idx)
            if pos >= 0:
                series.iloc[pos] = float(item.get("amount") if "amount" in item else item.get("value", 0.0))
        return series
    raise ValueError(f"Unsupported {name} payload; expected number, mapping, or list.")


def _parse_capex_items(items: Iterable[Mapping[str, Any]]) -> list[CapexItem]:
    parsed: list[CapexItem] = []
    for item in items:
        parsed.append(
            CapexItem(
                name=str(item.get("name")),
                amount=float(item.get("amount", 0.0)),
                capex_month=int(item.get("capex_month", 0)),
                depreciation_years=float(item.get("depreciation_years", 0.0)),
            )
        )
    return parsed


def _parse_debt_facilities(items: Iterable[Mapping[str, Any]]) -> list[DebtFacility]:
    facilities: list[DebtFacility] = []
    for item in items:
        specified = item.get("specified_principal_payments") or {}
        specified_parsed = {int(k): float(v) for k, v in specified.items()}
        facilities.append(
            DebtFacility(
                name=str(item.get("name")),
                principal=float(item.get("principal", 0.0)),
                annual_interest_rate=float(item.get("annual_interest_rate", 0.0)),
                draw_month=int(item.get("draw_month", 0)),
                grace_months=int(item.get("grace_months", 0)),
                term_months=int(item.get("term_months", 0)),
                repayment_type=str(item.get("repayment_type", "linear")),  # type: ignore[arg-type]
                specified_principal_payments=specified_parsed if specified_parsed else None,
            )
        )
    return facilities


def _parse_equity_injections(raw: Mapping[str, Any] | None) -> dict[int, float]:
    if raw is None:
        return {}
    return {int(k): float(v) for k, v in raw.items()}


def parse_inputs(payload: Mapping[str, Any]) -> MicrobreweryModelParameters:
    """Parse an API or CLI payload into model-ready structures."""

    if not isinstance(payload, Mapping):
        raise ValueError("Payload must be a mapping.")

    cfg_raw = payload.get("config", {}) or {}
    config = ModelConfig(
        start_date=str(cfg_raw.get("start_date", ModelConfig.start_date)),
        months=int(cfg_raw.get("months", ModelConfig.months)),
        pricing_cost_basis_month=int(cfg_raw.get("pricing_cost_basis_month", ModelConfig.pricing_cost_basis_month)),
        price_inflation_annual=float(cfg_raw.get("price_inflation_annual", ModelConfig.price_inflation_annual)),
        cost_inflation_annual=float(cfg_raw.get("cost_inflation_annual", ModelConfig.cost_inflation_annual)),
        tax_rate=float(cfg_raw.get("tax_rate", ModelConfig.tax_rate)),
        days_receivables=float(cfg_raw.get("days_receivables", ModelConfig.days_receivables)),
        days_inventory=float(cfg_raw.get("days_inventory", ModelConfig.days_inventory)),
        days_payables=float(cfg_raw.get("days_payables", ModelConfig.days_payables)),
        other_current_assets_pct_revenue=float(
            cfg_raw.get("other_current_assets_pct_revenue", ModelConfig.other_current_assets_pct_revenue)
        ),
        other_current_liabilities_pct_direct_costs=float(
            cfg_raw.get("other_current_liabilities_pct_direct_costs", ModelConfig.other_current_liabilities_pct_direct_costs)
        ),
        wacc_annual=float(cfg_raw.get("wacc_annual", ModelConfig.wacc_annual)),
        exit_month=cfg_raw.get("exit_month", ModelConfig.exit_month),
        exit_ev_ebitda_multiple=float(cfg_raw.get("exit_ev_ebitda_multiple", ModelConfig.exit_ev_ebitda_multiple)),
        initial_cash=float(cfg_raw.get("initial_cash", ModelConfig.initial_cash)),
    )

    div_raw = payload.get("dividend_policy", {}) or {}
    dividend_policy = DividendPolicy(
        enabled=bool(div_raw.get("enabled", DividendPolicy.enabled)),
        model=str(div_raw.get("model", DividendPolicy.model) or "cash_sweep"),  # type: ignore[arg-type]
        start_month=int(div_raw.get("start_month", DividendPolicy.start_month)),
        minimum_cash_position=float(div_raw.get("minimum_cash_position", DividendPolicy.minimum_cash_position)),
        payout_ratio=float(div_raw.get("payout_ratio", DividendPolicy.payout_ratio)),
    )

    idx = _timeline(config)

    skus = _ensure_dataframe(payload.get("skus", []), ["sku_id", "name", "direct_cost_per_unit", "markup_pct"], "skus")
    channels = _ensure_dataframe(payload.get("channels", []), ["channel", "price_factor"], "channels")
    sales_plan = _ensure_dataframe(payload.get("sales_plan", []), ["date", "sku_id", "channel", "units"], "sales_plan")
    sales_plan["date"] = pd.to_datetime(sales_plan["date"], errors="coerce")
    if sales_plan["date"].isna().any():
        raise ValueError("Sales plan contains invalid dates.")
    sales_plan["units"] = sales_plan["units"].astype(float)

    inputs = ModelInputs(
        skus=skus,
        channels=channels,
        sales_plan=sales_plan,
        opex_fixed_monthly=_series_from_payload(payload.get("opex_fixed_monthly"), idx, "opex_fixed_monthly"),
        other_income_monthly=_series_from_payload(payload.get("other_income_monthly"), idx, "other_income_monthly"),
        capex_items=_parse_capex_items(payload.get("capex_items", [])),
        debt_facilities=_parse_debt_facilities(payload.get("debt_facilities", [])),
        equity_injections=_parse_equity_injections(payload.get("equity_injections")),
    )

    # Validate early to surface human-friendly errors via /inputs/{model}/validate
    MicrobreweryFinancialModel(config, dividend_policy, inputs)
    return MicrobreweryModelParameters(config=config, dividend_policy=dividend_policy, inputs=inputs)


def load_inputs(path: Path | None = None) -> MicrobreweryModelParameters:
    """Load model inputs from JSON or fall back to bundled defaults."""

    resolved = path or DEFAULT_INPUTS_PATH
    payload = _load_json(resolved)
    return parse_inputs(payload)


def default_payload() -> dict[str, Any]:
    """Return a representative default payload for the microbrewery model."""

    config = {
        "start_date": "2025-01-01",
        "months": 72,
        "pricing_cost_basis_month": 18,
        "price_inflation_annual": 0.018,
        "cost_inflation_annual": 0.018,
        "tax_rate": 0.25,
        "days_receivables": 22,
        "days_inventory": 18,
        "days_payables": 32,
        "other_current_assets_pct_revenue": 0.05,
        "other_current_liabilities_pct_direct_costs": 0.05,
        "wacc_annual": 0.12,
        "exit_month": 60,
        "exit_ev_ebitda_multiple": 8.0,
        "initial_cash": 500_000.0,
    }
    dividend_policy = {
        "enabled": True,
        "model": "cash_sweep",
        "start_month": 48,
        "minimum_cash_position": 1_250_000.0,
        "payout_ratio": 0.25,
    }

    idx = pd.date_range(config["start_date"], periods=config["months"], freq="MS")
    u1 = phase_growth_series(idx, start_month=1, start_units=6_000, monthly_growth=0.035, cap_units=20_000)
    u2 = phase_growth_series(idx, start_month=2, start_units=4_500, monthly_growth=0.04, cap_units=18_000)
    channel_mix = {"Wholesale": 0.45, "Retail": 0.35, "E-Commerce": 0.12, "On-Premise": 0.08}

    sales_rows: list[dict[str, Any]] = []
    for date in idx:
        for sku_id, series in [(1, u1), (2, u2)]:
            total_units = float(series.loc[date])
            if total_units == 0.0:
                continue
            for channel, share in channel_mix.items():
                sales_rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "sku_id": sku_id,
                        "channel": channel,
                        "units": float(total_units * share),
                    }
                )

    return {
        "config": config,
        "dividend_policy": dividend_policy,
        "skus": [
            {
                "sku_id": 1,
                "name": "Pale Ale 330ml",
                "direct_cost_per_unit": 2.05,
                "markup_pct": 0.65,
                "relative_opex_weight": 1.0,
            },
            {
                "sku_id": 2,
                "name": "Pilsner 500ml",
                "direct_cost_per_unit": 2.55,
                "markup_pct": 0.62,
                "relative_opex_weight": 1.05,
            },
        ],
        "channels": [
            {"channel": "Wholesale", "price_factor": 1.4},
            {"channel": "Retail", "price_factor": 2.0},
            {"channel": "E-Commerce", "price_factor": 1.72},
            {"channel": "On-Premise", "price_factor": 1.0},
        ],
        "sales_plan": sales_rows,
        "opex_fixed_monthly": 95_000.0,
        "other_income_monthly": [{"month": 12, "amount": 12_500.0}],
        "capex_items": [
            {"name": "Land", "amount": 850_000.0, "capex_month": 0, "depreciation_years": 0},
            {"name": "Brewery equipment", "amount": 1_150_000.0, "capex_month": 1, "depreciation_years": 10},
            {"name": "Packaging line", "amount": 450_000.0, "capex_month": 6, "depreciation_years": 8},
        ],
        "debt_facilities": [
            {
                "name": "Mortgage",
                "principal": 700_000.0,
                "annual_interest_rate": 0.032,
                "draw_month": 0,
                "grace_months": 6,
                "term_months": 120,
                "repayment_type": "linear",
            },
            {
                "name": "Loan A",
                "principal": 400_000.0,
                "annual_interest_rate": 0.028,
                "draw_month": 5,
                "grace_months": 4,
                "term_months": 72,
                "repayment_type": "annuity",
            },
        ],
        "equity_injections": {
            "0": 4_200_000.0,
            "12": 650_000.0,
        },
    }


def write_default_inputs(path: Path = DEFAULT_INPUTS_PATH) -> None:
    """Materialise the generated defaults to disk (used during development)."""

    payload = default_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
