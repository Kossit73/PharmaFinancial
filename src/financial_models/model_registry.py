"""Registry of financial models exposed via API/CLI."""
from __future__ import annotations

import json
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Type

import pandas as pd

from .api.schemas import (
    BiotechModelRunRequest,
    BiotechModelRunResponse,
    BiotechValidationRequest,
    BroilerInputsPayload,
    BroilerModelRunRequest,
    BroilerModelRunResponse,
    BroilerValidationRequest,
    CassavaInputsPayload,
    CassavaModelRunRequest,
    CassavaModelRunResponse,
    CassavaValidationRequest,
    GoatModelRunRequest,
    GoatModelRunResponse,
    GoatValidationRequest,
    ModelRunResponse,
    PharmaModelRunRequest,
    PharmaValidationRequest,
    ScenarioToolResultPayload,
    TablePayload,
)
from .biotech import (
    BiotechInputs,
    ValuationEngine as BiotechValuationEngine,
    build_portfolio as build_biotech_portfolio,
    load_inputs as load_biotech_inputs,
    parse_inputs as parse_biotech_inputs,
)
from .broiler_chicken import BroilerModelParameters, generate_model_outputs as run_broiler_outputs, load_inputs as load_broiler_inputs, parse_inputs as parse_broiler_inputs
from .cassava_ethanol import CassavaModelParameters, load_inputs as load_cassava_inputs, parse_inputs as parse_cassava_inputs
from .microbrewery import (
    MicrobreweryFinancialModel,
    MicrobreweryModelParameters,
    ModelRunResult as MicrobreweryResult,
    load_inputs as load_microbrewery_inputs,
    parse_inputs as parse_microbrewery_inputs,
)
from .goat_farming import GoatModelParameters, load_inputs as load_goat_inputs, parse_inputs as parse_goat_inputs
from .pharma.inputs import ModelInputs, load_inputs as load_pharma_inputs, parse_inputs as parse_pharma_inputs
from .pharma.model import FinancialModel
from .core.table import Table
from .api.schemas.common import AIInsightsPayload
from .core.report import collect_biotech_report_sections, collect_report_sections
from .api.schemas.microbrewery import (
    MicrobreweryModelRunRequest,
    MicrobreweryModelRunResponse,
    MicrobreweryValidationRequest,
)


@dataclass
class ModelSpec:
    """Describes a registered model and how to execute it."""

    name: str
    load_inputs: Callable[[Path | None], Any]
    parse_inputs: Callable[[Mapping[str, Any]], Any]
    run_core: Callable[[Any], Any]
    build_response: Callable[[Any], Any]
    run_request_model: Type[Any]
    validate_request_model: Type[Any]
    response_model: Type[Any]
    cli_exporter: Callable[[Any, Path], None] | None = None
    build_report_sections: Callable[[Any, Any], list] | None = None
    report_title: str | None = None
    report_name: str | None = None


def _clean_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _table_payload(table: Table | None) -> TablePayload | None:
    if table is None:
        return None
    sanitized = {key: [_clean_value(v) for v in values] for key, values in table.as_dict().items()}
    return TablePayload(index_name=table.index_name, index=list(table.index), data=sanitized)


def _df_payload(df: pd.DataFrame | None, *, index_name: str = "Year") -> TablePayload | None:
    if df is None:
        return None
    def _normalise_index(values):
        normalised = []
        for value in values:
            if isinstance(value, (pd.Timestamp, datetime)):
                normalised.append(value.isoformat())
            else:
                normalised.append(_clean_value(value))
        return normalised

    data = {key: [_clean_value(v) for v in values] for key, values in df.to_dict(orient="list").items()}
    return TablePayload(index_name=index_name, index=_normalise_index(df.index), data=data)


def _ai_payload(insights) -> AIInsightsPayload | None:
    if insights is None:
        return None
    return AIInsightsPayload(
        enabled=bool(insights.enabled),
        generative_summary=insights.generative_summary,
        metadata=insights.metadata,
        ml_forecast=_table_payload(insights.ml_forecast),
    )


def _run_pharma_core(inputs: ModelInputs):
    model = FinancialModel(inputs)
    return model.run()


def _build_pharma_response(outputs) -> ModelRunResponse:
    return ModelRunResponse(
        summary_metrics=_table_payload(outputs.summary_metrics),
        income_statement=_table_payload(outputs.income_statement),
        balance_sheet=_table_payload(outputs.balance_sheet),
        cash_flow=_table_payload(outputs.cash_flow),
        goal_seek=_table_payload(outputs.goal_seek),
        break_even=_table_payload(outputs.break_even),
        payback=_table_payload(outputs.payback),
        discounted_payback=_table_payload(outputs.discounted_payback),
        monte_carlo=_table_payload(outputs.monte_carlo),
        scenario_results={name: _table_payload(table) for name, table in outputs.scenario_results.items()},
        sensitivity_results={name: _table_payload(table) for name, table in outputs.sensitivity_results.items()},
        scenario_tool_results={
            name: ScenarioToolResultPayload(rows=result.rows, interpretation=result.interpretation)
            for name, result in outputs.scenario_tool_results.items()
        },
        ai_insights=_ai_payload(outputs.ai_insights),
        risk_factor_diagnostics=_table_payload(outputs.risk_factor_diagnostics),
    )


def _export_pharma(outputs, output_dir: Path) -> None:
    """Write pharma outputs to CSV files."""

    def _write_table(path: Path, table) -> None:
        try:
            table.to_frame().to_csv(path)  # type: ignore[attr-defined]
        except Exception:
            table.to_csv(path)

    _write_table(
        output_dir / "income_statement.csv",
        outputs.income_statement.rounded(0, exclude_keywords=("Margin", "Return")),
    )
    _write_table(output_dir / "balance_sheet.csv", outputs.balance_sheet.rounded(0))
    _write_table(output_dir / "cash_flow.csv", outputs.cash_flow.rounded(0))
    _write_table(output_dir / "summary_metrics.csv", outputs.summary_metrics)
    _write_table(output_dir / "break_even.csv", outputs.break_even)
    _write_table(output_dir / "payback.csv", outputs.payback)
    _write_table(output_dir / "discounted_payback.csv", outputs.discounted_payback)

    for scenario, df in outputs.scenario_results.items():
        _write_table(output_dir / f"scenario_{scenario}.csv", df)

    for variable, df in outputs.sensitivity_results.items():
        _write_table(output_dir / f"sensitivity_{variable}.csv", df)

    _write_table(output_dir / "monte_carlo.csv", outputs.monte_carlo)


def _run_biotech_core(inputs: BiotechInputs):
    portfolio = build_biotech_portfolio(inputs)
    return BiotechValuationEngine(portfolio).run()


def _build_biotech_response(result) -> BiotechModelRunResponse:
    return BiotechModelRunResponse(
        rnpv=result.rnpv,
        consolidated=_df_payload(result.consolidated),
        dcf_table=_df_payload(result.dcf_table),
        per_product={name: _df_payload(df) for name, df in result.per_product.items()},
        per_product_prob={name: _df_payload(df) for name, df in result.per_product_prob.items()},
        ai_insights=_ai_payload(getattr(result, "ai_insights", None)),
    )


def _export_biotech(result, output_dir: Path) -> None:
    """Write biotech outputs to CSV files."""

    def _write_table(path: Path, table) -> None:
        try:
            table.to_frame().to_csv(path)  # type: ignore[attr-defined]
        except Exception:
            table.to_csv(path)

    _write_table(output_dir / "consolidated.csv", result.consolidated)
    _write_table(output_dir / "dcf_table.csv", result.dcf_table)
    for name, df in result.per_product.items():
        _write_table(output_dir / f"per_product_{name}.csv", df)
    for name, df in result.per_product_prob.items():
        _write_table(output_dir / f"per_product_prob_{name}.csv", df)
    (output_dir / "rnpv.txt").write_text(str(result.rnpv), encoding="utf-8")


def _run_cassava_core(params: CassavaModelParameters):
    result = params.model.build(params.scenario)
    return result


def _build_cassava_response(result) -> CassavaModelRunResponse:
    financials = result.get("financials")
    break_even = result.get("break_even")
    payback = result.get("payback")
    metrics = result.get("metrics", {})
    scenario = result.get("scenario", "")
    return CassavaModelRunResponse(
        income_statement_monthly=_df_payload(getattr(financials, "income_monthly", None), index_name="Month"),
        income_statement_annual=_df_payload(getattr(financials, "income_annual", None)),
        balance_sheet_monthly=_df_payload(getattr(financials, "balance_monthly", None), index_name="Month"),
        balance_sheet_annual=_df_payload(getattr(financials, "balance_annual", None)),
        cash_flow_monthly=_df_payload(getattr(financials, "cashflow_monthly", None), index_name="Month"),
        cash_flow_annual=_df_payload(getattr(financials, "cashflow_annual", None)),
        break_even=_df_payload(break_even, index_name="Month"),
        payback=_df_payload(payback, index_name="Year"),
        metrics={k: _clean_value(v) for k, v in metrics.items()},
        scenario=scenario,
    )


def _export_cassava(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    financials = result.get("financials")
    if financials:
        getattr(financials, "income_monthly", pd.DataFrame()).to_csv(output_dir / "income_monthly.csv")
        getattr(financials, "income_annual", pd.DataFrame()).to_csv(output_dir / "income_annual.csv")
        getattr(financials, "balance_monthly", pd.DataFrame()).to_csv(output_dir / "balance_monthly.csv")
        getattr(financials, "balance_annual", pd.DataFrame()).to_csv(output_dir / "balance_annual.csv")
        getattr(financials, "cashflow_monthly", pd.DataFrame()).to_csv(output_dir / "cashflow_monthly.csv")
        getattr(financials, "cashflow_annual", pd.DataFrame()).to_csv(output_dir / "cashflow_annual.csv")
    be = result.get("break_even")
    if hasattr(be, "to_csv"):
        be.to_csv(output_dir / "break_even.csv")
    pb = result.get("payback")
    if hasattr(pb, "to_csv"):
        pb.to_csv(output_dir / "payback.csv")
    metrics = result.get("metrics", {})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _run_broiler_core(params: BroilerModelParameters):
    return run_broiler_outputs(params.assumptions)


def _df_from_rows(rows):
    if not rows:
        return pd.DataFrame()
    try:
        import pandas as pd  # noqa: F401
    except Exception:
        return pd.DataFrame(rows)
    return pd.DataFrame([row.__dict__ if hasattr(row, "__dict__") else dict(row) for row in rows])


def _build_broiler_response(result) -> BroilerModelRunResponse:
    assumptions_schedule = result.get("assumptions_schedule")
    income_statement = result.get("financial_statements", {}).get("income_statement", [])
    balance_sheet = result.get("financial_statements", {}).get("balance_sheet", [])
    cash_flow_statement = result.get("financial_statements", {}).get("cash_flow_statement", [])
    cashflows = result.get("cashflows", [])
    revenue_summary = result.get("revenue_summary")
    advanced = result.get("advanced_analytics", {}) or {}
    valuation = result.get("valuation", {})
    def _adv_tables():
        payload = {}
        for name, df in advanced.items():
            payload[name] = _df_payload(df, index_name="Year")
        return payload

    return BroilerModelRunResponse(
        assumptions_schedule=_df_payload(assumptions_schedule, index_name="Year"),
        income_statement=_df_payload(_df_from_rows(income_statement), index_name="Year"),
        balance_sheet=_df_payload(_df_from_rows(balance_sheet), index_name="Year"),
        cash_flow_statement=_df_payload(_df_from_rows(cash_flow_statement), index_name="Year"),
        cashflows=_df_payload(_df_from_rows(cashflows), index_name="Year"),
        revenue_summary=_df_payload(revenue_summary, index_name="Year"),
        valuation={k: _clean_value(v) for k, v in valuation.items()},
        advanced_analytics=_adv_tables(),
    )


def _export_broiler(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    def _write(name, df):
        if df is None:
            return
        df.to_csv(output_dir / f"{name}.csv")
    _write("assumptions_schedule", result.get("assumptions_schedule"))
    fs = result.get("financial_statements", {}) or {}
    _write("income_statement", _df_from_rows(fs.get("income_statement", [])))
    _write("balance_sheet", _df_from_rows(fs.get("balance_sheet", [])))
    _write("cash_flow_statement", _df_from_rows(fs.get("cash_flow_statement", [])))
    _write("cashflows", _df_from_rows(result.get("cashflows", [])))
    _write("revenue_summary", result.get("revenue_summary"))
    for name, df in (result.get("advanced_analytics") or {}).items():
        _write(f"advanced_{name}", df)
    (output_dir / "valuation.json").write_text(json.dumps(result.get("valuation", {}), indent=2), encoding="utf-8")


@dataclass
class GoatRunResult:
    schedule: pd.DataFrame
    scenario: pd.DataFrame
    performance: pd.DataFrame
    cash_flow: pd.DataFrame
    position: pd.DataFrame
    kpis: pd.DataFrame
    break_even: pd.DataFrame
    advanced: Dict[str, Dict[str, Any]]
    valuation_summary: Dict[str, Any]


def _run_goat_core(params: GoatModelParameters) -> GoatRunResult:
    model = params.schedule.to_model()
    scenario_cfg = params.scenario or {}
    scenario_df = model.scenario(
        milk_price_pct=float(scenario_cfg.get("milk_price_pct") or 0.0),
        feed_cost_pct=float(scenario_cfg.get("feed_cost_pct") or 0.0),
    )
    performance = model.statement_of_financial_performance(scenario_df, annual=True)
    cash_flow = model.statement_of_cash_flow(scenario_df, annual=True)
    position = model.statement_of_financial_position(scenario_df, annual=True)
    kpis = model.kpis(scenario_df, annual=True)
    break_even = model.break_even(scenario_df, annual=True)
    advanced_raw = model.advanced_analytics(scenario_df, window=3, annual=True)
    advanced = {
        name: {
            "title": analysis.title,
            "description": analysis.description,
            "tables": analysis.tables,
        }
        for name, analysis in advanced_raw.items()
    }
    valuation_summary = {
        "WACC": model.wacc(),
        "NPV": model.npv(),
        "Terminal Value": model.terminal_value(),
    }
    return GoatRunResult(
        schedule=model.to_tidy(),
        scenario=scenario_df,
        performance=performance,
        cash_flow=cash_flow,
        position=position,
        kpis=kpis,
        break_even=break_even,
        advanced=advanced,
        valuation_summary=valuation_summary,
    )


def _build_goat_response(result: GoatRunResult) -> GoatModelRunResponse:
    def _analysis_payloads() -> Dict[str, Dict[str, Any]]:
        payloads: Dict[str, Dict[str, Any]] = {}
        for name, analysis in result.advanced.items():
            payloads[name] = {
                "title": analysis.get("title"),
                "description": analysis.get("description"),
                "tables": {
                    table_name: _df_payload(table, index_name="Period")
                    for table_name, table in analysis["tables"].items()
                    if table is not None
                },
            }
        return payloads

    return GoatModelRunResponse(
        schedule=_df_payload(result.schedule, index_name="Period"),
        scenario=_df_payload(result.scenario, index_name="Period"),
        performance=_df_payload(result.performance, index_name="Period"),
        cash_flow=_df_payload(result.cash_flow, index_name="Period"),
        position=_df_payload(result.position, index_name="Period"),
        kpis=_df_payload(result.kpis, index_name="Period"),
        break_even=_df_payload(result.break_even, index_name="Period"),
        advanced=_analysis_payloads(),
        valuation_summary={key: _clean_value(value) for key, value in result.valuation_summary.items()},
    )


def _export_goat(result: GoatRunResult, output_dir: Path) -> None:
    """Write goat outputs to CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result.schedule.to_csv(output_dir / "schedule.csv")
    result.scenario.to_csv(output_dir / "scenario.csv")
    result.performance.to_csv(output_dir / "performance.csv")
    result.cash_flow.to_csv(output_dir / "cash_flow.csv")
    result.position.to_csv(output_dir / "position.csv")
    result.kpis.to_csv(output_dir / "kpis.csv")
    result.break_even.to_csv(output_dir / "break_even.csv")
    for name, analysis in result.advanced.items():
        for table_name, table in analysis["tables"].items():
            table.to_csv(output_dir / f"{name}_{table_name}.csv")
    (output_dir / "valuation_summary.json").write_text(json.dumps(result.valuation_summary, indent=2), encoding="utf-8")


def _run_microbrewery_core(params: MicrobreweryModelParameters) -> MicrobreweryResult:
    model = MicrobreweryFinancialModel(params.config, params.dividend_policy, params.inputs)
    return model.run()


def _build_microbrewery_response(result: MicrobreweryResult) -> MicrobreweryModelRunResponse:
    return MicrobreweryModelRunResponse(
        monthly=_df_payload(result.monthly, index_name="Month"),
        annual=_df_payload(result.annual, index_name="Year"),
        prices=_df_payload(result.prices, index_name="Month"),
        debt_schedules={name: _df_payload(df, index_name="Month") for name, df in result.debt_schedules.items()},
        valuation={key: _clean_value(value) for key, value in result.valuation.items()},
    )


def _export_microbrewery(result: MicrobreweryResult, output_dir: Path) -> None:
    """Write microbrewery outputs to CSV files."""

    def _write_table(path: Path, df: pd.DataFrame) -> None:
        df.to_csv(path)

    _write_table(output_dir / "monthly.csv", result.monthly)
    _write_table(output_dir / "annual.csv", result.annual)
    _write_table(output_dir / "prices.csv", result.prices)
    for name, df in result.debt_schedules.items():
        _write_table(output_dir / f"debt_{name}.csv", df)
    (output_dir / "valuation.json").write_text(json.dumps(result.valuation, indent=2), encoding="utf-8")


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "pharma": ModelSpec(
        name="Pharmaceuticals",
        load_inputs=load_pharma_inputs,
        parse_inputs=parse_pharma_inputs,
        run_core=_run_pharma_core,
        build_response=_build_pharma_response,
        run_request_model=PharmaModelRunRequest,
        validate_request_model=PharmaValidationRequest,
        response_model=ModelRunResponse,
        cli_exporter=_export_pharma,
        build_report_sections=lambda inputs, outputs: collect_report_sections(FinancialModel(inputs), outputs),
        report_title="Pharmaceuticals Financial Report",
        report_name="pharma_financial_report",
    ),
    "biotech": ModelSpec(
        name="Biotech",
        load_inputs=load_biotech_inputs,
        parse_inputs=parse_biotech_inputs,
        run_core=_run_biotech_core,
        build_response=_build_biotech_response,
        run_request_model=BiotechModelRunRequest,
        validate_request_model=BiotechValidationRequest,
        response_model=BiotechModelRunResponse,
        cli_exporter=_export_biotech,
        build_report_sections=lambda _inputs, result: collect_biotech_report_sections(result),
        report_title="Biotech Valuation Report",
        report_name="biotech_financial_report",
    ),
    "microbrewery": ModelSpec(
        name="Microbrewery",
        load_inputs=load_microbrewery_inputs,
        parse_inputs=parse_microbrewery_inputs,
        run_core=_run_microbrewery_core,
        build_response=_build_microbrewery_response,
        run_request_model=MicrobreweryModelRunRequest,
        validate_request_model=MicrobreweryValidationRequest,
        response_model=MicrobreweryModelRunResponse,
        cli_exporter=_export_microbrewery,
    ),
    "goat_farming": ModelSpec(
        name="Goat Farming",
        load_inputs=load_goat_inputs,
        parse_inputs=parse_goat_inputs,
        run_core=_run_goat_core,
        build_response=_build_goat_response,
        run_request_model=GoatModelRunRequest,
        validate_request_model=GoatValidationRequest,
        response_model=GoatModelRunResponse,
        cli_exporter=_export_goat,
    ),
    "cassava_ethanol": ModelSpec(
        name="Cassava Bioethanol",
        load_inputs=load_cassava_inputs,
        parse_inputs=parse_cassava_inputs,
        run_core=_run_cassava_core,
        build_response=_build_cassava_response,
        run_request_model=CassavaModelRunRequest,
        validate_request_model=CassavaValidationRequest,
        response_model=CassavaModelRunResponse,
        cli_exporter=_export_cassava,
    ),
    "broiler_chicken": ModelSpec(
        name="Broiler Chicken",
        load_inputs=load_broiler_inputs,
        parse_inputs=parse_broiler_inputs,
        run_core=_run_broiler_core,
        build_response=_build_broiler_response,
        run_request_model=BroilerModelRunRequest,
        validate_request_model=BroilerValidationRequest,
        response_model=BroilerModelRunResponse,
        cli_exporter=_export_broiler,
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'")
    return MODEL_REGISTRY[key]


def list_models() -> Dict[str, ModelSpec]:
    return dict(MODEL_REGISTRY)
