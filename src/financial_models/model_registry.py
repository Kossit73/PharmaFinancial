"""Registry of financial models exposed via API/CLI."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Type

import pandas as pd

from .api.schemas import (
    BiotechModelRunRequest,
    BiotechModelRunResponse,
    BiotechValidationRequest,
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
from .pharma.inputs import ModelInputs, load_inputs as load_pharma_inputs, parse_inputs as parse_pharma_inputs
from .pharma.model import FinancialModel
from .pharma.table import Table
from .api.schemas.common import AIInsightsPayload


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
    return TablePayload(index_name=index_name, index=list(df.index), data=df.to_dict(orient="list"))


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
}


def get_model_spec(model_name: str) -> ModelSpec:
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'")
    return MODEL_REGISTRY[key]


def list_models() -> Dict[str, ModelSpec]:
    return dict(MODEL_REGISTRY)
