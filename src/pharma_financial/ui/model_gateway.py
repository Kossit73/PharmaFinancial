"""Gateway for running the financial model locally or via the HTTP API."""
from __future__ import annotations

import math
import os
from typing import Any, Mapping, Sequence, Tuple

try:  # pragma: no cover - optional dependency guard
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from ..core.ai import AIInsights
from ..core.inputs import ModelInputs
from ..core.model import FinancialModel, FinancialOutputs, ScenarioToolResult
from ..core.table import Table


def _clean_cell(value: Any) -> Any:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        as_float = float(value)
        if math.isnan(as_float) or math.isinf(as_float):
            return float("nan")
        return as_float
    return value


def _table_from_payload(payload: Mapping[str, Any] | None) -> Table | None:
    if not payload:
        return None
    index = list(payload.get("index", []))
    columns = {name: [_clean_cell(value) for value in values] for name, values in payload.get("data", {}).items()}
    index_name = payload.get("index_name") or "Year"
    return Table(index, columns, index_name=index_name)


def _ai_from_payload(payload: Mapping[str, Any] | None) -> AIInsights | None:
    if not payload:
        return None
    ml_forecast = _table_from_payload(payload.get("ml_forecast"))
    metadata = dict(payload.get("metadata") or {})
    return AIInsights(
        ml_forecast=ml_forecast,
        generative_summary=str(payload.get("generative_summary") or ""),
        enabled=bool(payload.get("enabled", False)),
        metadata=metadata,
    )


def _scenario_tool_results_from_payload(payload: Mapping[str, Any]) -> dict[str, ScenarioToolResult]:
    results: dict[str, ScenarioToolResult] = {}
    for name, raw in payload.items():
        rows = list(raw.get("rows", [])) if isinstance(raw, Mapping) else []
        interpretation = str(raw.get("interpretation", "")) if isinstance(raw, Mapping) else ""
        results[name] = ScenarioToolResult(rows=rows, interpretation=interpretation)
    return results


class ModelGateway:
    """Encapsulates the logic for running the financial model."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = (base_url or os.getenv("PHARMA_FINANCIAL_API_URL") or "").strip()
        self.timeout = timeout
        if self.use_api:
            if requests is None:
                raise RuntimeError("The 'requests' package is required to call the API gateway.")
            self.session = requests.Session()
        else:
            self.session = None

    @property
    def use_api(self) -> bool:
        return bool(self.base_url)

    def run_model(
        self,
        payload: Mapping[str, Any],
        inputs: ModelInputs,
    ) -> Tuple[FinancialModel, FinancialOutputs]:
        if not self.use_api:
            model = FinancialModel(inputs)
            outputs = model.run()
            return model, outputs

        response = self.session.post(  # type: ignore[union-attr]
            f"{self.base_url.rstrip('/')}/model/run",
            json={"inputs": payload},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return FinancialModel(inputs), self._outputs_from_payload(data)

    # ------------------------------------------------------------------ helpers
    def _outputs_from_payload(self, payload: Mapping[str, Any]) -> FinancialOutputs:
        summary_metrics = _table_from_payload(payload.get("summary_metrics"))
        income_statement = _table_from_payload(payload.get("income_statement"))
        balance_sheet = _table_from_payload(payload.get("balance_sheet"))
        cash_flow = _table_from_payload(payload.get("cash_flow"))
        goal_seek = _table_from_payload(payload.get("goal_seek"))
        break_even = _table_from_payload(payload.get("break_even"))
        payback = _table_from_payload(payload.get("payback"))
        discounted_payback = _table_from_payload(payload.get("discounted_payback"))
        monte_carlo = _table_from_payload(payload.get("monte_carlo"))
        scenario_results = {
            name: _table_from_payload(table)
            for name, table in (payload.get("scenario_results") or {}).items()
        }
        sensitivity_results = {
            name: _table_from_payload(table)
            for name, table in (payload.get("sensitivity_results") or {}).items()
        }
        scenario_tool_results = _scenario_tool_results_from_payload(payload.get("scenario_tool_results") or {})
        ai_insights = _ai_from_payload(payload.get("ai_insights"))
        risk_factor_diagnostics = _table_from_payload(payload.get("risk_factor_diagnostics"))

        return FinancialOutputs(
            income_statement=income_statement or Table([], {}),
            balance_sheet=balance_sheet or Table([], {}),
            cash_flow=cash_flow or Table([], {}),
            summary_metrics=summary_metrics or Table([], {}),
            goal_seek=goal_seek or Table([], {}),
            break_even=break_even or Table([], {}),
            payback=payback or Table([], {}),
            discounted_payback=discounted_payback or Table([], {}),
            scenario_results={k: v for k, v in scenario_results.items() if v is not None},
            sensitivity_results={k: v for k, v in sensitivity_results.items() if v is not None},
            monte_carlo=monte_carlo or Table([], {}),
            scenario_tool_results=scenario_tool_results,
            ai_insights=ai_insights,
            risk_factor_diagnostics=risk_factor_diagnostics,
        )
