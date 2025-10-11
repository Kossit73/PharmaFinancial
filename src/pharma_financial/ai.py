"""AI and machine-learning utilities for the financial model."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from .inputs import AIParameters
from .table import Table, build_table


@dataclass
class AIInsights:
    """Container for AI-generated artefacts accompanying the financial outputs."""

    ml_forecast: Optional[Table]
    generative_summary: str
    enabled: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


class MachineLearningAdvisor:
    """Derive machine-learning forecasts using lightweight algorithms."""

    def __init__(self, parameters: AIParameters):
        self.parameters = parameters

    def revenue_forecast(self, years: Sequence[int], revenues: Sequence[float]) -> Optional[Table]:
        horizon = max(int(self.parameters.forecast_horizon or 0), 0)
        if horizon <= 0:
            return None

        base_years = list(years)
        revenue_values = [float(value) for value in revenues]
        if not base_years or not revenue_values:
            return None

        methods = [
            method.strip().lower()
            for method in self.parameters.ml_methods
            if isinstance(method, str) and method.strip()
        ]
        if not methods:
            methods = ["linear_regression"]

        horizon_years = [base_years[-1] + step for step in range(1, horizon + 1)]
        full_years = base_years + horizon_years

        columns: dict[str, List[float]] = {}
        columns["Historical Net Revenue"] = revenue_values + [math.nan] * horizon

        for method in methods:
            forecasts = self._forecast(method, base_years, revenue_values, horizon)
            label = self._label(method)
            columns[label] = [math.nan] * len(base_years) + forecasts

        return build_table(full_years, columns)

    # ------------------------------------------------------------------ helpers
    def _forecast(
        self,
        method: str,
        years: Sequence[int],
        values: Sequence[float],
        horizon: int,
    ) -> List[float]:
        method = method.lower()
        if method == "cagr":
            return self._cagr(values, horizon)
        if method in {"moving_average", "rolling_mean"}:
            return self._moving_average(values, horizon)
        return self._linear_regression(years, values, horizon)

    def _linear_regression(
        self,
        years: Sequence[int],
        values: Sequence[float],
        horizon: int,
    ) -> List[float]:
        if not years:
            return [0.0 for _ in range(horizon)]
        n = len(years)
        if n == 1:
            return [float(values[-1]) for _ in range(horizon)]

        mean_x = sum(years) / n
        mean_y = sum(values) / n
        denominator = sum((x - mean_x) ** 2 for x in years)
        if abs(denominator) < 1e-12:
            slope = 0.0
        else:
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(years, values)) / denominator
        intercept = mean_y - slope * mean_x

        forecasts: List[float] = []
        last_year = years[-1]
        for step in range(1, horizon + 1):
            year = last_year + step
            forecasts.append(intercept + slope * year)
        return forecasts

    def _cagr(self, values: Sequence[float], horizon: int) -> List[float]:
        if not values:
            return [0.0 for _ in range(horizon)]
        start = float(values[0])
        end = float(values[-1])
        periods = max(len(values) - 1, 1)
        if start <= 0 or end <= 0:
            growth = 0.0
        else:
            growth = end / start
            growth = growth ** (1 / periods) - 1
        forecasts: List[float] = []
        current = float(values[-1])
        for _ in range(horizon):
            current *= 1 + growth
            forecasts.append(current)
        return forecasts

    def _moving_average(self, values: Sequence[float], horizon: int, window: int = 3) -> List[float]:
        history = [float(value) for value in values]
        if not history:
            return [0.0 for _ in range(horizon)]
        window = max(1, min(window, len(history)))
        forecasts: List[float] = []
        for _ in range(horizon):
            segment = history[-window:]
            average = sum(segment) / len(segment)
            forecasts.append(average)
            history.append(average)
        return forecasts

    def _label(self, method: str) -> str:
        aliases = {
            "linear_regression": "Linear Regression Forecast",
            "cagr": "CAGR Projection",
            "moving_average": "Moving Average Forecast",
            "rolling_mean": "Moving Average Forecast",
        }
        return aliases.get(method, method.replace("_", " ").title())


class GenerativeAdvisor:
    """Optionally call a generative model to provide narrative commentary."""

    def __init__(self, parameters: AIParameters):
        self.parameters = parameters
        self.metadata: dict[str, Any] = {
            "provider": parameters.provider,
            "model": parameters.model,
            "features": list(parameters.generative_features),
        }

    def summarise(
        self,
        *,
        summary: Table,
        income: Table,
        cash_flow: Table,
        ml_table: Optional[Table],
    ) -> str:
        if not self.parameters.enabled:
            self.metadata["status"] = "disabled"
            return (
                "AI enhancements are disabled. Enable them on the Input Landing Page "
                "and provide an API key to generate automated commentary."
            )

        prompt = self._build_prompt(summary, income, cash_flow, ml_table)
        response = self._invoke_model(prompt)
        if response:
            self.metadata["status"] = "model_response"
            return response

        self.metadata.setdefault("status", "fallback")
        return self._fallback(summary, income, cash_flow, ml_table)

    # ------------------------------------------------------------------ helpers
    def _build_prompt(
        self,
        summary: Table,
        income: Table,
        cash_flow: Table,
        ml_table: Optional[Table],
    ) -> str:
        metrics = summary.as_dict().get("Value", [])
        labels = summary.index
        metric_pairs = [f"{label}: {value:.2f}" for label, value in zip(labels, metrics)]

        latest_year = income.index[-1] if income.index else "latest year"
        net_income = income.column("Net Income")[-1] if "Net Income" in income.data else 0.0
        net_revenue = income.column("Net Revenue")[-1] if "Net Revenue" in income.data else 0.0
        ebitda = income.column("EBITDA")[-1] if "EBITDA" in income.data else 0.0

        cash_change = 0.0
        if "Net Change in Cash" in cash_flow.data:
            cash_change = cash_flow.column("Net Change in Cash")[-1]
        elif "Net Cash Flow for the Period" in cash_flow.data:
            cash_change = cash_flow.column("Net Cash Flow for the Period")[-1]
        elif "Net Increase/Decrease in Cash" in cash_flow.data:
            cash_change = cash_flow.column("Net Increase/Decrease in Cash")[-1]

        forecast_lines: list[str] = []
        if ml_table is not None:
            for column in ml_table.columns():
                if column == "Historical Net Revenue":
                    continue
                forecast_lines.append(
                    f"{column}: {ml_table.data[column][-1]:.2f} by {ml_table.index[-1]}"
                )

        prompt_lines = [
            "You are a financial analyst. Provide concise insights on the Pharmaceuticals model.",
            "Key investment metrics:",
            *metric_pairs,
            f"Latest year ({latest_year}) Net Revenue: {net_revenue:.2f}",
            f"Latest year EBITDA: {ebitda:.2f}",
            f"Latest year Net Income: {net_income:.2f}",
            f"Net change in cash: {cash_change:.2f}",
        ]

        if forecast_lines:
            prompt_lines.append("Forecast highlights:")
            prompt_lines.extend(forecast_lines)

        prompt_lines.append(
            "Comment on profitability, liquidity, risk exposure, and forward-looking outlook in no more than 150 words."
        )
        return "\n".join(str(line) for line in prompt_lines if str(line).strip())

    def _invoke_model(self, prompt: str) -> Optional[str]:
        api_key = self.parameters.api_key
        if not api_key:
            self.metadata["warning"] = "No API key supplied; using heuristic summary."
            return None

        provider = (self.parameters.provider or "").strip().lower()
        if provider == "openai":
            try:  # pragma: no cover - executed only when openai is installed
                import openai  # type: ignore
            except Exception as exc:  # pragma: no cover
                self.metadata["error"] = f"OpenAI SDK unavailable: {exc}"
                return None

            try:  # pragma: no cover - requires networked environment
                if hasattr(openai, "OpenAI"):
                    client = openai.OpenAI(api_key=api_key)
                    completion = client.chat.completions.create(
                        model=self.parameters.model,
                        messages=[
                            {"role": "system", "content": "You are a financial analyst."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                        max_tokens=400,
                    )
                    if completion.choices:
                        message = completion.choices[0].message.content
                        if message:
                            return message.strip()
                else:
                    openai.api_key = api_key
                    completion = openai.ChatCompletion.create(  # type: ignore[attr-defined]
                        model=self.parameters.model,
                        messages=[
                            {"role": "system", "content": "You are a financial analyst."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                        max_tokens=400,
                    )
                    if completion and completion.choices:
                        message = completion.choices[0].message["content"]
                        if message:
                            return str(message).strip()
            except Exception as exc:  # pragma: no cover
                self.metadata["error"] = f"OpenAI request failed: {exc}"
                return None
        else:
            self.metadata["warning"] = (
                f"Provider '{self.parameters.provider}' not implemented; "
                "falling back to heuristic summary."
            )
        return None

    def _fallback(
        self,
        summary: Table,
        income: Table,
        cash_flow: Table,
        ml_table: Optional[Table],
    ) -> str:
        def _lookup(metric: str) -> float:
            if metric in summary.index:
                position = summary.index.index(metric)
                return float(summary.data["Value"][position])
            return float("nan")

        npv = _lookup("NPV")
        irr = _lookup("IRR")
        payback = _lookup("Payback Period")

        net_income_series = income.column("Net Income") if "Net Income" in income.data else []
        latest_income = net_income_series[-1] if net_income_series else 0.0

        if "Ending Cash" in cash_flow.data:
            cash_series = cash_flow.column("Ending Cash")
        elif "Cash and Cash Equivalents at the End of the Period" in cash_flow.data:
            cash_series = cash_flow.column("Cash and Cash Equivalents at the End of the Period")
        else:
            cash_series = []
        latest_cash = cash_series[-1] if cash_series else 0.0

        forecast_note = ""
        if ml_table is not None and ml_table.columns():
            future_year = ml_table.index[-1]
            series = [
                f"{column}: {ml_table.data[column][-1]:.2f}"
                for column in ml_table.columns()
                if column != "Historical Net Revenue"
            ]
            if series:
                forecast_note = f"Forecast {future_year}: " + ", ".join(series)

        lines = [
            "AI-generated summary (heuristic):",
            f"- Net Present Value (NPV): {self._format_currency(npv)}",
            f"- Internal Rate of Return (IRR): {self._format_percentage(irr)}",
            f"- Payback Period: {self._format_years(payback)}",
            f"- Latest Net Income: {self._format_currency(latest_income)}",
            f"- Ending Cash Position: {self._format_currency(latest_cash)}",
        ]

        if forecast_note:
            lines.append(f"- {forecast_note}")

        lines.append(
            "These figures indicate the profitability profile and liquidity runway based on the configured assumptions."
        )
        return "\n".join(lines)

    def _format_currency(self, value: float) -> str:
        if value != value or math.isinf(value):
            return "n/a"
        return f"${value:,.2f}"

    def _format_percentage(self, value: float) -> str:
        if value != value or math.isinf(value):
            return "n/a"
        return f"{value * 100:.2f}%"

    def _format_years(self, value: float) -> str:
        if value != value or math.isinf(value):
            return "n/a"
        return f"{value:.2f} years"


__all__ = ["AIInsights", "MachineLearningAdvisor", "GenerativeAdvisor"]
