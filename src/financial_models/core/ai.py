"""AI and machine-learning utilities for the financial model."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from financial_models.pharma.inputs import AIParameters

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
        self.diagnostics: dict[str, Mapping[str, float]] = {}

    def revenue_forecast(self, years: Sequence[int], revenues: Sequence[float]) -> Optional[Table]:
        horizon = max(int(self.parameters.forecast_horizon or 0), 0)
        if horizon <= 0:
            return None

        base_years = list(years)
        revenue_values = [float(value) for value in revenues]
        if not base_years or not revenue_values:
            return None

        self.diagnostics = {}
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
            forecasts, in_sample = self._fit_and_forecast(method, base_years, revenue_values, horizon)
            label = self._label(method)
            columns[label] = [math.nan] * len(base_years) + forecasts
            diagnostics = self._error_metrics(revenue_values, in_sample)
            self.diagnostics[label] = diagnostics if diagnostics else {}

        return build_table(full_years, columns)

    # ------------------------------------------------------------------ helpers
    def _fit_and_forecast(
        self,
        method: str,
        years: Sequence[int],
        values: Sequence[float],
        horizon: int,
    ) -> tuple[List[float], List[float]]:
        method = method.lower()
        if method == "cagr":
            return self._cagr(values, horizon)
        if method in {"moving_average", "rolling_mean"}:
            return self._moving_average(values, horizon)
        if method in {"seasonal", "seasonal_trend", "seasonality"}:
            return self._seasonal_trend(years, values, horizon)
        return self._linear_regression(years, values, horizon)

    def _linear_regression(
        self,
        years: Sequence[int],
        values: Sequence[float],
        horizon: int,
    ) -> tuple[List[float], List[float]]:
        if not years:
            return [0.0 for _ in range(horizon)], []
        n = len(years)
        if n == 1:
            repeated = float(values[-1])
            return [repeated for _ in range(horizon)], [repeated]

        mean_x = sum(years) / n
        mean_y = sum(values) / n
        denominator = sum((x - mean_x) ** 2 for x in years) + max(self.parameters.regularization, 0.0)
        slope = 0.0 if abs(denominator) < 1e-12 else sum((x - mean_x) * (y - mean_y) for x, y in zip(years, values)) / denominator
        intercept = mean_y - slope * mean_x

        forecasts = [intercept + slope * (years[-1] + step) for step in range(1, horizon + 1)]
        in_sample = [intercept + slope * year for year in years]
        return self._clamp_forecasts(forecasts, values), in_sample

    def _cagr(self, values: Sequence[float], horizon: int) -> tuple[List[float], List[float]]:
        if not values:
            return [0.0 for _ in range(horizon)], []
        start = float(values[0])
        end = float(values[-1])
        periods = max(len(values) - 1, 1)
        growth = 0.0 if start <= 0 or end <= 0 else (end / start) ** (1 / periods) - 1
        forecasts: List[float] = []
        current = float(values[-1]) if values else 0.0
        for _ in range(horizon):
            current *= 1 + growth
            forecasts.append(current)
        in_sample: List[float] = []
        current = start
        for _ in range(len(values)):
            in_sample.append(current)
            current *= 1 + growth
        return self._clamp_forecasts(forecasts, values), in_sample

    def _moving_average(self, values: Sequence[float], horizon: int, window: int = 3) -> tuple[List[float], List[float]]:
        history = [float(value) for value in values]
        if not history:
            return [0.0 for _ in range(horizon)], []
        window = max(1, min(window, len(history)))
        forecasts: List[float] = []
        rolling = history[:]
        for _ in range(horizon):
            segment = rolling[-window:]
            average = sum(segment) / len(segment)
            forecasts.append(average)
            rolling.append(average)

        in_sample: List[float] = []
        for idx in range(len(history)):
            start = max(0, idx - window)
            segment = history[start:idx]
            in_sample.append(sum(segment) / len(segment) if segment else history[idx])
        return self._clamp_forecasts(forecasts, values), in_sample

    def _seasonal_trend(self, years: Sequence[int], values: Sequence[float], horizon: int) -> tuple[List[float], List[float]]:
        period = max(int(self.parameters.seasonality_period or 0), 0)
        if period < 2 or len(values) < period:
            return self._linear_regression(years, values, horizon)

        indices = list(range(len(values)))
        slope, intercept = self._trend_coefficients(indices, values)
        trend = [intercept + slope * idx for idx in indices]
        residuals = [actual - trend[idx] for idx, actual in enumerate(values)]

        seasonal: List[float] = [0.0 for _ in range(period)]
        counts: List[int] = [0 for _ in range(period)]
        for idx, residual in enumerate(residuals):
            bucket = idx % period
            seasonal[bucket] += residual
            counts[bucket] += 1
        seasonal = [seasonal[idx] / counts[idx] if counts[idx] else 0.0 for idx in range(period)]

        in_sample = [trend[idx] + seasonal[idx % period] for idx in indices]

        forecasts: List[float] = []
        base_index = len(values) - 1
        for step in range(1, horizon + 1):
            future_index = base_index + step
            trend_value = intercept + slope * future_index
            seasonal_value = seasonal[future_index % period]
            forecasts.append(trend_value + seasonal_value)
        return self._clamp_forecasts(forecasts, values), in_sample

    def _trend_coefficients(self, indices: Sequence[int], values: Sequence[float]) -> tuple[float, float]:
        if not indices:
            return 0.0, float(values[-1]) if values else 0.0
        n = len(indices)
        mean_x = sum(indices) / n
        mean_y = sum(values) / n if values else 0.0
        denominator = sum((x - mean_x) ** 2 for x in indices) + max(self.parameters.regularization, 0.0)
        slope = 0.0 if abs(denominator) < 1e-12 else sum((x - mean_x) * (y - mean_y) for x, y in zip(indices, values)) / denominator
        intercept = mean_y - slope * mean_x
        return slope, intercept

    def _clamp_forecasts(self, forecasts: Sequence[float], history: Sequence[float]) -> List[float]:
        if not forecasts:
            return []
        minimum = float(self.parameters.min_forecast)
        max_multiplier = max(float(self.parameters.max_forecast_multiplier or 0.0), 1.0)
        anchor = 1.0
        if history:
            history_values = [abs(float(value)) for value in history]
            anchor = max(max(history_values), abs(float(history[-1])), 1.0)
        max_bound = anchor * max_multiplier
        return [min(max(float(value), minimum), max_bound) for value in forecasts]

    def _error_metrics(self, actual: Sequence[float], predicted: Sequence[float]) -> Mapping[str, float]:
        if not actual or len(actual) != len(predicted):
            return {}
        errors = [float(a) - float(p) for a, p in zip(actual, predicted)]
        mae = sum(abs(err) for err in errors) / len(errors)
        rmse = math.sqrt(sum(err**2 for err in errors) / len(errors))
        mape_values = [abs((a - p) / a) * 100 for a, p in zip(actual, predicted) if abs(a) > 1e-9]
        mape = sum(mape_values) / len(mape_values) if mape_values else float("nan")
        mean_actual = sum(actual) / len(actual)
        sst = sum((float(a) - mean_actual) ** 2 for a in actual)
        r_squared = float("nan") if abs(sst) < 1e-12 else 1 - (sum((float(a) - float(p)) ** 2 for a, p in zip(actual, predicted)) / sst)
        return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R^2": r_squared}

    def _label(self, method: str) -> str:
        return {
            "linear_regression": "Linear Regression Forecast",
            "cagr": "CAGR Projection",
            "moving_average": "Moving Average Forecast",
            "rolling_mean": "Moving Average Forecast",
            "seasonal": "Seasonal Trend Forecast",
            "seasonality": "Seasonal Trend Forecast",
            "seasonal_trend": "Seasonal Trend Forecast",
        }.get(method, method.replace("_", " ").title())


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

    def _build_prompt(self, summary: Table, income: Table, cash_flow: Table, ml_table: Optional[Table]) -> str:
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
                forecast_lines.append(f"{column}: {ml_table.data[column][-1]:.2f} by {ml_table.index[-1]}")
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
        prompt_lines.append("Comment on profitability, liquidity, risk exposure, and forward-looking outlook in no more than 150 words.")
        return "\n".join(str(line) for line in prompt_lines if str(line).strip())

    def _invoke_model(self, prompt: str) -> Optional[str]:
        api_key = self.parameters.api_key
        if not api_key:
            self.metadata["warning"] = "No API key supplied; using heuristic summary."
            return None
        provider = (self.parameters.provider or "").strip().lower()
        if provider == "openai":
            try:
                import openai  # type: ignore
            except Exception as exc:
                self.metadata["error"] = f"OpenAI SDK unavailable: {exc}"
                return None
            try:
                if hasattr(openai, "OpenAI"):
                    client = openai.OpenAI(api_key=api_key)
                    completion = client.chat.completions.create(
                        model=self.parameters.model,
                        messages=[{"role": "system", "content": "You are a financial analyst."}, {"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=400,
                    )
                    if completion.choices:
                        message = completion.choices[0].message.content
                        if message:
                            return message.strip()
                else:
                    openai.api_key = api_key
                    completion = openai.ChatCompletion.create(
                        model=self.parameters.model,
                        messages=[{"role": "system", "content": "You are a financial analyst."}, {"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=400,
                    )
                    if completion and completion.choices:
                        message = completion.choices[0].message["content"]
                        if message:
                            return str(message).strip()
            except Exception as exc:
                self.metadata["error"] = f"OpenAI request failed: {exc}"
                return None
        else:
            self.metadata["warning"] = f"Provider '{self.parameters.provider}' not implemented; falling back to heuristic summary."
        return None

    def _fallback(self, summary: Table, income: Table, cash_flow: Table, ml_table: Optional[Table]) -> str:
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

        lines.append("These figures indicate the profitability profile and liquidity runway based on the configured assumptions.")
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
