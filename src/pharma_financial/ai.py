"""AI and machine-learning utilities for the financial model."""
from __future__ import annotations

import math
import re
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
        self.diagnostics: dict[str, Mapping[str, float]] = {}
        self.backtest_diagnostics: dict[str, Mapping[str, float]] = {}
        self.explainability: dict[str, Mapping[str, float]] = {}
        self.audit_log: dict[str, Any] = {}

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
        self.diagnostics = {}
        self.backtest_diagnostics = {}
        self.explainability = {}
        self.audit_log = {
            "forecast_horizon": horizon,
            "methods": methods,
            "history_points": len(revenue_values),
            "years": list(base_years),
        }

        horizon_years = [base_years[-1] + step for step in range(1, horizon + 1)]
        full_years = base_years + horizon_years

        columns: dict[str, List[float]] = {}
        columns["Historical Net Revenue"] = revenue_values + [math.nan] * horizon

        for method in methods:
            forecasts, in_sample = self._fit_and_forecast(method, base_years, revenue_values, horizon)
            label = self._label(method)
            columns[label] = [math.nan] * len(base_years) + forecasts
            diagnostics = self._error_metrics(revenue_values, in_sample)
            if diagnostics:
                self.diagnostics[label] = diagnostics
            else:
                self.diagnostics[label] = {}
            backtest = self._rolling_backtest(method, base_years, revenue_values)
            self.backtest_diagnostics[label] = backtest
            explanation = self._explain_method(method, base_years, revenue_values)
            if explanation:
                self.explainability[label] = explanation

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
        in_sample = [intercept + slope * year for year in years]
        return self._clamp_forecasts(forecasts, values), in_sample

    def _cagr(self, values: Sequence[float], horizon: int) -> tuple[List[float], List[float]]:
        if not values:
            return [0.0 for _ in range(horizon)], []
        start = float(values[0])
        end = float(values[-1])
        periods = max(len(values) - 1, 1)
        if start <= 0 or end <= 0:
            growth = 0.0
        else:
            growth = end / start
            growth = growth ** (1 / periods) - 1
        forecasts: List[float] = []
        in_sample: List[float] = []
        current = start
        for _ in range(len(values)):
            in_sample.append(current)
            current *= 1 + growth
        current = float(values[-1]) if values else 0.0
        for _ in range(horizon):
            current *= 1 + growth
            forecasts.append(current)
        return self._clamp_forecasts(forecasts, values), in_sample

    def _moving_average(
        self, values: Sequence[float], horizon: int, window: int = 3
    ) -> tuple[List[float], List[float]]:
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
            if segment:
                in_sample.append(sum(segment) / len(segment))
            else:
                in_sample.append(history[idx])
        return self._clamp_forecasts(forecasts, values), in_sample

    def _seasonal_trend(
        self, years: Sequence[int], values: Sequence[float], horizon: int
    ) -> tuple[List[float], List[float]]:
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
        seasonal = [
            seasonal[idx] / counts[idx] if counts[idx] else 0.0
            for idx in range(period)
        ]

        in_sample = [trend[idx] + seasonal[idx % period] for idx in indices]

        forecasts: List[float] = []
        base_index = len(values) - 1
        for step in range(1, horizon + 1):
            future_index = base_index + step
            trend_value = intercept + slope * future_index
            seasonal_value = seasonal[future_index % period]
            forecasts.append(trend_value + seasonal_value)
        return self._clamp_forecasts(forecasts, values), in_sample

    def _trend_coefficients(
        self, indices: Sequence[int], values: Sequence[float]
    ) -> tuple[float, float]:
        if not indices:
            return 0.0, float(values[-1]) if values else 0.0
        n = len(indices)
        mean_x = sum(indices) / n
        mean_y = sum(values) / n if values else 0.0
        denominator = sum((x - mean_x) ** 2 for x in indices) + max(self.parameters.regularization, 0.0)
        if abs(denominator) < 1e-12:
            slope = 0.0
        else:
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(indices, values)) / denominator
        intercept = mean_y - slope * mean_x
        return slope, intercept

    def _rolling_backtest(
        self,
        method: str,
        years: Sequence[int],
        values: Sequence[float],
        min_train: int = 3,
    ) -> Mapping[str, float]:
        if len(values) <= min_train:
            return {}
        predictions: List[float] = []
        actuals: List[float] = []
        for idx in range(min_train, len(values)):
            train_years = years[:idx]
            train_values = values[:idx]
            forecast, _ = self._fit_and_forecast(method, train_years, train_values, horizon=1)
            if not forecast:
                continue
            predictions.append(float(forecast[0]))
            actuals.append(float(values[idx]))
        return self._error_metrics(actuals, predictions)

    def _explain_method(
        self,
        method: str,
        years: Sequence[int],
        values: Sequence[float],
    ) -> Mapping[str, float]:
        method = method.lower()
        if method == "cagr":
            if not values:
                return {}
            start = float(values[0])
            end = float(values[-1])
            periods = max(len(values) - 1, 1)
            growth = 0.0
            if start > 0 and end > 0:
                growth = end / start
                growth = growth ** (1 / periods) - 1
            return {"cagr": growth}
        if method in {"moving_average", "rolling_mean"}:
            return {"window": 3.0}
        if method in {"seasonal", "seasonal_trend", "seasonality"}:
            period = max(int(self.parameters.seasonality_period or 0), 0)
            indices = list(range(len(values)))
            slope, intercept = self._trend_coefficients(indices, values)
            return {"trend_slope": slope, "trend_intercept": intercept, "seasonality_period": float(period)}
        if not years:
            return {}
        n = len(years)
        mean_x = sum(years) / n
        mean_y = sum(values) / n if values else 0.0
        denominator = sum((x - mean_x) ** 2 for x in years) + max(self.parameters.regularization, 0.0)
        slope = 0.0 if abs(denominator) < 1e-12 else sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(years, values)
        ) / denominator
        intercept = mean_y - slope * mean_x
        return {"slope": slope, "intercept": intercept}

    def _clamp_forecasts(
        self, forecasts: Sequence[float], history: Sequence[float]
    ) -> List[float]:
        if not forecasts:
            return []
        minimum = float(self.parameters.min_forecast)
        max_multiplier = max(float(self.parameters.max_forecast_multiplier or 0.0), 1.0)
        anchor = 1.0
        if history:
            history_values = [abs(float(value)) for value in history]
            anchor = max(max(history_values), abs(float(history[-1])), 1.0)
        max_bound = anchor * max_multiplier
        clamped: List[float] = []
        for value in forecasts:
            clamped.append(min(max(float(value), minimum), max_bound))
        return clamped

    def _error_metrics(
        self, actual: Sequence[float], predicted: Sequence[float]
    ) -> Mapping[str, float]:
        if not actual or len(actual) != len(predicted):
            return {}
        errors = [float(a) - float(p) for a, p in zip(actual, predicted)]
        if not errors:
            return {}
        mae = sum(abs(err) for err in errors) / len(errors)
        rmse = math.sqrt(sum(err ** 2 for err in errors) / len(errors))
        mape_values = [
            abs((a - p) / a) * 100
            for a, p in zip(actual, predicted)
            if abs(a) > 1e-9
        ]
        mape = sum(mape_values) / len(mape_values) if mape_values else float("nan")
        mean_actual = sum(actual) / len(actual)
        sst = sum((float(a) - mean_actual) ** 2 for a in actual)
        if abs(sst) < 1e-12:
            r_squared = float("nan")
        else:
            sse = sum((float(a) - float(p)) ** 2 for a, p in zip(actual, predicted))
            r_squared = 1 - sse / sst
        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "R^2": r_squared,
        }

    def _label(self, method: str) -> str:
        aliases = {
            "linear_regression": "Linear Regression Forecast",
            "cagr": "CAGR Projection",
            "moving_average": "Moving Average Forecast",
            "rolling_mean": "Moving Average Forecast",
            "seasonal": "Seasonal Trend Forecast",
            "seasonality": "Seasonal Trend Forecast",
            "seasonal_trend": "Seasonal Trend Forecast",
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
        self.metadata["prompt"] = prompt
        response = self._invoke_model(prompt)
        if response:
            filtered = self._filter_response(prompt, response)
            if filtered is not None:
                audited = self._audit_pharma_management_alignment(filtered)
                if audited is not None:
                    self.metadata["status"] = "model_response"
                    self.metadata["response"] = audited
                    return audited

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
                            {
                                "role": "system",
                                "content": (
                                    "Prioritise pharmaceutical management best practices: "
                                    "patient safety, GMP quality controls, regulatory readiness, "
                                    "supply resilience, and prudent working-capital stewardship."
                                ),
                            },
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
                            {
                                "role": "system",
                                "content": (
                                    "Prioritise pharmaceutical management best practices: "
                                    "patient safety, GMP quality controls, regulatory readiness, "
                                    "supply resilience, and prudent working-capital stewardship."
                                ),
                            },
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

    def _filter_response(self, prompt: str, response: str) -> Optional[str]:
        allowed_numbers = self._extract_numbers(prompt)
        response_numbers = self._extract_numbers(response)
        if not response_numbers:
            self.metadata["numeric_fidelity"] = "no_numbers"
            return response
        for number in response_numbers:
            if not self._number_in_set(number, allowed_numbers):
                self.metadata["numeric_fidelity"] = "failed"
                self.metadata["numeric_fidelity_details"] = {
                    "unexpected_number": number,
                    "allowed_samples": allowed_numbers[:10],
                }
                return None
        self.metadata["numeric_fidelity"] = "passed"
        return response

    def _audit_pharma_management_alignment(self, response: str) -> Optional[str]:
        required_domains = {
            "patient_safety": ("patient", "safety", "pharmacovigilance"),
            "quality_and_gmp": ("gmp", "quality", "validation", "batch"),
            "regulatory": ("regulatory", "compliance", "inspection", "authority"),
            "supply_continuity": ("supply", "inventory", "continuity", "shortage"),
        }
        lowered = response.lower()
        covered = {
            domain: any(keyword in lowered for keyword in keywords)
            for domain, keywords in required_domains.items()
        }
        self.metadata["pharma_management_audit"] = covered
        self.metadata["pharma_management_audit_status"] = (
            "passed" if all(covered.values()) else "failed"
        )
        if all(covered.values()):
            return response
        self.metadata["warning"] = (
            "Model response did not cover all pharmaceutical management practice domains; "
            "using deterministic best-practice fallback."
        )
        return None

    def _extract_numbers(self, text: str) -> List[float]:
        matches = re.findall(r"[-+]?(?:\\d+\\.?\\d*|\\d*\\.\\d+)", text)
        numbers: List[float] = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        return numbers

    def _number_in_set(self, value: float, allowed: Sequence[float]) -> bool:
        for candidate in allowed:
            if math.isclose(value, candidate, rel_tol=0.02, abs_tol=1.0):
                return True
        return False

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

        lines.extend(
            [
                "- Pharmaceutical management practice review:",
                "  - Patient safety and product-quality controls should remain the first operational priority.",
                "  - GMP readiness requires documented deviations, validated processes, and audit-ready records.",
                "  - Regulatory planning should include proactive authority engagement and submission timelines.",
                "  - Supply resilience should be monitored via inventory cover, critical suppliers, and shortage alerts.",
            ]
        )

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
