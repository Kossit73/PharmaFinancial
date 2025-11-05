"""Streamlit web application for the Longevity Pharmaceuticals financial model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd
import plotly.express as px
import streamlit as st

from .inputs import ModelInputs, load_inputs, parse_inputs
from .model import FinancialModel, FinancialOutputs

DEFAULT_INPUT_PATH = Path(__file__).resolve().parent / "data" / "default_inputs.json"
DEFAULT_INPUT_JSON = DEFAULT_INPUT_PATH.read_text(encoding="utf-8")

_SESSION_INPUTS_KEY = "model_inputs"
_SESSION_INPUT_SOURCE_KEY = "model_inputs_source"
_SESSION_FEEDBACK_KEY = "model_inputs_feedback"
_UPLOAD_WIDGET_KEY = "custom_assumptions_file"


def main() -> None:
    st.set_page_config(
        page_title="Longevity Pharmaceuticals Financial Model",
        page_icon="💊",
        layout="wide",
    )

    st.title("Longevity Pharmaceuticals Financial Model")
    st.caption(
        "Interactive financial modelling environment covering statements, "
        "scenario analysis, and Monte Carlo simulation."
    )

    tabs = st.tabs(
        [
            "Input Landing Page",
            "Key Metrics Dashboard",
            "Financial Performance",
            "Financial Position",
            "Cash Flow Statement",
            "Sensitivity Analysis",
            "Scenario / IFs Analysis",
            "Monte Carlo Simulation",
            "Break-even & Payback",
        ]
    )

    with tabs[0]:
        inputs = _render_inputs_tab()

    outputs = FinancialModel(inputs).run()

    with tabs[1]:
        _render_dashboard_tab(outputs)
    with tabs[2]:
        _render_statement_tab("Statement of Financial Performance", outputs.income_statement)
    with tabs[3]:
        _render_statement_tab("Statement of Financial Position", outputs.balance_sheet)
    with tabs[4]:
        _render_statement_tab("Statement of Cash Flows", outputs.cash_flow)
    with tabs[5]:
        _render_sensitivity(outputs)
    with tabs[6]:
        _render_scenarios(outputs)
    with tabs[7]:
        _render_monte_carlo(outputs)
    with tabs[8]:
        _render_break_even(outputs)


def _ensure_inputs_initialized() -> None:
    if _SESSION_INPUTS_KEY not in st.session_state:
        st.session_state[_SESSION_INPUTS_KEY] = load_inputs(DEFAULT_INPUT_PATH)
        st.session_state[_SESSION_INPUT_SOURCE_KEY] = "default"


def _render_inputs_tab() -> ModelInputs:
    _ensure_inputs_initialized()
    inputs: ModelInputs = st.session_state[_SESSION_INPUTS_KEY]

    st.subheader("Core Assumptions")
    assumption_rows = [
        {
            "Product": name.title(),
            "Production Cost": params.production_cost,
            "Selling Price": params.selling_price,
            "Freight Cost": params.freight_cost,
            "Markup": params.markup,
        }
        for name, params in inputs.unit_costs.items()
    ]
    st.dataframe(pd.DataFrame(assumption_rows), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Direct Labour Structure")
        st.dataframe(
            _dict_to_dataframe(inputs.direct_labor_costs, "Role", "Annual Cost"),
            use_container_width=True,
        )
    with col2:
        st.markdown("### Indirect Labour Structure")
        st.dataframe(
            _dict_to_dataframe(inputs.indirect_labor_costs, "Role", "Annual Cost"),
            use_container_width=True,
        )

    st.markdown("### Utility Schedule")
    utility_df = pd.DataFrame(
        {
            "Year": inputs.years,
            "Operating Days": inputs.utility_schedule.operating_days,
            "Operating Hours": inputs.utility_schedule.operating_hours,
        }
    )
    st.dataframe(utility_df, use_container_width=True)

    st.markdown("### Model Configuration")
    st.write("Upload a customised JSON assumptions file or use the bundled defaults.")

    feedback = st.session_state.pop(_SESSION_FEEDBACK_KEY, None)
    if feedback is not None:
        level, message = feedback
        if level == "success":
            st.success(message)
        else:
            st.error(message)

    uploaded = st.file_uploader(
        "Custom assumptions (JSON)",
        type="json",
        accept_multiple_files=False,
        key=_UPLOAD_WIDGET_KEY,
        on_change=_load_custom_inputs,
    )

    current_source = st.session_state.get(_SESSION_INPUT_SOURCE_KEY, "default")
    if current_source == "default":
        st.caption("Using default assumptions bundled with the project.")
    else:
        st.caption(f"Using assumptions from {current_source}.")

    st.download_button(
        label="Download default JSON",
        data=DEFAULT_INPUT_JSON,
        file_name="default_inputs.json",
        mime="application/json",
    )

    st.button(
        "Use bundled default assumptions",
        on_click=_reset_inputs_to_default,
    )

    return st.session_state[_SESSION_INPUTS_KEY]


def _load_custom_inputs() -> None:
    uploaded = st.session_state.get(_UPLOAD_WIDGET_KEY)
    if uploaded is None:
        return

    try:
        try:
            raw = json.loads(uploaded.getvalue().decode("utf-8"))
        except json.JSONDecodeError as exc:
            st.session_state[_SESSION_FEEDBACK_KEY] = (
                "error",
                f"Invalid JSON file: {exc}",
            )
            return

        try:
            inputs = parse_inputs(raw)
        except Exception as exc:  # pragma: no cover - user supplied input
            st.session_state[_SESSION_FEEDBACK_KEY] = (
                "error",
                f"Unable to parse inputs: {exc}",
            )
            return

        st.session_state[_SESSION_INPUTS_KEY] = inputs
        st.session_state[_SESSION_INPUT_SOURCE_KEY] = getattr(
            uploaded, "name", "uploaded file"
        )
        st.session_state[_SESSION_FEEDBACK_KEY] = (
            "success",
            "Loaded custom assumptions.",
        )
    finally:
        st.session_state.pop(_UPLOAD_WIDGET_KEY, None)


def _reset_inputs_to_default() -> None:
    st.session_state[_SESSION_INPUTS_KEY] = load_inputs(DEFAULT_INPUT_PATH)
    st.session_state[_SESSION_INPUT_SOURCE_KEY] = "default"
    st.session_state[_SESSION_FEEDBACK_KEY] = (
        "success",
        "Reverted to default assumptions.",
    )
    st.session_state.pop(_UPLOAD_WIDGET_KEY, None)


def _render_dashboard_tab(outputs: FinancialOutputs) -> None:
    income = _with_year(outputs.income_statement)

    col1, col2 = st.columns(2)
    with col1:
        fig_revenue = px.line(income, x="Year", y="Net Revenue", title="Net Revenue")
        st.plotly_chart(fig_revenue, use_container_width=True)
    with col2:
        fig_ebitda = px.line(income, x="Year", y="EBITDA", title="EBITDA")
        st.plotly_chart(fig_ebitda, use_container_width=True)

    st.markdown("### Investment Metrics")
    metrics = outputs.summary_metrics["Value"]
    metric_cols = st.columns(len(metrics))
    for col, (name, value) in zip(metric_cols, metrics.items()):
        with col:
            formatted = _format_number(value)
            st.metric(label=name, value=formatted)


def _render_statement_tab(title: str, df: pd.DataFrame) -> None:
    st.subheader(title)
    st.dataframe(_with_year(df), use_container_width=True)


def _render_sensitivity(outputs: FinancialOutputs) -> None:
    st.subheader("Sensitivity Analysis")
    if not outputs.sensitivity_results:
        st.info("No sensitivity configurations provided in the assumptions file.")
        return

    for variable, df in outputs.sensitivity_results.items():
        st.markdown(f"#### {variable}")
        st.dataframe(_with_year(df), use_container_width=True)


def _render_scenarios(outputs: FinancialOutputs) -> None:
    st.subheader("Scenario / IFs Analysis")
    for name, df in outputs.scenario_results.items():
        st.markdown(f"#### {name}")
        st.dataframe(_with_year(df), use_container_width=True)


def _render_monte_carlo(outputs: FinancialOutputs) -> None:
    st.subheader("Monte Carlo Simulation")
    fig = px.histogram(outputs.monte_carlo, x="NPV", nbins=40, title="NPV Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(outputs.monte_carlo.describe().T, use_container_width=True)


def _render_break_even(outputs: FinancialOutputs) -> None:
    st.subheader("Break-even Analysis")
    st.dataframe(outputs.break_even.reset_index().rename(columns={"index": "Product"}), use_container_width=True)

    st.markdown("### Payback Schedule")
    st.dataframe(_with_year(outputs.payback), use_container_width=True)

    st.markdown("### Discounted Payback Schedule")
    st.dataframe(_with_year(outputs.discounted_payback), use_container_width=True)


def _dict_to_dataframe(data: Mapping[str, float], index_label: str, value_label: str) -> pd.DataFrame:
    return (
        pd.DataFrame(list(data.items()), columns=[index_label, value_label])
        .sort_values(index_label)
        .reset_index(drop=True)
    )


def _with_year(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    if "Year" not in table.columns and not isinstance(df.index, pd.RangeIndex):
        table.insert(0, "Year", list(df.index))
    return table.reset_index(drop=True)


def _format_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:,.2f}K"
    return f"{value:,.2f}"


if __name__ == "__main__":  # pragma: no cover - Streamlit executes the script directly
    main()
