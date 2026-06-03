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


def _inject_app_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --pharma-ink: #0f172a;
            --pharma-muted: #475569;
            --pharma-brand: #1d4ed8;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(191, 219, 254, 0.30), transparent 34%),
                radial-gradient(circle at top right, rgba(196, 181, 253, 0.16), transparent 28%),
                linear-gradient(180deg, #f6f9ff 0%, #f4f7fb 58%, #eef4ff 100%);
        }
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }
        .designer-hero {
            margin-bottom: 1.2rem;
            padding: 1.8rem 1.9rem;
            border-radius: 28px;
            border: 1px solid rgba(29, 78, 216, 0.12);
            background:
                linear-gradient(135deg, rgba(233, 240, 255, 0.96), rgba(255, 255, 255, 0.94)),
                linear-gradient(135deg, rgba(29, 78, 216, 0.05), rgba(56, 189, 248, 0.06));
            box-shadow: 0 24px 48px rgba(15, 23, 42, 0.08);
        }
        .designer-kicker {
            margin: 0 0 0.45rem 0;
            font-size: 0.78rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--pharma-brand);
            font-weight: 700;
        }
        .designer-title {
            margin: 0;
            font-size: clamp(2rem, 2.8vw, 3.15rem);
            line-height: 1.02;
            color: var(--pharma-ink);
            font-weight: 800;
        }
        .designer-copy {
            max-width: 55rem;
            margin: 0.7rem 0 0 0;
            color: var(--pharma-muted);
            font-size: 1rem;
            line-height: 1.6;
        }
        .designer-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }
        .designer-badge {
            padding: 0.42rem 0.78rem;
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.92);
            color: var(--pharma-brand);
            font-size: 0.82rem;
            font-weight: 700;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.55rem;
            margin-bottom: 1rem;
        }
        div[data-baseweb="tab-list"] button {
            min-height: 3rem;
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.72);
            color: var(--pharma-muted);
            padding: 0.25rem 1rem;
        }
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            background: linear-gradient(135deg, #1d4ed8, #0891b2);
            color: white;
            border-color: transparent;
            box-shadow: 0 12px 24px rgba(8, 145, 178, 0.16);
        }
        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"] {
            border-radius: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_model_hero() -> None:
    badges = "".join(
        f'<span class="designer-badge">{label}</span>'
        for label in (
            "Scenario dashboards",
            "Financial statements",
            "Monte Carlo insights",
            "Executive layout",
        )
    )
    st.markdown(
        f"""
        <section class="designer-hero">
            <p class="designer-kicker">Pharma planning suite</p>
            <h1 class="designer-title">Longevity Pharmaceuticals Financial Model</h1>
            <p class="designer-copy">
                Review revenue, operations, financing, and risk assumptions in a cleaner executive shell
                built for management review and investor presentation.
            </p>
            <div class="designer-badges">{badges}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Longevity Pharmaceuticals Financial Model",
        page_icon="💊",
        layout="wide",
    )
    _inject_app_theme()
    _render_model_hero()

    inputs = _resolve_inputs()
    outputs = FinancialModel(inputs).run()

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
        _render_inputs_tab(inputs)
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


def _resolve_inputs() -> ModelInputs:
    st.sidebar.header("Model Configuration")
    st.sidebar.write(
        "Upload a customised JSON assumptions file or use the bundled defaults."
    )

    uploaded = st.sidebar.file_uploader(
        "Custom assumptions (JSON)", type="json", accept_multiple_files=False
    )
    if uploaded is not None:
        try:
            raw = json.loads(uploaded.getvalue().decode("utf-8"))
            inputs = parse_inputs(raw)
            st.sidebar.success("Loaded custom assumptions.")
            return inputs
        except json.JSONDecodeError as exc:
            st.sidebar.error(f"Invalid JSON file: {exc}")
        except Exception as exc:  # pragma: no cover - user supplied input
            st.sidebar.error(f"Unable to parse inputs: {exc}")

    st.sidebar.caption("Using default assumptions bundled with the project.")
    st.sidebar.download_button(
        label="Download default JSON",
        data=DEFAULT_INPUT_JSON,
        file_name="default_inputs.json",
        mime="application/json",
    )
    return load_inputs(DEFAULT_INPUT_PATH)


def _render_inputs_tab(inputs: ModelInputs) -> None:
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
