"""Streamlit web application for the Longevity Pharmaceuticals financial model."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import List, Tuple

import streamlit as st

from .inputs import ModelInputs, parse_inputs
from .model import FinancialModel, FinancialOutputs
from .table import Table

try:  # pragma: no cover - executed in environments with pandas available
    import pandas as pd
except Exception:  # pragma: no cover - fallback for environments without pandas
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependency for charting
    import plotly.express as px
except Exception:  # pragma: no cover - gracefully degrade when Plotly missing
    px = None  # type: ignore

DEFAULT_INPUT_PATH = Path(__file__).resolve().parent / "data" / "default_inputs.json"
DEFAULT_INPUT_JSON = DEFAULT_INPUT_PATH.read_text(encoding="utf-8")


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
        file_bytes = uploaded.getvalue()
        signature = f"{getattr(uploaded, 'name', 'upload')}:{hashlib.md5(file_bytes).hexdigest()}"
        if st.session_state.get("uploaded_signature") != signature:
            try:
                raw = json.loads(file_bytes.decode("utf-8"))
                _initialise_session_payload(raw)
                parse_inputs(raw)
                st.session_state["uploaded_signature"] = signature
                st.sidebar.success("Loaded custom assumptions.")
            except json.JSONDecodeError as exc:
                st.sidebar.error(f"Invalid JSON file: {exc}")
            except Exception as exc:  # pragma: no cover - user supplied input
                st.sidebar.error(f"Unable to parse inputs: {exc}")

    if st.session_state.get("uploaded_signature"):
        st.sidebar.caption(
            "Using uploaded assumptions. Adjust the tables below to update the model."
        )
    else:
        st.sidebar.caption("Using default assumptions bundled with the project.")
    st.sidebar.download_button(
        label="Download default JSON",
        data=DEFAULT_INPUT_JSON,
        file_name="default_inputs.json",
        mime="application/json",
    )
    if "input_payload" not in st.session_state:
        _initialise_session_payload(json.loads(DEFAULT_INPUT_JSON))

    payload = st.session_state["input_payload"]
    rows = st.session_state.setdefault(
        "core_assumption_rows", _payload_to_core_rows(payload)
    )
    _core_rows_to_payload(rows, payload)

    return parse_inputs(payload)


def _render_inputs_tab(inputs: ModelInputs) -> None:
    st.subheader("Core Assumptions")
    rows: List[dict] = st.session_state.get("core_assumption_rows", [])

    if not rows:
        st.info("No core assumptions configured. Use the form below to add entries.")

    updated_rows: list[dict] = []
    for index, row in enumerate(rows):
        container = st.container()
        with container:
            cols = st.columns([3, 2, 2, 2, 2, 1])
            description = cols[0].text_input(
                "Description",
                value=row.get("Product", ""),
                key=f"core_desc_{index}",
                help="Name of the product or assumption this row represents.",
            )
            production = cols[1].number_input(
                "Production Cost",
                value=float(row.get("Production Cost", 0.0)),
                key=f"core_prod_{index}",
                step=0.001,
                format="%.4f",
            )
            selling = cols[2].number_input(
                "Selling Price",
                value=float(row.get("Selling Price", 0.0)),
                key=f"core_sell_{index}",
                step=0.001,
                format="%.4f",
            )
            freight = cols[3].number_input(
                "Freight Cost",
                value=float(row.get("Freight Cost", 0.0)),
                key=f"core_freight_{index}",
                step=0.001,
                format="%.4f",
            )
            markup = cols[4].number_input(
                "Markup",
                value=float(row.get("Markup", 0.0)),
                key=f"core_markup_{index}",
                step=0.01,
                format="%.2f",
            )
            if cols[5].button("Remove", key=f"core_remove_{index}"):
                del rows[index]
                st.session_state["core_assumption_rows"] = rows
                st.experimental_rerun()

        updated_rows.append(
            {
                "Product": description.strip(),
                "Production Cost": production,
                "Selling Price": selling,
                "Freight Cost": freight,
                "Markup": markup,
            }
        )

    if updated_rows != rows:
        st.session_state["core_assumption_rows"] = updated_rows

    st.markdown("#### Add a core assumption")
    with st.form("add_core_assumption"):
        new_description = st.text_input(
            "Description", key="core_new_description", help="Label for the new row."
        )
        new_production = st.number_input(
            "Production Cost", value=0.0, step=0.001, format="%.4f", key="core_new_prod"
        )
        new_selling = st.number_input(
            "Selling Price", value=0.0, step=0.001, format="%.4f", key="core_new_sell"
        )
        new_freight = st.number_input(
            "Freight Cost", value=0.0, step=0.001, format="%.4f", key="core_new_freight"
        )
        new_markup = st.number_input(
            "Markup", value=0.0, step=0.01, format="%.2f", key="core_new_markup"
        )
        submitted = st.form_submit_button("Add")

    if submitted:
        if not new_description.strip():
            st.warning("Description is required to add a core assumption.")
        else:
            rows.append(
                {
                    "Product": new_description.strip(),
                    "Production Cost": new_production,
                    "Selling Price": new_selling,
                    "Freight Cost": new_freight,
                    "Markup": new_markup,
                }
            )
            st.session_state["core_assumption_rows"] = rows
            # Clearing the form widgets by removing their stored state avoids
            # manipulating widget-managed keys directly, which previously
            # triggered ``StreamlitAPIException`` in bare-mode executions.
            for key in (
                "core_new_description",
                "core_new_prod",
                "core_new_sell",
                "core_new_freight",
                "core_new_markup",
            ):
                st.session_state.pop(key, None)
            st.experimental_rerun()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Direct Labour Structure")
        st.dataframe(
            _ensure_dataframe(_dict_to_dataframe(inputs.direct_labor_costs, "Role", "Annual Cost")),
            use_container_width=True,
        )
    with col2:
        st.markdown("### Indirect Labour Structure")
        st.dataframe(
            _ensure_dataframe(_dict_to_dataframe(inputs.indirect_labor_costs, "Role", "Annual Cost")),
            use_container_width=True,
        )

    st.markdown("### Utility Schedule")
    utility_rows = [
        {
            "Year": year,
            "Operating Days": days,
            "Operating Hours": hours,
        }
        for year, days, hours in zip(
            inputs.years,
            inputs.utility_schedule.operating_days,
            inputs.utility_schedule.operating_hours,
        )
    ]
    st.dataframe(_ensure_dataframe(utility_rows), use_container_width=True)


def _render_dashboard_tab(outputs: FinancialOutputs) -> None:
    income = _with_year(outputs.income_statement)

    if px is None or pd is None:
        st.warning(
            "Plotly visualisations unavailable. Displaying financial metrics as tables instead."
        )
        st.dataframe(income, use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig_revenue = px.line(income, x="Year", y="Net Revenue", title="Net Revenue")
            st.plotly_chart(fig_revenue, use_container_width=True)
        with col2:
            fig_ebitda = px.line(income, x="Year", y="EBITDA", title="EBITDA")
            st.plotly_chart(fig_ebitda, use_container_width=True)

    st.markdown("### Investment Metrics")
    metric_pairs = _extract_metric_pairs(outputs.summary_metrics)
    if not metric_pairs:
        st.info("No investment metrics were generated for the current assumptions.")
        return

    metric_cols = st.columns(len(metric_pairs))
    for col, (name, value) in zip(metric_cols, metric_pairs):
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
    monte_carlo_df = _ensure_dataframe(outputs.monte_carlo)
    if px is None or pd is None:
        st.warning("Plotly unavailable. Displaying Monte Carlo results in tabular form.")
        st.dataframe(monte_carlo_df, use_container_width=True)
    else:
        fig = px.histogram(monte_carlo_df, x="NPV", nbins=40, title="NPV Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(monte_carlo_df.describe().T, use_container_width=True)


def _render_break_even(outputs: FinancialOutputs) -> None:
    st.subheader("Break-even Analysis")
    break_even_df = _ensure_dataframe(outputs.break_even)
    if pd is not None:
        break_even_df = break_even_df.reset_index().rename(columns={"index": "Product"})
    st.dataframe(break_even_df, use_container_width=True)

    st.markdown("### Payback Schedule")
    st.dataframe(_with_year(outputs.payback), use_container_width=True)

    st.markdown("### Discounted Payback Schedule")
    st.dataframe(_with_year(outputs.discounted_payback), use_container_width=True)


def _dict_to_dataframe(data: Mapping[str, float], index_label: str, value_label: str):
    if pd is None:
        return [
            {index_label: key, value_label: value}
            for key, value in sorted(data.items(), key=lambda item: item[0])
        ]
    return (
        pd.DataFrame(list(data.items()), columns=[index_label, value_label])
        .sort_values(index_label)
        .reset_index(drop=True)
    )


def _with_year(table) -> "pd.DataFrame | Table | list":
    frame = _ensure_dataframe(table)
    if pd is None:
        return frame
    result = frame.copy()
    if "Year" not in result.columns and not isinstance(frame.index, pd.RangeIndex):
        result.insert(0, "Year", list(frame.index))
    return result.reset_index(drop=True)


def _ensure_dataframe(table) -> "pd.DataFrame | list":
    if isinstance(table, list):
        if pd is None:
            return table
        return pd.DataFrame(table)
    if isinstance(table, Table):
        if pd is None:
            rows = []
            data = table.as_dict()
            for idx, label in enumerate(table.index):
                row = {table.index_name: label}
                for column, values in data.items():
                    row[column] = values[idx]
                rows.append(row)
            return rows
        return table.to_frame()
    if hasattr(table, "to_frame"):
        try:
            return table.to_frame()
        except Exception:
            pass
    return table


def _format_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:,.2f}K"
    return f"{value:,.2f}"


def _extract_metric_pairs(summary) -> Sequence[Tuple[str, float]]:
    if isinstance(summary, Table):
        return list(zip([str(label) for label in summary.index], summary.column("Value")))

    if pd is not None and hasattr(summary, "reset_index"):
        try:
            frame = summary.reset_index()
        except Exception:
            frame = pd.DataFrame(summary)
        label_column = summary.index.name if getattr(summary, "index", None) is not None else None
        if not label_column or label_column not in frame.columns:
            label_column = frame.columns[0]
        value_column = "Value" if "Value" in frame.columns else frame.columns[-1]
        return list(zip(frame[label_column].astype(str), frame[value_column].astype(float)))

    if isinstance(summary, list):
        pairs: list[Tuple[str, float]] = []
        for position, row in enumerate(summary, start=1):
            if isinstance(row, Mapping):
                label = row.get("Metric") or row.get("Year") or f"Metric {position}"
                value = float(row.get("Value", float("nan")))
                pairs.append((str(label), value))
        return pairs

    if isinstance(summary, Mapping):
        value = summary.get("Value")
        if isinstance(value, Mapping):
            return [(str(name), float(val)) for name, val in value.items()]

    return []


def _initialise_session_payload(payload: dict) -> None:
    st.session_state["input_payload"] = payload
    st.session_state["core_assumption_rows"] = _payload_to_core_rows(payload)


def _payload_to_core_rows(payload: Mapping) -> list[dict]:
    unit_costs = payload.get("unit_costs", {})
    markup = payload.get("markup", {})
    rows: list[dict] = []
    for name, values in unit_costs.items():
        rows.append(
            {
                "Product": str(name),
                "Production Cost": float(values.get("production", 0.0)),
                "Selling Price": float(values.get("price", 0.0)),
                "Freight Cost": float(values.get("freight", 0.0)),
                "Markup": float(markup.get(name, 0.0)),
            }
        )
    return rows


def _core_rows_to_payload(rows: Sequence[Mapping], payload: dict) -> None:
    unit_costs: dict[str, dict[str, float]] = {}
    markup: dict[str, float] = {}
    years = payload.get("years", [])
    existing_estimate = payload.get("production_estimate", {})
    production_estimate: dict[str, list[float]] = {}

    for row in rows:
        name = str(row.get("Product", "")).strip()
        if not name:
            continue
        unit_costs[name] = {
            "production": float(row.get("Production Cost", 0.0)),
            "price": float(row.get("Selling Price", 0.0)),
            "freight": float(row.get("Freight Cost", 0.0)),
        }
        markup[name] = float(row.get("Markup", 0.0))
        if isinstance(existing_estimate, Mapping) and name in existing_estimate:
            production_estimate[name] = list(existing_estimate[name])
        else:
            production_estimate[name] = [0.0 for _ in years]

    payload["unit_costs"] = unit_costs
    payload["markup"] = markup
    if production_estimate:
        payload["production_estimate"] = production_estimate


if __name__ == "__main__":  # pragma: no cover - Streamlit executes the script directly
    main()
