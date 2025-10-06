"""Streamlit web application for the Longevity Pharmaceuticals financial model."""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
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
DEFAULT_RISK_CATEGORIES = ["inherent", "climate", "political"]


def _streamlit_runtime_exists() -> bool:
    """Return ``True`` when the Streamlit runtime has been initialised."""

    try:  # pragma: no cover - depends on Streamlit internals
        from streamlit.runtime import exists
    except Exception:  # pragma: no cover - runtime API unavailable
        return False

    try:  # pragma: no cover - defensive against older Streamlit versions
        return bool(exists())
    except Exception:
        return False


def main() -> None:
    if not _streamlit_runtime_exists():  # pragma: no cover - requires Streamlit runner
        raise RuntimeError(
            "Streamlit runtime is not initialised. Launch the app with "
            "`streamlit run streamlit_app.py` to enable interactive inputs."
        )

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

    inflation_rows = st.session_state.setdefault(
        "inflation_rows", _payload_to_inflation_rows(payload)
    )
    _inflation_rows_to_payload(inflation_rows, payload)

    risk_rows = st.session_state.setdefault(
        "risk_rows", _payload_to_risk_rows(payload)
    )
    _risk_rows_to_payload(risk_rows, payload)

    return parse_inputs(payload)


def _render_inputs_tab(inputs: ModelInputs) -> None:
    payload = st.session_state["input_payload"]

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
            for key in (
                "core_new_description",
                "core_new_prod",
                "core_new_sell",
                "core_new_freight",
                "core_new_markup",
            ):
                st.session_state.pop(key, None)
            st.experimental_rerun()

    st.markdown("### Direct Labour Structure")
    _render_labor_section("direct", "direct_labor_rows", payload)

    st.markdown("### Indirect Labour Structure")
    _render_labor_section("indirect", "indirect_labor_rows", payload)

    st.markdown("### Utility Schedule")
    _render_utility_schedule(payload)

    st.markdown("### Cost & Financing Assumptions")
    _render_cost_and_financing(payload)

    st.markdown("### Tax Schedule")
    _render_tax_schedule(payload)

    st.markdown("### Inflation Schedule")
    _render_inflation_schedule(payload)

    st.markdown("### Risk Schedule")
    _render_risk_schedule(payload)

    st.markdown("### Sensitivity Analysis Configuration")
    _render_sensitivity_inputs(payload)

    st.markdown("### Scenario / IFs Configuration")
    _render_scenario_inputs(payload)

    st.markdown("### Monte Carlo Simulation")
    _render_monte_carlo_inputs(payload)

    _core_rows_to_payload(st.session_state.get("core_assumption_rows", []), payload)
    _risk_rows_to_payload(st.session_state.get("risk_rows", []), payload)
    _inflation_rows_to_payload(st.session_state.get("inflation_rows", []), payload)
    st.session_state["input_payload"] = payload


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


def _mapping_to_rows(mapping: Mapping[str, float], key_label: str, value_label: str) -> list[dict]:
    return [
        {key_label: str(name), value_label: float(cost)}
        for name, cost in mapping.items()
    ]


def _payload_to_sensitivity_rows(payload: Mapping) -> list[dict]:
    variables = (
        payload.get("sensitivity", {}).get("variables", {})
        if isinstance(payload.get("sensitivity"), Mapping)
        else {}
    )
    rows: list[dict] = []
    for name, values in variables.items():
        numeric = [float(value) for value in values]
        rows.append({"Variable": str(name), "Values": numeric})
    return rows


def _ensure_schedule_length(values: Iterable[float], length: int, fill: float = 0.0) -> List[float]:
    sequence = [float(value) for value in values]
    if length <= 0:
        return sequence
    if len(sequence) < length:
        sequence += [fill for _ in range(length - len(sequence))]
    return sequence[:length]


def _parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    if not text:
        return values
    for token in text.replace("\n", ",").split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return values


def _format_float_list(values: Iterable[float]) -> str:
    return ", ".join(f"{float(value):.4f}" for value in values)


def _render_labor_section(section: str, state_key: str, payload: dict) -> None:
    rows: list[dict] = st.session_state.get(state_key, [])
    updated: list[dict] = []
    for index, row in enumerate(rows):
        cols = st.columns([3, 2, 1])
        role = cols[0].text_input(
            "Role",
            value=row.get("Role", ""),
            key=f"{state_key}_role_{index}",
        )
        cost = cols[1].number_input(
            "Annual Cost",
            value=float(row.get("Annual Cost", 0.0)),
            key=f"{state_key}_cost_{index}",
            step=0.001,
            format="%.4f",
        )
        if cols[2].button("Remove", key=f"{state_key}_remove_{index}"):
            del rows[index]
            st.session_state[state_key] = rows
            st.experimental_rerun()
        updated.append({"Role": role.strip(), "Annual Cost": cost})

    if updated != rows:
        st.session_state[state_key] = updated

    with st.form(f"add_{state_key}"):
        new_role = st.text_input("Role", key=f"{state_key}_new_role")
        new_cost = st.number_input(
            "Annual Cost",
            value=0.0,
            step=0.001,
            format="%.4f",
            key=f"{state_key}_new_cost",
        )
        submitted = st.form_submit_button("Add")

    if submitted:
        if not new_role.strip():
            st.warning("Role is required to add a labour cost entry.")
        else:
            rows.append({"Role": new_role.strip(), "Annual Cost": new_cost})
            st.session_state[state_key] = rows
            for key in (f"{state_key}_new_role", f"{state_key}_new_cost"):
                st.session_state.pop(key, None)
            st.experimental_rerun()

    labor = payload.setdefault("labor", {})
    labor[section] = {
        row["Role"]: row["Annual Cost"]
        for row in st.session_state.get(state_key, [])
        if row.get("Role")
    }


def _render_utility_schedule(payload: dict) -> None:
    utility = payload.setdefault("utility_costs", {})
    electricity = utility.get("electricity_per_day", 0.0)
    water = utility.get("water_per_day", 0.0)
    steam = utility.get("steam_per_hour", 0.0)
    cols = st.columns(3)
    utility["electricity_per_day"] = cols[0].number_input(
        "Electricity per day",
        value=float(electricity),
        step=1.0,
        format="%.4f",
        key="utility_electricity",
    )
    utility["water_per_day"] = cols[1].number_input(
        "Water per day",
        value=float(water),
        step=1.0,
        format="%.4f",
        key="utility_water",
    )
    utility["steam_per_hour"] = cols[2].number_input(
        "Steam per hour",
        value=float(steam),
        step=0.1,
        format="%.4f",
        key="utility_steam",
    )

    years = payload.get("years", [])
    days = _ensure_schedule_length(utility.get("days", []), len(years), fill=0)
    hours = _ensure_schedule_length(utility.get("hours", []), len(years), fill=0)

    for index, year in enumerate(years):
        cols = st.columns([1, 2, 2])
        cols[0].markdown(f"**{year}**")
        days[index] = int(
            cols[1].number_input(
                "Operating Days",
                value=int(days[index]),
                key=f"utility_days_{index}",
                min_value=0,
            )
        )
        hours[index] = int(
            cols[2].number_input(
                "Operating Hours",
                value=int(hours[index]),
                key=f"utility_hours_{index}",
                min_value=0,
            )
        )

    utility["days"] = days
    utility["hours"] = hours


def _render_cost_and_financing(payload: dict) -> None:
    raw = payload.setdefault("raw_material_cost", {})
    raw["per_unit"] = st.number_input(
        "Raw material cost per unit",
        value=float(raw.get("per_unit", 0.0)),
        step=0.0001,
        format="%.4f",
        key="raw_material_per_unit",
    )
    annual_text = st.text_area(
        "Annual raw material spend (comma separated, optional)",
        value=_format_float_list(raw.get("annual", [])),
        key="raw_material_annual",
    )
    try:
        raw["annual"] = _parse_float_list(annual_text)
    except ValueError as exc:
        st.warning(f"Raw material schedule ignored: {exc}")

    financing = payload.setdefault("financing", {})
    finance_cols = st.columns(3)
    financing["initial_investment"] = finance_cols[0].number_input(
        "Initial investment",
        value=float(financing.get("initial_investment", 0.0)),
        step=0.1,
        format="%.4f",
        key="finance_initial",
    )
    financing["discount_rate"] = finance_cols[1].number_input(
        "Discount rate",
        value=float(financing.get("discount_rate", 0.0)),
        step=0.001,
        format="%.4f",
        key="finance_discount",
    )
    financing["share_capital"] = finance_cols[2].number_input(
        "Share capital",
        value=float(financing.get("share_capital", 0.0)),
        step=0.1,
        format="%.4f",
        key="finance_share_capital",
    )

    finance_cols = st.columns(3)
    financing["senior_debt_interest"] = finance_cols[0].number_input(
        "Senior debt interest",
        value=float(financing.get("senior_debt_interest", 0.0)),
        step=0.001,
        format="%.4f",
        key="finance_senior_interest",
    )
    financing["revolver_interest"] = finance_cols[1].number_input(
        "Revolver interest",
        value=float(financing.get("revolver_interest", 0.0)),
        step=0.001,
        format="%.4f",
        key="finance_revolver_interest",
    )
    financing["cash_interest"] = finance_cols[2].number_input(
        "Cash interest",
        value=float(financing.get("cash_interest", 0.0)),
        step=0.001,
        format="%.4f",
        key="finance_cash_interest",
    )

    financing["dividend_payout"] = st.number_input(
        "Dividend payout ratio",
        value=float(financing.get("dividend_payout", 0.0)),
        step=0.01,
        format="%.4f",
        key="finance_dividend",
    )


def _render_tax_schedule(payload: dict) -> None:
    tax = payload.setdefault("tax", {})
    years = payload.get("years", [])
    base_rate = st.number_input(
        "Base tax rate",
        value=float(tax.get("rate", 0.0)),
        step=0.01,
        format="%.4f",
        key="tax_base_rate",
    )
    tax["rate"] = base_rate
    tax["timing_adjustment"] = st.number_input(
        "Timing adjustment",
        value=float(tax.get("timing_adjustment", 0.0)),
        step=0.01,
        format="%.4f",
        key="tax_timing",
    )

    schedule = _ensure_schedule_length(tax.get("schedule", []), len(years), fill=base_rate)
    for index, year in enumerate(years):
        schedule[index] = st.number_input(
            f"Tax rate {year}",
            value=float(schedule[index]),
            step=0.01,
            format="%.4f",
            key=f"tax_rate_{index}",
        )
    tax["schedule"] = schedule


def _render_inflation_schedule(payload: dict) -> None:
    rows: list[dict] = st.session_state.get("inflation_rows", [])

    if not rows:
        st.info("No inflation assumptions configured. Use the form below to add entries.")

    header_cols = st.columns([3, 2, 1])
    header_cols[0].markdown("**Years**")
    header_cols[1].markdown("**Rate**")
    header_cols[2].markdown(" ")

    updated_rows: list[dict] = []
    for index, row in enumerate(rows):
        cols = st.columns([3, 2, 1])
        year_label = cols[0].text_input(
            "Years",
            value=row.get("Year", ""),
            key=f"inflation_year_{index}",
        )
        rate_value = cols[1].number_input(
            "Rate",
            value=float(row.get("Rate", 0.0)),
            min_value=0.0,
            step=0.001,
            format="%.4f",
            key=f"inflation_rate_{index}",
        )
        if cols[2].button("Remove", key=f"inflation_remove_{index}"):
            del rows[index]
            st.session_state["inflation_rows"] = rows
            st.experimental_rerun()
        updated_rows.append({"Year": year_label.strip(), "Rate": rate_value})

    if updated_rows != rows:
        st.session_state["inflation_rows"] = updated_rows

    with st.form("add_inflation_row"):
        new_year = st.text_input("Year label", key="inflation_new_year")
        new_rate = st.number_input(
            "Rate",
            value=0.0,
            min_value=0.0,
            step=0.001,
            format="%.4f",
            key="inflation_new_rate",
        )
        submitted = st.form_submit_button("Add")

    if submitted:
        if not new_year.strip():
            st.warning("Year label is required to add an inflation entry.")
        else:
            rows.append({"Year": new_year.strip(), "Rate": new_rate})
            st.session_state["inflation_rows"] = rows
            st.session_state.pop("inflation_new_year", None)
            st.session_state.pop("inflation_new_rate", None)
            st.experimental_rerun()


def _render_risk_schedule(payload: dict) -> None:
    rows = st.session_state.get("risk_rows", [])
    categories = _risk_categories(payload, rows)

    if not rows:
        st.info("No risk assumptions configured. Use the form below to add entries.")

    updated_rows: list[dict] = []
    for index, row in enumerate(rows):
        cols = st.columns([2] + [1 for _ in categories] + [0.6])
        year_label = cols[0].text_input(
            "Year", value=str(row.get("Year", "")), key=f"risk_year_{index}"
        )

        cleaned_row = {"Year": year_label.strip()}
        for position, category in enumerate(categories, start=1):
            value = float(row.get(category, 0.0))
            cleaned_row[category] = cols[position].number_input(
                f"{category.title()} Risk",
                value=value,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.4f",
                key=f"risk_{category}_{index}",
            )

        if cols[-1].button("Remove", key=f"risk_remove_{index}"):
            del rows[index]
            st.session_state["risk_rows"] = rows
            st.experimental_rerun()

        updated_rows.append(cleaned_row)

    if updated_rows != rows:
        st.session_state["risk_rows"] = updated_rows

    with st.form("add_risk_row"):
        new_year = st.text_input("Year", key="risk_new_year")
        new_values: dict[str, float] = {}
        for category in categories:
            new_values[category] = st.number_input(
                f"{category.title()} Risk",
                value=float(st.session_state.get(f"risk_new_{category}", 0.0)),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.4f",
                key=f"risk_new_{category}",
            )
        submitted = st.form_submit_button("Add")

    if submitted:
        if not new_year.strip():
            st.warning("Year label is required to add a risk entry.")
        else:
            rows.append({"Year": new_year.strip(), **new_values})
            st.session_state["risk_rows"] = rows
            st.session_state.pop("risk_new_year", None)
            for category in categories:
                st.session_state.pop(f"risk_new_{category}", None)
            st.experimental_rerun()


def _render_sensitivity_inputs(payload: dict) -> None:
    rows = st.session_state.get("sensitivity_rows", [])
    updated: list[dict] = []
    for index, row in enumerate(rows):
        cols = st.columns([3, 5, 1])
        variable = cols[0].text_input(
            "Variable",
            value=row.get("Variable", ""),
            key=f"sensitivity_var_{index}",
        )
        values_text = cols[1].text_input(
            "Multipliers",
            value=_format_float_list(row.get("Values", [])),
            help="Comma-separated multipliers applied during sensitivity analysis.",
            key=f"sensitivity_vals_{index}",
        )
        if cols[2].button("Remove", key=f"sensitivity_remove_{index}"):
            del rows[index]
            st.session_state["sensitivity_rows"] = rows
            st.experimental_rerun()
        try:
            values = _parse_float_list(values_text)
        except ValueError as exc:
            st.warning(f"Sensitivity entry ignored due to invalid number: {exc}")
            values = row.get("Values", [])
        updated.append({"Variable": variable.strip(), "Values": values})

    if updated != rows:
        st.session_state["sensitivity_rows"] = updated

    with st.form("add_sensitivity"):
        new_variable = st.text_input("Variable Name", key="sensitivity_new_variable")
        new_values_text = st.text_input(
            "Multipliers",
            key="sensitivity_new_values",
            help="Comma-separated list such as 0.9, 1.0, 1.1",
        )
        submitted = st.form_submit_button("Add Variable")

    if submitted:
        if not new_variable.strip():
            st.warning("Variable name is required for sensitivity analysis.")
        else:
            try:
                new_values = _parse_float_list(new_values_text)
            except ValueError as exc:
                st.warning(f"Unable to add sensitivity variable: {exc}")
                new_values = []
            if new_values:
                rows.append({"Variable": new_variable.strip(), "Values": new_values})
                st.session_state["sensitivity_rows"] = rows
                for key in ("sensitivity_new_variable", "sensitivity_new_values"):
                    st.session_state.pop(key, None)
                st.experimental_rerun()
            else:
                st.warning("At least one multiplier is required.")

    variables = {
        row["Variable"]: row["Values"]
        for row in st.session_state.get("sensitivity_rows", [])
        if row.get("Variable") and row.get("Values")
    }
    payload.setdefault("sensitivity", {})["variables"] = variables


def _render_scenario_inputs(payload: dict) -> None:
    scenarios = payload.setdefault("scenarios", {})
    updated: dict[str, dict[str, List[float]]] = {}
    scenario_items = list(scenarios.items())
    for index, (name, values) in enumerate(scenario_items):
        with st.expander(f"Scenario: {name}", expanded=False):
            new_name = st.text_input(
                "Scenario Name",
                value=name,
                key=f"scenario_name_{index}",
            )
            inflation_text = st.text_area(
                "Inflation Series",
                value=_format_float_list(values.get("inflation", [])),
                key=f"scenario_inflation_{index}",
            )
            interest_text = st.text_area(
                "Interest Series",
                value=_format_float_list(values.get("interest", [])),
                key=f"scenario_interest_{index}",
            )
            remove = st.checkbox(
                "Remove scenario",
                key=f"scenario_remove_{index}",
                value=False,
            )

        if remove:
            continue

        try:
            inflation_values = _parse_float_list(inflation_text)
            interest_values = _parse_float_list(interest_text)
        except ValueError as exc:
            st.warning(f"Scenario '{name}' ignored due to invalid number: {exc}")
            inflation_values = values.get("inflation", [])
            interest_values = values.get("interest", [])

        key_name = new_name.strip() or name
        updated[key_name] = {
            "inflation": inflation_values,
            "interest": interest_values,
        }

    with st.form("add_scenario"):
        new_name = st.text_input("Scenario Name", key="scenario_new_name")
        new_inflation = st.text_area(
            "Inflation Series",
            key="scenario_new_inflation",
        )
        new_interest = st.text_area(
            "Interest Series",
            key="scenario_new_interest",
        )
        submitted = st.form_submit_button("Add Scenario")

    if submitted:
        if not new_name.strip():
            st.warning("Scenario name is required.")
        else:
            try:
                inflation_values = _parse_float_list(new_inflation)
                interest_values = _parse_float_list(new_interest)
            except ValueError as exc:
                st.warning(f"Unable to add scenario: {exc}")
            else:
                updated[new_name.strip()] = {
                    "inflation": inflation_values,
                    "interest": interest_values,
                }
                for key in (
                    "scenario_new_name",
                    "scenario_new_inflation",
                    "scenario_new_interest",
                ):
                    st.session_state.pop(key, None)
                st.experimental_rerun()

    payload["scenarios"] = updated


def _render_monte_carlo_inputs(payload: dict) -> None:
    monte = payload.setdefault("monte_carlo", {})
    iterations = st.number_input(
        "Iterations",
        min_value=1,
        value=int(monte.get("iterations", 1000)),
        step=10,
        key="monte_iterations",
    )
    growth_range = list(monte.get("revenue_growth_range", [0.05, 0.15]))
    if len(growth_range) < 2:
        growth_range = [0.0, 0.0]
    min_growth = st.number_input(
        "Minimum revenue growth",
        value=float(growth_range[0]),
        format="%.4f",
        key="monte_growth_min",
    )
    max_growth = st.number_input(
        "Maximum revenue growth",
        value=float(growth_range[1]),
        format="%.4f",
        key="monte_growth_max",
    )
    if max_growth < min_growth:
        st.warning("Maximum growth cannot be less than minimum growth. Adjusted automatically.")
        max_growth = min_growth

    metric_options = [
        "NPV",
        "Average Net Income",
        "Average EBITDA",
        "Average Cash Flow",
    ]
    metrics = st.multiselect(
        "Metrics to capture",
        options=metric_options,
        default=[m for m in monte.get("metrics", ["NPV"]) if m in metric_options],
        key="monte_metrics",
    )
    if not metrics:
        metrics = ["NPV"]

    monte["iterations"] = int(iterations)
    monte["revenue_growth_range"] = [float(min_growth), float(max_growth)]
    monte["metrics"] = metrics


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
    st.session_state["direct_labor_rows"] = _mapping_to_rows(
        payload.get("labor", {}).get("direct", {}),
        "Role",
        "Annual Cost",
    )
    st.session_state["indirect_labor_rows"] = _mapping_to_rows(
        payload.get("labor", {}).get("indirect", {}),
        "Role",
        "Annual Cost",
    )
    st.session_state["sensitivity_rows"] = _payload_to_sensitivity_rows(payload)
    st.session_state["inflation_rows"] = _payload_to_inflation_rows(payload)
    st.session_state["risk_rows"] = _payload_to_risk_rows(payload)


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


def _payload_to_inflation_rows(payload: Mapping) -> list[dict]:
    years = list(payload.get("years", []))
    series = list(payload.get("inflation_series", []))
    default_rate = float(payload.get("inflation_rate", 0.0))

    rows: list[dict] = []
    if years:
        values = _ensure_schedule_length(series, len(years), fill=default_rate)
        for position, year in enumerate(years):
            rows.append({"Year": str(year), "Rate": float(values[position])})
    elif series:
        for index, value in enumerate(series, start=1):
            rows.append({"Year": f"Year {index}", "Rate": float(value)})
    else:
        rows.append({"Year": "Year 1", "Rate": default_rate})

    return rows


def _inflation_rows_to_payload(rows: Sequence[Mapping], payload: dict) -> None:
    if rows is None:
        return

    rates: list[float] = []
    labels: list[str] = []
    for index, row in enumerate(rows):
        label = str(row.get("Year", "")).strip()
        if not label:
            label = f"Year {index + 1}"
        labels.append(label)
        try:
            rate = float(row.get("Rate", 0.0))
        except (TypeError, ValueError):
            rate = 0.0
        rates.append(rate)

    if not rates:
        payload["inflation_series"] = []
        payload["inflation_labels"] = []
        return

    payload["inflation_series"] = list(rates)
    payload["inflation_labels"] = labels
    _align_payload_horizon(payload, labels, len(rates))


def _payload_to_risk_rows(payload: Mapping) -> list[dict]:
    source: Mapping[str, Sequence[float]] = payload.get("risk", {}) or {}
    risk: dict[str, list[float]] = {}
    for name, values in source.items():
        key = str(name).strip().lower()
        if not key:
            continue
        risk[key] = [float(value) for value in values]

    labels = list(payload.get("inflation_labels") or payload.get("years", []))
    categories = _risk_categories(payload)

    max_length = max([len(labels)] + [len(values) for values in risk.values()] or [0])
    if max_length == 0:
        max_length = 1

    if not labels:
        labels = [f"Year {index + 1}" for index in range(max_length)]
    elif len(labels) < max_length:
        labels = labels + [f"Year {index + 1}" for index in range(len(labels), max_length)]

    rows: list[dict] = []
    for index in range(max_length):
        label = labels[index] if index < len(labels) else f"Year {index + 1}"
        row = {"Year": str(label)}
        for category in categories:
            values = risk.get(category, [])
            row[category] = float(values[index]) if index < len(values) else 0.0
        rows.append(row)

    return rows


def _risk_rows_to_payload(rows: Sequence[Mapping], payload: dict) -> None:
    if rows is None:
        return

    categories = _risk_categories(payload, rows)
    if not rows:
        payload["risk"] = {category: [] for category in categories}
        return

    labels: list[str] = []
    risk_payload: dict[str, list[float]] = {category: [] for category in categories}

    for index, row in enumerate(rows):
        label = str(row.get("Year", "")).strip()
        if not label:
            label = f"Year {index + 1}"
        labels.append(label)
        for category in categories:
            try:
                value = float(row.get(category, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            risk_payload[category].append(min(max(value, 0.0), 1.0))

    payload["risk"] = risk_payload
    _align_payload_horizon(payload, labels, len(rows))

    if "risk_rows" in st.session_state:
        st.session_state["risk_rows"] = _payload_to_risk_rows(payload)


def _risk_categories(payload: Mapping | None = None, rows: Sequence[Mapping] | None = None) -> list[str]:
    categories: list[str] = []
    seen: set[str] = set()

    def _add(name: str | None) -> None:
        if not name:
            return
        key = str(name).strip().lower()
        if not key or key == "year" or key in seen:
            return
        seen.add(key)
        categories.append(key)

    for default in DEFAULT_RISK_CATEGORIES:
        _add(default)

    if payload and isinstance(payload.get("risk"), Mapping):
        for name in payload["risk"].keys():
            _add(str(name))

    if rows:
        for row in rows:
            for key in row.keys():
                _add(str(key))

    return categories


def _align_payload_horizon(payload: dict, labels: Sequence[str], target_length: int) -> None:
    if target_length <= 0:
        return

    years = list(payload.get("years", []))
    derived_years = _derive_years_from_labels(labels)
    payload["years"] = _resize_years(years, target_length, derived_years)

    payload["inflation_series"] = _resize_sequence(
        payload.get("inflation_series", []), target_length
    )

    production = payload.get("production_estimate", {})
    for name, series in list(production.items()):
        production[name] = _resize_sequence(series, target_length)

    utility = payload.setdefault("utility_costs", {})
    for field in ("days", "hours"):
        utility[field] = _resize_sequence(utility.get(field, []), target_length)

    tax = payload.setdefault("tax", {})
    schedule = tax.get("schedule", [])
    fill_rate = tax.get("rate", schedule[-1] if schedule else 0.0)
    tax["schedule"] = _resize_sequence(schedule, target_length, fill=fill_rate)

    risk = payload.setdefault("risk", {})
    for category, values in list(risk.items()):
        risk[category] = _resize_sequence(values, target_length)

    working = payload.get("working_capital", {}).get("days", {})
    for key, values in list(working.items()):
        working[key] = _resize_sequence(values, target_length)

    scenarios = payload.get("scenarios", {})
    for scenario in scenarios.values():
        if not isinstance(scenario, Mapping):
            continue
        if "inflation" in scenario:
            scenario["inflation"] = _resize_sequence(scenario.get("inflation", []), target_length)
        if "interest" in scenario:
            scenario["interest"] = _resize_sequence(scenario.get("interest", []), target_length)


def _resize_sequence(values: Iterable, target_length: int, fill=None) -> list:
    items = list(values)
    if target_length <= 0:
        return []
    if len(items) >= target_length:
        return items[:target_length]
    if fill is None:
        fill = items[-1] if items else 0
    items.extend([fill for _ in range(target_length - len(items))])
    return items


def _resize_years(current: Sequence[int], target_length: int, derived: Sequence[int]) -> list[int]:
    if target_length <= 0:
        return []
    if derived and len(derived) == target_length:
        return [int(value) for value in derived]
    existing = list(current)
    if len(existing) >= target_length:
        return [int(value) for value in existing[:target_length]]
    if existing:
        if len(existing) >= 2:
            step = existing[1] - existing[0]
        else:
            step = 1
        base = existing[-1]
        extension = [int(base + step * (index + 1)) for index in range(target_length - len(existing))]
        return [int(value) for value in existing + extension]
    return [index + 1 for index in range(target_length)]


def _derive_years_from_labels(labels: Sequence[str]) -> list[int]:
    derived: list[int] = []
    for label in labels:
        value = _parse_year_number(label)
        if value is None:
            return []
        derived.append(value)
    return derived


def _parse_year_number(label: str) -> int | None:
    if not label:
        return None
    try:
        return int(float(label))
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", label)
        if match:
            try:
                return int(float(match.group()))
            except ValueError:
                return None
    return None


if __name__ == "__main__":  # pragma: no cover - Streamlit executes the script directly
    if _streamlit_runtime_exists():
        main()
    else:  # pragma: no cover - guidance for incorrect invocation
        raise SystemExit(
            "This module is a Streamlit application. Launch it with "
            "`streamlit run streamlit_app.py` instead of executing it directly."
        )
