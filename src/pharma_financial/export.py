"""Shared utilities for Excel export controls in the Streamlit app."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from .inputs import ModelInputs
from .model import FinancialModel, FinancialOutputs

SESSION_MODEL_RESULTS_KEY = "model_results"
SESSION_SELECTED_SCENARIO_KEY = "selected_scenario_label"
SESSION_EXCEL_BYTES_KEY = "excel_bytes_map"
EXCEL_FILE_NAME = "Ecommerce_Financial_Model.xlsx"


@dataclass
class ExportContext:
    """Container bundling the artefacts required to render export controls."""

    inputs: ModelInputs
    base_model: FinancialModel
    base_outputs: FinancialOutputs
    snapshot: ModelInputs


def render_excel_download_controls(
    *,
    widget_prefix: str,
    with_year: Callable[[pd.DataFrame], pd.DataFrame],
    inputs: Optional[ModelInputs] = None,
    base_model: Optional[FinancialModel] = None,
    base_outputs: Optional[FinancialOutputs] = None,
    header: bool = True,
) -> None:
    """Render scenario-aware Excel export controls.

    Parameters
    ----------
    widget_prefix:
        Prefix used to namespace Streamlit widget keys for a specific tab.
    with_year:
        Callable that ensures a "Year" column is present in rendered tables.
    inputs, base_model, base_outputs:
        Optional overrides for the data required to construct the export
        workbook. When omitted the values are resolved from ``st.session_state``.
    header:
        Whether to emit the "Excel Model Export" section header.
    """

    context = _resolve_export_context(inputs, base_model, base_outputs)
    if header:
        st.markdown("### Excel Model Export")

    if context is None:
        st.info("Excel export controls will appear after assumptions load.")
        return

    selected_label, scenario_key = _select_scenario(context.inputs, widget_prefix)
    model, results = _resolve_model_results(selected_label, scenario_key, context)
    _render_download_controls(model, results, selected_label, widget_prefix, with_year)


def reset_excel_export_state(*widget_prefixes: str) -> None:
    """Clear cached Excel exports and widget state across the provided prefixes."""

    st.session_state[SESSION_EXCEL_BYTES_KEY] = {}
    st.session_state.pop(SESSION_MODEL_RESULTS_KEY, None)
    st.session_state.pop(SESSION_SELECTED_SCENARIO_KEY, None)
    for prefix in widget_prefixes:
        st.session_state.pop(f"{prefix}_scenario_select", None)


def _resolve_export_context(
    inputs: Optional[ModelInputs],
    base_model: Optional[FinancialModel],
    base_outputs: Optional[FinancialOutputs],
) -> Optional[ExportContext]:
    if inputs is None:
        inputs = st.session_state.get("model_inputs")
    if base_model is None:
        base_model = st.session_state.get("base_model")
    if base_outputs is None:
        base_outputs = st.session_state.get("base_outputs")

    if inputs is None or base_model is None or base_outputs is None:
        return None

    snapshot = st.session_state.get("input_snapshot")
    if snapshot is None:
        snapshot = copy.deepcopy(inputs)
        st.session_state["input_snapshot"] = snapshot

    return ExportContext(inputs, base_model, base_outputs, snapshot)


def _select_scenario(
    inputs: ModelInputs, widget_prefix: str
) -> Tuple[str, Optional[str]]:
    scenario_map = _scenario_label_map(inputs)
    scenario_labels = list(scenario_map.keys())
    default_label = st.session_state.get(
        SESSION_SELECTED_SCENARIO_KEY, scenario_labels[0]
    )
    if default_label not in scenario_labels:
        default_label = scenario_labels[0]

    selectbox_key = f"{widget_prefix}_scenario_select"
    if (
        selectbox_key in st.session_state
        and st.session_state[selectbox_key] not in scenario_labels
    ):
        st.session_state.pop(selectbox_key, None)

    selected_label = st.selectbox(
        "Select scenario for Excel export",
        scenario_labels,
        index=scenario_labels.index(default_label),
        key=selectbox_key,
    )
    st.session_state[SESSION_SELECTED_SCENARIO_KEY] = selected_label
    return selected_label, scenario_map[selected_label]


def _resolve_model_results(
    selected_label: str, scenario_key: Optional[str], context: ExportContext
) -> Tuple[FinancialModel, FinancialOutputs]:
    if scenario_key is None:
        model = context.base_model
        results = context.base_outputs
    else:
        model, results = _build_scenario_payload(
            selected_label, scenario_key, context.snapshot
        )

    st.session_state[SESSION_MODEL_RESULTS_KEY] = (model, results)
    return model, results


def _render_download_controls(
    model: FinancialModel,
    results: FinancialOutputs,
    selected_label: str,
    widget_prefix: str,
    with_year: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    excel_map: Dict[str, bytes] = st.session_state.setdefault(
        SESSION_EXCEL_BYTES_KEY, {}
    )
    excel_bytes = excel_map.get(selected_label)
    widget_suffix = _scenario_widget_key(selected_label)
    button_prefix = f"{widget_prefix}_{widget_suffix}"

    download_container = st.container()
    with download_container:
        if not excel_bytes:
            if st.button(
                "Prepare Excel Model", key=f"prepare_{button_prefix}"
            ):
                with st.spinner("Preparing Excel workbook..."):
                    excel_bytes = _generate_excel_bytes(
                        model, results, selected_label, with_year=with_year
                    )
                excel_map[selected_label] = excel_bytes
                st.session_state[SESSION_EXCEL_BYTES_KEY] = excel_map
        if excel_bytes:
            st.download_button(
                "Download Excel Model",
                data=excel_bytes,
                file_name=EXCEL_FILE_NAME,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{button_prefix}",
            )
            if st.button(
                "Clear Prepared Excel", key=f"clear_{button_prefix}"
            ):
                excel_map.pop(selected_label, None)
                st.session_state[SESSION_EXCEL_BYTES_KEY] = excel_map
                excel_bytes = None
        if not excel_bytes:
            st.info("Click 'Prepare Excel Model' to generate the workbook for download.")


def _scenario_label_map(inputs: ModelInputs) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {"Base Case": None}
    for key in sorted(inputs.scenarios.keys()):
        label = _format_scenario_label(key)
        if label in mapping:
            suffix = 2
            candidate = f"{label} {suffix}"
            while candidate in mapping:
                suffix += 1
                candidate = f"{label} {suffix}"
            label = candidate
        mapping[label] = key
    return mapping


def _format_scenario_label(raw: str) -> str:
    return raw.replace("_", " ").title()


def _scenario_widget_key(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    return sanitized or "scenario"


def _build_scenario_payload(
    label: str, scenario_key: str, snapshot: ModelInputs
) -> Tuple[FinancialModel, FinancialOutputs]:
    scenario_inputs = copy.deepcopy(snapshot)
    scenario_config = scenario_inputs.scenarios.get(scenario_key, {})

    inflation = scenario_config.get("inflation")
    if inflation:
        scenario_inputs.inflation_series = [float(value) for value in inflation]

    interest = scenario_config.get("interest")
    if interest:
        scenario_inputs.financing.discount_rate = float(interest[0])

    model = FinancialModel(scenario_inputs)
    model.scenario = label
    results = model.run()
    return model, results


def _sheet_name(label: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in label)
    sanitized = sanitized[:31]
    return sanitized or "Sheet"


def _generate_excel_bytes(
    model: FinancialModel,
    results: FinancialOutputs,
    selected_scenario: str,
    *,
    with_year: Callable[[pd.DataFrame], pd.DataFrame],
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        with_year(results.income_statement).to_excel(
            writer, sheet_name="Income Statement", index=False
        )
        with_year(results.balance_sheet).to_excel(
            writer, sheet_name="Balance Sheet", index=False
        )
        with_year(results.cash_flow).to_excel(
            writer, sheet_name="Cash Flow", index=False
        )
        results.summary_metrics.reset_index().to_excel(
            writer, sheet_name="Summary Metrics", index=False
        )
        results.break_even.reset_index().rename(
            columns={"index": "Product"}
        ).to_excel(writer, sheet_name="Break Even", index=False)
        with_year(results.payback).to_excel(
            writer, sheet_name="Payback", index=False
        )
        with_year(results.discounted_payback).to_excel(
            writer, sheet_name="Discounted Payback", index=False
        )
        results.monte_carlo.describe().T.to_excel(
            writer, sheet_name="Monte Carlo Stats"
        )

        for name, df in results.scenario_results.items():
            sheet_name = _sheet_name(f"Scenario_{name}")
            with_year(df).to_excel(writer, sheet_name=sheet_name, index=False)

        for name, df in results.sensitivity_results.items():
            sheet_name = _sheet_name(f"Sensitivity_{name}")
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        assumption_rows = [
            {
                "Product": product.title(),
                "Production Cost": params.production_cost,
                "Selling Price": params.selling_price,
                "Freight Cost": params.freight_cost,
                "Markup": params.markup,
            }
            for product, params in model.inputs.unit_costs.items()
        ]
        if assumption_rows:
            pd.DataFrame(assumption_rows).to_excel(
                writer, sheet_name="Assumptions", index=False
            )

        production_df = pd.DataFrame(
            model.inputs.production_estimate, index=model.inputs.years
        )
        production_df.index.name = "Year"
        production_df.reset_index().to_excel(
            writer, sheet_name="Production Plan", index=False
        )

        utility_df = pd.DataFrame(
            {
                "Year": model.inputs.years,
                "Operating Days": model.inputs.utility_schedule.operating_days,
                "Operating Hours": model.inputs.utility_schedule.operating_hours,
            }
        )
        utility_df.to_excel(writer, sheet_name="Utility Schedule", index=False)

        metadata = pd.DataFrame(
            {
                "Scenario": [selected_scenario],
                "Discount Rate": [model.inputs.financing.discount_rate],
                "Initial Investment": [model.inputs.financing.initial_investment],
                "Tax Rate": [model.inputs.tax_rate],
            }
        )
        metadata.to_excel(writer, sheet_name="Scenario Summary", index=False)

    buffer.seek(0)
    return buffer.getvalue()
