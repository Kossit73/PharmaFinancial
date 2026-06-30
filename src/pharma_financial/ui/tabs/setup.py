"""Setup-facing tabs for the redesigned pharma workspace."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .. import shell
from ..editors import core_assumptions
from ...inputs import ModelInputs
from ...model import FinancialModel, FinancialOutputs


def _count_rows(rows: object) -> int:
    return len(rows) if isinstance(rows, list) else 0


def _count_unique_values(rows: object, field: str) -> int:
    if not isinstance(rows, list):
        return 0
    values = {
        str(row.get(field, "")).strip()
        for row in rows
        if isinstance(row, Mapping) and str(row.get(field, "")).strip()
    }
    return len(values)


def _format_summary(parts: Sequence[str], fallback: str) -> str:
    summary = " | ".join(part for part in parts if part)
    return summary or fallback


def _render_collapsible_section(
    legacy,
    *,
    title: str,
    section_key: str,
    summary: str,
    description: str,
    payload: dict,
    render_body: Callable[[dict], None],
    hydrate_state: Callable[[dict], None] | None = None,
    add_hint: str | None = None,
) -> None:
    st = legacy.st
    mode_key = f"{section_key}_panel_mode"
    refresh_key = f"{section_key}_panel_refresh"
    mode = str(st.session_state.get(mode_key, "collapsed"))

    header_cols = st.columns([6, 1, 1, 1])
    header_cols[0].markdown(f"### {title}")
    if header_cols[1].button("Edit", key=f"{section_key}_panel_edit"):
        st.session_state[mode_key] = "edit"
        st.session_state[refresh_key] = True
        mode = "edit"
        legacy._rerun()

    if add_hint:
        if header_cols[2].button("Add", key=f"{section_key}_panel_add"):
            st.session_state[mode_key] = "add"
            st.session_state[refresh_key] = True
            mode = "add"
            legacy._rerun()
    else:
        header_cols[2].empty()

    if mode != "collapsed":
        if header_cols[3].button("Hide", key=f"{section_key}_panel_hide"):
            st.session_state[mode_key] = "collapsed"
            mode = "collapsed"
            legacy._rerun()
    else:
        header_cols[3].empty()

    if mode == "collapsed":
        action_text = "Click Edit to review the full schedule."
        if add_hint:
            action_text = "Click Edit to review the full schedule or Add to open the entry form."
        st.caption(f"{summary} {action_text}")
        return

    if st.session_state.pop(refresh_key, False) and hydrate_state is not None:
        hydrate_state(payload)

    st.caption(description)
    if mode == "add" and add_hint:
        st.info(add_hint)
    render_body(payload)


def _hydrate_core_assumptions(legacy, payload: dict) -> None:
    rows = legacy._payload_to_core_rows(payload)
    legacy.st.session_state["core_assumption_rows"] = rows
    legacy._prime_core_widget_state(rows)


def _hydrate_commission(legacy, payload: dict) -> None:
    legacy.st.session_state["commission_rows"] = legacy._payload_to_commission_rows(payload)


def _hydrate_labor(legacy, payload: dict) -> None:
    mode = legacy._payload_labor_mode(payload)
    legacy.st.session_state["labor_mode"] = mode
    if mode == "Basic":
        labor = payload.get("labor", {}) if isinstance(payload.get("labor"), Mapping) else {}
        legacy.st.session_state["direct_labor_rows"] = legacy._mapping_to_rows(
            labor.get("direct", {}),
            "Role",
            "Annual Cost",
        )
        legacy.st.session_state["indirect_labor_rows"] = legacy._mapping_to_rows(
            labor.get("indirect", {}),
            "Role",
            "Annual Cost",
        )
        return

    legacy.st.session_state["labor_model_rows"] = legacy._payload_to_labor_model_rows(payload)
    legacy.st.session_state["labor_model_settings_rows"] = legacy._payload_to_labor_model_settings_rows(payload)


def _hydrate_fixed_variable(legacy, payload: dict) -> None:
    legacy.st.session_state["fixed_variable_rows"] = legacy._payload_to_fixed_variable_rows(payload)


def _hydrate_utility(legacy, payload: dict) -> None:
    legacy.st.session_state["utility_entries"] = legacy._payload_to_utility_entries(payload)
    legacy.st.session_state.pop("utility_increment_preview_rows", None)


def _hydrate_receivable(legacy, payload: dict) -> None:
    legacy.st.session_state["receivable_rows"] = legacy._payload_to_receivable_rows(payload)
    legacy.st.session_state.pop("receivable_increment_preview_rows", None)


def _hydrate_inventory(legacy, payload: dict) -> None:
    legacy.st.session_state["inventory_rows"] = legacy._payload_to_inventory_rows(payload)
    legacy.st.session_state.pop("inventory_increment_preview_rows", None)


def _hydrate_depreciation(legacy, payload: dict) -> None:
    legacy.st.session_state["depreciation_rows"] = legacy._payload_to_depreciation_rows(payload)


def _hydrate_financing(legacy, payload: dict) -> None:
    legacy.st.session_state["senior_debt_rows"] = legacy._payload_to_debt_rows(payload, "senior_debt")
    legacy.st.session_state["revolver_rows"] = legacy._payload_to_debt_rows(payload, "revolver")
    legacy.st.session_state["overdraft_rows"] = legacy._payload_to_debt_rows(payload, "overdraft")


def _hydrate_tax(legacy, payload: dict) -> None:
    entries = legacy._payload_to_tax_entries(payload)
    legacy.st.session_state["tax_entries"] = entries
    legacy.st.session_state["tax_rows"] = [
        {
            "Year": str(entry.get("label", f"Year {index + 1}")),
            "Rate": float(entry.get("rate", 0.0) or 0.0),
        }
        for index, entry in enumerate(entries)
    ]
    legacy.st.session_state.pop("tax_increment_preview_rows", None)


def _hydrate_inflation(legacy, payload: dict) -> None:
    legacy.st.session_state["inflation_rows"] = legacy._payload_to_inflation_rows(payload)
    legacy.st.session_state.pop("inflation_increment_preview_rows", None)


def _hydrate_risk(legacy, payload: dict) -> None:
    legacy.st.session_state["risk_rows"] = legacy._payload_to_risk_rows(payload)
    legacy.st.session_state.pop("risk_increment_preview_rows", None)


def _core_assumptions_summary(legacy, payload: dict) -> str:
    rows = legacy._payload_to_core_rows(payload)
    return _format_summary(
        [f"{_count_rows(rows)} product rows configured"],
        "No core assumptions configured yet.",
    )


def _commission_summary(legacy, payload: dict) -> str:
    rows = legacy._payload_to_commission_rows(payload)
    return _format_summary(
        [
            f"{_count_rows(rows)} commission rows",
            f"{_count_unique_values(rows, 'Product')} products",
        ],
        "No distributor commission assumptions configured yet.",
    )


def _labor_summary(legacy, payload: dict) -> str:
    mode = legacy._payload_labor_mode(payload)
    if mode == "Advanced":
        roles = legacy._payload_to_labor_model_rows(payload)
        settings = legacy._payload_to_labor_model_settings_rows(payload)
        return _format_summary(
            [
                "Advanced mode",
                f"{_count_rows(roles)} role rows",
                f"{_count_rows(settings)} yearly settings rows",
            ],
            "Advanced labour model not configured yet.",
        )

    labor = payload.get("labor", {}) if isinstance(payload.get("labor"), Mapping) else {}
    direct = labor.get("direct", {}) if isinstance(labor, Mapping) else {}
    indirect = labor.get("indirect", {}) if isinstance(labor, Mapping) else {}
    return _format_summary(
        [
            "Basic mode",
            f"{len(direct) if isinstance(direct, Mapping) else 0} direct roles",
            f"{len(indirect) if isinstance(indirect, Mapping) else 0} indirect roles",
        ],
        "Basic labour assumptions not configured yet.",
    )


def _fixed_variable_summary(legacy, payload: dict) -> str:
    rows = legacy._payload_to_fixed_variable_rows(payload)
    return _format_summary(
        [f"{_count_rows(rows)} product cost rows"],
        "No fixed or variable cost rows configured yet.",
    )


def _utility_summary(legacy, payload: dict) -> str:
    rows = legacy._payload_to_utility_entries(payload)
    return _format_summary(
        [f"{_count_rows(rows)} yearly utility rows"],
        "No utility schedule configured yet.",
    )


def _receivable_summary(legacy, payload: dict) -> str:
    rows = legacy._payload_to_receivable_rows(payload)
    return _format_summary(
        [f"{_count_rows(rows)} yearly receivable rows"],
        "No receivable assumptions configured yet.",
    )


def _inventory_summary(legacy, payload: dict) -> str:
    rows = legacy._payload_to_inventory_rows(payload)
    return _format_summary(
        [f"{_count_rows(rows)} yearly inventory/AP rows"],
        "No inventory or payable assumptions configured yet.",
    )


def _depreciation_summary(legacy, payload: dict) -> str:
    rows = legacy._payload_to_depreciation_rows(payload)
    return _format_summary(
        [f"{_count_rows(rows)} fixed asset rows"],
        "No fixed asset rows configured yet.",
    )


def _financing_summary(legacy, payload: dict) -> str:
    senior = legacy._payload_to_debt_rows(payload, "senior_debt")
    revolver = legacy._payload_to_debt_rows(payload, "revolver")
    overdraft = legacy._payload_to_debt_rows(payload, "overdraft")
    return _format_summary(
        [
            f"{_count_rows(senior)} senior debt rows",
            f"{_count_rows(revolver)} revolver rows",
            f"{_count_rows(overdraft)} overdraft rows",
        ],
        "No financing schedules configured yet.",
    )


def _tax_summary(legacy, payload: dict) -> str:
    entries = legacy._payload_to_tax_entries(payload)
    tax = payload.get("tax", {}) if isinstance(payload.get("tax"), Mapping) else {}
    base_rate = float(tax.get("rate", 0.0) or 0.0) * 100.0
    return _format_summary(
        [
            f"{_count_rows(entries)} yearly tax rows",
            f"{base_rate:.1f}% base rate",
        ],
        "No tax schedule configured yet.",
    )


def _inflation_summary(legacy, payload: dict) -> str:
    rows = legacy._payload_to_inflation_rows(payload)
    base_rate = float(payload.get("inflation_rate", 0.0) or 0.0) * 100.0
    return _format_summary(
        [
            f"{_count_rows(rows)} yearly inflation rows",
            f"{base_rate:.1f}% base rate",
        ],
        "No inflation schedule configured yet.",
    )


def _risk_summary(legacy, payload: dict) -> str:
    rows = legacy._payload_to_risk_rows(payload)
    categories = legacy._risk_categories(payload, rows)
    return _format_summary(
        [
            f"{_count_rows(rows)} yearly risk rows",
            f"{len(categories)} risk categories",
        ],
        "No risk schedule configured yet.",
    )


def render_setup_and_validation(
    inputs: ModelInputs,
    model: FinancialModel | None,
    outputs: FinancialOutputs | None,
    digest: str,
) -> None:
    from ... import app as legacy

    del digest
    st = legacy.st
    payload = st.session_state["input_payload"]
    shell.render_section_header(
        "Setup & Validation",
        "Control the model horizon, trigger recalculation, and review whether the current assumptions are ready for analysis.",
    )

    st.button("Run Model", key="run_model", on_click=legacy._request_model_run)

    product_count = len(getattr(inputs, "products", []) or [])
    year_count = len(getattr(inputs, "years", []) or [])
    scenario_count = len(getattr(inputs, "scenarios", {}) or {})
    monte_carlo = getattr(inputs, "monte_carlo", None)
    iteration_count = int(getattr(monte_carlo, "iterations", 0) or 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products", product_count)
    col2.metric("Projection Years", year_count)
    col3.metric("Configured Scenarios", scenario_count)
    col4.metric("Monte Carlo Iterations", iteration_count)

    if outputs is not None:
        quality = legacy._summary_metric(outputs, "Assumption Data Quality Score")
        gate_ratio = legacy._summary_metric(outputs, "Investor Gate Pass Ratio")
        gate_status = legacy._summary_metric(outputs, "Investor Gate Status")
        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Assumption Quality",
            legacy._format_display(float(quality), 1) if quality is not None else "N/A",
        )
        metric_cols[1].metric(
            "Gate Pass Ratio",
            legacy._format_percentage(float(gate_ratio), 1) if gate_ratio is not None else "N/A",
        )
        metric_cols[2].metric(
            "Investment Gate",
            "Pass" if gate_status == 1.0 else "Review",
        )

    st.markdown("### Projection Horizon")
    legacy._render_projection_horizon(payload)

    st.markdown("### Readiness Checklist")
    checklist = [
        ("At least one product is configured", product_count > 0),
        ("Projection horizon spans at least 3 years", year_count >= 3),
        ("Risk schedule is configured", bool(st.session_state.get("risk_rows"))),
        ("Debt or financing assumptions are present", bool(st.session_state.get("senior_debt_rows")) or bool(st.session_state.get("revolver_rows")) or bool(st.session_state.get("overdraft_rows"))),
    ]
    for label, passed in checklist:
        if passed:
            st.success(label)
        else:
            st.warning(label)

    if model is not None and outputs is not None:
        st.caption(
            "The current workspace is live. Updates in the operations and funding tabs will refresh this run on the next rerender."
        )


def render_commercial_operations(
    inputs: ModelInputs,
    model: FinancialModel | None,
    outputs: FinancialOutputs | None,
    digest: str,
) -> None:
    from ... import app as legacy

    del inputs, model, outputs, digest
    payload = legacy.st.session_state["input_payload"]
    shell.render_section_header(
        "Commercial & Operations",
        "Manage products, price build-up, labor structure, utilities, and operating cost drivers.",
    )

    _render_collapsible_section(
        legacy,
        title="Core Assumptions",
        section_key="core_assumptions",
        summary=_core_assumptions_summary(legacy, payload),
        description="Review or update the product-level production, pricing, and capacity assumptions.",
        payload=payload,
        render_body=core_assumptions.render_core_assumptions_section,
        hydrate_state=lambda current_payload: _hydrate_core_assumptions(legacy, current_payload),
        add_hint="The add form opens inside this section below the existing product rows.",
    )
    _render_collapsible_section(
        legacy,
        title="Distributors Commission Input Table",
        section_key="commission_schedule",
        summary=_commission_summary(legacy, payload),
        description="Edit product-year commission assumptions or open the add form for a new distributor entry.",
        payload=payload,
        render_body=legacy._render_distributor_commission,
        hydrate_state=lambda current_payload: _hydrate_commission(legacy, current_payload),
        add_hint="The add form appears at the bottom of this section after it expands.",
    )
    _render_collapsible_section(
        legacy,
        title="Labour Structure",
        section_key="labor_schedule",
        summary=_labor_summary(legacy, payload),
        description="Switch between basic and advanced labour modelling, then edit the related rows in place.",
        payload=payload,
        render_body=legacy._render_labor_mode_section,
        hydrate_state=lambda current_payload: _hydrate_labor(legacy, current_payload),
    )
    _render_collapsible_section(
        legacy,
        title="Fixed & Variable Costs Input Table",
        section_key="fixed_variable_schedule",
        summary=_fixed_variable_summary(legacy, payload),
        description="Review product-level fixed and variable cost rows or open the add form for another cost line.",
        payload=payload,
        render_body=legacy._render_fixed_variable_costs,
        hydrate_state=lambda current_payload: _hydrate_fixed_variable(legacy, current_payload),
        add_hint="The add form appears below the existing cost rows inside this section.",
    )
    _render_collapsible_section(
        legacy,
        title="Utility Schedule",
        section_key="utility_schedule",
        summary=_utility_summary(legacy, payload),
        description="Edit year-by-year utility assumptions and yearly increment settings when needed.",
        payload=payload,
        render_body=legacy._render_utility_schedule,
        hydrate_state=lambda current_payload: _hydrate_utility(legacy, current_payload),
    )


def render_funding_working_capital(
    inputs: ModelInputs,
    model: FinancialModel | None,
    outputs: FinancialOutputs | None,
    digest: str,
) -> None:
    from ... import app as legacy

    del inputs, model, outputs, digest
    payload = legacy.st.session_state["input_payload"]
    shell.render_section_header(
        "Funding & Working Capital",
        "Manage collections, inventory, fixed assets, financing, tax, inflation, and risk assumptions.",
    )

    _render_collapsible_section(
        legacy,
        title="Accounts Receivable Input Table",
        section_key="receivable_schedule",
        summary=_receivable_summary(legacy, payload),
        description="Edit working-capital collection assumptions and the yearly increment helper only when you need them.",
        payload=payload,
        render_body=legacy._render_receivable_inputs,
        hydrate_state=lambda current_payload: _hydrate_receivable(legacy, current_payload),
    )
    _render_collapsible_section(
        legacy,
        title="Inventory & Accounts Payable Input Table",
        section_key="inventory_schedule",
        summary=_inventory_summary(legacy, payload),
        description="Open this section to adjust inventory days, payable days, and their yearly progression.",
        payload=payload,
        render_body=legacy._render_inventory_inputs,
        hydrate_state=lambda current_payload: _hydrate_inventory(legacy, current_payload),
    )
    _render_collapsible_section(
        legacy,
        title="Fixed Assets Schedule",
        section_key="depreciation_schedule",
        summary=_depreciation_summary(legacy, payload),
        description="Review the depreciation schedule or expand the section to add another fixed asset line.",
        payload=payload,
        render_body=legacy._render_depreciation_schedule,
        hydrate_state=lambda current_payload: _hydrate_depreciation(legacy, current_payload),
        add_hint="Use the add form inside this section to capture a new fixed asset row.",
    )
    _render_collapsible_section(
        legacy,
        title="Cost & Financing Assumptions",
        section_key="financing_schedule",
        summary=_financing_summary(legacy, payload),
        description="Expand this section only when updating raw material, financing, or debt schedule assumptions.",
        payload=payload,
        render_body=legacy._render_cost_and_financing,
        hydrate_state=lambda current_payload: _hydrate_financing(legacy, current_payload),
        add_hint="Debt add forms remain inside this section once it expands.",
    )
    _render_collapsible_section(
        legacy,
        title="Tax Schedule",
        section_key="tax_schedule",
        summary=_tax_summary(legacy, payload),
        description="Edit the base tax rate, timing adjustment, and yearly tax schedule only when needed.",
        payload=payload,
        render_body=legacy._render_tax_schedule,
        hydrate_state=lambda current_payload: _hydrate_tax(legacy, current_payload),
    )
    _render_collapsible_section(
        legacy,
        title="Inflation Schedule",
        section_key="inflation_schedule",
        summary=_inflation_summary(legacy, payload),
        description="Open the inflation schedule to revise the base rate or apply the yearly increment helper.",
        payload=payload,
        render_body=legacy._render_inflation_schedule,
        hydrate_state=lambda current_payload: _hydrate_inflation(legacy, current_payload),
    )
    _render_collapsible_section(
        legacy,
        title="Risk Schedule",
        section_key="risk_schedule",
        summary=_risk_summary(legacy, payload),
        description="Expand the risk schedule when you need to change category-level risk assumptions by year.",
        payload=payload,
        render_body=legacy._render_risk_schedule,
        hydrate_state=lambda current_payload: _hydrate_risk(legacy, current_payload),
    )
