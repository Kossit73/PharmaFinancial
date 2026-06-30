"""Setup-facing tabs for the redesigned pharma workspace."""

from __future__ import annotations

from .. import shell
from ..editors import core_assumptions
from ...inputs import ModelInputs
from ...model import FinancialModel, FinancialOutputs


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

    core_assumptions.render_core_assumptions_section(payload)
    legacy.st.markdown("### Distributors Commission Input Table")
    legacy._render_distributor_commission(payload)
    legacy.st.markdown("### Labour Structure")
    legacy._render_labor_mode_section(payload)
    legacy.st.markdown("### Fixed & Variable Costs Input Table")
    legacy._render_fixed_variable_costs(payload)
    legacy.st.markdown("### Utility Schedule")
    legacy._render_utility_schedule(payload)


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

    legacy.st.markdown("### Accounts Receivable Input Table")
    legacy._render_receivable_inputs(payload)
    legacy.st.markdown("### Inventory & Accounts Payable Input Table")
    legacy._render_inventory_inputs(payload)
    legacy.st.markdown("### Fixed Assets Schedule")
    legacy._render_depreciation_schedule(payload)
    legacy.st.markdown("### Cost & Financing Assumptions")
    legacy._render_cost_and_financing(payload)
    legacy.st.markdown("### Tax Schedule")
    legacy._render_tax_schedule(payload)
    legacy.st.markdown("### Inflation Schedule")
    legacy._render_inflation_schedule(payload)
    legacy.st.markdown("### Risk Schedule")
    legacy._render_risk_schedule(payload)

