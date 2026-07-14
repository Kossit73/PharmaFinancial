"""Financial statement presentation layer for the redesigned pharma workspace."""

from __future__ import annotations

from ...inputs import ModelInputs
from ...model import FinancialModel, FinancialOutputs
from .. import shell


def render_financial_statements(
    inputs: ModelInputs,
    model: FinancialModel | None,
    outputs: FinancialOutputs | None,
    digest: str,
) -> None:
    from ... import app as legacy

    del inputs, digest
    if model is None or outputs is None:
        legacy.st.info("Run the model to view the financial statements.")
        return

    shell.render_section_header(
        "Financial Statements",
        "Review the formal statements and the core roll-forwards that support them.",
    )
    active_panel = legacy.st.radio(
        "Statement",
        ["Income Statement", "Financial Position", "Cash Flow", "Working Capital & Inventory"],
        horizontal=True,
        key="pharma_active_statement_panel",
        label_visibility="collapsed",
    )
    if active_panel == "Income Statement":
        legacy._render_income_statement(model, outputs)
    if active_panel == "Financial Position":
        legacy._render_statement_tab("Statement of Financial Position", outputs.balance_sheet)
    if active_panel == "Cash Flow":
        legacy._render_statement_tab("Statement of Cash Flows", outputs.cash_flow)
    if active_panel == "Working Capital & Inventory":
        try:
            working_capital = model.working_capital_schedule()
            legacy.st.markdown("### Working Capital Schedule")
            legacy.st.dataframe(legacy._with_year(working_capital), width="stretch")
        except Exception as exc:  # pragma: no cover - defensive UI feedback
            legacy.st.warning(f"Unable to compute working capital schedule: {exc}")
        try:
            inventory_table = model.inventory_schedule()
            legacy.st.markdown("### Inventory Schedule")
            legacy.st.dataframe(legacy._with_year(inventory_table), width="stretch")
        except Exception as exc:  # pragma: no cover - defensive UI feedback
            legacy.st.warning(f"Unable to compute inventory schedule: {exc}")

