"""Investment-case presentation layer for the redesigned pharma workspace."""

from __future__ import annotations

from ...inputs import ModelInputs
from ...model import FinancialModel, FinancialOutputs
from .. import shell


def render_investment_case(
    inputs: ModelInputs,
    model: FinancialModel | None,
    outputs: FinancialOutputs | None,
    digest: str,
) -> None:
    from ... import app as legacy

    del inputs
    if model is None or outputs is None:
        legacy.st.info("Update the setup tabs and click Run Model to generate the investment case.")
        return

    shell.render_section_header(
        "Investment Case",
        "Use this view for management and investment-committee discussion before dropping into detailed statements.",
    )
    summary_tab, dashboard_tab = legacy.st.tabs(["Executive Summary", "Performance Dashboard"])
    with summary_tab:
        legacy._render_executive_summary(model, outputs, digest)
    with dashboard_tab:
        legacy._render_dashboard_tab(model, outputs, digest)

