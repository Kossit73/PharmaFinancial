"""Investment-case presentation layer for the redesigned pharma workspace."""

from __future__ import annotations

from ...inputs import ModelInputs
from ...model import FinancialModel, FinancialOutputs
from .. import shell


def _render_table_like(legacy, value: object) -> None:
    frame = legacy._ensure_dataframe(value)
    legacy.st.dataframe(frame, use_container_width=True)


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
    summary_tab, returns_tab, liquidity_tab, bridge_tab = legacy.st.tabs(
        [
            "IC Summary",
            "Returns & Covenants",
            "Liquidity & Funding Need",
            "Value Driver Bridge",
        ]
    )
    with summary_tab:
        legacy._render_executive_summary(model, outputs, digest)
        legacy.st.markdown("### Bankability Gate")
        _render_table_like(legacy, outputs.bankability_gate)
    with returns_tab:
        legacy.st.markdown("### Summary Metrics")
        _render_table_like(legacy, outputs.summary_metrics)
        legacy.st.markdown("### Covenant Headroom")
        _render_table_like(legacy, outputs.covenant_headroom)
    with liquidity_tab:
        legacy.st.markdown("### Sources & Uses")
        _render_table_like(legacy, outputs.sources_and_uses)
        legacy.st.markdown("### Liquidity Bridge")
        _render_table_like(legacy, outputs.liquidity_bridge)
    with bridge_tab:
        legacy.st.markdown("### Commercial Diagnostics")
        _render_table_like(legacy, outputs.commercial_diagnostics)
        legacy.st.markdown("### Downside Case Summary")
        _render_table_like(legacy, outputs.downside_case_summary)
