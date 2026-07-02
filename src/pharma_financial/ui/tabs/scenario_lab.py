"""Scenario and downside analysis tabs for the redesigned pharma workspace."""

from __future__ import annotations

from ...inputs import ModelInputs
from ...model import FinancialModel, FinancialOutputs
from .. import shell


def _render_table_like(legacy, value: object) -> None:
    frame = legacy._ensure_dataframe(value)
    legacy.st.dataframe(frame, use_container_width=True)


def render_scenario_lab(
    inputs: ModelInputs,
    model: FinancialModel | None,
    outputs: FinancialOutputs | None,
    digest: str,
) -> None:
    from ... import app as legacy

    del inputs
    if model is None or outputs is None:
        legacy.st.info("Run the model to unlock sensitivity, scenarios, Monte Carlo, and break-even analysis.")
        return

    shell.render_section_header(
        "Scenario Lab",
        "Stress the base case, compare outcomes, and review probabilistic downside before finalising recommendations.",
    )
    sensitivity_tab, downside_tab, monte_tab, breach_tab = legacy.st.tabs(
        [
            "Key Sensitivities",
            "Pharma Downside Cases",
            "Monte Carlo",
            "Breach Monitor",
        ]
    )
    with sensitivity_tab:
        legacy._render_sensitivity(model, outputs, digest)
    with downside_tab:
        legacy.st.markdown("### Downside Case Summary")
        _render_table_like(legacy, outputs.downside_case_summary)
        legacy.st.markdown("### Scenario Compare")
        legacy._render_scenarios(outputs)
        legacy.st.markdown("### Break-even & Payback")
        legacy._render_break_even(outputs)
    with monte_tab:
        legacy._render_monte_carlo(model, outputs, digest)
    with breach_tab:
        legacy.st.markdown("### Bankability Gate")
        _render_table_like(legacy, outputs.bankability_gate)
        legacy.st.markdown("### Covenant Headroom")
        _render_table_like(legacy, outputs.covenant_headroom)
        legacy.st.markdown("### Data Quality Exceptions")
        _render_table_like(legacy, outputs.data_quality_exceptions or [])
