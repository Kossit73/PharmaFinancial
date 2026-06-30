"""Scenario and downside analysis tabs for the redesigned pharma workspace."""

from __future__ import annotations

from ...inputs import ModelInputs
from ...model import FinancialModel, FinancialOutputs
from .. import shell


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
    sensitivity_tab, scenarios_tab, monte_tab, break_even_tab = legacy.st.tabs(
        [
            "Sensitivity",
            "Scenario Compare",
            "Monte Carlo",
            "Break-even & Payback",
        ]
    )
    with sensitivity_tab:
        legacy._render_sensitivity(model, outputs, digest)
    with scenarios_tab:
        legacy._render_scenarios(outputs)
    with monte_tab:
        legacy._render_monte_carlo(model, outputs, digest)
    with break_even_tab:
        legacy._render_break_even(outputs)

