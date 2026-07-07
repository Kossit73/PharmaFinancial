"""Workspace tab registry for the redesigned pharma experience."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..inputs import ModelInputs
from ..model import FinancialModel, FinancialOutputs


TabRenderer = Callable[[ModelInputs, FinancialModel | None, FinancialOutputs | None, str], None]


@dataclass(frozen=True)
class WorkspaceTab:
    key: str
    title: str
    requires_model: bool
    render: TabRenderer


def build_workspace_tabs() -> list[WorkspaceTab]:
    from .tabs import assistant, investment_case, scenario_lab, setup, statements

    return [
        WorkspaceTab("setup", "Setup & Validation", False, setup.render_setup_and_validation),
        WorkspaceTab(
            "commercial",
            "Commercial & Operations",
            False,
            setup.render_commercial_operations,
        ),
        WorkspaceTab(
            "funding",
            "Funding & Working Capital",
            False,
            setup.render_funding_working_capital,
        ),
        WorkspaceTab(
            "investment_case",
            "Investment Case",
            True,
            investment_case.render_investment_case,
        ),
        WorkspaceTab(
            "statements",
            "Financial Statements",
            True,
            statements.render_financial_statements,
        ),
        WorkspaceTab("scenario_lab", "Scenario Lab", True, scenario_lab.render_scenario_lab),
        WorkspaceTab(
            "knowledge",
            "Knowledge & Reports",
            True,
            assistant.render_knowledge_and_reports,
        ),
    ]
