"""Pharmaceutical financial modelling toolkit."""

from .pharma.ai import AIInsights
from .pharma.inputs import AIParameters, load_inputs, parse_inputs
from .pharma.model import FinancialModel, FinancialOutputs, IRRResult, ScenarioToolResult

__all__ = [
    "load_inputs",
    "parse_inputs",
    "FinancialModel",
    "FinancialOutputs",
    "ScenarioToolResult",
    "AIInsights",
    "AIParameters",
    "IRRResult",
]
