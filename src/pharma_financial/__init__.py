"""Pharmaceutical financial modelling toolkit."""
from .core.ai import AIInsights
from .core.inputs import AIParameters, load_inputs, parse_inputs
from .core.model import FinancialModel, FinancialOutputs, IRRResult, ScenarioToolResult

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
