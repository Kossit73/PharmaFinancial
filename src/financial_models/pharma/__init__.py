"""Pharma package shim pointing to the core implementation."""
from __future__ import annotations

from financial_models.core.inputs import ModelInputs, load_inputs, parse_inputs
from financial_models.core.model import (
    CASH_FLOW_BEGIN_COLUMN,
    CASH_FLOW_END_COLUMN,
    CASH_FLOW_NET_COLUMN,
    FinancialModel,
    FinancialOutputs,
    IRRResult,
    ScenarioToolResult,
)
from financial_models.core.table import Table

__all__ = [
    "ModelInputs",
    "FinancialModel",
    "FinancialOutputs",
    "IRRResult",
    "ScenarioToolResult",
    "Table",
    "load_inputs",
    "parse_inputs",
    "CASH_FLOW_BEGIN_COLUMN",
    "CASH_FLOW_END_COLUMN",
    "CASH_FLOW_NET_COLUMN",
]
