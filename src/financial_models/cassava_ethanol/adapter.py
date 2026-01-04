"""Adapter for integrating the cassava bioethanol model into the shared registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .financial_model import CassavaBioethanolModel
from .inputs import default_input_page


@dataclass
class CassavaModelParameters:
    model: CassavaBioethanolModel
    scenario: str = "FARM_ONLY"


def parse_inputs(payload: Mapping[str, Any] | None) -> CassavaModelParameters:
    scenario = "FARM_ONLY"
    if payload and isinstance(payload, Mapping):
        scenario = str(payload.get("scenario") or scenario).upper()
    model = CassavaBioethanolModel(default_input_page())
    return CassavaModelParameters(model=model, scenario=scenario)


def load_inputs(path=None) -> CassavaModelParameters:
    return parse_inputs(None)
