"""Adapter for integrating the cassava bioethanol model into the shared registry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

from .financial_model import CassavaBioethanolModel
from .inputs import default_input_page

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "default_inputs.json"


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
    target = Path(path) if path else DEFAULT_DATA_PATH
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8"))
        return parse_inputs(data)
    if path is None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"scenario": "FARM_ONLY"}, indent=2), encoding="utf-8")
    return parse_inputs(None)
