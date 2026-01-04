"""Input loader and parser for the broiler chicken model."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

from .assumptions import Assumptions
from .model import apply_overrides

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "default_inputs.json"


class BroilerModelParameters:
    def __init__(self, assumptions: Assumptions):
        self.assumptions = assumptions


def parse_inputs(payload: Mapping[str, Any] | None) -> BroilerModelParameters:
    """Parse payload into an Assumptions object.

    Accepts a flat mapping of assumption fields; unknown keys are ignored.
    """

    base = Assumptions()
    if not payload:
        return BroilerModelParameters(base)

    current = asdict(base)
    updates = {}
    for key, value in payload.items():
        if key in current:
            updates[key] = value
    if updates:
        updated = replace(base, **apply_overrides(base, updates).__dict__)
        return BroilerModelParameters(updated)
    return BroilerModelParameters(base)


def load_inputs(path: Path | None = None) -> BroilerModelParameters:
    target = path or DEFAULT_DATA_PATH
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("Assumptions file must decode to an object")
        return parse_inputs(data)
    base = Assumptions()
    if path is None:
        DEFAULT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_DATA_PATH.write_text(json.dumps(asdict(base), indent=2), encoding="utf-8")
    return BroilerModelParameters(base)
