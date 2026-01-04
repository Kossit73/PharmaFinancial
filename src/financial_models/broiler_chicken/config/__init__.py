"""Configuration loaders for the broiler model."""

from __future__ import annotations

"""Configuration loaders for analytics- and simulation-related settings."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_CUSTOM_SIM_PATH = _CONFIG_DIR / "custom_simulations.json"
_DEFAULT_MONTE_CARLO_PATH = _CONFIG_DIR / "monte_carlo_distributions.json"


def _load_json_file(path: Path) -> Any:
    data = json.loads(path.read_text())
    return data


def load_custom_simulation_definitions(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    target = path or _DEFAULT_CUSTOM_SIM_PATH
    if not target.exists():
        return []
    data = _load_json_file(target)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError(f"Custom simulation configuration must be a list of objects: {target}")


def load_monte_carlo_distributions(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    target = path or _DEFAULT_MONTE_CARLO_PATH
    if not target.exists():
        return []
    data = _load_json_file(target)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError(f"Monte Carlo configuration must be a list of objects: {target}")


__all__ = [
    "load_custom_simulation_definitions",
    "load_monte_carlo_distributions",
]
