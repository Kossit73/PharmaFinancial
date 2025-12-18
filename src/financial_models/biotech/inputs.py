"""Helpers for loading and parsing biotech model inputs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from .model import ModelConfig, Portfolio, Product, ProductConfig


DEFAULT_INPUTS_PATH = Path(__file__).resolve().parent / "data" / "default_inputs.json"


@dataclass
class BiotechInputs:
    """Container for biotech model assumptions."""

    model_config: ModelConfig
    products: List[ProductConfig]


def _coerce_model_config(payload: Mapping[str, object]) -> ModelConfig:
    return ModelConfig(**{k: payload.get(k) for k in ModelConfig.__dataclass_fields__.keys()})  # type: ignore[attr-defined]


def _coerce_product(payload: Mapping[str, object]) -> ProductConfig:
    return ProductConfig(**{k: payload.get(k) for k in ProductConfig.__dataclass_fields__.keys()})  # type: ignore[attr-defined]


def parse_inputs(payload: Mapping[str, object]) -> BiotechInputs:
    """Parse a mapping into biotech inputs."""

    if not isinstance(payload, Mapping):
        raise ValueError("Inputs must be a mapping.")
    model_cfg_raw = payload.get("model_config")
    products_raw = payload.get("products")
    if not isinstance(model_cfg_raw, Mapping):
        raise ValueError("model_config must be an object.")
    if not isinstance(products_raw, Iterable):
        raise ValueError("products must be a list.")

    model_config = _coerce_model_config(model_cfg_raw)
    products: List[ProductConfig] = []
    for prod in products_raw:
        if not isinstance(prod, Mapping):
            raise ValueError("Each product must be an object.")
        products.append(_coerce_product(prod))

    return BiotechInputs(model_config=model_config, products=products)


def load_inputs(path: Path | None = None) -> BiotechInputs:
    """Load biotech inputs from JSON or fall back to defaults."""

    target = path or DEFAULT_INPUTS_PATH
    data = json.loads(Path(target).read_text(encoding="utf-8"))
    return parse_inputs(data)


def build_portfolio(inputs: BiotechInputs) -> Portfolio:
    """Construct a Portfolio from parsed biotech inputs."""

    products = [Product(cfg, inputs.model_config) for cfg in inputs.products]
    return Portfolio(products, inputs.model_config)
