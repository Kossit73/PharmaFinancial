"""Registry of financial models exposed via API/CLI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Type

from .core.inputs import ModelInputs, load_inputs as load_pharma_inputs, parse_inputs as parse_pharma_inputs
from .core.model import FinancialModel
from .api.schemas.pharma import PharmaModelRunRequest, PharmaValidationRequest


@dataclass
class ModelSpec:
    """Describes a registered model and how to execute it."""

    name: str
    load_inputs: Callable[[], ModelInputs]
    parse_inputs: Callable[[Mapping[str, Any]], ModelInputs]
    model_factory: Callable[[ModelInputs], Any]
    run_request_model: Type[Any]
    validate_request_model: Type[Any]


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "pharma": ModelSpec(
        name="Pharmaceuticals",
        load_inputs=load_pharma_inputs,
        parse_inputs=parse_pharma_inputs,
        model_factory=FinancialModel,
        run_request_model=PharmaModelRunRequest,
        validate_request_model=PharmaValidationRequest,
    ),
    # add additional models here (e.g., "biotech": ModelSpec(...))
}


def get_model_spec(model_name: str) -> ModelSpec:
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'")
    return MODEL_REGISTRY[key]


def list_models() -> Dict[str, ModelSpec]:
    return dict(MODEL_REGISTRY)
