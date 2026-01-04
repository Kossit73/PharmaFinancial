"""Broiler chicken financial model package (integrated adapter)."""

from .assumptions import Assumptions
from .inputs import BroilerModelParameters, load_inputs, parse_inputs
from .model import generate_model_outputs

__all__ = [
    "Assumptions",
    "BroilerModelParameters",
    "generate_model_outputs",
    "load_inputs",
    "parse_inputs",
]
