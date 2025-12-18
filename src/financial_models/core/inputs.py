"""Compatibility shim re-exporting pharma inputs."""

from importlib import import_module

_pharma_inputs = import_module("financial_models.pharma.inputs")
globals().update({k: v for k, v in vars(_pharma_inputs).items() if not k.startswith("_")})

__all__ = [name for name in globals().keys() if not name.startswith("_")]
