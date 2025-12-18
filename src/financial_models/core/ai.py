"""Compatibility shim re-exporting pharma AI helpers."""

from importlib import import_module

_pharma_ai = import_module("financial_models.pharma.ai")
globals().update({k: v for k, v in vars(_pharma_ai).items() if not k.startswith("_")})

__all__ = [name for name in globals().keys() if not name.startswith("_")]
