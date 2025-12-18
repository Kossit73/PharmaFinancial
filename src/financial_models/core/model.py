"""Compatibility shim re-exporting pharma financial model."""

from importlib import import_module

_pharma_model = import_module("financial_models.pharma.model")
globals().update({k: v for k, v in vars(_pharma_model).items() if not k.startswith("_")})

__all__ = [name for name in globals().keys() if not name.startswith("_")]
