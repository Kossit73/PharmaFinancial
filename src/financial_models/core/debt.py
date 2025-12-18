"""Compatibility shim re-exporting pharma debt helpers."""

from importlib import import_module

_pharma_debt = import_module("financial_models.pharma.debt")
globals().update({k: v for k, v in vars(_pharma_debt).items() if not k.startswith("_")})

__all__ = [name for name in globals().keys() if not name.startswith("_")]
