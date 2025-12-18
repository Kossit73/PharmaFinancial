"""Compatibility shim re-exporting pharma table helpers."""

from importlib import import_module

_pharma_table = import_module("financial_models.pharma.table")
globals().update({k: v for k, v in vars(_pharma_table).items() if not k.startswith("_")})

__all__ = [name for name in globals().keys() if not name.startswith("_")]
