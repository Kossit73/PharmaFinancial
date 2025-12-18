"""Backwards-compatible shims for ``financial_models.model``."""

from .pharma import model as _core_model
from .pharma.model import *  # noqa: F401,F403

CASH_FLOW_BEGIN_COLUMN = _core_model.CASH_FLOW_BEGIN_COLUMN
CASH_FLOW_END_COLUMN = _core_model.CASH_FLOW_END_COLUMN
CASH_FLOW_NET_COLUMN = _core_model.CASH_FLOW_NET_COLUMN

__all__ = list(getattr(_core_model, "__all__", [])) + [
    "CASH_FLOW_BEGIN_COLUMN",
    "CASH_FLOW_END_COLUMN",
    "CASH_FLOW_NET_COLUMN",
]
