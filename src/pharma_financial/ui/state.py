"""Workspace state and caching helpers extracted from the legacy app shell."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from collections.abc import Mapping

from ..inputs import ModelInputs, parse_inputs
from ..model import FinancialModel, FinancialOutputs


def cached_parse_inputs(payload: Mapping[str, object]) -> tuple[ModelInputs, str]:
    from .. import app as legacy

    digest = legacy._payload_digest(payload)
    cached = legacy._INPUT_CACHE.get(digest)
    if cached is None:
        normalised = legacy._normalise_payload(payload)
        cached = parse_inputs(normalised)
        legacy._INPUT_CACHE[digest] = cached
    return cached, digest


def cached_model_run(inputs: ModelInputs, digest: str) -> tuple[FinancialModel, FinancialOutputs]:
    from .. import app as legacy

    cached = legacy._MODEL_CACHE.get(digest)
    if cached is not None:
        return cached

    model = FinancialModel(inputs)
    outputs = model.run_core()
    legacy._MODEL_CACHE[digest] = (model, outputs)
    return model, outputs


def clear_analysis_cache() -> None:
    from .. import app as legacy

    for key in legacy._ANALYSIS_CACHE_KEYS:
        legacy.st.session_state.pop(key, None)
        legacy.st.session_state.pop(f"{key}_digest", None)


def analysis_cache_value(key: str, digest: str, fallback):
    from .. import app as legacy

    cached = legacy.st.session_state.get(key)
    cached_digest = legacy.st.session_state.get(f"{key}_digest")
    if cached is None or cached_digest != digest:
        return fallback
    return cached


def store_analysis_cache(key: str, digest: str, value) -> None:
    from .. import app as legacy

    legacy.st.session_state[key] = value
    legacy.st.session_state[f"{key}_digest"] = digest


def merge_analysis_outputs(outputs: FinancialOutputs, digest: str) -> FinancialOutputs:
    return replace(
        outputs,
        sensitivity_results=analysis_cache_value(
            "sensitivity_results", digest, outputs.sensitivity_results
        ),
        monte_carlo=analysis_cache_value(
            "monte_carlo_results", digest, outputs.monte_carlo
        ),
        ai_insights=analysis_cache_value("ai_insights", digest, outputs.ai_insights),
    )


def resolve_model_outputs(
    inputs: ModelInputs, digest: str
) -> tuple[FinancialModel | None, FinancialOutputs | None]:
    from .. import app as legacy

    run_requested = bool(legacy.st.session_state.pop("run_requested", False))
    last_digest = legacy.st.session_state.get("last_run_digest")
    model = legacy.st.session_state.get("last_model")
    outputs = legacy.st.session_state.get("last_outputs")

    if run_requested or last_digest != digest:
        model, outputs = cached_model_run(inputs, digest)
        legacy.st.session_state["last_model"] = model
        legacy.st.session_state["last_outputs"] = outputs
        legacy.st.session_state["last_run_digest"] = digest
        clear_analysis_cache()

    return model, outputs


def get_state() -> dict:
    from .. import app as legacy
    import copy as _copy

    state: dict[str, Any] = {}
    payload = legacy.st.session_state.get("input_payload")
    if payload is not None:
        state["input_payload"] = _copy.deepcopy(payload)

    for key in ["labor_mode"] + list(legacy._PHARMA_ROW_KEYS):
        value = legacy.st.session_state.get(key)
        if value is not None:
            state[key] = list(value) if isinstance(value, set) else value

    return state


def set_state(state: dict) -> None:
    from .. import app as legacy

    for key, value in state.items():
        if key == "break_even_overrides" and isinstance(value, list):
            legacy.st.session_state[key] = set(value)
        else:
            legacy.st.session_state[key] = value

