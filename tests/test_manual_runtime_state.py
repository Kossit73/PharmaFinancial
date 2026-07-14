from __future__ import annotations

from types import SimpleNamespace

from pharma_financial import app
from pharma_financial.ui import state


def test_draft_digest_change_does_not_run_model(monkeypatch) -> None:
    last_model = object()
    last_outputs = object()
    session_state = {
        "last_model": last_model,
        "last_outputs": last_outputs,
        "last_run_digest": "run-v1",
    }
    monkeypatch.setattr(app, "st", SimpleNamespace(session_state=session_state))
    monkeypatch.setattr(app, "_refresh_runtime_session_state", lambda: False)
    monkeypatch.setattr(
        state,
        "cached_model_run",
        lambda *_: (_ for _ in ()).throw(AssertionError("draft edit ran model")),
    )

    model, outputs = state.resolve_model_outputs(object(), "draft-v2")

    assert model is last_model
    assert outputs is last_outputs
    assert session_state["last_run_digest"] == "run-v1"
    assert session_state["pharma_results_stale"] is True


def test_explicit_run_activates_digest_and_clears_analysis(monkeypatch) -> None:
    model = object()
    outputs = object()
    session_state = {"run_requested": True}
    monkeypatch.setattr(app, "st", SimpleNamespace(session_state=session_state))
    monkeypatch.setattr(app, "_refresh_runtime_session_state", lambda: False)
    monkeypatch.setattr(app, "_sync_workbook_cache_with_payload_digest", lambda digest: True)
    monkeypatch.setattr(state, "cached_model_run", lambda *_: (model, outputs))
    monkeypatch.setattr(state, "clear_analysis_cache", lambda: None)

    resolved_model, resolved_outputs = state.resolve_model_outputs(object(), "draft-v2")

    assert resolved_model is model
    assert resolved_outputs is outputs
    assert session_state["last_run_digest"] == "draft-v2"
    assert session_state["pharma_results_stale"] is False
