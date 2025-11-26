import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:  # pragma: no cover - FastAPI optional in minimal environments
    from fastapi.testclient import TestClient  # type: ignore
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if TestClient is None:  # pragma: no cover
    raise unittest.SkipTest("FastAPI is not installed; skipping API tests.")

from pharma_financial.api.server import create_app, get_paystack_client
from pharma_financial.services.paystack import SubscriptionStatus


def _load_default_inputs() -> dict:
    inputs_path = ROOT / "src" / "pharma_financial" / "data" / "default_inputs.json"
    return json.loads(inputs_path.read_text())


def test_healthcheck_endpoint():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_run_endpoint_returns_outputs():
    client = TestClient(create_app())
    payload = {"inputs": _load_default_inputs()}
    response = client.post("/model/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary_metrics" in data
    assert "income_statement" in data
    assert "NPV" in data["summary_metrics"]["index"]


def test_inputs_validate_endpoint_flags_errors():
    client = TestClient(create_app())
    good_response = client.post("/inputs/validate", json={"inputs": _load_default_inputs()})
    assert good_response.status_code == 200
    assert good_response.json()["valid"] is True

    bad_response = client.post("/inputs/validate", json={"inputs": {"years": []}})
    assert bad_response.status_code == 200
    assert bad_response.json()["valid"] is False


def test_subscription_check_endpoint_uses_dependency_override():
    app = create_app()

    class DummyClient:
        def has_active_subscription(self, email: str) -> SubscriptionStatus:
            return SubscriptionStatus(email=email, is_active=True, message="ok", payload={"source": "test"})

    app.dependency_overrides[get_paystack_client] = lambda: DummyClient()

    client = TestClient(app)
    response = client.post("/subscriptions/check", json={"email": "user@example.com"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["is_active"] is True
    assert payload["message"] == "ok"
    assert payload["payload"]["source"] == "test"


def test_subscription_status_endpoints_roundtrip():
    with tempfile.TemporaryDirectory() as tempdir, mock.patch.dict(
        os.environ, {"SUBSCRIPTION_STORE_PATH": str(Path(tempdir) / "subs.db")}
    ):
        client = TestClient(create_app())
        email = "user@example.com"
        upsert = client.post(
            "/subscriptions/status",
            json={
                "email": email,
                "is_active": True,
                "status_message": "Active",
                "payload": {"source": "test"},
                "source": "testcase",
                "ttl_seconds": 60,
            },
        )
        assert upsert.status_code == 200
        data = upsert.json()
        assert data["email"] == email
        assert data["is_active"] is True
        status = client.get("/subscriptions/status", params={"email": email})
        assert status.status_code == 200
        fetched = status.json()
        assert fetched["status_message"] == "Active"
        delete = client.delete("/subscriptions/status", params={"email": email})
        assert delete.status_code == 204
        missing = client.get("/subscriptions/status", params={"email": email})
        assert missing.status_code == 404
