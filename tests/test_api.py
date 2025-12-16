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

from financial_models.api.server import API_TOKEN_HEADER, create_app, get_paystack_client
from financial_models.services.paystack import SubscriptionStatus


def _load_default_inputs() -> dict:
    inputs_path = ROOT / "src" / "financial_models" / "data" / "default_inputs.json"
    return json.loads(inputs_path.read_text())


def test_healthcheck_endpoint():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_run_endpoint_returns_outputs():
    client = TestClient(create_app())
    payload = {"inputs": _load_default_inputs()}
    response = client.post("/model/pharma/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary_metrics" in data
    assert "income_statement" in data
    assert "NPV" in data["summary_metrics"]["index"]


def test_inputs_validate_endpoint_flags_errors():
    client = TestClient(create_app())
    good_response = client.post("/inputs/pharma/validate", json={"inputs": _load_default_inputs()})
    assert good_response.status_code == 200
    assert good_response.json()["valid"] is True

    bad_response = client.post("/inputs/pharma/validate", json={"inputs": {"years": []}})
    assert bad_response.status_code == 200
    assert bad_response.json()["valid"] is False


def test_model_run_rejects_unknown_model_type():
    client = TestClient(create_app())
    payload = {"inputs": _load_default_inputs()}
    response = client.post("/model/unknown-model/run", json=payload)
    assert response.status_code == 404


def test_model_run_versioned_path():
    client = TestClient(create_app())
    payload = {"inputs": _load_default_inputs()}
    response = client.post("/model/pharma/run", json=payload)
    assert response.status_code == 200


def test_inputs_validate_versioned_path():
    client = TestClient(create_app())
    good_response = client.post("/inputs/pharma/validate", json={"inputs": _load_default_inputs()})
    assert good_response.status_code == 200
    assert good_response.json()["valid"] is True


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


def test_api_key_required_when_configured():
    payload = {"inputs": _load_default_inputs()}
    with mock.patch.dict(os.environ, {"PHARMA_FINANCIAL_API_TOKEN": "super-secret"}):
        client = TestClient(create_app())
        unauthenticated = client.post("/model/pharma/run", json=payload)
        assert unauthenticated.status_code == 401
        authorised = client.post("/model/pharma/run", json=payload, headers={API_TOKEN_HEADER: "super-secret"})
        assert authorised.status_code == 200


def test_google_authentication_enforced_when_audience_configured():
    payload = {"inputs": _load_default_inputs()}
    with mock.patch.dict(os.environ, {"PHARMA_FINANCIAL_GOOGLE_AUDIENCE": "client-1"}):
        client = TestClient(create_app())
        response = client.post("/model/pharma/run", json=payload)
        assert response.status_code == 401


def test_google_authentication_with_valid_bearer_token():
    payload = {"inputs": _load_default_inputs()}
    with mock.patch.dict(os.environ, {"PHARMA_FINANCIAL_GOOGLE_AUDIENCE": "client-1"}):
        with mock.patch("financial_models.api.server._verify_google_token", return_value={"sub": "abc", "email": "a@b.com"}) as verify:
            client = TestClient(create_app())
            response = client.post(
                "/model/pharma/run",
                json=payload,
                headers={"Authorization": "Bearer valid-token"},
            )
            assert response.status_code == 200
            verify.assert_called_once_with("valid-token", ["client-1"])


def test_auth_register_and_login_allows_model_run(tmp_path):
    payload = {"inputs": _load_default_inputs()}
    env = {
        "FINANCIAL_MODELS_AUTH_SECRET": "dev-secret",
        "FINANCIAL_MODELS_USER_DB": str(tmp_path / "users.db"),
    }
    with mock.patch.dict(os.environ, env):
        client = TestClient(create_app())
        register = client.post("/auth/register", params={"email": "user@example.com", "password": "pass"})
        assert register.status_code == 200
        token = register.json()["access_token"]
        login = client.post(
            "/auth/login",
            data={"username": "user@example.com", "password": "pass"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert login.status_code == 200
        bearer = {"Authorization": f"Bearer {token}"}
        authorised = client.post("/model/pharma/run", json=payload, headers=bearer)
        assert authorised.status_code == 200


def test_auth_update_and_delete(tmp_path):
    env = {
        "FINANCIAL_MODELS_AUTH_SECRET": "dev-secret",
        "FINANCIAL_MODELS_USER_DB": str(tmp_path / "users.db"),
    }
    with mock.patch.dict(os.environ, env):
        client = TestClient(create_app())
        register = client.post("/auth/register", params={"email": "user@example.com", "password": "pass", "name": "Old"})
        assert register.status_code == 200
        token = register.json()["access_token"]
        bearer = {"Authorization": f"Bearer {token}"}
        update = client.patch("/auth/me", json={"name": "New Name"}, headers=bearer)
        assert update.status_code == 200
        assert update.json()["name"] == "New Name"
        delete = client.delete("/auth/users/user@example.com", headers=bearer)
        assert delete.status_code == 204
        relogin = client.post(
            "/auth/login",
            data={"username": "user@example.com", "password": "pass"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert relogin.status_code == 401


def test_auth_list_users(tmp_path):
    env = {
        "FINANCIAL_MODELS_AUTH_SECRET": "dev-secret",
        "FINANCIAL_MODELS_USER_DB": str(tmp_path / "users.db"),
    }
    with mock.patch.dict(os.environ, env):
        client = TestClient(create_app())
        r1 = client.post("/auth/register", params={"email": "one@example.com", "password": "pass"})
        token1 = r1.json()["access_token"]
        r2 = client.post("/auth/register", params={"email": "two@example.com", "password": "pass"})
        assert r2.status_code == 200
        bearer = {"Authorization": f"Bearer {token1}"}
        listing = client.get("/auth/users", headers=bearer)
        assert listing.status_code == 200
        users = listing.json()["users"]
        emails = {u["email"] for u in users}
        assert "one@example.com" in emails
        assert "two@example.com" in emails
