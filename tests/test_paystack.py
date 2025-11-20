import unittest

from pharma_financial.paystack import PaystackClient, PaystackError


class DummyResponse:
    def __init__(self, *, status_code=200, payload=None, reason="OK"):
        self.status_code = status_code
        self._payload = payload or {}
        self.reason = reason

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, headers=None, timeout=None, **kwargs):
        if not self.responses:
            raise AssertionError("No stub responses remaining for request: %s %s" % (method, url))
        self.calls.append({"method": method, "url": url, "headers": headers, "timeout": timeout, "kwargs": kwargs})
        response_payload = self.responses.pop(0)
        return DummyResponse(**response_payload)


class PaystackClientTest(unittest.TestCase):
    def test_active_subscription_detected(self):
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": [{"email": "user@example.com", "customer_code": "CUS_123", "id": 42}],
                    },
                },
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {"subscriptions": []},
                    },
                },
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {"subscriptions": []},
                    },
                },
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": [{"status": "active", "subscription_code": "SUB_111"}],
                    },
                },
            ]
        )
        client = PaystackClient(secret_key="sk_test", session=session)

        status = client.has_active_subscription("user@example.com")

        self.assertTrue(status.is_active)
        self.assertEqual(status.email, "user@example.com")
        self.assertEqual(len(session.calls), 4)

    def test_active_subscription_detected_inline_customer_detail(self):
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": [{"email": "user@example.com", "customer_code": "CUS_789"}],
                    },
                },
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {
                            "subscriptions": [
                                {"status": "active", "subscription_code": "SUB_INLINE"},
                            ]
                        },
                    },
                },
            ]
        )
        client = PaystackClient(secret_key="sk_test", session=session)

        status = client.has_active_subscription("user@example.com")

        self.assertTrue(status.is_active)
        self.assertEqual(status.payload.get("subscription_code"), "SUB_INLINE")
        self.assertEqual(len(session.calls), 2)

    def test_inactive_when_customer_missing(self):
        session = DummySession(
            [
                {
                    "status_code": 404,
                    "payload": {"status": False, "message": "Customer not found", "data": None},
                }
            ]
        )
        client = PaystackClient(secret_key="sk_test", session=session)

        status = client.has_active_subscription("missing@example.com")

        self.assertFalse(status.is_active)
        self.assertEqual(len(session.calls), 1)

    def test_successful_transaction_grants_access(self):
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": [{"email": "user@example.com", "customer_code": "CUS_456", "id": 77}],
                    },
                },
                {
                    "status_code": 200,
                    "payload": {"status": True, "message": "", "data": {"subscriptions": []}},
                },
                {
                    "status_code": 200,
                    "payload": {"status": True, "message": "", "data": {"subscriptions": []}},
                },
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": [],
                    },
                },
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": [],
                    },
                },
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": [],
                    },
                },
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": [{"status": "success", "reference": "TRX_1"}],
                    },
                },
            ]
        )
        client = PaystackClient(secret_key="sk_test", session=session)

        status = client.has_active_subscription("user@example.com")

        self.assertTrue(status.is_active)
        self.assertIn("Successful transaction", status.message)
        self.assertEqual(len(session.calls), 6)

    def test_checkout_requires_plan_or_amount(self):
        client = PaystackClient(secret_key="sk_test", session=DummySession([]))

        with self.assertRaises(PaystackError):
            client.create_subscription_checkout("user@example.com")

    def test_get_subscriptions_handles_pagination(self):
        first_batch = [{"status": "inactive", "subscription_code": f"SUB_{i}"} for i in range(50)]
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {"status": True, "message": "", "data": first_batch},
                },
                {
                    "status_code": 200,
                    "payload": {"status": True, "message": "", "data": [{"status": "active", "subscription_code": "SUB_LAST"}]},
                },
            ]
        )
        client = PaystackClient(secret_key="sk_test", session=session)

        subs = client.get_subscriptions_for_customer("CUS_PAGE")

        self.assertEqual(len(subs), 51)
        self.assertEqual(session.calls[0]["kwargs"]["params"]["page"], 1)
        self.assertEqual(session.calls[1]["kwargs"]["params"]["page"], 2)

    def test_verify_transaction_returns_payload(self):
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {"status": "success", "reference": "TRX_123"},
                    },
                }
            ]
        )
        client = PaystackClient(secret_key="sk_test", session=session)

        tx = client.verify_transaction("TRX_123")

        self.assertEqual(tx.get("reference"), "TRX_123")
        self.assertTrue(session.calls[0]["url"].endswith("/transaction/verify/TRX_123"))

    def test_checkout_initialization_returns_link(self):
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {"authorization_url": "https://checkout.paystack.com/abc"},
                    },
                }
            ]
        )
        client = PaystackClient(secret_key="sk_test", session=session)

        url = client.create_subscription_checkout("user@example.com", plan_code="PLAN_TEST")

        self.assertEqual(url, "https://checkout.paystack.com/abc")
        payload = session.calls[0]["kwargs"].get("json", {})
        self.assertEqual(payload.get("plan"), "PLAN_TEST")
        self.assertEqual(payload.get("email"), "user@example.com")

    def test_checkout_includes_callback_when_configured(self):
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {"authorization_url": "https://checkout.paystack.com/callback"},
                    },
                }
            ]
        )
        client = PaystackClient(
            secret_key="sk_test",
            session=session,
            plan_code="PLAN_TEST",
            callback_url="https://app.example.com/paystack/callback",
        )

        client.create_subscription_checkout("user@example.com")

        payload = session.calls[0]["kwargs"].get("json", {})
        self.assertEqual(payload.get("callback_url"), "https://app.example.com/paystack/callback")

    def test_checkout_callback_can_be_overridden(self):
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {"authorization_url": "https://checkout.paystack.com/callback"},
                    },
                }
            ]
        )
        client = PaystackClient(
            secret_key="sk_test",
            session=session,
            plan_code="PLAN_TEST",
            callback_url="https://app.example.com/paystack/callback",
        )

        client.create_subscription_checkout(
            "user@example.com", callback_url="https://override.example.com/paystack"
        )

        payload = session.calls[0]["kwargs"].get("json", {})
        self.assertEqual(payload.get("callback_url"), "https://override.example.com/paystack")
        metadata = payload.get("metadata") or {}
        self.assertEqual(metadata.get("cancel_action"), "https://override.example.com/paystack")

    def test_checkout_metadata_includes_cancel_action(self):
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {"authorization_url": "https://checkout.paystack.com/abc"},
                    },
                }
            ]
        )
        client = PaystackClient(
            secret_key="sk_test",
            session=session,
            plan_code="PLAN_TEST",
            callback_url="https://app.example.com/paystack/callback",
        )

        client.create_subscription_checkout("user@example.com", metadata={"source": "app"})

        metadata = session.calls[0]["kwargs"]["json"].get("metadata") or {}
        self.assertEqual(metadata.get("source"), "app")
        self.assertEqual(metadata.get("cancel_action"), "https://app.example.com/paystack/callback")

    def test_checkout_metadata_cancel_action_override(self):
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {"authorization_url": "https://checkout.paystack.com/abc"},
                    },
                }
            ]
        )
        client = PaystackClient(
            secret_key="sk_test",
            session=session,
            plan_code="PLAN_TEST",
            callback_url="https://app.example.com/paystack/callback",
        )

        client.create_subscription_checkout(
            "user@example.com",
            metadata={"source": "app"},
            cancel_action_url="https://app.example.com/paystack/cancelled",
        )

        metadata = session.calls[0]["kwargs"]["json"].get("metadata") or {}
        self.assertEqual(metadata.get("source"), "app")
        self.assertEqual(metadata.get("cancel_action"), "https://app.example.com/paystack/cancelled")

    def test_checkout_uses_default_amount_when_configured(self):
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {"authorization_url": "https://checkout.paystack.com/abc"},
                    },
                }
            ]
        )
        client = PaystackClient(
            secret_key="sk_test", session=session, plan_code="PLAN_TEST", default_amount_kobo=50000
        )

        client.create_subscription_checkout("user@example.com")

        payload = session.calls[0]["kwargs"].get("json", {})
        self.assertEqual(payload.get("amount"), 50000)

    def test_checkout_infers_amount_from_plan_lookup(self):
        session = DummySession(
            [
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {"amount": 75000},
                    },
                },
                {
                    "status_code": 200,
                    "payload": {
                        "status": True,
                        "message": "",
                        "data": {"authorization_url": "https://checkout.paystack.com/xyz"},
                    },
                },
            ]
        )
        client = PaystackClient(secret_key="sk_test", session=session, plan_code="PLAN_PLAN")

        client.create_subscription_checkout("user@example.com")

        self.assertEqual(len(session.calls), 2)
        self.assertTrue(session.calls[0]["url"].endswith("/plan/PLAN_PLAN"))
        payload = session.calls[1]["kwargs"].get("json", {})
        self.assertEqual(payload.get("amount"), 75000)
