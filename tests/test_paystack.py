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
                    "payload": {"status": True, "message": "", "data": {"customer_code": "CUS_123"}},
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

    def test_checkout_requires_plan_or_amount(self):
        client = PaystackClient(secret_key="sk_test", session=DummySession([]))

        with self.assertRaises(PaystackError):
            client.create_subscription_checkout("user@example.com")

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
