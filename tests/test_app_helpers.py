import importlib
import json
import json
import sys
import types
import unittest
from pathlib import Path


class _NoOp:
    def __call__(self, *args, **kwargs):  # pragma: no cover - defensive noop
        return None

    def __getattr__(self, _name):  # pragma: no cover - defensive noop
        return self


class DummyStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self._no_op = _NoOp()
        self.calls: list[str] = []

    def rerun(self):  # pragma: no cover - behaviour exercised in tests
        self.calls.append("rerun")
        raise RuntimeError("rerun not available")

    def experimental_rerun(self):  # pragma: no cover - exercised in tests
        self.calls.append("experimental_rerun")
        raise RuntimeError("experimental rerun not available")

    def __getattr__(self, name):  # pragma: no cover - fallback for unused attrs
        return getattr(self._no_op, name, self._no_op)


class RerunHelperTest(unittest.TestCase):
    def setUp(self):
        self.original_streamlit = sys.modules.get("streamlit")
        self.original_runtime = sys.modules.get("streamlit.runtime")

        stub = DummyStreamlit()
        runtime = types.ModuleType("streamlit.runtime")
        runtime.exists = lambda: False  # type: ignore[attr-defined]

        sys.modules["streamlit"] = stub
        sys.modules["streamlit.runtime"] = runtime

        if "pharma_financial.app" in sys.modules:
            del sys.modules["pharma_financial.app"]

        importlib.invalidate_caches()
        self.stub = stub
        self.app = importlib.import_module("pharma_financial.app")

    def tearDown(self):
        if self.original_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = self.original_streamlit

        if self.original_runtime is None:
            sys.modules.pop("streamlit.runtime", None)
        else:
            sys.modules["streamlit.runtime"] = self.original_runtime

        if "pharma_financial.app" in sys.modules:
            del sys.modules["pharma_financial.app"]

    def test_rerun_helper_handles_missing_runtime(self):
        self.app._rerun()
        self.assertEqual(self.stub.calls, ["rerun", "experimental_rerun"])

    def test_core_rows_calculations_match_inputs(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(encoding="utf-8")
        )
        rows = self.app._payload_to_core_rows(payload)
        self.assertTrue(rows)

        for row in rows:
            units = float(row["Total Production Units"])
            selling = float(row["Selling Price"])
            production = float(row["Production Cost"])
            freight = float(row["Freight Cost"])
            markup = float(row.get("Markup", 0.0))
            capacity = float(row.get("Max Capacity", 0.0))

            self.assertAlmostEqual(row["Total Revenue"], units * selling, places=8)
            self.assertAlmostEqual(
                row["Total Cost"], units * (production + freight + markup), places=8
            )
            if capacity > 0:
                self.assertLessEqual(units, capacity + 1e-9)

    def test_inventory_rows_roundtrip(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(encoding="utf-8")
        )
        rows = self.app._payload_to_inventory_rows(payload)
        self.assertTrue(rows)

        rows[0]["inventory_days"] = float(rows[0]["inventory_days"]) + 5
        rows[0]["accounts_payable_days"] = float(rows[0]["accounts_payable_days"]) + 3
        rows[0]["days_in_year"] = float(rows[0]["days_in_year"]) - 1

        self.app._inventory_rows_to_payload(rows, payload)

        working = payload.get("working_capital", {})
        calendar = working.get("calendar_days", [])
        day_mapping = working.get("days", {})
        inventory = day_mapping.get("inventory", [])
        payables = day_mapping.get("accounts_payable", [])

        self.assertAlmostEqual(calendar[0], rows[0]["days_in_year"], places=6)
        self.assertAlmostEqual(inventory[0], rows[0]["inventory_days"], places=6)
        self.assertAlmostEqual(payables[0], rows[0]["accounts_payable_days"], places=6)

    def test_receivable_rows_roundtrip(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(encoding="utf-8")
        )
        rows = self.app._payload_to_receivable_rows(payload)
        self.assertTrue(rows)

        rows[0]["accounts_receivable_days"] = float(rows[0]["accounts_receivable_days"]) + 4
        rows[0]["prepaid_expense_days"] = float(rows[0]["prepaid_expense_days"]) + 2
        rows[0]["other_asset_days"] = float(rows[0]["other_asset_days"]) + 1
        rows[0]["days_in_year"] = float(rows[0]["days_in_year"]) + 2

        self.app._receivable_rows_to_payload(rows, payload)

        working = payload.get("working_capital", {})
        calendar = working.get("calendar_days", [])
        day_mapping = working.get("days", {})
        receivable = day_mapping.get("accounts_receivable", [])
        prepaid = day_mapping.get("prepaid_expenses", [])
        other_assets = day_mapping.get("other_assets", [])

        self.assertAlmostEqual(calendar[0], rows[0]["days_in_year"], places=6)
        self.assertAlmostEqual(receivable[0], rows[0]["accounts_receivable_days"], places=6)
        self.assertAlmostEqual(prepaid[0], rows[0]["prepaid_expense_days"], places=6)
        self.assertAlmostEqual(other_assets[0], rows[0]["other_asset_days"], places=6)

    def test_ai_settings_roundtrip(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(encoding="utf-8")
        )
        settings = self.app._payload_to_ai_settings(payload)
        settings["provider"] = "Azure OpenAI"
        settings["model"] = "gpt-custom"
        settings["ml_methods"] = ["linear_regression", "moving_average"]
        settings["generative_features"] = ["summary", "risk_review"]
        settings["api_key"] = "test-key"
        self.app._ai_settings_to_payload(settings, payload)
        updated = self.app._payload_to_ai_settings(payload)
        self.assertEqual(updated["provider"], "Azure OpenAI")
        self.assertEqual(updated["model"], "gpt-custom")
        self.assertIn("moving_average", updated["ml_methods"])
        self.assertIn("risk_review", updated["generative_features"])
        self.assertEqual(updated["api_key"], "test-key")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

