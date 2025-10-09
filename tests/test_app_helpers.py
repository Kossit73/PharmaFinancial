import importlib
import json
import sys
import types
import unittest
from io import BytesIO
from pathlib import Path

try:  # pragma: no cover - optional dependency check
    from docx import Document
except Exception:  # pragma: no cover - import guard for missing optional dependency
    Document = None  # type: ignore

try:  # pragma: no cover - optional dependency check
    from fpdf import FPDF
except Exception:  # pragma: no cover - import guard for missing optional dependency
    FPDF = None  # type: ignore

try:  # pragma: no cover - optional dependency check
    from openpyxl import Workbook
except Exception:  # pragma: no cover - import guard for missing optional dependency
    Workbook = None  # type: ignore


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
        self.app._INPUT_CACHE.clear()
        self.app._MODEL_CACHE.clear()

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

    def test_projection_horizon_dropdown_updates_years(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(
                encoding="utf-8"
            )
        )

        new_years = list(range(2026, 2031))
        labels = [str(year) for year in new_years]
        self.app._align_payload_horizon(payload, labels, len(new_years), update_years=True)

        self.assertEqual(payload["years"], new_years)
        self.assertEqual(len(payload.get("inflation_series", [])), len(new_years))

        working = payload.get("working_capital", {})
        calendar = working.get("calendar_days", []) if isinstance(working, dict) else []
        self.assertEqual(len(calendar), len(new_years))

    def test_align_payload_preserves_calendar_years_when_not_updating(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(
                encoding="utf-8"
            )
        )

        original_years = list(payload["years"])
        labels = [f"Year {index + 1}" for index in range(len(original_years))]
        self.app._align_payload_horizon(payload, labels, len(labels))

        self.assertEqual(payload["years"], original_years)

    def test_clone_payload_returns_independent_copy(self):
        payload = {"a": {"b": 1}}
        clone = self.app._clone_payload(payload)

        self.assertEqual(clone, payload)
        clone["a"]["b"] = 2
        self.assertEqual(payload["a"]["b"], 1)

    def test_generate_workspace_label_advances_index(self):
        existing = {"Workspace 1": {}, "Workspace 2": {}}
        label = self.app._generate_workspace_label(existing)

        self.assertEqual(label, "Workspace 3")

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

    def test_commission_rows_roundtrip(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(encoding="utf-8")
        )
        rows = self.app._payload_to_commission_rows(payload)
        self.assertTrue(rows)

        rows[0]["Commission (%)"] = float(rows[0]["Commission (%)"]) + 1.5
        rows[0]["Revenue Share (%)"] = 80.0
        rows[0]["Payment Days"] = int(rows[0]["Payment Days"]) + 10

        self.app._commission_rows_to_payload(rows, payload)

        commission_section = payload.get("distributor_commission", {})
        stored_rows = commission_section.get("rows", [])
        self.assertTrue(stored_rows)
        first = stored_rows[0]
        self.assertAlmostEqual(first["rate"] * 100.0, rows[0]["Commission (%)"], places=6)
        self.assertAlmostEqual(first["revenue_share"] * 100.0, rows[0]["Revenue Share (%)"], places=6)
        self.assertEqual(first["payment_days"], rows[0]["Payment Days"])

    def test_utility_rows_extend_projection_horizon(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(encoding="utf-8")
        )
        rows = self.app._payload_to_utility_rows(payload)
        original_years = list(payload["years"])
        new_year = original_years[-1] + 1

        new_row = dict(rows[-1])
        new_row["label"] = str(new_year)
        new_row["year"] = new_year
        rows.append(new_row)

        self.app._utility_rows_to_payload(rows, payload)

        self.assertEqual(len(payload["years"]), len(rows))
        self.assertEqual(payload["years"][-1], new_year)

    def test_receivable_rows_do_not_shrink_horizon(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(encoding="utf-8")
        )
        original_years = list(payload["years"])
        rows = self.app._payload_to_receivable_rows(payload)

        trimmed = rows[:2]
        self.app._receivable_rows_to_payload(trimmed, payload)

        self.assertEqual(payload["years"], original_years)

    def test_cached_parse_and_model_run_reuse_digest(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(encoding="utf-8")
        )
        inputs_a, digest_a = self.app._cached_parse_inputs(payload)
        inputs_b, digest_b = self.app._cached_parse_inputs(payload)

        self.assertIs(inputs_a, inputs_b)
        self.assertEqual(digest_a, digest_b)

        model1, outputs1 = self.app._cached_model_run(inputs_a, digest_a)
        model2, outputs2 = self.app._cached_model_run(inputs_b, digest_b)

        self.assertIs(model1, model2)
        self.assertIs(outputs1, outputs2)

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


class UploadLoaderTests(unittest.TestCase):
    def setUp(self):
        if "pharma_financial.app" in sys.modules:
            importlib.reload(sys.modules["pharma_financial.app"])
            self.app = sys.modules["pharma_financial.app"]
        else:
            self.app = importlib.import_module("pharma_financial.app")

        self.sample_payload = {
            "example": True,
            "years": [2024, 2025],
        }
        self.json_text = json.dumps(self.sample_payload)

    def test_load_from_json_bytes(self):
        loaded = self.app._load_payload_from_bytes(self.json_text.encode("utf-8"), ".json")
        self.assertEqual(loaded["example"], True)

    def test_load_from_csv_fragment(self):
        csv_text = f"header,value\njson,{self.json_text}\n"
        loaded = self.app._load_payload_from_bytes(csv_text.encode("utf-8"), ".csv")
        self.assertEqual(loaded["years"], [2024, 2025])

    @unittest.skipUnless(Workbook is not None, "openpyxl is required for this test")
    def test_load_from_excel_fragment(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet["A1"] = "Assumptions"
        sheet["A2"] = self.json_text
        buffer = BytesIO()
        workbook.save(buffer)
        loaded = self.app._load_payload_from_bytes(buffer.getvalue(), ".xlsx")
        self.assertIn("example", loaded)

    @unittest.skipUnless(Document is not None, "python-docx is required for this test")
    def test_load_from_docx_fragment(self):
        document = Document()
        document.add_paragraph("Example assumptions")
        document.add_paragraph(self.json_text)
        buffer = BytesIO()
        document.save(buffer)
        loaded = self.app._load_payload_from_bytes(buffer.getvalue(), ".docx")
        self.assertEqual(loaded["years"], [2024, 2025])

    @unittest.skipUnless(FPDF is not None, "fpdf is required for this test")
    def test_load_from_pdf_fragment(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"Example assumptions\n{self.json_text}")
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        loaded = self.app._load_payload_from_bytes(pdf_bytes, ".pdf")
        self.assertTrue(loaded["example"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

