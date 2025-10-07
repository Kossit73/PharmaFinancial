import json
import unittest

from pharma_financial.report import (
    ReportGenerationError,
    ReportSection,
    ReportTable,
    generate_report,
)
from pharma_financial.table import build_table


class ReportGenerationTest(unittest.TestCase):
    def setUp(self):
        table = build_table([2024, 2025], {"Metric": [1.0, 2.0]})
        self.sections = [
            ReportSection("Key Metrics Dashboard", [ReportTable("Summary", table)]),
        ]

    def test_generate_json_report(self):
        data, mime, filename = generate_report(self.sections, "JSON")
        self.assertEqual(mime, "application/json")
        self.assertTrue(filename.endswith(".json"))
        payload = json.loads(data.decode("utf-8"))
        self.assertIn("sections", payload)
        self.assertEqual(payload["sections"][0]["title"], "Key Metrics Dashboard")

    def test_generate_csv_report(self):
        data, mime, filename = generate_report(self.sections, "CSV")
        self.assertEqual(mime, "text/csv")
        self.assertTrue(filename.endswith(".csv"))
        text = data.decode("utf-8")
        self.assertIn("# Section: Key Metrics Dashboard", text)
        self.assertIn("Metric", text)

    def test_invalid_format_raises(self):
        with self.assertRaises(ReportGenerationError):
            generate_report(self.sections, "invalid")

    def test_empty_sections_raise(self):
        with self.assertRaises(ReportGenerationError):
            generate_report([], "JSON")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
