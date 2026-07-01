import json
import sys
import unittest
from pathlib import Path
import tempfile
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_financial.inputs import load_inputs
from pharma_financial.model import FinancialModel
from pharma_financial.report import ReportSection, ReportTable, generate_report


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot_dir = Path(__file__).resolve().parent / "snapshots"
        cls.snapshot_dir.mkdir(parents=True, exist_ok=True)
        inputs = load_inputs(Path("src/pharma_financial/data/default_inputs.json"))
        model = FinancialModel(inputs)
        cls.outputs = model.run()

    def test_summary_metrics_csv_snapshot(self):
        snapshot_path = self.snapshot_dir / "summary_metrics.csv"
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir) / "summary_metrics.csv"
            self.outputs.summary_metrics.to_csv(temp_path)
            actual = self._read_csv(temp_path)
        expected = self._read_csv(snapshot_path)
        self.assertEqual(actual, expected)

    def test_report_json_snapshot(self):
        sections = [
            ReportSection(
                "Summary Metrics",
                [ReportTable("Summary Metrics", self.outputs.summary_metrics)],
            )
        ]
        report_bytes, _, _ = generate_report(sections, "JSON")
        payload = json.loads(report_bytes.decode("utf-8"))
        payload["generated"] = "TIMESTAMP"
        snapshot_path = self.snapshot_dir / "report.json"
        expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
        self.assertEqual(payload, expected)

    def _read_csv(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = []
            for row in reader:
                normalized = []
                for value in row:
                    if value == "" or value.lower() == "nan":
                        normalized.append(None)
                    else:
                        normalized.append(value)
                rows.append(normalized)
            return rows


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
