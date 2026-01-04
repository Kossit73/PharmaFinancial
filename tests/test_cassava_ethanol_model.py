import sys
import unittest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_models.model_registry import get_model_spec  # noqa: E402


class CassavaModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = get_model_spec("cassava_ethanol")
        cls.params = cls.spec.load_inputs(None)
        cls.result = cls.spec.run_core(cls.params)

    def test_financials_present(self):
        financials = self.result.get("financials")
        self.assertIsNotNone(financials)
        self.assertFalse(getattr(financials, "income_monthly").empty)
        self.assertFalse(getattr(financials, "balance_monthly").empty)
        self.assertFalse(getattr(financials, "cashflow_monthly").empty)

    def test_metrics_present(self):
        metrics = self.result.get("metrics")
        self.assertTrue(metrics)
        self.assertIn("Scenario", metrics)


if __name__ == "__main__":
    unittest.main()
