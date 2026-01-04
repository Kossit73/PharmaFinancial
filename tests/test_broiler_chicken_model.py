import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_models.model_registry import get_model_spec  # noqa: E402


class BroilerModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = get_model_spec("broiler_chicken")
        cls.params = cls.spec.load_inputs(None)
        cls.result = cls.spec.run_core(cls.params)

    def test_financial_statements(self):
        fs = self.result.get("financial_statements", {})
        self.assertTrue(fs)
        self.assertTrue(fs.get("income_statement"))
        self.assertTrue(fs.get("balance_sheet"))
        self.assertTrue(fs.get("cash_flow_statement"))

    def test_valuation_present(self):
        valuation = self.result.get("valuation", {})
        self.assertIn("npv", valuation)


if __name__ == "__main__":
    unittest.main()
