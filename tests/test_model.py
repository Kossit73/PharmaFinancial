import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_financial.inputs import load_inputs
from pharma_financial.model import FinancialModel, npf_irr


class FinancialModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_inputs(Path("src/pharma_financial/data/default_inputs.json"))
        cls.model = FinancialModel(cls.inputs)
        cls.outputs = cls.model.run()

    def test_income_statement_columns(self):
        income = self.outputs.income_statement
        self.assertEqual(income.index, self.inputs.years)
        self.assertIn("Net Revenue", income.data)
        self.assertIn("EBITDA", income.data)

    def test_cash_flow_consistency(self):
        cash_flow = self.outputs.cash_flow
        net_change = cash_flow.column("Net Change in Cash")
        ending = cash_flow.column("Ending Cash")
        self.assertEqual(len(net_change), len(self.inputs.years))
        self.assertEqual(len(ending), len(self.inputs.years))

    def test_summary_metrics_index(self):
        summary = self.outputs.summary_metrics
        expected = ["NPV", "IRR", "Payback Period", "Discounted Payback"]
        self.assertEqual(summary.index, expected)

    def test_npf_irr_handles_short_series(self):
        self.assertTrue(npf_irr([100]) != float("inf"))


if __name__ == "__main__":
    unittest.main()

