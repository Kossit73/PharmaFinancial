import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import pandas as pd  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency guard
    pd = None

if pd is not None:
    from financial_models.microbrewery.inputs import load_inputs, parse_inputs
    from financial_models.microbrewery.model import MicrobreweryFinancialModel


@unittest.skipUnless(pd is not None, "pandas required for microbrewery model")
class MicrobreweryModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = load_inputs(Path("src/financial_models/microbrewery/data/default_inputs.json"))
        cls.model = MicrobreweryFinancialModel(cls.params.config, cls.params.dividend_policy, cls.params.inputs)
        cls.result = cls.model.run()

    def test_monthly_has_key_lines(self):
        monthly = self.result.monthly
        self.assertEqual(len(monthly), self.params.config.months)
        for column in [
            "revenue",
            "direct_costs",
            "ebitda",
            "net_income",
            "cash",
            "debt_ending_balance",
        ]:
            self.assertIn(column, monthly.columns)

    def test_debt_schedules_exist(self):
        facilities = self.params.inputs.debt_facilities or []
        if facilities:
            for facility in facilities:
                self.assertIn(facility.name, self.result.debt_schedules)

    def test_valuation_metrics_present(self):
        valuation = self.result.valuation
        for key in [
            "enterprise_value_dcf",
            "equity_value_exit",
            "equity_irr_annual",
            "equity_moic",
        ]:
            self.assertIn(key, valuation)
            self.assertIsInstance(valuation[key], float)

    def test_parse_rejects_bad_payload(self):
        with self.assertRaises(ValueError):
            parse_inputs({"config": {}, "skus": [], "channels": [], "sales_plan": []})


if __name__ == "__main__":
    unittest.main()
