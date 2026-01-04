import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_models.goat_farming import GoatModelParameters
from financial_models.goat_farming.inputs import load_inputs, parse_inputs


class GoatModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = load_inputs()
        model = cls.params.schedule.to_model()
        scenario_cfg = cls.params.scenario or {}
        cls.scenario = model.scenario(
            milk_price_pct=scenario_cfg.get("milk_price_pct", 0.0) or 0.0,
            feed_cost_pct=scenario_cfg.get("feed_cost_pct", 0.0) or 0.0,
        )
        cls.performance = model.statement_of_financial_performance(cls.scenario, annual=True)
        cls.cash_flow = model.statement_of_cash_flow(cls.scenario, annual=True)
        cls.position = model.statement_of_financial_position(cls.scenario, annual=True)
        cls.kpis = model.kpis(cls.scenario, annual=True)
        cls.break_even = model.break_even(cls.scenario, annual=True)
        cls.advanced = model.advanced_analytics(cls.scenario, window=3, annual=True)

    def test_schedule_and_statements_present(self):
        self.assertIsInstance(self.params, GoatModelParameters)
        self.assertFalse(self.params.schedule.data.empty)
        for frame in [
            self.scenario,
            self.performance,
            self.cash_flow,
            self.position,
            self.kpis,
            self.break_even,
        ]:
            self.assertFalse(frame.empty)

    def test_advanced_outputs_exist(self):
        self.assertTrue(self.advanced)
        for analysis in self.advanced.values():
            self.assertIn("tables", analysis)
            self.assertTrue(analysis["tables"])

    def test_validation_rejects_missing_schedule(self):
        with self.assertRaises(ValueError):
            parse_inputs({})


if __name__ == "__main__":
    unittest.main()
