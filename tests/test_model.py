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

    def test_total_units_respect_capacity(self):
        totals = self.inputs.total_production_units
        capacity = self.inputs.production_capacity
        for product in self.inputs.products:
            cap = capacity.get(product, 0.0)
            if cap > 0:
                self.assertLessEqual(totals.get(product, 0.0), cap)

    def test_npf_irr_handles_short_series(self):
        self.assertTrue(npf_irr([100]) != float("inf"))

    def test_utility_costs_feed_total_expenses(self):
        costs = self.model.cost_structure()
        utilities = costs.column("Utilities")
        expected_utilities = self.inputs.utility_schedule.annual_totals()
        self.assertEqual(len(utilities), len(expected_utilities))
        for actual, expected in zip(utilities, expected_utilities):
            self.assertAlmostEqual(actual, expected, places=6)

        total_expenses = costs.column("Total Expenses")
        raw = costs.column("Raw Materials")
        direct = costs.column("Direct Labor")
        general = costs.column("General & Admin")
        for idx in range(len(total_expenses)):
            expected_total = raw[idx] + utilities[idx] + direct[idx] + general[idx]
            self.assertAlmostEqual(total_expenses[idx], expected_total, places=6)

    def test_interest_matches_financing_inputs(self):
        financing = self.inputs.financing
        interest_column = self.outputs.income_statement.column("Interest")
        manual: list[float] = []
        for year in self.inputs.years:
            total = 0.0
            for entry in financing.senior_debt_entries:
                if entry.year == year:
                    total += entry.amount * financing.senior_debt_interest
            for entry in financing.revolver_entries:
                if entry.year == year:
                    total += entry.amount * financing.revolver_interest
            for entry in financing.overdraft_entries:
                if entry.year == year:
                    total += entry.amount * financing.cash_interest
            manual.append(-total)

        self.assertEqual(len(interest_column), len(manual))
        for actual, expected in zip(interest_column, manual):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_liabilities_include_outstanding_balances(self):
        financing = self.inputs.financing
        liabilities = self.outputs.balance_sheet.column("Total Liabilities")
        manual: list[float] = []
        for year in self.inputs.years:
            total = 0.0
            for entry in financing.senior_debt_entries:
                if entry.year == year:
                    total += entry.outstanding
            for entry in financing.revolver_entries:
                if entry.year == year:
                    total += entry.outstanding
            for entry in financing.overdraft_entries:
                if entry.year == year:
                    total += entry.outstanding
            manual.append(total)

        self.assertEqual(len(liabilities), len(manual))
        for actual, expected in zip(liabilities, manual):
            self.assertAlmostEqual(actual, expected, places=6)


if __name__ == "__main__":
    unittest.main()

