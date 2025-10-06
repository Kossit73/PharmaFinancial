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
        self.assertIn("Total Depreciation Expense", income.data)

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

    def test_goal_seek_metric_matches_income_statement(self):
        self.assertIsNotNone(self.inputs.goal_seek)
        goal_table = self.outputs.goal_seek
        self.assertTrue(goal_table.index)
        metric_name = goal_table.index[0]
        self.assertEqual(metric_name, self.inputs.goal_seek.metric)
        actual_value = goal_table.data["Actual"][0]
        income_last = self.outputs.income_statement.column(metric_name)[-1]
        self.assertAlmostEqual(actual_value, income_last, places=6)

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

    def test_depreciation_schedule_feeds_statements(self):
        depreciation = self.model.depreciation_schedule()
        income_dep = self.outputs.income_statement.column("Depreciation")
        dep_expense = self.outputs.income_statement.column("Total Depreciation Expense")
        self.assertEqual(depreciation, income_dep)
        for actual, expense in zip(depreciation, dep_expense):
            self.assertAlmostEqual(actual, expense, places=6)

        details, per_year_depr, per_year_nb = self.model._depreciation_rollforward()
        self.assertTrue(details)

        net_ppe = self.outputs.balance_sheet.column("Net PP&E")
        for idx, year in enumerate(self.inputs.years):
            self.assertAlmostEqual(per_year_depr.get(year, 0.0), depreciation[idx], places=6)
            self.assertAlmostEqual(per_year_nb.get(year, 0.0), net_ppe[idx], places=6)

        by_asset: dict[str, list[dict]] = {}
        for entry in details:
            by_asset.setdefault(entry["asset_type"], []).append(entry)

        for asset_entries in by_asset.values():
            asset_entries.sort(key=lambda item: item["year"])
            previous_net_book = None
            previous_cumulative = None
            for entry in asset_entries:
                opening_net = float(entry["opening_net_book"])
                acquisition = float(entry["acquisition"])
                total_cost = float(entry["total_asset_cost"])
                total_dep = float(entry["total_depreciation"])
                cumulative_dep = float(entry["cumulative_depreciation"])
                rate = float(entry["depreciation_rate"])
                base = float(entry["depreciable_base"])

                if previous_net_book is not None:
                    self.assertAlmostEqual(opening_net, previous_net_book, places=5)

                expected_total = acquisition + opening_net
                self.assertAlmostEqual(total_cost, expected_total, places=6)

                expected_base = opening_net + (acquisition * 0.5)
                self.assertAlmostEqual(base, expected_base, places=6)

                base_cumulative = 0.0 if previous_cumulative is None else previous_cumulative
                allowable = max(total_cost - base_cumulative, 0.0)
                expected_depreciation = min(base * rate, allowable)
                self.assertAlmostEqual(total_dep, expected_depreciation, places=5)

                expected_cumulative = base_cumulative + expected_depreciation
                self.assertAlmostEqual(cumulative_dep, expected_cumulative, places=5)

                expected_net_book = max(total_cost - expected_cumulative, 0.0)
                self.assertAlmostEqual(float(entry["net_book_value"]), expected_net_book, places=5)

                previous_net_book = float(entry["net_book_value"])
                previous_cumulative = cumulative_dep


if __name__ == "__main__":
    unittest.main()

