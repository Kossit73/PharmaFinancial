import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_financial.inputs import load_inputs, parse_inputs
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
        interest_column = self.outputs.income_statement.column("Interest")
        senior_interest, _ = self.model._senior_debt_schedules()
        revolver_interest, _ = self.model._revolver_schedules()
        overdraft_interest, _ = self.model._overdraft_schedules()
        manual: list[float] = []
        for idx in range(len(self.inputs.years)):
            total = senior_interest[idx]
            total += revolver_interest[idx]
            total += overdraft_interest[idx]
            manual.append(-total)

        self.assertEqual(len(interest_column), len(manual))
        for actual, expected in zip(interest_column, manual):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_liabilities_include_working_capital_and_debt(self):
        balance_sheet = self.outputs.balance_sheet
        liabilities = balance_sheet.column("Total Liabilities")
        working_capital = self.model._working_capital_balances()
        payables = working_capital.column("Accounts Payable")
        other_current = working_capital.column("Other Liabilities")
        _, senior_outstanding = self.model._senior_debt_schedules()
        _, revolver_outstanding = self.model._revolver_schedules()
        _, overdraft_outstanding = self.model._overdraft_schedules()

        manual: list[float] = []
        for idx in range(len(self.inputs.years)):
            total = senior_outstanding[idx]
            total += revolver_outstanding[idx]
            total += overdraft_outstanding[idx]
            total += payables[idx]
            total += other_current[idx]
            manual.append(total)

        self.assertEqual(len(liabilities), len(manual))
        for actual, expected in zip(liabilities, manual):
            self.assertAlmostEqual(actual, expected, places=6)

        overdraft_column = balance_sheet.column("Overdraft")
        for actual, expected in zip(overdraft_column, overdraft_outstanding):
            self.assertAlmostEqual(actual, expected, places=6)

        payable_column = balance_sheet.column("Accounts Payable")
        for actual, expected in zip(payable_column, payables):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_monte_carlo_defaults_include_revenue_growth(self):
        self.assertIn("revenue_growth", self.inputs.monte_carlo.variables)

    def test_monte_carlo_handles_multiple_variables(self):
        payload = json.loads(Path("src/pharma_financial/data/default_inputs.json").read_text())
        payload["monte_carlo"]["variables"] = [
            "revenue_growth",
            "raw_material_cost",
            "labor_cost",
            "tax_rate",
            "utility_cost",
            "senior_debt",
            "other",
        ]
        inputs = parse_inputs(payload)
        model = FinancialModel(inputs)
        table = model.monte_carlo_simulation()
        self.assertEqual(table.index_name, "Iteration")
        self.assertEqual(len(table.index), inputs.monte_carlo.iterations)
        expected_columns = set(["NPV"] + [metric for metric in inputs.monte_carlo.metrics if metric != "NPV"])
        self.assertTrue(expected_columns.issubset(set(table.columns())))

    def test_inventory_schedule_reconciles_to_balance_sheet(self):
        schedule = self.model.inventory_schedule()
        calculated = schedule.column("Calculated Inventory")
        balance = schedule.column("Balance Sheet Inventory")
        variance = schedule.column("Variance")
        inventory_days = schedule.column("Inventory Days")
        days_in_year = schedule.column("Days in Year")
        configured_days = list(self.inputs.working_capital_days.inventory)
        calendar_days = list(self.inputs.working_capital_days.calendar_days)

        for idx, (calc, actual, diff) in enumerate(zip(calculated, balance, variance)):
            self.assertAlmostEqual(calc, actual, places=6)
            self.assertAlmostEqual(diff, 0.0, places=6)

            expected_day = 0.0
            if configured_days:
                expected_day = configured_days[idx] if idx < len(configured_days) else configured_days[-1]
            self.assertAlmostEqual(inventory_days[idx], expected_day, places=6)

            if calendar_days:
                expected_calendar = (
                    calendar_days[idx]
                    if idx < len(calendar_days)
                    else calendar_days[-1]
                )
                self.assertAlmostEqual(days_in_year[idx], expected_calendar, places=6)

    def test_working_capital_schedule_matches_balance_sheet(self):
        schedule = self.model.working_capital_schedule()
        balance = self.outputs.balance_sheet
        self.assertEqual(schedule.index, self.inputs.years)

        days = self.inputs.working_capital_days
        day_expectations = {
            "Days in Year": days.calendar_days,
            "Accounts Receivable Days": days.accounts_receivable,
            "Inventory Days": days.inventory,
            "Prepaid Expenses Days": days.prepaid_expenses,
            "Other Assets Days": days.other_assets,
            "Accounts Payable Days": days.accounts_payable,
            "Other Liabilities Days": days.other_liabilities,
        }

        for column, configured in day_expectations.items():
            schedule_values = schedule.column(column)
            configured_values = list(configured)
            for idx, actual in enumerate(schedule_values):
                if configured_values:
                    expected = (
                        configured_values[idx]
                        if idx < len(configured_values)
                        else configured_values[-1]
                    )
                else:
                    expected = 0.0
                self.assertAlmostEqual(actual, expected, places=6)

        for column in [
            "Accounts Receivable",
            "Inventory",
            "Prepaid Expenses",
            "Other Assets",
        ]:
            schedule_values = schedule.column(column)
            balance_values = balance.column(column)
            for actual, expected in zip(schedule_values, balance_values):
                self.assertAlmostEqual(actual, expected, places=6)

        payables = schedule.column("Accounts Payable")
        other_liabilities = schedule.column("Other Liabilities")
        # Total liabilities on the balance sheet already aggregates debt balances.
        # Ensure working-capital payables remain non-negative.
        for value in payables + other_liabilities:
            self.assertGreaterEqual(value, 0.0)

        net_working = schedule.column("Net Working Capital")
        changes = schedule.column("Change in Net Working Capital")
        for idx, value in enumerate(net_working):
            if idx == 0:
                expected_change = value
            else:
                expected_change = value - net_working[idx - 1]
            self.assertAlmostEqual(changes[idx], expected_change, places=6)

    def test_break_even_table_contains_expected_columns(self):
        break_even = self.outputs.break_even
        expected_columns = {
            "Fixed Cost",
            "Variable Cost per Unit",
            "Selling Price",
            "Contribution Margin",
            "Contribution Margin Ratio",
            "Break-even Units",
            "Break-even Revenue",
            "Margin of Safety (Units)",
        }
        for column in expected_columns:
            self.assertIn(column, break_even.data)

    def test_break_even_overrides_respected(self):
        default_path = Path("src/pharma_financial/data/default_inputs.json")
        raw = json.loads(default_path.read_text(encoding="utf-8"))
        override_row = {
            "product": "Tablets",
            "fixed_cost": 1_000_000.0,
            "selling_price": 0.06,
            "variable_cost": 0.03,
            "target_profit": 100_000.0,
            "expected_volume": 5_000_000.0,
        }
        raw["break_even"] = {"rows": [override_row]}

        inputs = parse_inputs(raw)
        model = FinancialModel(inputs)
        table = model.break_even_analysis()

        self.assertIn("Tablets", table.index)
        idx = table.index.index("Tablets")
        break_even_units = table.data["Break-even Units"][idx]
        contribution = table.data["Contribution Margin"][idx]

        expected_units = (override_row["fixed_cost"] + override_row["target_profit"]) / contribution
        self.assertAlmostEqual(break_even_units, expected_units, places=6)

    def test_senior_debt_outstanding_clears_by_horizon(self):
        _, outstanding = self.model._senior_debt_schedules()
        self.assertTrue(outstanding)
        self.assertAlmostEqual(outstanding[-1], 0.0, places=6)

    def test_revolver_outstanding_clears_by_duration(self):
        _, outstanding = self.model._revolver_schedules()
        if outstanding:
            self.assertAlmostEqual(outstanding[-1], 0.0, places=6)

    def test_balance_sheet_balances(self):
        balance = self.outputs.balance_sheet
        assets = balance.column("Total Assets")
        liabilities_equity = balance.column("Total Liabilities & Equity")
        for actual_assets, actual_total in zip(assets, liabilities_equity):
            self.assertAlmostEqual(actual_assets, actual_total, places=6)

    def test_overdraft_outstanding_clears_by_duration(self):
        _, outstanding = self.model._overdraft_schedules()
        if outstanding:
            self.assertAlmostEqual(outstanding[-1], 0.0, places=6)

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

        by_asset: dict[tuple[str, int], list[dict]] = {}
        for entry in details:
            acquisition_year = int(entry.get("acquisition_year", entry["year"]))
            key = (entry["asset_type"], acquisition_year)
            by_asset.setdefault(key, []).append(entry)

        for asset_key, asset_entries in by_asset.items():
            asset_entries.sort(key=lambda item: item["year"])
            previous_net_book = None
            previous_cumulative = None
            asset_life = asset_entries[0].get("asset_life")
            configured_life = int(asset_life) if asset_life else None

            for entry in asset_entries:
                opening_net = float(entry["opening_net_book"])
                acquisition = float(entry["acquisition"])
                total_cost = float(entry["total_asset_cost"])
                total_dep = float(entry["total_depreciation"])
                cumulative_dep = float(entry["cumulative_depreciation"])
                method = str(entry.get("method", "straight_line"))
                configured_rate = float(entry.get("configured_rate", entry.get("depreciation_rate", 0.0)))
                life_index = int(entry.get("life_year_index", 0))
                self.assertIn(method, {"straight_line", "reducing_balance"})

                if previous_net_book is not None:
                    self.assertAlmostEqual(opening_net, previous_net_book, places=5)

                expected_total = acquisition + opening_net
                self.assertAlmostEqual(total_cost, expected_total, places=6)

                base_cumulative = 0.0 if previous_cumulative is None else previous_cumulative
                allowable = max(total_cost - base_cumulative, 0.0)

                if method == "straight_line" and configured_life and configured_life > 0:
                    remaining = max(configured_life - life_index, 1)
                    expected_depreciation = allowable / remaining if remaining else allowable
                else:
                    if method == "reducing_balance":
                        expected_base = opening_net + (acquisition * 0.5)
                    else:
                        expected_base = opening_net + acquisition
                    expected_depreciation = expected_base * configured_rate
                    if configured_life and configured_life > 0 and life_index >= configured_life - 1:
                        expected_depreciation = allowable
                    elif expected_depreciation > allowable:
                        expected_depreciation = allowable

                self.assertAlmostEqual(total_dep, expected_depreciation, places=5)

                expected_cumulative = base_cumulative + expected_depreciation
                self.assertAlmostEqual(cumulative_dep, expected_cumulative, places=5)

                expected_net_book = max(total_cost - expected_cumulative, 0.0)
                self.assertAlmostEqual(float(entry["net_book_value"]), expected_net_book, places=5)

                previous_net_book = expected_net_book
                previous_cumulative = expected_cumulative

            if configured_life and configured_life > 0 and len(asset_entries) >= configured_life:
                final_entry = asset_entries[configured_life - 1]
                self.assertAlmostEqual(float(final_entry["net_book_value"]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

