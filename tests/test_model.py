import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_financial.inputs import apply_scenario_files, load_inputs, parse_inputs
from pharma_financial.model import (
    CASH_FLOW_BEGIN_COLUMN,
    CASH_FLOW_END_COLUMN,
    CASH_FLOW_NET_COLUMN,
    FinancialModel,
    IRRResult,
    npf_irr,
)


class FinancialModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_inputs(Path("src/pharma_financial/data/default_inputs.json"))
        cls.model = FinancialModel(cls.inputs)
        cls.outputs = cls.model.run()

    def test_income_statement_columns(self):
        income = self.outputs.income_statement
        self.assertEqual(income.index, self.inputs.years)
        for column in [
            "Gross Revenue",
            "Distributors Commission",
            "Net Revenue",
            "Cost of Sales",
            "Gross Profit",
            "General & Admin",
            "EBITDA",
            "Total Depreciation Expense",
            "EBIT",
            "Gross Profit Margin",
            "EBITDA Margin",
            "EBIT Margin",
            "Return on Equity",
        ]:
            self.assertIn(column, income.data)

    def test_cash_flow_consistency(self):
        cash_flow = self.outputs.cash_flow
        net_change = cash_flow.column(CASH_FLOW_NET_COLUMN)
        ending = cash_flow.column(CASH_FLOW_END_COLUMN)
        self.assertEqual(len(net_change), len(self.inputs.years))
        self.assertEqual(len(ending), len(self.inputs.years))

    def test_cash_flow_ifrs_relationships(self):
        model = self.model
        inputs = self.inputs

        cash_flow = model.cash_flow_statement()
        income = model.income_statement()
        depreciation = model.depreciation_schedule()
        working_capital = model._working_capital_balances()

        def _difference(values):
            result = []
            previous = None
            for value in values:
                value = float(value)
                if previous is None:
                    result.append(value)
                else:
                    result.append(value - previous)
                previous = value
            return result

        net_income = income.column("Net Income")
        taxes = income.column("Taxes")
        interest = income.column("Interest")

        operating_profit = [
            ni + tax + interest_value
            for ni, tax, interest_value in zip(net_income, taxes, interest)
        ]

        inventory_change = _difference(working_capital.column("Inventory"))
        receivable_change = _difference(working_capital.column("Accounts Receivable"))
        payable_change = _difference(working_capital.column("Accounts Payable"))
        prepaid_change = _difference(working_capital.column("Prepaid Expenses"))
        other_asset_change = _difference(working_capital.column("Other Assets"))
        other_liability_change = _difference(working_capital.column("Other Liabilities"))

        expected_cfo = [
            op
            + dep
            - inv
            - ar
            + ap
            - pre
            - other_asset
            + other_liab
            for op, dep, inv, ar, ap, pre, other_asset, other_liab in zip(
                operating_profit,
                depreciation,
                inventory_change,
                receivable_change,
                payable_change,
                prepaid_change,
                other_asset_change,
                other_liability_change,
            )
        ]

        cfo_column = cash_flow.column("Cash Flow from Operations")
        for actual, expected in zip(cfo_column, expected_cfo):
            self.assertAlmostEqual(actual, expected, places=6)

        interest_paid = [-value for value in interest]
        taxes_paid = [-value for value in taxes]

        expected_net_ops = [
            cfo + interest_val + tax_val
            for cfo, interest_val, tax_val in zip(
                expected_cfo, interest_paid, taxes_paid
            )
        ]
        net_ops_column = cash_flow.column("Net Cash Generated from Operating Activities")
        for actual, expected in zip(net_ops_column, expected_net_ops):
            self.assertAlmostEqual(actual, expected, places=6)

        capex = model._capex_series()
        expected_investing = [-value for value in capex]
        investing_column = cash_flow.column("Net Cash Used in Investing Activities")
        for actual, expected in zip(investing_column, expected_investing):
            self.assertAlmostEqual(actual, expected, places=6)

        financing_components = model._financing_cash_flow_components()
        expected_financing = [
            sum(values)
            for values in zip(*financing_components.values(), strict=False)
        ]
        financing_column = cash_flow.column("Net Cash Used in Financing Activities")
        for actual, expected in zip(financing_column, expected_financing):
            self.assertAlmostEqual(actual, expected, places=6)

        net_flow_column = cash_flow.column(CASH_FLOW_NET_COLUMN)
        for idx, (expected, op, inv, fin) in enumerate(
            zip(net_flow_column, expected_net_ops, expected_investing, expected_financing)
        ):
            self.assertAlmostEqual(expected, op + inv + fin, places=6)

        cumulative = []
        running_total = 0.0
        for value in net_flow_column:
            running_total += float(value)
            cumulative.append(running_total)

        expected_beginning = [0.0] + cumulative[:-1]
        beginning_column = cash_flow.column(CASH_FLOW_BEGIN_COLUMN)
        self.assertEqual(len(beginning_column), len(expected_beginning))
        for actual, expected in zip(beginning_column, expected_beginning):
            self.assertAlmostEqual(actual, expected, places=6)

        ending_column = cash_flow.column(CASH_FLOW_END_COLUMN)
        for begin, change, ending in zip(beginning_column, net_flow_column, ending_column):
            self.assertAlmostEqual(begin + change, ending, places=6)

        net_increase_column = cash_flow.column("Net Increase/Decrease in Cash")
        for actual, expected in zip(net_increase_column, net_flow_column):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_summary_metrics_index(self):
        summary = self.outputs.summary_metrics
        expected_prefix = ["NPV", "IRR", "Payback Period", "Discounted Payback"]
        self.assertEqual(summary.index[:4], expected_prefix)
        for metric in [
            "Profitability Index",
            "Revenue CAGR",
            "Investor Viability Score",
            "Probability NPV < 0",
        ]:
            self.assertIn(metric, summary.index)

    def test_summary_metrics_regression(self):
        summary = self.outputs.summary_metrics.column("Value")
        self.assertGreater(summary[0], 0.0)
        self.assertTrue(math.isfinite(summary[1]))
        self.assertGreater(summary[1], 0.0)
        self.assertTrue(math.isfinite(summary[2]))
        self.assertGreaterEqual(summary[2], 0.0)
        self.assertTrue(math.isfinite(summary[3]))
        self.assertGreaterEqual(summary[3], summary[2])

    def test_payback_period_returns_elapsed_years(self):
        payback = self.model._payback_period([-10.0, 4.0, 8.0])
        self.assertAlmostEqual(payback, 1.75, places=6)

    def test_production_schedule_matches_inputs(self):
        production = self.model._production()
        year_count = len(self.inputs.years)

        def _pad(series):
            values = [float(value) for value in series]
            if not values:
                return [0.0 for _ in range(year_count)]
            if len(values) >= year_count:
                return values[:year_count]
            return values + [values[-1] for _ in range(year_count - len(values))]

        for product, series in self.inputs.production_estimate.items():
            expected = _pad(series)
            self.assertEqual(len(production[product]), year_count)
            for actual, expected_value in zip(production[product], expected):
                self.assertAlmostEqual(actual, expected_value, places=6)

    def test_working_capital_schedule_changes(self):
        schedule = self.model.working_capital_schedule()
        net_working_capital = schedule.column("Net Working Capital")
        changes = schedule.column("Change in Net Working Capital")
        self.assertEqual(len(net_working_capital), len(changes))
        expected_changes = []
        previous = None
        for value in net_working_capital:
            if previous is None:
                expected_changes.append(float(value))
            else:
                expected_changes.append(float(value) - previous)
            previous = float(value)
        for actual, expected in zip(changes, expected_changes):
            self.assertAlmostEqual(actual, expected, places=6)

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

    def test_scenario_tools_present_interpretations(self):
        tools = self.inputs.scenario_tools
        self.assertTrue(tools)
        outputs = self.outputs.scenario_tool_results
        self.assertTrue(outputs)
        for key, result in outputs.items():
            self.assertIn(key, tools)
            self.assertTrue(result.rows)
            self.assertIsInstance(result.interpretation, str)
            self.assertTrue(result.interpretation.strip())

    def test_npf_irr_handles_short_series(self):
        result = npf_irr([100])
        self.assertIsInstance(result, IRRResult)
        self.assertFalse(result.converged)
        self.assertTrue(math.isnan(result.value))

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
        cost_of_sales = costs.column("Cost of Sales")

        base_indirect = sum(self.inputs.indirect_labor_costs.values())
        risk_factors = self.model._risk_cost_factors()
        utility_share = self.inputs.utility_cost_of_sales_share

        def _risk(idx: int) -> float:
            if not risk_factors:
                return 1.0
            if idx < len(risk_factors):
                return risk_factors[idx]
            return risk_factors[-1]

        for idx in range(len(total_expenses)):
            expected_cost_of_sales = raw[idx] + direct[idx] + utilities[idx] * utility_share
            self.assertAlmostEqual(
                cost_of_sales[idx], expected_cost_of_sales, places=6
            )

            expected_general = (
                base_indirect * self.model._inflation[idx] * _risk(idx)
                + utilities[idx] * (1.0 - utility_share)
            )
            self.assertAlmostEqual(general[idx], expected_general, places=6)

            expected_total = cost_of_sales[idx] + general[idx]
            self.assertAlmostEqual(total_expenses[idx], expected_total, places=6)

    def test_cost_structure_includes_per_product_raw_material_columns(self):
        costs = self.model.cost_structure()
        aggregate = costs.column("Raw Materials")

        per_product_totals = [0.0 for _ in costs.index]
        for product in self.inputs.products:
            column = f"Raw Materials - {product}"
            self.assertIn(column, costs.data)
            values = costs.column(column)
            for idx, value in enumerate(values):
                per_product_totals[idx] += value

        for idx, value in enumerate(aggregate):
            self.assertAlmostEqual(value, per_product_totals[idx], places=6)

    def test_custom_projection_years(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(encoding="utf-8")
        )
        payload["years"] = [2026, 2027, 2028, 2029]
        parsed = parse_inputs(payload)
        self.assertEqual(parsed.years, [2026, 2027, 2028, 2029])
        horizon = len(parsed.years)
        for series in parsed.production_estimate.values():
            self.assertEqual(len(series), horizon)
        self.assertEqual(len(parsed.inflation_series), horizon)
        self.assertEqual(len(parsed.utility_schedule.electricity_per_day), horizon)

    def test_scenario_files_are_loaded(self):
        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(encoding="utf-8")
        )
        payload["scenarios"] = {}
        apply_scenario_files(payload, Path("src/pharma_financial/data"))
        self.assertIn("base", payload["scenarios"])
        self.assertIn("best", payload["scenarios"])
        self.assertIn("worst", payload["scenarios"])

    def test_income_statement_relationships(self):
        income = self.outputs.income_statement
        gross_revenue = income.column("Gross Revenue")
        commission = income.column("Distributors Commission")
        net_revenue = income.column("Net Revenue")
        cost_of_sales = income.column("Cost of Sales")
        gross_profit = income.column("Gross Profit")
        general_admin = income.column("General & Admin")
        ebitda = income.column("EBITDA")
        depreciation = income.column("Total Depreciation Expense")
        ebit = income.column("EBIT")
        gross_margin = income.column("Gross Profit Margin")
        ebitda_margin = income.column("EBITDA Margin")
        ebit_margin = income.column("EBIT Margin")
        for idx in range(len(self.inputs.years)):
            self.assertAlmostEqual(
                net_revenue[idx], gross_revenue[idx] - commission[idx], places=6
            )
            self.assertAlmostEqual(
                gross_profit[idx], net_revenue[idx] - cost_of_sales[idx], places=6
            )
            self.assertAlmostEqual(
                ebitda[idx], gross_profit[idx] - general_admin[idx], places=6
            )
            self.assertAlmostEqual(
                ebit[idx], ebitda[idx] - depreciation[idx], places=6
            )
            self.assertAlmostEqual(
                gross_margin[idx],
                gross_profit[idx] / gross_revenue[idx]
                if gross_revenue[idx] else 0.0,
                places=6,
            )
            self.assertAlmostEqual(
                ebitda_margin[idx],
                ebitda[idx] / net_revenue[idx] if net_revenue[idx] else 0.0,
                places=6,
            )
            self.assertAlmostEqual(
                ebit_margin[idx],
                ebit[idx] / net_revenue[idx] if net_revenue[idx] else 0.0,
                places=6,
            )

    def test_revenue_schedule_breakdown_matches_net_revenue(self):
        schedule = self.model.revenue_schedule()
        gross = schedule.column("Gross Revenue")
        commission = schedule.column("Distributors Commission")
        net = schedule.column("Net Revenue")

        self.assertEqual(len(gross), len(self.inputs.years))
        self.assertEqual(len(commission), len(self.inputs.years))
        self.assertEqual(len(net), len(self.inputs.years))

        for g, c, n in zip(gross, commission, net):
            self.assertAlmostEqual(g - c, n, places=6)

        product_columns = [schedule.column(product) for product in self.inputs.products]
        for idx in range(len(self.inputs.years)):
            expected_total = sum(column[idx] for column in product_columns)
            self.assertAlmostEqual(expected_total, gross[idx], places=5)

    def test_gross_revenue_matches_pricing_assumptions(self):
        schedule = self.model.revenue_schedule()
        gross = schedule.column("Gross Revenue")

        inflation_series = list(self.inputs.inflation_series)
        risk_schedule = dict(self.inputs.risk_schedule)
        risk_weights = dict(self.inputs.risk_weights)
        price_adjustments = dict(self.inputs.price_adjustments)

        expected: list[float] = []
        cumulative_inflation: list[float] = []
        running = 1.0
        for rate in inflation_series:
            running *= 1.0 + float(rate)
            cumulative_inflation.append(running)

        production = self.inputs.production_estimate

        for idx, year in enumerate(self.inputs.years):
            inflation_factor = cumulative_inflation[idx] if idx < len(cumulative_inflation) else cumulative_inflation[-1]
            risk_factor = 1.0
            for name, values in risk_schedule.items():
                if not values:
                    continue
                rate = values[idx] if idx < len(values) else values[-1]
                weight = risk_weights.get(name, {})
                revenue_weight = float(weight.get("revenue", 1.0)) if weight else 1.0
                risk_factor *= max(0.0, 1.0 - float(rate) * revenue_weight)

            total = 0.0
            for product in self.inputs.products:
                units = 0.0
                if product in production:
                    schedule = production[product]
                    if idx < len(schedule):
                        units = float(schedule[idx])
                    elif schedule:
                        units = float(schedule[-1])
                price = float(self.inputs.unit_costs[product].selling_price)
                adjustments = price_adjustments.get(product, [1.0])
                price_factor = adjustments[idx] if idx < len(adjustments) else adjustments[-1]
                total += units * price * price_factor * inflation_factor * risk_factor
            expected.append(total)

        self.assertEqual(len(expected), len(gross))
        for actual, expected_value in zip(gross, expected):
            self.assertAlmostEqual(actual, expected_value, places=6)

    def test_distributor_commission_uses_configured_rates(self):
        schedule = self.model.revenue_schedule()
        commission_column = schedule.column("Distributors Commission")
        parameters = self.model._commission_parameters()
        inflation = self.model._inflation
        risk = self.model._risk_factors()
        price_adjustments = dict(self.inputs.price_adjustments)

        expected: list[float] = []
        production = self.inputs.production_estimate

        for idx, year in enumerate(self.inputs.years):
            year_params = parameters.get(int(year), {})
            total = 0.0
            for product in self.inputs.products:
                units = 0.0
                if product in production:
                    schedule = production[product]
                    if idx < len(schedule):
                        units = float(schedule[idx])
                    elif schedule:
                        units = float(schedule[-1])
                price = self.inputs.unit_costs[product].selling_price
                price_series = price_adjustments.get(product, [1.0])
                price_factor = price_series[idx] if idx < len(price_series) else price_series[-1]
                factor = risk[idx] if idx < len(risk) else (risk[-1] if risk else 1.0)
                inflation_factor = inflation[idx] if idx < len(inflation) else inflation[-1]
                gross_value = units * price * price_factor * inflation_factor * factor
                rate, share, _ = year_params.get(product, (0.0, 1.0, 0))
                total += gross_value * rate * share
            expected.append(total)

        self.assertEqual(len(commission_column), len(expected))
        for actual, expected_value in zip(commission_column, expected):
            self.assertAlmostEqual(actual, expected_value, places=6)

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
            manual.append(total)

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

    def test_ai_insights_available(self):
        insights = self.outputs.ai_insights
        self.assertIsNotNone(insights)
        if insights.enabled:
            self.assertIsNotNone(insights.ml_forecast)
            self.assertTrue(insights.generative_summary.strip())

    def test_risk_factor_diagnostics_matches_schedule(self):
        risk_table = self.outputs.risk_factor_diagnostics
        self.assertIsNotNone(risk_table)
        combined = risk_table.column("Combined Factor")
        risk_series = self.model._risk_factors()
        self.assertEqual(len(combined), len(risk_series))
        for actual, expected in zip(combined, risk_series):
            self.assertAlmostEqual(actual, expected)

    def test_monte_carlo_seed_produces_reproducible_results(self):
        payload = json.loads(Path("src/pharma_financial/data/default_inputs.json").read_text())
        payload["monte_carlo"]["seed"] = 42
        inputs = parse_inputs(payload)
        model = FinancialModel(inputs)
        table_a = model.monte_carlo_simulation().as_dict()
        table_b = model.monte_carlo_simulation().as_dict()
        self.assertEqual(table_a, table_b)

    def test_inventory_schedule_includes_per_product_material_and_inventory(self):
        schedule = self.model.inventory_schedule()
        aggregate_purchased = schedule.column("Raw Materials") if "Raw Materials" in schedule.data else None
        if aggregate_purchased is None:
            aggregate_purchased = self.model.cost_structure().column("Raw Materials")

        purchased_sum = [0.0 for _ in schedule.index]
        inventory_sum = [0.0 for _ in schedule.index]
        for product in self.inputs.products:
            purchased_col = f"Material Purchased - {product}"
            inventory_col = f"Inventory - {product}"
            self.assertIn(purchased_col, schedule.data)
            self.assertIn(inventory_col, schedule.data)
            for idx, value in enumerate(schedule.column(purchased_col)):
                purchased_sum[idx] += value
            for idx, value in enumerate(schedule.column(inventory_col)):
                inventory_sum[idx] += value

        for idx, value in enumerate(aggregate_purchased):
            self.assertAlmostEqual(value, purchased_sum[idx], places=6)

        inventory_days = schedule.column("Inventory Days")
        days_in_year = schedule.column("Days in Year")
        for idx, value in enumerate(inventory_sum):
            day_length = days_in_year[idx] if days_in_year[idx] else 0.0
            expected_material_inventory = (
                aggregate_purchased[idx] / day_length * inventory_days[idx]
                if day_length
                else 0.0
            )
            self.assertAlmostEqual(value, expected_material_inventory, places=6)

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

    def test_capex_includes_fixed_asset_acquisitions(self):
        capex_series = self.model._capex_series()
        year_index = {year: idx for idx, year in enumerate(self.inputs.years)}
        acquisitions: dict[int, float] = {}
        for row in self.inputs.depreciation_schedule:
            acquisitions[row.year] = acquisitions.get(row.year, 0.0) + float(row.acquisition or 0.0)

        for year, amount in acquisitions.items():
            idx = year_index.get(year)
            if idx is None:
                continue
            self.assertGreaterEqual(capex_series[idx], amount)

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

    def test_fixed_variable_cost_overrides(self):
        default_path = Path("src/pharma_financial/data/default_inputs.json")
        raw = json.loads(default_path.read_text(encoding="utf-8"))
        raw["fixed_variable_costs"] = {
            "rows": [
                {
                    "product": "Tablets",
                    "fixed_cost": 123.0,
                    "variable_cost": 0.99,
                }
            ]
        }

        inputs = parse_inputs(raw)
        model = FinancialModel(inputs)

        variable_costs = model._variable_costs()
        self.assertAlmostEqual(variable_costs.get("Tablets"), 0.99, places=6)

        table = model.break_even_analysis()
        self.assertIn("Tablets", table.index)
        idx = table.index.index("Tablets")
        fixed_cost_value = table.data["Fixed Cost"][idx]
        self.assertAlmostEqual(fixed_cost_value, 123.0, places=6)

    def test_variable_cost_override_updates_cost_of_sales(self):
        base_costs = self.outputs.income_statement.column("Cost of Sales")

        payload = json.loads(
            Path("src/pharma_financial/data/default_inputs.json").read_text(encoding="utf-8")
        )
        payload.setdefault("fixed_variable_costs", {})["rows"] = [
            {
                "product": "Tablets",
                "variable_cost": 0.5,
            }
        ]

        parsed = parse_inputs(payload)
        override_model = FinancialModel(parsed)
        override_costs = override_model.income_statement().column("Cost of Sales")

        self.assertNotEqual(base_costs[0], override_costs[0])
        self.assertGreater(override_costs[0], base_costs[0])

    def test_break_even_fixed_cost_defaults_zero_without_overrides(self):
        inputs = load_inputs(Path("src/pharma_financial/data/default_inputs.json"))
        model = FinancialModel(inputs)
        break_even = model.break_even_analysis()
        self.assertTrue(break_even.index)
        for value in break_even.data["Fixed Cost"]:
            self.assertAlmostEqual(value, 0.0, places=6)

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
        income_dep = self.outputs.income_statement.column("Total Depreciation Expense")
        self.assertEqual(depreciation, income_dep)
        for actual, expense in zip(depreciation, income_dep):
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

                base_cumulative = 0.0 if previous_cumulative is None else previous_cumulative
                expected_total = acquisition + opening_net + base_cumulative
                self.assertAlmostEqual(total_cost, expected_total, places=6)

                allowable = max(total_cost - base_cumulative, 0.0)

                if method == "straight_line" and configured_life and configured_life > 0:
                    remaining = max(configured_life - life_index, 1)
                    expected_depreciation = allowable / remaining if remaining else allowable
                else:
                    if method == "reducing_balance":
                        expected_base = opening_net + (acquisition * 0.5)
                    else:
                        expected_base = total_cost
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

    def test_labor_model_v1_generates_labor_kpis(self):
        payload = json.loads(Path("src/pharma_financial/data/default_inputs.json").read_text())
        payload["labor"]["model_v1"] = {
            "roles": [
                {
                    "name": "Operators",
                    "labor_type": "direct",
                    "behavior": "variable",
                    "headcount": 4,
                    "salary": 0.5,
                    "planned_headcount": [4 for _ in payload["years"]],
                    "benefits_rate": 0.1,
                    "overtime_rate": 0.05,
                    "burden_rate": 0.15,
                    "productivity_target": [100000 for _ in payload["years"]],
                },
                {
                    "name": "Admin",
                    "labor_type": "indirect",
                    "behavior": "fixed",
                    "headcount": 2,
                    "salary": 0.4,
                    "planned_headcount": [2 for _ in payload["years"]],
                    "benefits_rate": 0.12,
                    "overtime_rate": 0.0,
                    "burden_rate": 0.1,
                    "productivity_target": [1 for _ in payload["years"]],
                },
            ],
            "settings": {
                "shifts": [1 for _ in payload["years"]],
                "utilization": [0.85 for _ in payload["years"]],
                "operating_hours_per_shift": [2080 for _ in payload["years"]],
                "absenteeism": [0.03 for _ in payload["years"]],
                "overtime_cap": [0.1 for _ in payload["years"]],
                "hiring_delay_quarters": [1 for _ in payload["years"]],
                "contractor_hours": [0 for _ in payload["years"]],
                "contractor_rate": [0 for _ in payload["years"]],
                "transition_training_cost": [0 for _ in payload["years"]],
                "supervision_increment": [0 for _ in payload["years"]],
                "shift_allowance": [0 for _ in payload["years"]],
                "wage_escalation_direct": [0.03 for _ in payload["years"]],
                "wage_escalation_indirect": [0.02 for _ in payload["years"]],
            },
        }
        payload["sensitivity"]["variables"] = {"wage_direct": [0.9, 1.0, 1.1]}

        inputs = parse_inputs(payload)
        self.assertIsNotNone(inputs.labor_model)
        model = FinancialModel(inputs)
        summary = model.summary_metrics()

        self.assertIn("Average Labor Cost per Unit", summary.index)
        self.assertIn("Average Units per Labor Hour", summary.index)
        self.assertIn("Average Fixed Labor Share", summary.index)

        sensitivity = model.sensitivity_analysis()
        self.assertIn("wage_direct", sensitivity)
        self.assertEqual(len(sensitivity["wage_direct"].index), 3)


    def test_summary_includes_investor_gate_and_risk_metrics(self):
        payload = json.loads(Path("src/pharma_financial/data/default_inputs.json").read_text())
        payload["monte_carlo"]["iterations"] = 20
        payload["monte_carlo"]["metrics"] = ["NPV", "IRR"]

        model = FinancialModel(parse_inputs(payload))
        summary = model.summary_metrics()

        self.assertIn("Investor Gate Pass Ratio", summary.index)
        self.assertIn("Investor Gate Status", summary.index)
        self.assertIn("Minimum Ending Cash", summary.index)
        self.assertIn("Evidence Coverage Ratio", summary.index)
        self.assertIn("Assumption Data Quality Score", summary.index)
        self.assertIn("Probability NPV < 0", summary.index)

    def test_bankability_outputs_are_available(self):
        outputs = self.model.run_core()
        self.assertIsNotNone(outputs.bankability_gate)
        self.assertIsNotNone(outputs.sources_and_uses)
        self.assertIsNotNone(outputs.liquidity_bridge)
        self.assertIsNotNone(outputs.covenant_headroom)
        self.assertIsNotNone(outputs.downside_case_summary)
        self.assertIsInstance(outputs.evidence_register, list)
        self.assertIsInstance(outputs.data_quality_exceptions, list)
        self.assertGreaterEqual(len(outputs.evidence_register), 1)
        self.assertGreaterEqual(len(outputs.data_quality_exceptions), 1)



if __name__ == "__main__":
    unittest.main()
