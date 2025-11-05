"""Core financial model implementation for the Longevity Pharmaceuticals project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .inputs import ModelInputs, ProductParameters


ProductName = str


@dataclass
class FinancialOutputs:
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    summary_metrics: pd.DataFrame
    break_even: pd.DataFrame
    payback: pd.DataFrame
    discounted_payback: pd.DataFrame
    scenario_results: Dict[str, pd.DataFrame]
    sensitivity_results: Dict[str, pd.DataFrame]
    monte_carlo: pd.DataFrame


def _inflation_multiplier(series: Iterable[float]) -> np.ndarray:
    return np.array([(1 + rate) for rate in series], dtype=float)


class FinancialModel:
    """Encapsulates the computations for the Longevity Pharmaceuticals model."""

    def __init__(self, inputs: ModelInputs):
        self.inputs = inputs
        self.years = inputs.years
        self.products = inputs.products
        self._inflation = np.cumprod(_inflation_multiplier(inputs.inflation_series))
        self.scenario = "Base Case"

    # ---------------------------- Helper sections ---------------------------- #
    def _production_df(self) -> pd.DataFrame:
        data = {
            product: pd.Series(values, index=self.years)
            for product, values in self.inputs.production_estimate.items()
        }
        df = pd.DataFrame(data)
        df["Total Units"] = df.sum(axis=1)
        return df

    def _unit_prices(self) -> pd.Series:
        prices = {}
        for product in self.products:
            params: ProductParameters = self.inputs.unit_costs[product]
            prices[product] = params.selling_price
        return pd.Series(prices)

    def _unit_costs(self) -> pd.Series:
        costs = {}
        for product in self.products:
            params: ProductParameters = self.inputs.unit_costs[product]
            costs[product] = params.production_cost
        return pd.Series(costs)

    def _freight_costs(self) -> pd.Series:
        freight = {}
        for product in self.products:
            freight[product] = self.inputs.unit_costs[product].freight_cost
        return pd.Series(freight)

    # ---------------------------- Revenue schedule --------------------------- #
    def revenue_schedule(self) -> pd.DataFrame:
        production = self._production_df()
        prices = self._unit_prices()
        freight = self._freight_costs()

        inflation = self._inflation
        inflation_series = pd.Series(inflation, index=self.years)

        gross_revenue = production[self.products].multiply(prices, axis=1).multiply(
            inflation_series, axis=0
        )
        freight_costs = production[self.products].multiply(freight, axis=1)
        net_revenue = gross_revenue - freight_costs
        summary = gross_revenue.assign(**{"Distributors Commission": freight_costs.sum(axis=1)})
        summary["Net Revenue"] = net_revenue.sum(axis=1)
        summary["Gross Revenue"] = gross_revenue.sum(axis=1)
        return summary

    # ------------------------ Cost of goods and expenses --------------------- #
    def cost_structure(self) -> pd.DataFrame:
        production = self._production_df()
        unit_raw_material = self.inputs.raw_material_cost_per_unit
        raw_material_cost = production["Total Units"] * unit_raw_material

        # Utility cost per unit using per-day assumptions.
        utility = self.inputs.utility_schedule
        electricity = np.array(utility.operating_days, dtype=float) * utility.electricity_per_day
        water = np.array(utility.operating_days, dtype=float) * utility.water_per_day
        steam = np.array(utility.operating_hours, dtype=float) * utility.steam_per_hour
        total_utility = electricity + water + steam
        utility_cost = pd.Series(total_utility, index=self.years)

        direct_labor_total = self._annual_direct_labor(production)
        indirect_labor_total = self._annual_indirect_labor()

        cost_of_sales = raw_material_cost + utility_cost + direct_labor_total
        general_admin = indirect_labor_total
        total_costs = cost_of_sales + general_admin

        df = pd.DataFrame(
            {
                "Raw Materials": raw_material_cost,
                "Utilities": utility_cost,
                "Direct Labor": direct_labor_total,
                "Cost of Sales": cost_of_sales,
                "General & Admin": general_admin,
                "Total Costs": total_costs,
            }
        )
        return df

    def _annual_direct_labor(self, production: pd.DataFrame) -> pd.Series:
        inflation = self._inflation
        base_costs = sum(self.inputs.direct_labor_costs.values())
        total_units = production["Total Units"]
        baseline_units = total_units.iloc[0] if total_units.iloc[0] else 1.0
        scaling = total_units / baseline_units
        scaling = scaling.fillna(0)
        return pd.Series(base_costs * scaling * inflation, index=self.years)

    def _annual_indirect_labor(self) -> pd.Series:
        inflation = self._inflation
        base_costs = sum(self.inputs.indirect_labor_costs.values())
        return pd.Series(base_costs * inflation, index=self.years)

    # ------------------------------- Depreciation ---------------------------- #
    def depreciation_schedule(self) -> pd.Series:
        depreciation = pd.Series(0.0, index=self.years)
        for item in self.inputs.depreciation_items:
            if item.useful_life is None or item.useful_life <= 0:
                continue
            annual = item.annual_depreciation
            depreciation += annual
        additions = self.inputs.capital_expenditure.get("annual_additions", {})
        for year, value in additions.items():
            years_remaining = len(self.years) - self.years.index(int(year))
            annual = value / max(years_remaining, 1)
            depreciation.loc[int(year):] += annual
        return depreciation

    # ------------------------------ Statements -------------------------------- #
    def income_statement(self) -> pd.DataFrame:
        revenue = self.revenue_schedule()
        costs = self.cost_structure()
        depreciation = self.depreciation_schedule()
        ebitda = revenue["Net Revenue"] - costs["Total Costs"]
        ebit = ebitda - depreciation

        interest = self._interest_schedule()
        ebt = ebit - interest
        taxes = ebt * self.inputs.tax_rate
        net_income = ebt - taxes

        df = pd.DataFrame(
            {
                "Gross Revenue": revenue["Gross Revenue"],
                "Net Revenue": revenue["Net Revenue"],
                "Total Costs": costs["Total Costs"],
                "EBITDA": ebitda,
                "Depreciation": depreciation,
                "EBIT": ebit,
                "Interest": interest,
                "EBT": ebt,
                "Taxes": taxes,
                "Net Income": net_income,
            }
        )
        df["EBITDA Margin"] = (df["EBITDA"] / df["Net Revenue"]).replace([np.inf, -np.inf], np.nan)
        df["EBIT Margin"] = (df["EBIT"] / df["Net Revenue"]).replace([np.inf, -np.inf], np.nan)
        df["Return on Equity"] = df["Net Income"] / self.inputs.financing.share_capital
        return df

    def _interest_schedule(self) -> pd.Series:
        financing = self.inputs.financing
        debt_balance = self._senior_debt_balance()
        interest = -debt_balance * financing.senior_debt_interest
        return interest

    def cash_flow_statement(self) -> pd.DataFrame:
        income = self.income_statement()
        depreciation = self.depreciation_schedule()
        working_capital_change = self._working_capital_changes()
        operating_cash_flow = income["Net Income"] + depreciation - working_capital_change

        capex = self._capex_series()
        investing_cash_flow = -capex

        financing_cash_flow = self._financing_cash_flow()

        cash_change = operating_cash_flow + investing_cash_flow + financing_cash_flow
        cash_begin = cash_change.cumsum().shift(1, fill_value=0)
        cash_end = cash_begin + cash_change

        df = pd.DataFrame(
            {
                "Operating Cash Flow": operating_cash_flow,
                "Investing Cash Flow": investing_cash_flow,
                "Financing Cash Flow": financing_cash_flow,
                "Net Change in Cash": cash_change,
                "Beginning Cash": cash_begin,
                "Ending Cash": cash_end,
            }
        )
        return df

    def balance_sheet(self) -> pd.DataFrame:
        cash_flow = self.cash_flow_statement()
        working_capital = self._working_capital_balances()
        net_ppe = self._net_ppe_schedule()

        total_current_assets = (
            cash_flow["Ending Cash"]
            + working_capital["Accounts Receivable"]
            + working_capital["Inventory"]
            + working_capital["Prepaid Expenses"]
            + working_capital["Other Assets"]
        )
        total_assets = total_current_assets + net_ppe

        debt = self._senior_debt_balance()
        equity = self._equity_schedule(cash_flow, self.income_statement())

        total_liabilities = debt
        shareholders_equity = equity

        df = pd.DataFrame(
            {
                "Cash": cash_flow["Ending Cash"],
                "Accounts Receivable": working_capital["Accounts Receivable"],
                "Inventory": working_capital["Inventory"],
                "Prepaid Expenses": working_capital["Prepaid Expenses"],
                "Other Assets": working_capital["Other Assets"],
                "Net PP&E": net_ppe,
                "Total Assets": total_assets,
                "Total Liabilities": total_liabilities,
                "Shareholders' Equity": shareholders_equity,
            }
        )
        df["Total Liabilities & Equity"] = df["Total Liabilities"] + df["Shareholders' Equity"]
        return df

    # ----------------------------- Supporting schedules ---------------------- #
    def _capex_series(self) -> pd.Series:
        capex = pd.Series(0.0, index=self.years)
        capex.iloc[0] = self.inputs.capital_expenditure["initial"]
        for year_str, value in self.inputs.capital_expenditure.get("annual_additions", {}).items():
            year = int(year_str)
            if year in capex.index:
                capex.loc[year] += value
        return capex

    def _working_capital_balances(self) -> pd.DataFrame:
        revenue = self.revenue_schedule()["Net Revenue"]
        cost_of_sales = self.cost_structure()["Cost of Sales"]
        days = self.inputs.working_capital_days
        days_in_year = pd.Series([365 + (year % 4 == 0) for year in self.years], index=self.years)

        ar = revenue / days_in_year * pd.Series(days.accounts_receivable, index=self.years)
        inventory = cost_of_sales / days_in_year * pd.Series(days.inventory, index=self.years)
        prepaid = cost_of_sales / days_in_year * pd.Series(days.prepaid_expenses, index=self.years)
        other_assets = cost_of_sales / days_in_year * pd.Series(days.other_assets, index=self.years)
        ap = cost_of_sales / days_in_year * pd.Series(days.accounts_payable, index=self.years)
        other_liabilities = cost_of_sales / days_in_year * pd.Series(days.other_liabilities, index=self.years)

        balances = pd.DataFrame(
            {
                "Accounts Receivable": ar,
                "Inventory": inventory,
                "Prepaid Expenses": prepaid,
                "Other Assets": other_assets,
                "Accounts Payable": ap,
                "Other Liabilities": other_liabilities,
            }
        )
        balances["Net Working Capital"] = (
            balances["Accounts Receivable"]
            + balances["Inventory"]
            + balances["Prepaid Expenses"]
            + balances["Other Assets"]
            - balances["Accounts Payable"]
            - balances["Other Liabilities"]
        )
        return balances

    def _working_capital_changes(self) -> pd.Series:
        balances = self._working_capital_balances()["Net Working Capital"]
        return balances.diff().fillna(balances)

    def _net_ppe_schedule(self) -> pd.Series:
        capex = self._capex_series()
        depreciation = self.depreciation_schedule()
        net_ppe = capex.cumsum() - depreciation.cumsum()
        return net_ppe

    def _senior_debt_balance(self) -> pd.Series:
        capex = self._capex_series()
        schedule = self.inputs.financing.senior_debt_schedule
        debt = pd.Series(self.inputs.financing.initial_investment, index=self.years)
        cumulative = self.inputs.financing.initial_investment
        for idx, year in enumerate(self.years):
            cumulative += schedule.get(year, 0.0)
            debt.iloc[idx] = max(cumulative, 0)
        return debt

    def _financing_cash_flow(self) -> pd.Series:
        financing = self.inputs.financing
        dividends = self.income_statement()["Net Income"] * financing.dividend_payout
        senior_debt = pd.Series(
            {year: value for year, value in financing.senior_debt_schedule.items()}, dtype=float
        )
        senior_debt = senior_debt.reindex(self.years, fill_value=0.0)
        share_issuance = pd.Series(0.0, index=self.years)
        share_issuance.iloc[0] = financing.share_capital
        financing_cash_flow = senior_debt + share_issuance - dividends
        return financing_cash_flow

    def _equity_schedule(self, cash_flow: pd.DataFrame, income_statement: pd.DataFrame) -> pd.Series:
        financing = self.inputs.financing
        dividends = income_statement["Net Income"] * financing.dividend_payout
        retained = (income_statement["Net Income"] - dividends).cumsum()
        equity = retained + financing.share_capital
        return equity

    # ---------------------- Scenario, sensitivity, Monte Carlo --------------- #
    def scenario_analysis(self) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        base_inflation = list(self.inputs.inflation_series)
        base_discount = self.inputs.financing.discount_rate
        for name, scenario in self.inputs.scenarios.items():
            self.inputs.inflation_series = scenario.get("inflation", base_inflation)
            self._inflation = np.cumprod(_inflation_multiplier(self.inputs.inflation_series))
            self.inputs.financing.discount_rate = scenario.get("interest", [base_discount])[0]
            results[name] = self.income_statement()[["Net Revenue", "EBITDA", "EBIT", "Net Income"]]
        self.inputs.inflation_series = base_inflation
        self.inputs.financing.discount_rate = base_discount
        self._inflation = np.cumprod(_inflation_multiplier(self.inputs.inflation_series))
        return results

    def sensitivity_analysis(self) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for variable, adjustments in self.inputs.sensitivity.variables.items():
            outputs = []
            for multiplier in adjustments:
                if variable == "tablet_price":
                    original_price = self.inputs.unit_costs["Tablets"].selling_price
                    self.inputs.unit_costs["Tablets"].selling_price = original_price * multiplier
                elif variable == "raw_material_cost":
                    original_raw = self.inputs.raw_material_cost_per_unit
                    self.inputs.raw_material_cost_per_unit = original_raw * multiplier
                elif variable == "discount_rate":
                    original_rate = self.inputs.financing.discount_rate
                    self.inputs.financing.discount_rate = multiplier
                summary = self.summary_metrics()
                outputs.append({"Multiplier": multiplier, "NPV": summary.loc["NPV", "Value"], "IRR": summary.loc["IRR", "Value"]})
                # Reset to original after each run
                if variable == "tablet_price":
                    self.inputs.unit_costs["Tablets"].selling_price = original_price
                elif variable == "raw_material_cost":
                    self.inputs.raw_material_cost_per_unit = original_raw
                elif variable == "discount_rate":
                    self.inputs.financing.discount_rate = original_rate
            results[variable] = pd.DataFrame(outputs)
        return results

    def monte_carlo_simulation(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        iterations = self.inputs.monte_carlo.iterations
        low, high = self.inputs.monte_carlo.revenue_growth_range
        growth_rates = rng.uniform(low, high, size=(iterations, len(self.years)))
        base_revenue = self.income_statement()["Net Revenue"].values
        simulated_revenue = base_revenue * (1 + growth_rates)
        npv = []
        for revenue_series in simulated_revenue:
            cash_flows = revenue_series - self.cost_structure()["Total Costs"].values
            discounted = cash_flows / (1 + self.inputs.financing.discount_rate) ** np.arange(1, len(cash_flows) + 1)
            npv.append(discounted.sum() - self.inputs.financing.initial_investment)
        return pd.DataFrame({"NPV": npv})

    # ----------------------------- Summary metrics --------------------------- #
    def summary_metrics(self) -> pd.DataFrame:
        cash_flow = self.cash_flow_statement()
        discount_rate = self.inputs.financing.discount_rate
        cash_flows = cash_flow["Net Change in Cash"].values
        discounted = cash_flows / (1 + discount_rate) ** np.arange(1, len(cash_flows) + 1)
        npv = discounted.sum() - self.inputs.financing.initial_investment
        irr = npf_irr(np.concatenate(([-self.inputs.financing.initial_investment], cash_flows)))
        payback_years = self._payback_period(cash_flow["Net Change in Cash"])
        discounted_payback_years = self._payback_period(pd.Series(discounted, index=self.years))
        metrics = pd.DataFrame(
            {
                "Metric": ["NPV", "IRR", "Payback Period", "Discounted Payback"],
                "Value": [npv, irr, payback_years, discounted_payback_years],
            }
        ).set_index("Metric")
        return metrics

    def break_even_analysis(self) -> pd.DataFrame:
        costs = self.cost_structure()
        break_even_units = {}
        for product in self.products:
            params = self.inputs.unit_costs[product]
            margin = params.selling_price - params.production_cost
            if margin <= 0:
                break_even_units[product] = np.nan
            else:
                fixed_cost = costs["Total Costs"].mean() * 0.25
                break_even_units[product] = fixed_cost / margin
        return (
            pd.DataFrame.from_dict(break_even_units, orient="index", columns=["Units"])
            .rename_axis("Product")
        )

    def _payback_period(self, cash_flows: pd.Series) -> float:
        cumulative = cash_flows.cumsum()
        for idx, value in enumerate(cumulative):
            if value >= self.inputs.financing.initial_investment:
                return self.years[idx]
        return float("nan")

    def payback_schedule(self) -> pd.DataFrame:
        cash_flows = self.cash_flow_statement()["Net Change in Cash"]
        cumulative = cash_flows.cumsum() - self.inputs.financing.initial_investment
        return pd.DataFrame({"Cash Flow": cash_flows, "Cumulative": cumulative})

    def discounted_payback_schedule(self) -> pd.DataFrame:
        discount_rate = self.inputs.financing.discount_rate
        cash_flows = self.cash_flow_statement()["Net Change in Cash"]
        discounted = cash_flows / (1 + discount_rate) ** np.arange(1, len(cash_flows) + 1)
        cumulative = discounted.cumsum() - self.inputs.financing.initial_investment
        return pd.DataFrame({"Discounted Cash Flow": discounted, "Cumulative": cumulative})

    # ------------------------------ Public API -------------------------------- #
    def run(self) -> FinancialOutputs:
        income = self.income_statement()
        balance = self.balance_sheet()
        cash_flow = self.cash_flow_statement()
        summary = self.summary_metrics()
        break_even = self.break_even_analysis()
        payback = self.payback_schedule()
        discounted_payback = self.discounted_payback_schedule()
        scenarios = self.scenario_analysis()
        sensitivity = self.sensitivity_analysis()
        monte_carlo = self.monte_carlo_simulation()
        return FinancialOutputs(
            income_statement=income,
            balance_sheet=balance,
            cash_flow=cash_flow,
            summary_metrics=summary,
            break_even=break_even,
            payback=payback,
            discounted_payback=discounted_payback,
            scenario_results=scenarios,
            sensitivity_results=sensitivity,
            monte_carlo=monte_carlo,
        )


def npf_irr(cashflows: Iterable[float]) -> float:
    """Internal rate of return implementation resilient to non-convergence."""
    cashflows = np.asarray(list(cashflows), dtype=float)
    if cashflows.size < 2:
        return float("nan")
    try:
        import numpy_financial as npf  # type: ignore

        return float(npf.irr(cashflows))
    except Exception:
        # Fallback to Newton-Raphson
        guess = 0.1
        for _ in range(100):
            denominator = (1 + guess) ** np.arange(cashflows.size)
            npv = np.sum(cashflows / denominator)
            d_npv = np.sum(-np.arange(cashflows.size) * cashflows / denominator / (1 + guess))
            if abs(d_npv) < 1e-8:
                break
            new_guess = guess - npv / d_npv
            if abs(new_guess - guess) < 1e-8:
                return float(new_guess)
            guess = new_guess
        return float("nan")


__all__ = ["FinancialModel", "FinancialOutputs", "npf_irr"]
