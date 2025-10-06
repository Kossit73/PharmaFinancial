"""Core financial model built without third-party scientific dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping

from .inputs import DebtEntry, ModelInputs, ProductParameters
from .table import Table, build_table


Number = float | int


@dataclass
class FinancialOutputs:
    income_statement: Table
    balance_sheet: Table
    cash_flow: Table
    summary_metrics: Table
    break_even: Table
    payback: Table
    discounted_payback: Table
    scenario_results: Dict[str, Table]
    sensitivity_results: Dict[str, Table]
    monte_carlo: Table


def _cumulative(values: Iterable[Number]) -> List[float]:
    total = 0.0
    result: List[float] = []
    for value in values:
        total += float(value)
        result.append(total)
    return result


def _difference(values: Iterable[Number]) -> List[float]:
    result: List[float] = []
    previous: float | None = None
    for value in values:
        if previous is None:
            result.append(float(value))
        else:
            result.append(float(value) - previous)
        previous = float(value)
    return result


def _shift(values: List[float], fill_value: float = 0.0) -> List[float]:
    return [fill_value] + values[:-1]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("nan")
    return numerator / denominator


class FinancialModel:
    """Implements the Longevity Pharmaceuticals financial engine."""

    def __init__(self, inputs: ModelInputs):
        self.inputs = inputs
        self.years = inputs.years
        self.products = inputs.products
        self._inflation = self._build_inflation_factors(inputs.inflation_series)
        self._risk_factors_cache: List[float] | None = None

    # ------------------------------------------------------------------ core
    def _build_inflation_factors(self, series: Iterable[Number]) -> List[float]:
        factors: List[float] = []
        running = 1.0
        for rate in series:
            running *= 1.0 + float(rate)
            factors.append(running)
        return factors

    def _production(self) -> Dict[str, List[float]]:
        return {name: [float(v) for v in values] for name, values in self.inputs.production_estimate.items()}

    def _unit_prices(self) -> Dict[str, float]:
        return {name: params.selling_price for name, params in self.inputs.unit_costs.items()}

    def _unit_costs(self) -> Dict[str, float]:
        return {name: params.production_cost for name, params in self.inputs.unit_costs.items()}

    def _freight_costs(self) -> Dict[str, float]:
        return {name: params.freight_cost for name, params in self.inputs.unit_costs.items()}

    # -------------------------------------------------------------- schedules
    def revenue_schedule(self) -> Table:
        production = self._production()
        prices = self._unit_prices()
        freight = self._freight_costs()

        gross_totals: List[float] = []
        commission: List[float] = []
        net_revenue: List[float] = []

        columns: MutableMapping[str, List[float]] = {product: [] for product in self.products}

        for idx, year in enumerate(self.years):
            gross_year = 0.0
            commission_year = 0.0
            for product in self.products:
                units = production[product][idx]
                gross_value = units * prices[product] * self._inflation[idx]
                columns[product].append(gross_value)
                gross_year += gross_value
                commission_year += units * freight[product]
            gross_totals.append(gross_year)
            commission.append(commission_year)
            net_revenue.append(gross_year - commission_year)

        columns["Gross Revenue"] = gross_totals
        columns["Distributors Commission"] = commission
        columns["Net Revenue"] = net_revenue
        return build_table(self.years, columns)

    def cost_structure(self) -> Table:
        production = self._production()
        total_units = [sum(production[product][idx] for product in self.products) for idx in range(len(self.years))]

        raw_material_cost = [units * self.inputs.raw_material_cost_per_unit for units in total_units]

        utility = self.inputs.utility_schedule
        utilities: List[float] = []
        for idx in range(len(self.years)):
            electricity = (
                utility.electricity_per_day[idx]
                * utility.electricity_rate[idx]
                * utility.electricity_days[idx]
            )
            water = (
                utility.water_per_day[idx]
                * utility.water_rate[idx]
                * utility.water_days[idx]
            )
            steam = (
                utility.steam_per_hour[idx]
                * utility.steam_rate[idx]
                * utility.steam_days[idx]
                * utility.steam_hours[idx]
            )
            utilities.append(electricity + water + steam)

        base_direct = sum(self.inputs.direct_labor_costs.values())
        baseline_units = total_units[0] or 1.0
        direct_labor = [base_direct * (units / baseline_units) * self._inflation[idx] for idx, units in enumerate(total_units)]

        base_indirect = sum(self.inputs.indirect_labor_costs.values())
        indirect_labor = [base_indirect * self._inflation[idx] for idx in range(len(self.years))]

        cost_of_sales = [raw + util + direct for raw, util, direct in zip(raw_material_cost, utilities, direct_labor)]
        total_expenses = [cos + indirect for cos, indirect in zip(cost_of_sales, indirect_labor)]

        return build_table(
            self.years,
            {
                "Raw Materials": raw_material_cost,
                "Utilities": utilities,
                "Direct Labor": direct_labor,
                "Cost of Sales": cost_of_sales,
                "General & Admin": indirect_labor,
                "Total Expenses": total_expenses,
            },
        )

    def depreciation_schedule(self) -> List[float]:
        depreciation = [0.0 for _ in self.years]
        for item in self.inputs.depreciation_items:
            if not item.useful_life or item.useful_life <= 0:
                continue
            annual = item.value / item.useful_life
            for idx in range(len(self.years)):
                depreciation[idx] += annual

        additions = self.inputs.capital_expenditure.get("annual_additions", {})
        for year_str, value in additions.items():
            year = int(year_str)
            if year not in self.years:
                continue
            start_idx = self.years.index(year)
            years_remaining = len(self.years) - start_idx
            annual = value / max(years_remaining, 1)
            for idx in range(start_idx, len(self.years)):
                depreciation[idx] += annual
        return depreciation

    # ----------------------------------------------------------- main outputs
    def income_statement(self) -> Table:
        revenue = self.revenue_schedule()
        costs = self.cost_structure()
        depreciation = self.depreciation_schedule()

        net_revenue = revenue.column("Net Revenue")
        total_expenses = costs.column("Total Expenses")

        ebitda = [rev - cost for rev, cost in zip(net_revenue, total_expenses)]
        ebit = [ea - dep for ea, dep in zip(ebitda, depreciation)]
        interest = self._interest_schedule()
        ebt = [e - i for e, i in zip(ebit, interest)]
        taxes = [value * rate for value, rate in zip(ebt, self._tax_schedule())]
        risk_factors = self._risk_factors()
        net_income = [
            (value - tax) * risk_factors[idx]
            for idx, (value, tax) in enumerate(zip(ebt, taxes))
        ]

        ebitda_margin = [_safe_ratio(e, r) for e, r in zip(ebitda, net_revenue)]
        ebit_margin = [_safe_ratio(e, r) for e, r in zip(ebit, net_revenue)]
        roe = [_safe_ratio(n, self.inputs.financing.share_capital) for n in net_income]

        return build_table(
            self.years,
            {
                "Gross Revenue": revenue.column("Gross Revenue"),
                "Net Revenue": net_revenue,
                "Total Expenses": total_expenses,
                "EBITDA": ebitda,
                "Depreciation": depreciation,
                "EBIT": ebit,
                "Interest": interest,
                "EBT": ebt,
                "Taxes": taxes,
                "Net Income": net_income,
                "EBITDA Margin": ebitda_margin,
                "EBIT Margin": ebit_margin,
                "Return on Equity": roe,
            },
        )

    def _interest_schedule(self) -> List[float]:
        financing = self.inputs.financing
        senior_amounts = self._instrument_values(financing.senior_debt_entries, "amount")
        revolver_amounts = self._instrument_values(financing.revolver_entries, "amount")
        overdraft_amounts = self._instrument_values(financing.overdraft_entries, "amount")

        interest: List[float] = []
        for idx in range(len(self.years)):
            total_interest = (
                senior_amounts[idx] * financing.senior_debt_interest
                + revolver_amounts[idx] * financing.revolver_interest
                + overdraft_amounts[idx] * financing.cash_interest
            )
            interest.append(-total_interest)
        return interest

    def _tax_schedule(self) -> List[float]:
        if getattr(self.inputs, "tax_rates", None):
            return list(self.inputs.tax_rates)
        return [self.inputs.tax_rate for _ in self.years]

    def _risk_factors(self) -> List[float]:
        if self._risk_factors_cache is not None:
            return self._risk_factors_cache
        schedule = getattr(self.inputs, "risk_schedule", {})
        factors: List[float] = []
        for idx in range(len(self.years)):
            factor = 1.0
            for values in schedule.values():
                if not values:
                    continue
                rate = values[idx] if idx < len(values) else values[-1]
                factor *= max(0.0, 1.0 - float(rate))
            factors.append(factor)
        self._risk_factors_cache = factors
        return factors

    def cash_flow_statement(self) -> Table:
        income = self.income_statement()
        depreciation = self.depreciation_schedule()
        working_capital_change = self._working_capital_changes()
        operating_cash_flow = [ni + dep - wc for ni, dep, wc in zip(income.column("Net Income"), depreciation, working_capital_change)]

        capex = self._capex_series()
        investing_cash_flow = [-value for value in capex]

        financing_cash_flow = self._financing_cash_flow()

        cash_change = [op + inv + fin for op, inv, fin in zip(operating_cash_flow, investing_cash_flow, financing_cash_flow)]
        beginning_cash = _shift(_cumulative(cash_change), fill_value=0.0)
        ending_cash = [begin + change for begin, change in zip(beginning_cash, cash_change)]

        return build_table(
            self.years,
            {
                "Operating Cash Flow": operating_cash_flow,
                "Investing Cash Flow": investing_cash_flow,
                "Financing Cash Flow": financing_cash_flow,
                "Net Change in Cash": cash_change,
                "Beginning Cash": beginning_cash,
                "Ending Cash": ending_cash,
            },
        )

    def balance_sheet(self) -> Table:
        cash_flow = self.cash_flow_statement()
        working_capital = self._working_capital_balances()
        net_ppe = self._net_ppe_schedule()

        total_current_assets = [
            cash + ar + inv + pre + other
            for cash, ar, inv, pre, other in zip(
                cash_flow.column("Ending Cash"),
                working_capital.column("Accounts Receivable"),
                working_capital.column("Inventory"),
                working_capital.column("Prepaid Expenses"),
                working_capital.column("Other Assets"),
            )
        ]
        total_assets = [tca + ppe for tca, ppe in zip(total_current_assets, net_ppe)]

        liabilities = self._liability_balance()
        equity = self._equity_schedule(cash_flow, income=self.income_statement())

        total_liabilities = liabilities
        shareholders_equity = equity
        total_liabilities_equity = [l + e for l, e in zip(total_liabilities, shareholders_equity)]

        return build_table(
            self.years,
            {
                "Cash": cash_flow.column("Ending Cash"),
                "Accounts Receivable": working_capital.column("Accounts Receivable"),
                "Inventory": working_capital.column("Inventory"),
                "Prepaid Expenses": working_capital.column("Prepaid Expenses"),
                "Other Assets": working_capital.column("Other Assets"),
                "Net PP&E": net_ppe,
                "Total Assets": total_assets,
                "Total Liabilities": total_liabilities,
                "Shareholders' Equity": shareholders_equity,
                "Total Liabilities & Equity": total_liabilities_equity,
            },
        )

    # -------------------------------------------------------------- schedules
    def _capex_series(self) -> List[float]:
        capex = [0.0 for _ in self.years]
        capex[0] = self.inputs.capital_expenditure.get("initial", 0.0)
        for year_str, value in self.inputs.capital_expenditure.get("annual_additions", {}).items():
            year = int(year_str)
            if year in self.years:
                capex[self.years.index(year)] += float(value)
        return capex

    def _working_capital_balances(self) -> Table:
        revenue = self.revenue_schedule().column("Net Revenue")
        cost_of_sales = self.cost_structure().column("Cost of Sales")
        days = self.inputs.working_capital_days

        days_in_year = [366 if year % 4 == 0 else 365 for year in self.years]

        def _calc(series: Iterable[int], base: List[float]) -> List[float]:
            return [value / denominator * day for value, denominator, day in zip(base, days_in_year, series)]

        ar = _calc(days.accounts_receivable, revenue)
        inventory = _calc(days.inventory, cost_of_sales)
        prepaid = _calc(days.prepaid_expenses, cost_of_sales)
        other_assets = _calc(days.other_assets, cost_of_sales)
        ap = _calc(days.accounts_payable, cost_of_sales)
        other_liabilities = _calc(days.other_liabilities, cost_of_sales)

        net_working_capital = [
            a + inv + pre + other - pay - other_liab
            for a, inv, pre, other, pay, other_liab in zip(ar, inventory, prepaid, other_assets, ap, other_liabilities)
        ]

        return build_table(
            self.years,
            {
                "Accounts Receivable": ar,
                "Inventory": inventory,
                "Prepaid Expenses": prepaid,
                "Other Assets": other_assets,
                "Accounts Payable": ap,
                "Other Liabilities": other_liabilities,
                "Net Working Capital": net_working_capital,
            },
        )

    def _working_capital_changes(self) -> List[float]:
        balances = self._working_capital_balances().column("Net Working Capital")
        return _difference(balances)

    def _net_ppe_schedule(self) -> List[float]:
        capex = self._capex_series()
        depreciation = self.depreciation_schedule()
        cumulative_capex = _cumulative(capex)
        cumulative_depreciation = _cumulative(depreciation)
        return [cap - dep for cap, dep in zip(cumulative_capex, cumulative_depreciation)]

    def _instrument_values(self, entries: List[DebtEntry], attribute: str) -> List[float]:
        per_year: Dict[int, float] = {}
        for entry in entries:
            value = float(getattr(entry, attribute))
            per_year[entry.year] = per_year.get(entry.year, 0.0) + value
        return [per_year.get(year, 0.0) for year in self.years]

    def _instrument_changes(self, entries: List[DebtEntry]) -> List[float]:
        schedule = self._instrument_values(entries, "outstanding")
        if not schedule:
            return [0.0 for _ in self.years]
        return _difference(schedule)

    def _liability_balance(self) -> List[float]:
        financing = self.inputs.financing
        senior = self._instrument_values(financing.senior_debt_entries, "outstanding")
        revolver = self._instrument_values(financing.revolver_entries, "outstanding")
        overdraft = self._instrument_values(financing.overdraft_entries, "outstanding")
        return [senior[idx] + revolver[idx] + overdraft[idx] for idx in range(len(self.years))]

    def _financing_cash_flow(self) -> List[float]:
        financing = self.inputs.financing
        dividends = [ni * financing.dividend_payout for ni in self.income_statement().column("Net Income")]
        senior_changes = self._instrument_changes(financing.senior_debt_entries)
        revolver_changes = self._instrument_changes(financing.revolver_entries)
        overdraft_changes = self._instrument_changes(financing.overdraft_entries)
        debt_changes = [
            senior_changes[idx] + revolver_changes[idx] + overdraft_changes[idx]
            for idx in range(len(self.years))
        ]
        share_issuance = [0.0 for _ in self.years]
        share_issuance[0] = financing.share_capital
        return [change + share - div for change, share, div in zip(debt_changes, share_issuance, dividends)]

    def _equity_schedule(self, cash_flow: Table, income: Table) -> List[float]:
        financing = self.inputs.financing
        net_income = income.column("Net Income")
        dividends = [ni * financing.dividend_payout for ni in net_income]
        retained = _cumulative([ni - div for ni, div in zip(net_income, dividends)])
        return [financing.share_capital + value for value in retained]

    # ---------------------------------------------------- analysis & metrics
    def scenario_analysis(self) -> Dict[str, Table]:
        results: Dict[str, Table] = {}
        base_inflation = list(self.inputs.inflation_series)
        base_discount = self.inputs.financing.discount_rate
        for name, scenario in self.inputs.scenarios.items():
            self.inputs.inflation_series = scenario.get("inflation", base_inflation)
            self.inputs.financing.discount_rate = scenario.get("interest", [base_discount])[0]
            self._inflation = self._build_inflation_factors(self.inputs.inflation_series)
            income = self.income_statement()
            results[name] = income.select(["Net Revenue", "EBITDA", "EBIT", "Net Income"])
        self.inputs.inflation_series = base_inflation
        self.inputs.financing.discount_rate = base_discount
        self._inflation = self._build_inflation_factors(self.inputs.inflation_series)
        self._risk_factors_cache = None
        return results

    def sensitivity_analysis(self) -> Dict[str, Table]:
        results: Dict[str, Table] = {}
        for variable, adjustments in self.inputs.sensitivity.variables.items():
            multipliers = []
            npvs = []
            irrs = []
            for multiplier in adjustments:
                original_tablet_price = self.inputs.unit_costs["Tablets"].selling_price
                original_raw = self.inputs.raw_material_cost_per_unit
                original_discount = self.inputs.financing.discount_rate

                if variable == "tablet_price":
                    self.inputs.unit_costs["Tablets"].selling_price = original_tablet_price * multiplier
                elif variable == "raw_material_cost":
                    self.inputs.raw_material_cost_per_unit = original_raw * multiplier
                elif variable == "discount_rate":
                    self.inputs.financing.discount_rate = multiplier

                metrics = self.summary_metrics()
                multipliers.append(multiplier)
                npvs.append(metrics.column("Value")[0])
                irrs.append(metrics.column("Value")[1])

                self.inputs.unit_costs["Tablets"].selling_price = original_tablet_price
                self.inputs.raw_material_cost_per_unit = original_raw
                self.inputs.financing.discount_rate = original_discount
            index = list(range(1, len(multipliers) + 1))
            results[variable] = build_table(index, {"Multiplier": multipliers, "NPV": npvs, "IRR": irrs}, index_name="Case")
        return results

    def monte_carlo_simulation(self) -> Table:
        iterations = self.inputs.monte_carlo.iterations
        low, high = self.inputs.monte_carlo.revenue_growth_range
        base_revenue = self.income_statement().column("Net Revenue")
        total_costs = self.cost_structure().column("Total Expenses")
        discount_rate = self.inputs.financing.discount_rate
        depreciation = self.depreciation_schedule()
        interest = self._interest_schedule()
        tax_schedule = self._tax_schedule()
        risk_factors = self._risk_factors()

        import random

        metric_names = [metric.strip() for metric in self.inputs.monte_carlo.metrics]
        allowed_metrics = {
            "NPV",
            "Average Net Income",
            "Average EBITDA",
            "Average Cash Flow",
        }
        metrics_to_track = [metric for metric in metric_names if metric in allowed_metrics]
        if "NPV" not in metrics_to_track:
            metrics_to_track.insert(0, "NPV")

        results: Dict[str, List[float]] = {metric: [] for metric in metrics_to_track}
        for _ in range(iterations):
            growth_rates = [random.uniform(low, high) for _ in self.years]
            simulated_revenue = [rev * (1 + growth) for rev, growth in zip(base_revenue, growth_rates)]
            ebitda = [rev - cost for rev, cost in zip(simulated_revenue, total_costs)]
            ebit = [ea - depreciation[idx] for idx, ea in enumerate(ebitda)]
            ebt = [ea - interest[idx] for idx, ea in enumerate(ebit)]
            taxes = [ea * tax_schedule[idx] for idx, ea in enumerate(ebt)]
            net_income = [
                (ebt[idx] - taxes[idx]) * risk_factors[idx]
                for idx in range(len(self.years))
            ]
            cash_flows = [ebitda[idx] * risk_factors[idx] for idx in range(len(self.years))]
            discounted = [
                cf / (1 + discount_rate) ** (idx + 1)
                for idx, cf in enumerate(cash_flows)
            ]
            npv = sum(discounted) - self.inputs.financing.initial_investment

            averages = {
                "Average Net Income": sum(net_income) / len(net_income),
                "Average EBITDA": sum(ebitda) / len(ebitda),
                "Average Cash Flow": sum(cash_flows) / len(cash_flows),
            }

            for metric in metrics_to_track:
                if metric == "NPV":
                    results[metric].append(npv)
                else:
                    results[metric].append(averages[metric])

        return build_table(range(1, iterations + 1), results, index_name="Iteration")

    def summary_metrics(self) -> Table:
        cash_flow = self.cash_flow_statement().column("Net Change in Cash")
        discount_rate = self.inputs.financing.discount_rate
        discounted = [cf / (1 + discount_rate) ** (idx + 1) for idx, cf in enumerate(cash_flow)]
        npv_value = sum(discounted) - self.inputs.financing.initial_investment
        irr_value = npf_irr([-self.inputs.financing.initial_investment] + cash_flow)
        payback_years = self._payback_period(cash_flow)
        discounted_payback_years = self._payback_period(discounted)
        return build_table(
            ["NPV", "IRR", "Payback Period", "Discounted Payback"],
            {"Value": [npv_value, irr_value, payback_years, discounted_payback_years]},
            index_name="Metric",
        )

    def break_even_analysis(self) -> Table:
        costs = self.cost_structure()
        fixed_cost_total = sum(costs.column("Total Expenses")) / len(self.years)
        break_even_units: Dict[str, List[float]] = {}
        for product in self.products:
            params: ProductParameters = self.inputs.unit_costs[product]
            margin = params.selling_price - params.production_cost
            if margin <= 0:
                break_even_units[product] = [float("nan")]
            else:
                break_even_units[product] = [fixed_cost_total / margin]
        return build_table(list(break_even_units.keys()), {"Units": [values[0] for values in break_even_units.values()]}, index_name="Product")

    def _payback_period(self, cash_flows: Iterable[Number]) -> float:
        cumulative = _cumulative(cash_flows)
        invested = self.inputs.financing.initial_investment
        for idx, value in enumerate(cumulative):
            if value >= invested:
                return float(self.years[idx])
        return float("nan")

    def payback_schedule(self) -> Table:
        cash_flows = self.cash_flow_statement().column("Net Change in Cash")
        cumulative = [value - self.inputs.financing.initial_investment for value in _cumulative(cash_flows)]
        return build_table(self.years, {"Cash Flow": cash_flows, "Cumulative": cumulative})

    def discounted_payback_schedule(self) -> Table:
        discount_rate = self.inputs.financing.discount_rate
        cash_flows = self.cash_flow_statement().column("Net Change in Cash")
        discounted = [cf / (1 + discount_rate) ** (idx + 1) for idx, cf in enumerate(cash_flows)]
        cumulative = [value - self.inputs.financing.initial_investment for value in _cumulative(discounted)]
        return build_table(self.years, {"Discounted Cash Flow": discounted, "Cumulative": cumulative})

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


def npf_irr(cashflows: Iterable[Number]) -> float:
    values = [float(value) for value in cashflows]
    if len(values) < 2:
        return float("nan")
    guess = 0.1
    for _ in range(100):
        denominator = [(1 + guess) ** idx for idx in range(len(values))]
        npv = sum(value / denom for value, denom in zip(values, denominator))
        derivative = sum(-idx * value / denom / (1 + guess) for idx, (value, denom) in enumerate(zip(values, denominator)))
        if abs(derivative) < 1e-8:
            break
        next_guess = guess - npv / derivative
        if abs(next_guess - guess) < 1e-8:
            return float(next_guess)
        guess = next_guess
    return float("nan")


__all__ = ["FinancialModel", "FinancialOutputs", "npf_irr"]

