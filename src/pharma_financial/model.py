"""Core financial model built without third-party scientific dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

from .inputs import DebtEntry, ModelInputs, ProductParameters
from .table import Table, build_table


Number = float | int


@dataclass
class FinancialOutputs:
    income_statement: Table
    balance_sheet: Table
    cash_flow: Table
    summary_metrics: Table
    goal_seek: Table
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


def _value_for_year(table: Table, column: str, year: Optional[int]) -> float:
    if column not in table.data:
        return float("nan")
    values = table.column(column)
    if not values:
        return float("nan")
    if year is None:
        return values[-1]
    try:
        position = table.index.index(year)
    except ValueError:
        position = len(values) - 1
    position = max(0, min(position, len(values) - 1))
    return values[position]


class FinancialModel:
    """Implements the Longevity Pharmaceuticals financial engine."""

    def __init__(self, inputs: ModelInputs):
        self.inputs = inputs
        self.years = inputs.years
        self.products = inputs.products
        self._inflation = self._build_inflation_factors(inputs.inflation_series)
        self._risk_factors_cache: List[float] | None = None
        self._depreciation_cache: "tuple[list[dict], dict[int, float], dict[int, float]] | None" = None
        self._senior_interest_cache: List[float] | None = None
        self._senior_outstanding_cache: List[float] | None = None
        self._revolver_interest_cache: List[float] | None = None
        self._revolver_outstanding_cache: List[float] | None = None
        self._overdraft_interest_cache: List[float] | None = None
        self._overdraft_outstanding_cache: List[float] | None = None

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

    def _depreciation_rollforward(self) -> "tuple[list[dict], dict[int, float], dict[int, float]]":
        if self._depreciation_cache is not None:
            return self._depreciation_cache

        per_year_depreciation: dict[int, float] = {int(year): 0.0 for year in self.years}
        per_year_net_book: dict[int, float] = {int(year): 0.0 for year in self.years}
        details: list[dict] = []

        year_index = {year: idx for idx, year in enumerate(self.years)}
        enumerated_schedule = list(enumerate(self.inputs.depreciation_schedule))
        enumerated_schedule.sort(key=lambda item: (item[1].asset_type, item[1].year, item[0]))

        for _, row in enumerated_schedule:
            asset = row.asset_type
            method = (row.method or "straight_line").lower()
            if method not in {"straight_line", "reducing_balance"}:
                method = "straight_line"

            start_position = year_index.get(row.year)
            if start_position is None:
                continue

            configured_life = row.asset_life if row.asset_life not in (None, 0) else None
            if configured_life is not None and configured_life < 0:
                configured_life = None

            available_years = len(self.years) - start_position
            if available_years <= 0:
                continue

            if configured_life is None:
                life_span = available_years
            else:
                life_span = min(configured_life, available_years)
            if life_span <= 0:
                continue

            previous_net_book = float(row.opening_net_book or 0.0)
            previous_cumulative = float(row.opening_cumulative or 0.0)

            for offset in range(life_span):
                year_idx = start_position + offset
                year = self.years[year_idx]
                acquisition_amount = row.acquisition if offset == 0 else 0.0
                opening_net_book = previous_net_book
                opening_cumulative = previous_cumulative
                total_asset_cost = opening_net_book + acquisition_amount
                allowable = max(total_asset_cost - opening_cumulative, 0.0)

                if configured_life is not None and method == "straight_line":
                    remaining_periods = max(configured_life - offset, 1)
                    total_depreciation = allowable / remaining_periods if remaining_periods else allowable
                else:
                    if method == "reducing_balance":
                        depreciation_base = opening_net_book + (acquisition_amount * 0.5)
                    else:
                        depreciation_base = total_asset_cost
                    total_depreciation = depreciation_base * row.depreciation_rate
                    if configured_life is not None and offset >= configured_life - 1:
                        total_depreciation = allowable
                    elif total_depreciation > allowable:
                        total_depreciation = allowable

                cumulative_depreciation = opening_cumulative + total_depreciation
                net_book_value = max(total_asset_cost - cumulative_depreciation, 0.0)

                per_year_depreciation[year] = (
                    per_year_depreciation.get(year, 0.0) + total_depreciation
                )
                per_year_net_book[year] = per_year_net_book.get(year, 0.0) + net_book_value

                details.append(
                    {
                        "asset_type": asset,
                        "year": year,
                        "acquisition_year": row.year,
                        "acquisition": acquisition_amount,
                        "opening_net_book": opening_net_book,
                        "opening_cumulative": opening_cumulative,
                        "total_asset_cost": total_asset_cost,
                        "configured_rate": row.depreciation_rate,
                        "depreciation_rate": (
                            total_depreciation / total_asset_cost if total_asset_cost else 0.0
                        ),
                        "total_depreciation": total_depreciation,
                        "cumulative_depreciation": cumulative_depreciation,
                        "net_book_value": net_book_value,
                        "method": method,
                        "asset_life": configured_life,
                        "life_year_index": offset,
                        "life_span": life_span,
                    }
                )

                previous_net_book = net_book_value
                previous_cumulative = cumulative_depreciation

        self._depreciation_cache = (details, per_year_depreciation, per_year_net_book)
        return self._depreciation_cache

    def depreciation_schedule(self) -> List[float]:
        _, per_year, _ = self._depreciation_rollforward()
        return [per_year.get(year, 0.0) for year in self.years]

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
                "Total Depreciation Expense": depreciation,
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

    def _compute_amortisation(
        self, entries: List[DebtEntry], rate: float
    ) -> tuple[List[float], List[float]]:
        length = len(self.years)
        interest_schedule = [0.0 for _ in range(length)]
        outstanding_schedule = [0.0 for _ in range(length)]

        if length == 0 or not entries:
            return interest_schedule, outstanding_schedule

        year_index = {year: position for position, year in enumerate(self.years)}

        for entry in entries:
            start_idx = year_index.get(entry.year)
            if start_idx is None:
                continue

            duration = max(int(entry.duration or 0), 1)
            principal = max(float(entry.amount), float(entry.outstanding))
            opening_outstanding = min(float(entry.outstanding), principal)
            cumulative_interest = principal - opening_outstanding

            for offset in range(duration):
                idx = start_idx + offset
                if idx >= length:
                    break

                current_outstanding = max(principal - cumulative_interest, 0.0)
                if current_outstanding <= 0.0:
                    break

                remaining_periods = duration - offset
                principal_share = (
                    current_outstanding / remaining_periods if remaining_periods > 0 else current_outstanding
                )
                payment = max(current_outstanding * rate, principal_share)
                if payment > current_outstanding:
                    payment = current_outstanding

                cumulative_interest += payment
                outstanding_after = max(principal - cumulative_interest, 0.0)

                interest_schedule[idx] += payment
                outstanding_schedule[idx] += outstanding_after

        return interest_schedule, outstanding_schedule

    def _senior_debt_schedules(self) -> tuple[List[float], List[float]]:
        if (
            self._senior_interest_cache is not None
            and self._senior_outstanding_cache is not None
        ):
            return self._senior_interest_cache, self._senior_outstanding_cache

        interest_schedule, outstanding_schedule = self._compute_amortisation(
            self.inputs.financing.senior_debt_entries,
            float(self.inputs.financing.senior_debt_interest),
        )

        self._senior_interest_cache = interest_schedule
        self._senior_outstanding_cache = outstanding_schedule
        return interest_schedule, outstanding_schedule

    def _revolver_schedules(self) -> tuple[List[float], List[float]]:
        if (
            self._revolver_interest_cache is not None
            and self._revolver_outstanding_cache is not None
        ):
            return self._revolver_interest_cache, self._revolver_outstanding_cache

        interest_schedule, outstanding_schedule = self._compute_amortisation(
            self.inputs.financing.revolver_entries,
            float(self.inputs.financing.revolver_interest),
        )

        self._revolver_interest_cache = interest_schedule
        self._revolver_outstanding_cache = outstanding_schedule
        return interest_schedule, outstanding_schedule

    def _overdraft_schedules(self) -> tuple[List[float], List[float]]:
        if (
            self._overdraft_interest_cache is not None
            and self._overdraft_outstanding_cache is not None
        ):
            return self._overdraft_interest_cache, self._overdraft_outstanding_cache

        interest_schedule, outstanding_schedule = self._compute_amortisation(
            self.inputs.financing.overdraft_entries,
            float(self.inputs.financing.cash_interest),
        )

        self._overdraft_interest_cache = interest_schedule
        self._overdraft_outstanding_cache = outstanding_schedule
        return interest_schedule, outstanding_schedule

    def _interest_schedule(self) -> List[float]:
        financing = self.inputs.financing
        senior_interest, _ = self._senior_debt_schedules()
        revolver_interest, _ = self._revolver_schedules()
        overdraft_interest, _ = self._overdraft_schedules()

        interest: List[float] = []
        for idx in range(len(self.years)):
            total_interest = (
                senior_interest[idx]
                + revolver_interest[idx]
                + overdraft_interest[idx]
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
        accounts_payable = working_capital.column("Accounts Payable")
        other_liabilities = working_capital.column("Other Liabilities")

        _, senior_outstanding = self._senior_debt_schedules()
        _, revolver_outstanding = self._revolver_schedules()
        _, overdraft_outstanding = self._overdraft_schedules()

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

        total_liabilities = [
            ap
            + other
            + senior
            + revolver
            + overdraft
            for ap, other, senior, revolver, overdraft in zip(
                accounts_payable,
                other_liabilities,
                senior_outstanding,
                revolver_outstanding,
                overdraft_outstanding,
            )
        ]
        shareholders_equity = [
            total_asset - total_liability
            for total_asset, total_liability in zip(total_assets, total_liabilities)
        ]
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
                "Accounts Payable": accounts_payable,
                "Other Liabilities": other_liabilities,
                "Overdraft": overdraft_outstanding,
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

    def _calendar_days(self) -> List[float]:
        days = list(getattr(self.inputs.working_capital_days, "calendar_days", []) or [])
        if not days:
            days = [366 if year % 4 == 0 else 365 for year in self.years]

        if len(days) < len(self.years):
            fill = days[-1] if days else 365
            days = days + [fill for _ in range(len(self.years) - len(days))]

        return [float(value) for value in days[: len(self.years)]]

    def _working_capital_balances(self) -> Table:
        revenue = self.revenue_schedule().column("Net Revenue")
        cost_of_sales = self.cost_structure().column("Cost of Sales")
        days = self.inputs.working_capital_days

        days_in_year = [float(value) for value in self._calendar_days()]

        def _expand(series: Iterable[int]) -> List[float]:
            values = [float(value) for value in series]
            expanded: List[float] = []
            carry = 0.0
            for index in range(len(self.years)):
                if index < len(values):
                    carry = values[index]
                expanded.append(carry)
            return expanded

        def _calc(series: List[float], base: List[float]) -> List[float]:
            calculated: List[float] = []
            for value, denominator, day in zip(base, days_in_year, series):
                if abs(denominator) < 1e-12:
                    calculated.append(0.0)
                else:
                    calculated.append(value / denominator * day)
            return calculated

        ar_days = _expand(days.accounts_receivable)
        inventory_days = _expand(days.inventory)
        prepaid_days = _expand(days.prepaid_expenses)
        other_asset_days = _expand(days.other_assets)
        ap_days = _expand(days.accounts_payable)
        other_liability_days = _expand(days.other_liabilities)

        ar = _calc(ar_days, revenue)
        inventory = _calc(inventory_days, cost_of_sales)
        prepaid = _calc(prepaid_days, cost_of_sales)
        other_assets = _calc(other_asset_days, cost_of_sales)
        ap = _calc(ap_days, cost_of_sales)
        other_liabilities = _calc(other_liability_days, cost_of_sales)

        net_working_capital = [
            a + inv + pre + other - pay - other_liab
            for a, inv, pre, other, pay, other_liab in zip(ar, inventory, prepaid, other_assets, ap, other_liabilities)
        ]

        return build_table(
            self.years,
            {
                "Days in Year": days_in_year,
                "Accounts Receivable Days": ar_days,
                "Accounts Receivable": ar,
                "Inventory Days": inventory_days,
                "Inventory": inventory,
                "Prepaid Expenses Days": prepaid_days,
                "Prepaid Expenses": prepaid,
                "Other Assets Days": other_asset_days,
                "Other Assets": other_assets,
                "Accounts Payable Days": ap_days,
                "Accounts Payable": ap,
                "Other Liabilities Days": other_liability_days,
                "Other Liabilities": other_liabilities,
                "Net Working Capital": net_working_capital,
            },
        )

    def _working_capital_changes(self) -> List[float]:
        balances = self._working_capital_balances().column("Net Working Capital")
        return _difference(balances)

    def working_capital_schedule(self) -> Table:
        """Expose working-capital balances alongside year-over-year changes."""

        balances = self._working_capital_balances()
        changes = self._working_capital_changes()
        return balances.with_columns(**{"Change in Net Working Capital": changes})

    def inventory_schedule(self) -> Table:
        """Reconcile inventory inputs to the balance-sheet values."""

        cost_structure = self.cost_structure()
        working_capital = self._working_capital_balances()

        cost_of_sales = cost_structure.column("Cost of Sales")
        balance_inventory = working_capital.column("Inventory")
        inventory_days_source = list(self.inputs.working_capital_days.inventory)
        days_in_year = self._calendar_days()

        inventory_days: List[float] = []
        calculated_inventory: List[float] = []
        variance: List[float] = []
        turnover: List[float] = []

        for idx, cost in enumerate(cost_of_sales):
            if inventory_days_source:
                if idx < len(inventory_days_source):
                    inventory_day_value = float(inventory_days_source[idx])
                else:
                    inventory_day_value = float(inventory_days_source[-1])
            else:
                inventory_day_value = 0.0

            day_length = float(days_in_year[idx]) if days_in_year[idx] else 0.0
            inventory_days.append(inventory_day_value)

            calculated = cost / day_length * inventory_day_value if day_length else 0.0
            calculated_inventory.append(calculated)

            actual_inventory = balance_inventory[idx] if idx < len(balance_inventory) else calculated
            difference = calculated - actual_inventory
            if abs(difference) < 1e-6:
                difference = 0.0
            variance.append(difference)

            if actual_inventory:
                turnover.append(cost / actual_inventory)
            else:
                turnover.append(float("nan"))

        return build_table(
            self.years,
            {
                "Cost of Sales": cost_of_sales,
                "Days in Year": days_in_year,
                "Inventory Days": inventory_days,
                "Calculated Inventory": calculated_inventory,
                "Balance Sheet Inventory": balance_inventory,
                "Variance": variance,
                "Inventory Turnover": turnover,
            },
        )

    def _net_ppe_schedule(self) -> List[float]:
        if self.inputs.depreciation_schedule:
            _, _, per_year_net = self._depreciation_rollforward()
            return [per_year_net.get(year, 0.0) for year in self.years]

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

    def _liability_balance(self) -> List[float]:
        financing = self.inputs.financing
        _, senior = self._senior_debt_schedules()
        _, revolver = self._revolver_schedules()
        _, overdraft = self._overdraft_schedules()
        return [senior[idx] + revolver[idx] + overdraft[idx] for idx in range(len(self.years))]

    def _financing_cash_flow(self) -> List[float]:
        financing = self.inputs.financing
        dividends = [ni * financing.dividend_payout for ni in self.income_statement().column("Net Income")]
        _, senior_outstanding = self._senior_debt_schedules()
        senior_changes = _difference(senior_outstanding)
        _, revolver_outstanding = self._revolver_schedules()
        revolver_changes = _difference(revolver_outstanding)
        _, overdraft_outstanding = self._overdraft_schedules()
        overdraft_changes = _difference(overdraft_outstanding)
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

    def goal_seek_metrics(
        self,
        summary: Optional[Table] = None,
        income: Optional[Table] = None,
        cash_flow: Optional[Table] = None,
    ) -> Table:
        config = getattr(self.inputs, "goal_seek", None)
        if config is None:
            return build_table([], {"Target": []}, index_name="Metric")

        source = (config.source or "income_statement").lower()
        metric_name = config.metric

        actual = float("nan")
        if source == "summary":
            summary_table = summary or self.summary_metrics()
            if metric_name in summary_table.index:
                position = summary_table.index.index(metric_name)
                actual = summary_table.data["Value"][position]
        elif source == "cash_flow":
            cash_table = cash_flow or self.cash_flow_statement()
            actual = _value_for_year(cash_table, metric_name, config.year)
        else:
            income_table = income or self.income_statement()
            actual = _value_for_year(income_table, metric_name, config.year)

        gap = config.target - actual
        multiplier = float("nan")
        if abs(actual) > 1e-9:
            multiplier = config.target / actual

        return build_table(
            [metric_name],
            {
                "Target": [config.target],
                "Actual": [actual],
                "Gap": [gap],
                "Required Multiplier": [multiplier],
            },
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
        goal_seek = self.goal_seek_metrics(summary=summary, income=income, cash_flow=cash_flow)
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
            goal_seek=goal_seek,
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

