"""Core financial model built without third-party scientific dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .ai import AIInsights, GenerativeAdvisor, MachineLearningAdvisor
from .debt import amortise_entries
from .inputs import BreakEvenRow, DebtEntry, ModelInputs, ProductParameters
from .table import Table, build_table


CASH_FLOW_NET_COLUMN = "Net Cash Flow for the Period"
CASH_FLOW_BEGIN_COLUMN = "Cash and Cash Equivalents at the Beginning of the Period"
CASH_FLOW_END_COLUMN = "Cash and Cash Equivalents at the End of the Period"


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
    scenario_tool_results: Mapping[str, "ScenarioToolResult"]
    risk_factor_diagnostics: Optional[Table] = None
    ai_insights: Optional[AIInsights] = None


@dataclass
class ScenarioToolResult:
    rows: List[Mapping[str, Any]]
    interpretation: str


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
    """Implements the Pharmaceuticals financial engine."""

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
        self._commission_cache: dict[int, dict[str, tuple[float, float, int]]] | None = None
        self._distributor_receivable_cache: List[float] | None = None
        self._revenue_schedule_cache: Table | None = None
        self._cost_structure_cache: Table | None = None
        self._income_statement_cache: Table | None = None
        self._cash_flow_cache: Table | None = None
        self._balance_sheet_cache: Table | None = None
        self._working_capital_cache: Table | None = None
        self._summary_metrics_cache: Table | None = None
        self._irr_result: "IRRResult | None" = None
        self._risk_factor_diagnostics_cache: Table | None = None

    # ------------------------------------------------------------------ core
    def _build_inflation_factors(self, series: Iterable[Number]) -> List[float]:
        factors: List[float] = []
        running = 1.0
        for rate in series:
            running *= 1.0 + float(rate)
            factors.append(running)
        return factors

    def _invalidate_statement_caches(self) -> None:
        self._revenue_schedule_cache = None
        self._cost_structure_cache = None
        self._income_statement_cache = None
        self._cash_flow_cache = None
        self._balance_sheet_cache = None
        self._working_capital_cache = None
        self._summary_metrics_cache = None
        self._irr_result = None
        self._risk_factors_cache = None
        self._risk_factor_diagnostics_cache = None
        self._distributor_receivable_cache = None

    def _production(self) -> Dict[str, List[float]]:
        return {name: [float(v) for v in values] for name, values in self.inputs.production_estimate.items()}

    def _unit_prices(self) -> Dict[str, float]:
        return {name: params.selling_price for name, params in self.inputs.unit_costs.items()}

    def _unit_costs(self) -> Dict[str, float]:
        return {name: params.production_cost for name, params in self.inputs.unit_costs.items()}

    def _commission_parameters(self) -> dict[int, dict[str, tuple[float, float, int]]]:
        if self._commission_cache is not None:
            return self._commission_cache

        fallback_rates: dict[str, float] = {}
        for name in self.inputs.unit_costs.keys():
            # Default distributor commission rate is 5%
            fallback_rates[name] = 0.05

        schedule: dict[int, dict[str, tuple[float, float, int]]] = {}
        for idx, year in enumerate(self.years):
            per_product: dict[str, tuple[float, float, int]] = {}
            for product in self.products:
                base_rate = fallback_rates.get(product, 0.05)
                per_product[product] = (base_rate, 1.0, 0)
            schedule[int(year)] = per_product

        for row in self.inputs.distributor_commission:
            year = int(row.year)
            if year not in schedule:
                continue
            product = row.product
            if product not in schedule[year]:
                continue
            rate = max(float(row.rate), 0.0)
            share = float(row.revenue_share) if row.revenue_share > 0 else 1.0
            payment_days = max(int(row.payment_days), 0)
            schedule[year][product] = (rate, share, payment_days)

        self._commission_cache = schedule
        return schedule

    def _freight_costs(self) -> Dict[str, float]:
        return {name: params.freight_cost for name, params in self.inputs.unit_costs.items()}

    def _variable_costs(self) -> Dict[str, float]:
        overrides = getattr(self.inputs, "variable_cost_overrides", {})
        return {product: float(overrides.get(product, 0.0)) for product in self.products}

    def _total_units(self) -> Dict[str, float]:
        return {name: float(value) for name, value in self.inputs.total_production_units.items()}

    # -------------------------------------------------------------- schedules
    def revenue_schedule(self) -> Table:
        """Build the revenue schedule using the production ramp."""

        if self._revenue_schedule_cache is not None:
            return self._revenue_schedule_cache

        prices = self._unit_prices()
        production = self._production()
        commission_params = self._commission_parameters()
        risk_factors = self._risk_factors()

        gross_totals: List[float] = []
        commission: List[float] = []
        net_revenue: List[float] = []
        distributor_receivable_weighted: List[float] = [0.0 for _ in self.years]

        columns: MutableMapping[str, List[float]] = {product: [] for product in self.products}

        for idx, year in enumerate(self.years):
            gross_year = 0.0
            commission_year = 0.0
            risk = risk_factors[idx] if idx < len(risk_factors) else (
                risk_factors[-1] if risk_factors else 1.0
            )
            try:
                inflation_factor = self._inflation[idx]
            except IndexError:  # pragma: no cover - defensive guard
                inflation_factor = self._inflation[-1] if self._inflation else 1.0
            year_rates = commission_params.get(int(year), {})
            for product in self.products:
                product_units = 0.0
                if product in production:
                    values = production[product]
                    if idx < len(values):
                        product_units = values[idx]
                    elif values:
                        product_units = values[-1]
                price = float(prices.get(product, 0.0))
                gross_value = product_units * price * inflation_factor * risk
                columns[product].append(gross_value)
                gross_year += gross_value
                rate, share, payment_days = year_rates.get(product, (0.0, 1.0, 0))
                effective_share = max(share, 0.0)
                commission_rate = max(rate, 0.0)
                commission_amount = gross_value * effective_share * commission_rate
                distributor_portion = gross_value * effective_share - commission_amount
                distributor_receivable_weighted[idx] += distributor_portion * max(payment_days, 0)
                commission_year += commission_amount
            gross_totals.append(gross_year)
            commission.append(commission_year)
            net_revenue.append(gross_year - commission_year)

        self._distributor_receivable_cache = distributor_receivable_weighted

        columns["Gross Revenue"] = gross_totals
        columns["Distributors Commission"] = commission
        columns["Net Revenue"] = net_revenue
        table = build_table(self.years, columns)
        self._revenue_schedule_cache = table
        return table

    def cost_structure(self) -> Table:
        if self._cost_structure_cache is not None:
            return self._cost_structure_cache

        production = self._production()
        total_units = [sum(production[product][idx] for product in self.products) for idx in range(len(self.years))]

        risk_factors = self._risk_factors()

        def _risk_for_index(index: int) -> float:
            if risk_factors:
                if index < len(risk_factors):
                    return risk_factors[index]
                return risk_factors[-1]
            return 1.0

        variable_lookup = self._variable_costs()

        raw_material_cost: List[float] = []
        for idx in range(len(self.years)):
            risk = _risk_for_index(idx)
            try:
                inflation = self._inflation[idx]
            except IndexError:  # pragma: no cover - defensive guard
                inflation = self._inflation[-1] if self._inflation else 1.0

            total = 0.0
            for product in self.products:
                product_units = production[product][idx]
                variable_cost = variable_lookup.get(product, 0.0)
                total += product_units * variable_cost * inflation * risk
            raw_material_cost.append(total)

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
        direct_labor = [
            base_direct
            * (units / baseline_units)
            * self._inflation[idx]
            * _risk_for_index(idx)
            for idx, units in enumerate(total_units)
        ]

        base_indirect = sum(self.inputs.indirect_labor_costs.values())
        indirect_labor = [
            base_indirect * self._inflation[idx] * _risk_for_index(idx)
            for idx in range(len(self.years))
        ]

        utility_cost_of_sales = [value * 0.8 for value in utilities]
        utility_general_admin = [value - cos_share for value, cos_share in zip(utilities, utility_cost_of_sales)]

        cost_of_sales = [
            raw + util_share + direct
            for raw, util_share, direct in zip(
                raw_material_cost, utility_cost_of_sales, direct_labor
            )
        ]
        general_admin = [
            base + util_share
            for base, util_share in zip(indirect_labor, utility_general_admin)
        ]
        total_expenses = [cos + ga for cos, ga in zip(cost_of_sales, general_admin)]

        table = build_table(
            self.years,
            {
                "Raw Materials": raw_material_cost,
                "Utilities": utilities,
                "Direct Labor": direct_labor,
                "Cost of Sales": cost_of_sales,
                "General & Admin": general_admin,
                "Total Expenses": total_expenses,
            },
        )
        self._cost_structure_cache = table
        return table

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
                total_asset_cost = (
                    opening_net_book + opening_cumulative + acquisition_amount
                )
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
        if self._income_statement_cache is not None:
            return self._income_statement_cache

        revenue = self.revenue_schedule()
        costs = self.cost_structure()
        depreciation = self.depreciation_schedule()

        gross_revenue = revenue.column("Gross Revenue")
        distributors_commission = revenue.column("Distributors Commission")
        net_revenue = revenue.column("Net Revenue")

        cost_of_sales = costs.column("Cost of Sales")
        general_admin = costs.column("General & Admin")

        gross_profit = [rev - cost for rev, cost in zip(net_revenue, cost_of_sales)]
        ebitda = [gp - ga for gp, ga in zip(gross_profit, general_admin)]
        ebit = [eb - dep for eb, dep in zip(ebitda, depreciation)]
        interest = self._interest_schedule()
        ebt = [e - i for e, i in zip(ebit, interest)]
        taxes: List[float] = []
        net_income: List[float] = []
        for idx, (value, rate) in enumerate(zip(ebt, self._tax_schedule())):
            if value <= 0:
                tax = 0.0
                base_net = value
            else:
                tax = value * rate
                base_net = value - tax

            taxes.append(tax)
            net_income.append(base_net)

        gross_profit_margin = [_safe_ratio(gp, gr) for gp, gr in zip(gross_profit, gross_revenue)]
        ebitda_margin = [_safe_ratio(e, r) for e, r in zip(ebitda, net_revenue)]
        ebit_margin = [_safe_ratio(e, r) for e, r in zip(ebit, net_revenue)]
        roe = [_safe_ratio(n, self.inputs.financing.share_capital) for n in net_income]

        columns: MutableMapping[str, List[float]] = {
            "Gross Revenue": gross_revenue,
            "Distributors Commission": distributors_commission,
            "Net Revenue": net_revenue,
            "Cost of Sales": cost_of_sales,
            "Gross Profit": gross_profit,
            "General & Admin": general_admin,
            "EBITDA": ebitda,
            "Total Depreciation Expense": depreciation,
            "EBIT": ebit,
            "Interest": interest,
            "EBT": ebt,
            "Taxes": taxes,
            "Net Income": net_income,
            "Gross Profit Margin": gross_profit_margin,
            "EBITDA Margin": ebitda_margin,
            "EBIT Margin": ebit_margin,
            "Return on Equity": roe,
        }

        table = build_table(self.years, columns)
        self._income_statement_cache = table
        return table

    def _compute_amortisation(
        self, entries: List[DebtEntry], rate: float
    ) -> tuple[List[float], List[float]]:
        length = len(self.years)
        interest_schedule, _, outstanding_schedule, _ = amortise_entries(
            entries, rate, self.years
        )
        if len(interest_schedule) < length:
            interest_schedule.extend([0.0] * (length - len(interest_schedule)))
        if len(outstanding_schedule) < length:
            outstanding_schedule.extend([0.0] * (length - len(outstanding_schedule)))
        return interest_schedule[:length], outstanding_schedule[:length]

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
            interest.append(total_interest)
        return interest

    def _tax_schedule(self) -> List[float]:
        if getattr(self.inputs, "tax_rates", None):
            return list(self.inputs.tax_rates)
        return [self.inputs.tax_rate for _ in self.years]

    def _risk_factors(self) -> List[float]:
        if self._risk_factors_cache is not None:
            return self._risk_factors_cache
        schedule = getattr(self.inputs, "risk_schedule", {})
        factor_columns: Dict[str, List[float]] = {}
        combined: List[float] = []

        if schedule:
            for name in schedule.keys():
                factor_columns[f"{name} Factor"] = []

        for idx in range(len(self.years)):
            combined_factor = 1.0
            for name, values in schedule.items():
                if not values:
                    continue
                series = list(values)
                rate = series[idx] if idx < len(series) else series[-1]
                if not 0.0 <= float(rate) <= 1.0:
                    raise ValueError(
                        f"Risk schedule '{name}' has value {rate} outside the [0, 1] range"
                    )
                factor = max(0.0, 1.0 - float(rate))
                factor_columns.setdefault(f"{name} Factor", []).append(factor)
                combined_factor *= factor
            combined.append(combined_factor)

        if not combined:
            combined = [1.0 for _ in self.years]

        diagnostics_columns: Dict[str, List[float]] = {}
        diagnostics_columns.update(factor_columns)
        diagnostics_columns["Combined Factor"] = combined
        diagnostics_columns["Combined Shock (%)"] = [
            (1.0 - value) * 100.0 for value in combined
        ]
        self._risk_factor_diagnostics_cache = build_table(self.years, diagnostics_columns)
        self._risk_factors_cache = combined
        return combined

    def risk_factor_diagnostics(self) -> Table:
        if self._risk_factor_diagnostics_cache is None:
            self._risk_factors()
        if self._risk_factor_diagnostics_cache is None:
            defaults = {
                "Combined Factor": [1.0 for _ in self.years],
                "Combined Shock (%)": [0.0 for _ in self.years],
            }
            self._risk_factor_diagnostics_cache = build_table(self.years, defaults)
        return self._risk_factor_diagnostics_cache

    def cash_flow_statement(self) -> Table:
        if self._cash_flow_cache is not None:
            return self._cash_flow_cache

        income = self.income_statement()
        depreciation = self.depreciation_schedule()
        working_balances = self._working_capital_balances()

        net_income = income.column("Net Income")
        tax_expense = income.column("Taxes")
        interest_expense = income.column("Interest")
        depreciation_expense = list(depreciation)

        operating_profit = [
            ni + tax + interest
            for ni, tax, interest in zip(
                net_income,
                tax_expense,
                interest_expense,
            )
        ]

        inventory_change = _difference(working_balances.column("Inventory"))
        receivable_change = _difference(working_balances.column("Accounts Receivable"))
        payable_change = _difference(working_balances.column("Accounts Payable"))
        prepaid_change = _difference(working_balances.column("Prepaid Expenses"))
        other_asset_change = _difference(working_balances.column("Other Assets"))
        other_liability_change = _difference(working_balances.column("Other Liabilities"))

        inventory_adjustment = [-value for value in inventory_change]
        receivable_adjustment = [-value for value in receivable_change]
        payable_adjustment = [value for value in payable_change]
        prepaid_adjustment = [-value for value in prepaid_change]
        other_asset_adjustment = [-value for value in other_asset_change]
        other_liability_adjustment = [value for value in other_liability_change]

        cash_flow_from_operations = [
            op
            + dep
            + inv
            + ar
            + ap
            + pre
            + other_asset_adj
            + other_liab
            for op, dep, inv, ar, ap, pre, other_asset_adj, other_liab in zip(
                operating_profit,
                depreciation_expense,
                inventory_adjustment,
                receivable_adjustment,
                payable_adjustment,
                prepaid_adjustment,
                other_asset_adjustment,
                other_liability_adjustment,
                strict=False,
            )
        ]

        interest_paid = [-value for value in interest_expense]
        taxes_paid = [-value for value in tax_expense]

        net_cash_from_operations = [
            cfo + interest + tax
            for cfo, interest, tax in zip(
                cash_flow_from_operations,
                interest_paid,
                taxes_paid,
                strict=False,
            )
        ]

        capex = self._capex_series()
        capital_expenditure = [-value for value in capex]
        net_cash_from_investing = list(capital_expenditure)

        financing_components = self._financing_cash_flow_components()
        net_cash_from_financing = [
            sum(values)
            for values in zip(*financing_components.values(), strict=False)
        ]

        net_cash_flow = [
            op + inv + fin
            for op, inv, fin in zip(
                net_cash_from_operations,
                net_cash_from_investing,
                net_cash_from_financing,
                strict=False,
            )
        ]

        beginning_cash = _shift(_cumulative(net_cash_flow), fill_value=0.0)
        ending_cash = [begin + change for begin, change in zip(beginning_cash, net_cash_flow)]

        columns: Dict[str, List[float]] = {
            "Cash Flow from Operations": cash_flow_from_operations,
            "Net Cash Generated from Operating Activities": net_cash_from_operations,
            "Net Cash Used in Investing Activities": net_cash_from_investing,
            "Net Cash Used in Financing Activities": net_cash_from_financing,
            CASH_FLOW_NET_COLUMN: net_cash_flow,
            CASH_FLOW_BEGIN_COLUMN: beginning_cash,
            CASH_FLOW_END_COLUMN: ending_cash,
            "Net Increase/Decrease in Cash": net_cash_flow,
        }

        table = build_table(self.years, columns)
        self._cash_flow_cache = table
        return table

    def balance_sheet(self) -> Table:
        if self._balance_sheet_cache is not None:
            return self._balance_sheet_cache

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
                cash_flow.column(CASH_FLOW_END_COLUMN),
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

        table = build_table(
            self.years,
            {
                "Cash": cash_flow.column(CASH_FLOW_END_COLUMN),
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
        self._balance_sheet_cache = table
        return table

    # -------------------------------------------------------------- schedules
    def _capex_series(self) -> List[float]:
        capex = [0.0 for _ in self.years]
        if not capex:
            return capex

        config = self.inputs.capital_expenditure or {}

        first_year_total = 0.0
        for key, value in config.items():
            if key == "annual_additions":
                continue
            try:
                first_year_total += float(value)
            except (TypeError, ValueError):
                continue
        capex[0] += first_year_total

        additions = config.get("annual_additions", {})
        if isinstance(additions, Mapping):
            for year_value, addition in additions.items():
                try:
                    year = int(year_value)
                except (TypeError, ValueError):
                    continue
                if year not in self.years:
                    continue
                try:
                    capex[self.years.index(year)] += float(addition)
                except (TypeError, ValueError):
                    continue

        year_index = {year: idx for idx, year in enumerate(self.years)}
        for row in self.inputs.depreciation_schedule:
            try:
                acquisition = float(row.acquisition or 0.0)
            except (TypeError, ValueError):
                acquisition = 0.0
            if not acquisition:
                continue
            idx = year_index.get(row.year)
            if idx is None:
                continue
            capex[idx] += acquisition

        return capex

    def _calendar_days(self) -> List[float]:
        days = list(getattr(self.inputs.working_capital_days, "calendar_days", []) or [])
        if not days:
            days = [366 if year % 4 == 0 else 365 for year in self.years]

        if len(days) < len(self.years):
            fill = days[-1] if days else 365
            days = days + [fill for _ in range(len(self.years) - len(days))]

        return [float(value) for value in days[: len(self.years)]]

    def _distributor_receivable_balances(
        self, *, weighted: Optional[Sequence[float]] = None
    ) -> List[float]:
        if weighted is None:
            if self._distributor_receivable_cache is None:
                self.revenue_schedule()
            weighted_series = list(self._distributor_receivable_cache or [0.0 for _ in self.years])
        else:
            weighted_series = [float(value) for value in weighted]
            if len(weighted_series) < len(self.years):
                weighted_series = weighted_series + [0.0] * (len(self.years) - len(weighted_series))
            else:
                weighted_series = weighted_series[: len(self.years)]

        days_in_year = self._calendar_days()
        balances: List[float] = []
        for weighted_value, day_count in zip(weighted_series, days_in_year):
            if day_count:
                balances.append(weighted_value / day_count)
            else:
                balances.append(0.0)
        return balances

    def _working_capital_balances_from_series(
        self,
        revenue: Sequence[float],
        cost_of_sales: Sequence[float],
        *,
        distributor_weighted: Optional[Sequence[float]] = None,
    ) -> Table:
        days = self.inputs.working_capital_days

        def _pad(series: Sequence[float]) -> List[float]:
            values = [float(value) for value in series]
            if len(values) < len(self.years):
                fill = values[-1] if values else 0.0
                values = values + [fill for _ in range(len(self.years) - len(values))]
            return values[: len(self.years)]

        revenue_series = _pad(revenue)
        cost_series = _pad(cost_of_sales)

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

        ar_base = _calc(ar_days, revenue_series)
        distributor_receivables = self._distributor_receivable_balances(weighted=distributor_weighted)
        ar = [base + dist for base, dist in zip(ar_base, distributor_receivables)]
        inventory = _calc(inventory_days, cost_series)
        prepaid = _calc(prepaid_days, cost_series)
        other_assets = _calc(other_asset_days, cost_series)
        ap = _calc(ap_days, cost_series)
        other_liabilities = _calc(other_liability_days, cost_series)

        net_working_capital = [
            a + inv + pre + other - pay - other_liab
            for a, inv, pre, other, pay, other_liab in zip(ar, inventory, prepaid, other_assets, ap, other_liabilities)
        ]

        return build_table(
            self.years,
            {
                "Days in Year": days_in_year,
                "Accounts Receivable Days": ar_days,
                "Accounts Receivable (Base)": ar_base,
                "Distributor Receivables": distributor_receivables,
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

    def _working_capital_balances(self) -> Table:
        if self._working_capital_cache is not None:
            return self._working_capital_cache

        revenue = self.revenue_schedule().column("Net Revenue")
        cost_of_sales = self.cost_structure().column("Cost of Sales")
        weighted = self._distributor_receivable_cache or [0.0 for _ in self.years]
        table = self._working_capital_balances_from_series(
            revenue,
            cost_of_sales,
            distributor_weighted=weighted,
        )
        self._working_capital_cache = table
        return table

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

    def _dividend_payments(self) -> List[float]:
        net_income = self.income_statement().column("Net Income")
        payout = self.inputs.financing.dividend_payout
        return [-max(ni, 0.0) * payout for ni in net_income]

    def _financing_cash_flow_components(self) -> Dict[str, List[float]]:
        financing = self.inputs.financing
        _, senior_outstanding = self._senior_debt_schedules()
        senior_changes = _difference(senior_outstanding)
        _, revolver_outstanding = self._revolver_schedules()
        revolver_changes = _difference(revolver_outstanding)
        _, overdraft_outstanding = self._overdraft_schedules()
        overdraft_changes = _difference(overdraft_outstanding)

        debt_movements = [
            senior_changes[idx] + revolver_changes[idx] + overdraft_changes[idx]
            for idx in range(len(self.years))
        ]

        share_issuance = [0.0 for _ in self.years]
        if financing.share_capital:
            share_issuance[0] = float(financing.share_capital)

        initial_investment = [0.0 for _ in self.years]
        if financing.initial_investment:
            initial_investment[0] = -float(financing.initial_investment)

        dividends_paid = self._dividend_payments()

        return {
            "Debt Drawdown/(Repayment)": debt_movements,
            "Share Capital Raised": share_issuance,
            "Initial Investment": initial_investment,
            "Dividends Paid": dividends_paid,
        }

    def _equity_schedule(self, cash_flow: Table, income: Table) -> List[float]:
        financing = self.inputs.financing
        net_income = income.column("Net Income")
        dividends = [max(ni, 0.0) * financing.dividend_payout for ni in net_income]
        retained = _cumulative([ni - div for ni, div in zip(net_income, dividends)])
        return [financing.share_capital + value for value in retained]

    # ---------------------------------------------------- analysis & metrics
    def scenario_analysis(self) -> Dict[str, Table]:
        results: Dict[str, Table] = {}
        base_inflation = list(self.inputs.inflation_series)
        base_discount = float(self.inputs.financing.discount_rate)
        try:
            for name, scenario in self.inputs.scenarios.items():
                inflation_override = scenario.get("inflation", base_inflation)
                inflation_series = [float(value) for value in inflation_override]
                interest_values = scenario.get("interest", [base_discount])
                try:
                    discount_rate = float(interest_values[0]) if interest_values else base_discount
                except (TypeError, ValueError, IndexError):
                    discount_rate = base_discount

                self.inputs.inflation_series = inflation_series
                self.inputs.financing.discount_rate = discount_rate
                self._inflation = self._build_inflation_factors(self.inputs.inflation_series)
                self._invalidate_statement_caches()
                income = self.income_statement()
                results[name] = income.select(["Net Revenue", "EBITDA", "EBIT", "Net Income"])
        finally:
            self.inputs.inflation_series = base_inflation
            self.inputs.financing.discount_rate = base_discount
            self._inflation = self._build_inflation_factors(self.inputs.inflation_series)
            self._invalidate_statement_caches()
        return results

    def scenario_toolkit(self, scenarios: Mapping[str, Table]) -> Dict[str, ScenarioToolResult]:
        configured = getattr(self.inputs, "scenario_tools", {}) or {}
        if not scenarios or not configured:
            return {}

        scenario_names = list(scenarios.keys())
        if not scenario_names:
            return {}

        base_name = None
        for name in scenario_names:
            if name.lower() == "base":
                base_name = name
                break
        if base_name is None:
            base_name = scenario_names[0]

        def _final_values(variable: str) -> Dict[str, float]:
            values: Dict[str, float] = {}
            for scenario_name, table in scenarios.items():
                if variable not in table.data:
                    continue
                series = table.column(variable)
                if not series:
                    continue
                values[scenario_name] = float(series[-1])
            return values

        def _series_for(variable: str, scenario_name: str) -> List[float]:
            table = scenarios.get(scenario_name)
            if table is None or variable not in table.data:
                return []
            return table.column(variable)

        results: Dict[str, ScenarioToolResult] = {}

        # Decision tree analysis
        decision_variables = configured.get("decision_tree", [])
        if decision_variables:
            decision_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            for variable in decision_variables:
                finals = _final_values(variable)
                if not finals:
                    continue
                best_scenario = max(finals, key=finals.get)
                worst_scenario = min(finals, key=finals.get)
                best_value = finals[best_scenario]
                worst_value = finals[worst_scenario]
                expected = sum(finals.values()) / len(finals)
                decision_rows.append(
                    {
                        "Variable": variable,
                        "Best Scenario": best_scenario,
                        "Best Value": best_value,
                        "Worst Scenario": worst_scenario,
                        "Worst Value": worst_value,
                        "Expected Value": expected,
                    }
                )
                narratives.append(
                    f"{variable} peaks under {best_scenario} and softens the most under {worst_scenario}."
                )
            if decision_rows:
                interpretation = " ".join(narratives) or "Decision tree insights derived from configured scenarios."
                results["decision_tree"] = ScenarioToolResult(rows=decision_rows, interpretation=interpretation)

        # Stress testing
        stress_variables = configured.get("stress_testing", [])
        if stress_variables:
            stress_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            for variable in stress_variables:
                finals = _final_values(variable)
                if not finals:
                    continue
                upside = max(finals.values())
                downside = min(finals.values())
                base_value = finals.get(base_name, sum(finals.values()) / len(finals))
                stress_range = upside - downside
                stress_rows.append(
                    {
                        "Variable": variable,
                        "Base": base_value,
                        "Downside": downside,
                        "Upside": upside,
                        "Stress Range": stress_range,
                    }
                )
                narratives.append(
                    f"{variable} endures a swing of {stress_range:,.2f} across configured stress scenarios."
                )
            if stress_rows:
                results["stress_testing"] = ScenarioToolResult(
                    rows=stress_rows,
                    interpretation=" ".join(narratives) or "Stress testing compares upside and downside spans.",
                )

        # Backtesting
        backtesting_variables = configured.get("backtesting", [])
        if backtesting_variables:
            back_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            for variable in backtesting_variables:
                base_series = _series_for(variable, base_name)
                if not base_series:
                    continue
                comparisons: List[float] = []
                final_errors: List[float] = []
                for scenario_name, table in scenarios.items():
                    if scenario_name == base_name or variable not in table.data:
                        continue
                    series = _series_for(variable, scenario_name)
                    if len(series) != len(base_series):
                        continue
                    errors = [abs(a - b) for a, b in zip(series, base_series)]
                    if not errors:
                        continue
                    comparisons.append(sum(errors) / len(errors))
                    final_errors.append(abs(series[-1] - base_series[-1]))
                if not comparisons:
                    continue
                average_error = sum(comparisons) / len(comparisons)
                max_error = max(final_errors) if final_errors else 0.0
                back_rows.append(
                    {
                        "Variable": variable,
                        "Reference Scenario": base_name,
                        "Mean Absolute Error": average_error,
                        "Worst Final Deviation": max_error,
                    }
                )
                narratives.append(
                    f"{variable} deviates on average by {average_error:,.2f} from the {base_name} path."
                )
            if back_rows:
                results["backtesting"] = ScenarioToolResult(
                    rows=back_rows,
                    interpretation=" ".join(narratives) or "Backtesting compares alternative scenarios to the reference path.",
                )

        # Walk-forward testing
        walk_forward_variables = configured.get("walk_forward", configured.get("walk_forward_testing", []))
        if not isinstance(walk_forward_variables, list):
            walk_forward_variables = list(walk_forward_variables)  # type: ignore[arg-type]
        if walk_forward_variables:
            walk_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            for variable in walk_forward_variables:
                base_series = _series_for(variable, base_name)
                if len(base_series) < 2:
                    continue
                growth_rates = []
                for idx in range(1, len(base_series)):
                    previous = base_series[idx - 1]
                    current = base_series[idx]
                    growth_rates.append(_safe_ratio(current - previous, previous))
                if not growth_rates:
                    continue
                average_growth = sum(growth_rates) / len(growth_rates)
                variance = sum((rate - average_growth) ** 2 for rate in growth_rates) / len(growth_rates)
                volatility = variance ** 0.5
                walk_rows.append(
                    {
                        "Variable": variable,
                        "Average Growth": average_growth,
                        "Volatility": volatility,
                    }
                )
                narratives.append(
                    f"{variable} grows on average {average_growth:.2%} with volatility of {volatility:.2%}."
                )
            if walk_rows:
                results["walk_forward"] = ScenarioToolResult(
                    rows=walk_rows,
                    interpretation=" ".join(narratives) or "Walk-forward analysis summarises stability in the reference scenario.",
                )

        # Driver-based modelling
        driver_variables = configured.get("driver_based", configured.get("driver_based_modeling", []))
        if not isinstance(driver_variables, list):
            driver_variables = list(driver_variables)  # type: ignore[arg-type]
        if driver_variables:
            driver_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            revenue_series = _series_for("Net Revenue", base_name)
            for variable in driver_variables:
                variable_series = _series_for(variable, base_name)
                if not revenue_series or not variable_series:
                    continue
                ratios = [
                    _safe_ratio(var, rev)
                    for var, rev in zip(variable_series, revenue_series)
                    if abs(rev) > 1e-12
                ]
                if not ratios:
                    continue
                final_ratio = ratios[-1]
                average_ratio = sum(ratios) / len(ratios)
                driver_rows.append(
                    {
                        "Variable": variable,
                        "Average Contribution": average_ratio,
                        "Latest Contribution": final_ratio,
                    }
                )
                narratives.append(
                    f"{variable} contributes {final_ratio:.2%} of revenue in the latest projection."
                )
            if driver_rows:
                results["driver_based"] = ScenarioToolResult(
                    rows=driver_rows,
                    interpretation=" ".join(narratives) or "Driver-based modelling links value drivers to revenue.",
                )

        # Real options analysis
        roa_variables = configured.get("real_options", configured.get("real_options_analysis", []))
        if not isinstance(roa_variables, list):
            roa_variables = list(roa_variables)  # type: ignore[arg-type]
        if roa_variables:
            roa_rows: List[Mapping[str, Any]] = []
            narratives: List[str] = []
            for variable in roa_variables:
                finals = _final_values(variable)
                if not finals:
                    continue
                base_value = finals.get(base_name)
                if base_value is None:
                    base_value = sum(finals.values()) / len(finals)
                upside = max(finals.values())
                option_value = max(upside - base_value, 0.0)
                roa_rows.append(
                    {
                        "Variable": variable,
                        "Reference": base_value,
                        "Best Case": upside,
                        "Option Value": option_value,
                    }
                )
                narratives.append(
                    f"{variable} offers an upside optionality of {option_value:,.2f} relative to the reference path."
                )
            if roa_rows:
                results["real_options"] = ScenarioToolResult(
                    rows=roa_rows,
                    interpretation=" ".join(narratives) or "Real options quantify strategic upside over the reference scenario.",
                )

        return results

    def sensitivity_analysis(self) -> Dict[str, Table]:
        results: Dict[str, Table] = {}
        for variable, adjustments in self.inputs.sensitivity.variables.items():
            multipliers = []
            npvs = []
            irrs = []
            original_variables = dict(getattr(self.inputs, "variable_cost_overrides", {}))
            for multiplier in adjustments:
                original_tablet_price = self.inputs.unit_costs["Tablets"].selling_price
                original_discount = self.inputs.financing.discount_rate

                if variable == "tablet_price":
                    self.inputs.unit_costs["Tablets"].selling_price = original_tablet_price * multiplier
                elif variable == "raw_material_cost":
                    scaled = {
                        product: value * multiplier for product, value in original_variables.items()
                    }
                    self.inputs.variable_cost_overrides = scaled
                elif variable == "discount_rate":
                    self.inputs.financing.discount_rate = multiplier

                self._invalidate_statement_caches()
                metrics = self.summary_metrics()
                multipliers.append(multiplier)
                npvs.append(metrics.column("Value")[0])
                irrs.append(metrics.column("Value")[1])

                self.inputs.unit_costs["Tablets"].selling_price = original_tablet_price
                self.inputs.variable_cost_overrides = dict(original_variables)
                self.inputs.financing.discount_rate = original_discount
                self._invalidate_statement_caches()
            index = list(range(1, len(multipliers) + 1))
            results[variable] = build_table(index, {"Multiplier": multipliers, "NPV": npvs, "IRR": irrs}, index_name="Case")
        return results

    def monte_carlo_simulation(self) -> Table:
        params = self.inputs.monte_carlo
        iterations = params.iterations
        bounds = [float(value) for value in params.revenue_growth_range]
        if len(bounds) >= 2:
            low, high = sorted((bounds[0], bounds[1]))
        elif bounds:
            span = abs(bounds[0])
            low, high = -span, span
        else:
            low, high = -0.05, 0.05

        revenue_table = self.revenue_schedule()
        base_revenue = revenue_table.column("Net Revenue")
        costs = self.cost_structure()
        raw_materials = costs.column("Raw Materials")
        utilities = costs.column("Utilities")
        direct_labor = costs.column("Direct Labor")
        indirect_labor = costs.column("General & Admin")
        base_cost_of_sales = costs.column("Cost of Sales")
        depreciation = self.depreciation_schedule()
        interest = self._interest_schedule()
        tax_schedule = self._tax_schedule()
        discount_rate = self.inputs.financing.discount_rate
        capex = self._capex_series()
        financing_components = self._financing_cash_flow_components()
        other_financing = {
            name: list(series)
            for name, series in financing_components.items()
            if name != "Dividends Paid"
        }
        base_weighted = self._distributor_receivable_cache or [0.0 for _ in self.years]

        import random

        rng = random.Random()
        if params.seed is not None:
            rng.seed(params.seed)
        distribution = (params.distribution or "uniform").lower()

        def _sample() -> float:
            if distribution == "normal":
                mean = (low + high) / 2
                std_dev = (high - low) / 6 if high != low else max(abs(high), 1.0) / 6
                value = rng.gauss(mean, std_dev)
                return max(min(value, high), low)
            if distribution in {"triangular", "triangle"}:
                mode = (low + high) / 2
                return rng.triangular(low, high, mode)
            return rng.uniform(low, high)

        metric_names = [metric.strip() for metric in params.metrics]
        allowed_metrics = {
            "NPV",
            "Average Net Income",
            "Average EBITDA",
            "Average Cash Flow",
        }
        metrics_to_track = [metric for metric in metric_names if metric in allowed_metrics]
        if "NPV" not in metrics_to_track:
            metrics_to_track.insert(0, "NPV")

        variable_codes = [
            value
            for value in getattr(params, "variables", ["revenue_growth"])
            if value
        ]
        if not variable_codes:
            variable_codes = ["revenue_growth"]
        if "revenue_growth" not in variable_codes:
            variable_codes.insert(0, "revenue_growth")

        results: Dict[str, List[float]] = {metric: [] for metric in metrics_to_track}

        for _ in range(iterations):
            if "revenue_growth" in variable_codes:
                growth_rates = [_sample() for _ in self.years]
            else:
                growth_rates = [0.0 for _ in self.years]

            raw_factor = 1.0 + (_sample() if "raw_material_cost" in variable_codes else 0.0)
            labor_factor = 1.0 + (_sample() if "labor_cost" in variable_codes else 0.0)
            utility_factor = 1.0 + (_sample() if "utility_cost" in variable_codes else 0.0)
            interest_factor = 1.0 + (_sample() if "senior_debt" in variable_codes else 0.0)
            tax_factor = 1.0 + (_sample() if "tax_rate" in variable_codes else 0.0)
            if tax_factor < 0:
                tax_factor = 0.0

            risk_adjustment = 1.0
            if "other" in variable_codes:
                risk_adjustment = max(0.0, 1.0 - _sample())
            risk_series = [risk_adjustment for _ in self.years] if "other" in variable_codes else [1.0 for _ in self.years]

            simulated_revenue = [
                base * (1.0 + growth) * risk_series[idx]
                for idx, (base, growth) in enumerate(zip(base_revenue, growth_rates))
            ]

            raw_series = [value * raw_factor * risk_series[idx] for idx, value in enumerate(raw_materials)]
            utility_series = [max(0.0, value * utility_factor) for value in utilities]
            direct_series = [value * labor_factor * risk_series[idx] for idx, value in enumerate(direct_labor)]
            indirect_series = [value * labor_factor * risk_series[idx] for idx, value in enumerate(indirect_labor)]

            utility_cost_share = [value * 0.8 for value in utility_series]
            utility_admin_share = [value - cost_share for value, cost_share in zip(utility_series, utility_cost_share)]
            simulated_cost_of_sales = [
                raw_series[idx] + utility_cost_share[idx] + direct_series[idx]
                for idx in range(len(self.years))
            ]
            general_admin_series = [
                indirect_series[idx] + utility_admin_share[idx]
                for idx in range(len(self.years))
            ]

            gross_profit = [
                simulated_revenue[idx] - simulated_cost_of_sales[idx]
                for idx in range(len(self.years))
            ]
            ebitda = [
                gross_profit[idx] - general_admin_series[idx]
                for idx in range(len(self.years))
            ]
            ebit = [ebitda[idx] - depreciation[idx] for idx in range(len(self.years))]

            interest_series = [
                interest[idx] * interest_factor
                for idx in range(len(self.years))
            ] if "senior_debt" in variable_codes else list(interest)

            ebt = [ebit[idx] - interest_series[idx] for idx in range(len(self.years))]
            effective_tax = [
                min(1.0, max(0.0, rate * tax_factor))
                for rate in tax_schedule
            ] if "tax_rate" in variable_codes else list(tax_schedule)
            taxes = [
                ebt[idx] * effective_tax[idx] if ebt[idx] > 0 else 0.0
                for idx in range(len(self.years))
            ]
            net_income = [ebt[idx] - taxes[idx] for idx in range(len(self.years))]

            weighted_adjusted: List[float] = []
            for base_weight, base_rev, sim_rev in zip(base_weighted, base_revenue, simulated_revenue):
                if abs(base_rev) > 1e-9:
                    ratio = sim_rev / base_rev
                else:
                    ratio = 1.0
                weighted_adjusted.append(base_weight * ratio)

            working_balances = self._working_capital_balances_from_series(
                simulated_revenue,
                simulated_cost_of_sales,
                distributor_weighted=weighted_adjusted,
            )

            inventory_change = _difference(working_balances.column("Inventory"))
            receivable_change = _difference(working_balances.column("Accounts Receivable"))
            payable_change = _difference(working_balances.column("Accounts Payable"))
            prepaid_change = _difference(working_balances.column("Prepaid Expenses"))
            other_asset_change = _difference(working_balances.column("Other Assets"))
            other_liability_change = _difference(working_balances.column("Other Liabilities"))

            operating_profit = [
                net_income[idx] + taxes[idx] + interest_series[idx]
                for idx in range(len(self.years))
            ]
            inventory_adjustment = [-value for value in inventory_change]
            receivable_adjustment = [-value for value in receivable_change]
            payable_adjustment = [value for value in payable_change]
            prepaid_adjustment = [-value for value in prepaid_change]
            other_asset_adjustment = [-value for value in other_asset_change]
            other_liability_adjustment = [value for value in other_liability_change]

            cash_flow_from_operations = [
                operating_profit[idx]
                + depreciation[idx]
                + inventory_adjustment[idx]
                + receivable_adjustment[idx]
                + payable_adjustment[idx]
                + prepaid_adjustment[idx]
                + other_asset_adjustment[idx]
                + other_liability_adjustment[idx]
                for idx in range(len(self.years))
            ]

            interest_paid = [-value for value in interest_series]
            taxes_paid = [-value for value in taxes]
            net_cash_from_operations = [
                cash_flow_from_operations[idx]
                + interest_paid[idx]
                + taxes_paid[idx]
                for idx in range(len(self.years))
            ]

            capital_expenditure = [-value for value in capex]
            net_cash_from_investing = list(capital_expenditure)

            dividends = [-max(ni, 0.0) * self.inputs.financing.dividend_payout for ni in net_income]
            net_cash_from_financing = [
                dividends[idx]
                + sum(series[idx] for series in other_financing.values())
                for idx in range(len(self.years))
            ]

            net_cash_flow = [
                net_cash_from_operations[idx]
                + net_cash_from_investing[idx]
                + net_cash_from_financing[idx]
                for idx in range(len(self.years))
            ]

            discounted = [
                cf / (1 + discount_rate) ** (idx + 1)
                for idx, cf in enumerate(net_cash_flow)
            ]
            npv = sum(discounted)

            averages = {
                "Average Net Income": sum(net_income) / len(net_income),
                "Average EBITDA": sum(ebitda) / len(ebitda),
                "Average Cash Flow": sum(net_cash_flow) / len(net_cash_flow),
            }

            for metric in metrics_to_track:
                if metric == "NPV":
                    results[metric].append(npv)
                else:
                    results[metric].append(averages[metric])

        return build_table(range(1, iterations + 1), results, index_name="Iteration")

    def summary_metrics(self) -> Table:
        if self._summary_metrics_cache is not None:
            return self._summary_metrics_cache

        cash_flow = self.cash_flow_statement().column(CASH_FLOW_NET_COLUMN)
        discount_rate = self.inputs.financing.discount_rate
        discounted = [cf / (1 + discount_rate) ** (idx + 1) for idx, cf in enumerate(cash_flow)]
        npv_value = sum(discounted)
        irr_result = npf_irr(cash_flow)
        irr_value = irr_result.value
        payback_years = self._payback_period(cash_flow)
        discounted_payback_years = self._payback_period(discounted)
        table = build_table(
            ["NPV", "IRR", "Payback Period", "Discounted Payback"],
            {"Value": [npv_value, irr_value, payback_years, discounted_payback_years]},
            index_name="Metric",
        )
        self._summary_metrics_cache = table
        self._irr_result = irr_result
        return table

    def irr_diagnostics(self) -> Optional["IRRResult"]:
        if self._irr_result is None:
            self.summary_metrics()
        return self._irr_result

    def ai_enhancements(
        self,
        *,
        income: Table,
        summary: Table,
        cash_flow: Table,
    ) -> Optional[AIInsights]:
        config = getattr(self.inputs, "ai", None)
        if config is None:
            return None

        revenues: Sequence[float] = []
        if "Net Revenue" in income.data:
            revenues = income.column("Net Revenue")

        ml_table: Optional[Table] = None
        advisor: Optional[MachineLearningAdvisor] = None
        if revenues and config.forecast_horizon > 0:
            advisor = MachineLearningAdvisor(config)
            ml_table = advisor.revenue_forecast(self.years, revenues)

        generative = GenerativeAdvisor(config)
        summary_text = generative.summarise(
            summary=summary,
            income=income,
            cash_flow=cash_flow,
            ml_table=ml_table,
        )

        metadata = dict(generative.metadata)
        if advisor is not None:
            metadata["ml_diagnostics"] = {
                name: dict(values) for name, values in advisor.diagnostics.items()
            }
        metadata["risk_factor_overview"] = self.risk_factor_diagnostics().as_dict()
        irr_info = self.irr_diagnostics()
        if irr_info is not None:
            metadata["irr_diagnostics"] = {
                "value": irr_info.value,
                "iterations": irr_info.iterations,
                "method": irr_info.method,
                "converged": irr_info.converged,
                "message": irr_info.message,
            }

        return AIInsights(
            ml_forecast=ml_table,
            generative_summary=summary_text,
            enabled=config.enabled,
            metadata=metadata,
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
        configured: Dict[str, BreakEvenRow] = {
            row.product: row for row in self.inputs.break_even_rows
        }

        price_lookup = self._unit_prices()
        variable_lookup = self._variable_costs()
        fixed_overrides = getattr(self.inputs, "fixed_cost_overrides", {})
        total_units_lookup = self._total_units()

        product_order: List[str] = []
        seen: Dict[str, None] = {}
        for name in list(self.products) + list(configured.keys()):
            if name not in seen:
                seen[name] = None
                product_order.append(name)

        columns: Dict[str, List[float]] = {
            "Fixed Cost": [],
            "Variable Cost per Unit": [],
            "Selling Price": [],
            "Contribution Margin": [],
            "Contribution Margin Ratio": [],
            "Target Profit": [],
            "Break-even Units": [],
            "Break-even Revenue": [],
            "Expected Volume": [],
            "Margin of Safety (Units)": [],
            "Margin of Safety (%)": [],
        }

        for product in product_order:
            params: ProductParameters | None = self.inputs.unit_costs.get(product)
            override = configured.get(product)

            selling_price = (
                override.selling_price
                if override is not None
                else price_lookup.get(product, params.selling_price if params else 0.0)
            )
            variable_cost = (
                override.variable_cost
                if override is not None
                else variable_lookup.get(product, 0.0)
            )
            if override is not None:
                fixed_cost = override.fixed_cost
            else:
                fixed_cost = float(fixed_overrides.get(product, 0.0) or 0.0)
            target_profit = override.target_profit if override is not None else 0.0
            expected_volume = (
                override.expected_volume
                if override is not None
                else total_units_lookup.get(product, 0.0)
            )

            contribution = selling_price - variable_cost
            ratio = _safe_ratio(contribution, selling_price)

            if contribution <= 0:
                break_even_units = float("nan")
                break_even_revenue = float("nan")
            else:
                break_even_units = (fixed_cost + target_profit) / contribution
                break_even_revenue = break_even_units * selling_price

            if expected_volume > 0 and break_even_units == break_even_units:
                margin_of_safety_units = expected_volume - break_even_units
                margin_of_safety_pct = _safe_ratio(margin_of_safety_units, expected_volume)
            else:
                margin_of_safety_units = float("nan")
                margin_of_safety_pct = float("nan")

            columns["Fixed Cost"].append(fixed_cost)
            columns["Variable Cost per Unit"].append(variable_cost)
            columns["Selling Price"].append(selling_price)
            columns["Contribution Margin"].append(contribution)
            columns["Contribution Margin Ratio"].append(ratio)
            columns["Target Profit"].append(target_profit)
            columns["Break-even Units"].append(break_even_units)
            columns["Break-even Revenue"].append(break_even_revenue)
            columns["Expected Volume"].append(expected_volume)
            columns["Margin of Safety (Units)"].append(margin_of_safety_units)
            columns["Margin of Safety (%)"].append(margin_of_safety_pct)

        return build_table(product_order, columns, index_name="Product")

    def _payback_period(self, cash_flows: Iterable[Number]) -> float:
        cumulative = _cumulative(cash_flows)
        for idx, value in enumerate(cumulative):
            if value >= 0:
                return float(self.years[idx])
        return float("nan")

    def payback_schedule(self) -> Table:
        cash_flows = self.cash_flow_statement().column(CASH_FLOW_NET_COLUMN)
        cumulative = _cumulative(cash_flows)
        return build_table(self.years, {"Cash Flow": cash_flows, "Cumulative": cumulative})

    def discounted_payback_schedule(self) -> Table:
        discount_rate = self.inputs.financing.discount_rate
        cash_flows = self.cash_flow_statement().column(CASH_FLOW_NET_COLUMN)
        discounted = [cf / (1 + discount_rate) ** (idx + 1) for idx, cf in enumerate(cash_flows)]
        cumulative = _cumulative(discounted)
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
        scenario_tools = self.scenario_toolkit(scenarios)
        sensitivity = self.sensitivity_analysis()
        monte_carlo = self.monte_carlo_simulation()
        ai_insights = self.ai_enhancements(
            income=income,
            summary=summary,
            cash_flow=cash_flow,
        )
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
            scenario_tool_results=scenario_tools,
            risk_factor_diagnostics=self.risk_factor_diagnostics(),
            ai_insights=ai_insights,
        )
@dataclass
class IRRResult:
    value: float
    iterations: int
    method: str
    converged: bool
    message: str = ""

    def __float__(self) -> float:
        return float(self.value)


def npf_irr(cashflows: Iterable[Number]) -> IRRResult:
    values = [float(value) for value in cashflows]
    if len(values) < 2:
        return IRRResult(
            value=float("nan"),
            iterations=0,
            method="insufficient_data",
            converged=False,
            message="At least two cash flow periods are required to compute IRR.",
        )

    def _npv(rate: float) -> float:
        total = 0.0
        for idx, value in enumerate(values):
            denominator = (1 + rate) ** idx
            if abs(denominator) < 1e-18:
                return float("inf")
            total += value / denominator
        return total

    def _derivative(rate: float) -> float:
        total = 0.0
        for idx, value in enumerate(values):
            if idx == 0:
                continue
            denominator = (1 + rate) ** (idx + 1)
            if abs(denominator) < 1e-18:
                return 0.0
            total += -idx * value / denominator
        return total

    guess = 0.1
    max_iterations = 100
    tolerance = 1e-8
    for iteration in range(1, max_iterations + 1):
        guess = max(guess, -0.999999)
        value = _npv(guess)
        derivative = _derivative(guess)
        if abs(derivative) < 1e-12:
            break
        next_guess = guess - value / derivative
        if abs(next_guess - guess) < tolerance:
            return IRRResult(
                value=next_guess,
                iterations=iteration,
                method="newton",
                converged=True,
            )
        guess = next_guess

    bracket_points = [-0.9, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    lower = None
    upper = None
    previous_rate = bracket_points[0]
    previous_value = _npv(previous_rate)
    for rate in bracket_points[1:]:
        current_value = _npv(rate)
        if previous_value == 0.0:
            return IRRResult(value=previous_rate, iterations=0, method="bracket", converged=True)
        if current_value == 0.0:
            return IRRResult(value=rate, iterations=0, method="bracket", converged=True)
        if previous_value * current_value < 0:
            lower, upper = previous_rate, rate
            break
        previous_rate, previous_value = rate, current_value

    if lower is None or upper is None:
        rate0, rate1 = 0.0, 0.2
        value0, value1 = _npv(rate0), _npv(rate1)
        for iteration in range(1, max_iterations + 1):
            denominator = value1 - value0
            if abs(denominator) < 1e-12:
                break
            rate2 = rate1 - value1 * (rate1 - rate0) / denominator
            if abs(rate2 - rate1) < tolerance:
                return IRRResult(
                    value=rate2,
                    iterations=iteration,
                    method="secant",
                    converged=True,
                )
            rate0, value0 = rate1, value1
            rate1 = rate2
            value1 = _npv(rate1)
        return IRRResult(
            value=float("nan"),
            iterations=max_iterations,
            method="secant",
            converged=False,
            message="Unable to locate a valid IRR using Newton or secant methods.",
        )

    npv_lower = _npv(lower)
    npv_upper = _npv(upper)
    midpoint = (lower + upper) / 2
    for iteration in range(1, max_iterations + 1):
        midpoint = (lower + upper) / 2
        value = _npv(midpoint)
        if abs(value) < tolerance or abs(upper - lower) < tolerance:
            return IRRResult(
                value=midpoint,
                iterations=iteration,
                method="bisection",
                converged=True,
            )
        if npv_lower * value < 0:
            upper = midpoint
            npv_upper = value
        else:
            lower = midpoint
            npv_lower = value

    return IRRResult(
        value=midpoint,
        iterations=max_iterations,
        method="bisection",
        converged=False,
        message="Bisection method did not converge within the iteration limit.",
    )


__all__ = [
    "FinancialModel",
    "FinancialOutputs",
    "ScenarioToolResult",
    "AIInsights",
    "npf_irr",
    "IRRResult",
]

