"""Financing calculations, cash flows, and financial statements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .assumptions import Assumptions
from .production import AnnualSummary


@dataclass
class CashFlowRow:
    year: int
    revenue: float
    variable_costs: float
    fixed_costs: float
    operating_expense: float
    ebitda: float
    depreciation: float
    interest_expense: float
    taxes: float
    net_income: float
    operating_cash_flow: float
    maintenance_capex: float
    debt_service: float
    principal_payment: float
    free_cash_flow: float
    discount_factor: float
    present_value: float
    ending_debt: float
    cumulative_cash: float
    calendar_year: Optional[int] = None


@dataclass
class IncomeStatementRow:
    year: int
    revenue: float
    cogs: float
    gross_profit: float
    operating_expenses: float
    ebitda: float
    depreciation: float
    ebit: float
    interest: float
    taxes: float
    net_income: float
    ebitda_margin: float
    net_margin: float
    calendar_year: Optional[int] = None


@dataclass
class BalanceSheetRow:
    year: int
    cash: float
    working_capital: float
    net_ppe: float
    total_assets: float
    debt: float
    equity: float
    retained_earnings: float
    debt_to_equity: float | None
    calendar_year: Optional[int] = None


@dataclass
class CashFlowStatementRow:
    year: int
    operating_cash_flow: float
    investing_cash_flow: float
    financing_cash_flow: float
    net_change_in_cash: float
    ending_cash: float
    calendar_year: Optional[int] = None


def _pmt(rate: float, term_years: int, principal: float) -> float:
    if rate == 0:
        return principal / term_years
    factor = (1 + rate) ** term_years
    return principal * rate * factor / (factor - 1)


def amortization_schedule(
    principal: float, rate: float, term_years: int
) -> List[Dict[str, float]]:
    if principal <= 0 or term_years <= 0:
        return []
    rate = float(rate)
    payment = _pmt(rate, term_years, principal)
    schedule = []
    balance = principal
    for year in range(1, term_years + 1):
        interest = balance * rate
        principal_paid = payment - interest
        balance = max(0.0, balance - principal_paid)
        schedule.append(
            {
                "year": year,
                "payment": payment,
                "interest": interest,
                "principal": principal_paid,
                "balance": balance,
            }
        )
    return schedule


def discounted_cash_flow(
    assumptions: Assumptions, base_annual: AnnualSummary
) -> Tuple[List[CashFlowRow], List[Dict[str, float]]]:
    total_capex = assumptions.capex_housing + assumptions.capex_equipment
    equity = total_capex * (1 - assumptions.debt_ratio)
    debt = total_capex * assumptions.debt_ratio
    loan_schedule = amortization_schedule(
        debt, assumptions.debt_interest_rate, assumptions.debt_term_years
    )

    rows: List[CashFlowRow] = []
    depreciation = base_annual.depreciation
    revenue = base_annual.revenue
    upfront_cash = -(equity + assumptions.working_capital)
    cumulative_cash = upfront_cash
    start_year = int(assumptions.production_start_year) if assumptions.production_start_year else 0
    rows.append(
        CashFlowRow(
            year=0,
            revenue=0.0,
            variable_costs=0.0,
            fixed_costs=0.0,
            operating_expense=0.0,
            ebitda=0.0,
            depreciation=0.0,
            interest_expense=0.0,
            taxes=0.0,
            net_income=0.0,
            operating_cash_flow=upfront_cash,
            maintenance_capex=0.0,
            debt_service=0.0,
            principal_payment=0.0,
            free_cash_flow=upfront_cash,
            discount_factor=1.0,
            present_value=upfront_cash,
            ending_debt=debt,
            cumulative_cash=cumulative_cash,
            calendar_year=start_year - 1 if start_year else None,
        )
    )

    projection_years = int(assumptions.production_horizon_years)
    if projection_years <= 0:
        projection_years = 1

    for year in range(1, projection_years + 1):
        revenue *= 1 + assumptions.price_growth
        variable_costs = (
            base_annual.feed_cost
            + base_annual.chick_cost
            + base_annual.processing_cost
            + base_annual.health_cost
        ) * ((1 + assumptions.cost_inflation) ** year)
        fixed_costs = (
            base_annual.energy_cost
            + base_annual.labor_cost
            + base_annual.overhead_cost
        ) * ((1 + assumptions.cost_inflation) ** year)

        ebitda = revenue - variable_costs - fixed_costs
        interest_expense = 0.0
        debt_service = 0.0
        principal_payment = 0.0
        ending_balance = 0.0
        if year <= len(loan_schedule):
            sched = loan_schedule[year - 1]
            interest_expense = sched["interest"]
            debt_service = sched["payment"]
            principal_payment = sched["principal"]
            ending_balance = sched["balance"]

        ebit = ebitda - depreciation
        taxable_income = max(0.0, ebit - interest_expense)
        taxes = taxable_income * assumptions.tax_rate
        net_income = ebit - interest_expense - taxes
        operating_cash_flow = ebitda - taxes
        free_cash_flow = (
            operating_cash_flow
            - assumptions.maintenance_capex_annual
            - debt_service
        )
        discount_factor = (1 + assumptions.discount_rate) ** year
        present_value = free_cash_flow / discount_factor
        cumulative_cash += free_cash_flow

        rows.append(
            CashFlowRow(
                year=year,
                revenue=revenue,
                variable_costs=variable_costs,
                fixed_costs=fixed_costs,
                operating_expense=variable_costs + fixed_costs,
                ebitda=ebitda,
                depreciation=depreciation,
                interest_expense=interest_expense,
                taxes=taxes,
                net_income=net_income,
                operating_cash_flow=operating_cash_flow,
                maintenance_capex=assumptions.maintenance_capex_annual,
                debt_service=debt_service,
                principal_payment=principal_payment,
                free_cash_flow=free_cash_flow,
                discount_factor=discount_factor,
                present_value=present_value,
                ending_debt=ending_balance,
                cumulative_cash=cumulative_cash,
                calendar_year=start_year + year - 1 if start_year else None,
            )
        )

    if start_year:
        for entry in loan_schedule:
            year_value = int(entry.get("year", 0))
            entry["calendar_year"] = start_year + year_value - 1 if year_value else start_year

    return rows, loan_schedule


def build_financial_statements(
    assumptions: Assumptions,
    cashflows: List[CashFlowRow],
    loan_schedule: List[Dict[str, float]],
) -> Dict[str, List[Any]]:
    total_capex = assumptions.capex_housing + assumptions.capex_equipment
    depreciation = (
        assumptions.capex_housing + assumptions.capex_equipment
    ) / assumptions.depreciation_years
    equity_base = total_capex * (1 - assumptions.debt_ratio)
    equity_total = equity_base + assumptions.working_capital

    income_rows: List[IncomeStatementRow] = []
    cash_statement: List[CashFlowStatementRow] = []
    balance_rows: List[BalanceSheetRow] = []

    investing_cash = -(total_capex + assumptions.working_capital)
    financing_cash = equity_total + (total_capex * assumptions.debt_ratio)
    net_change = investing_cash + financing_cash
    cash_balance = net_change
    start_year = int(assumptions.production_start_year) if assumptions.production_start_year else 0

    cash_statement.append(
        CashFlowStatementRow(
            year=0,
            operating_cash_flow=0.0,
            investing_cash_flow=investing_cash,
            financing_cash_flow=financing_cash,
            net_change_in_cash=net_change,
            ending_cash=cash_balance,
            calendar_year=start_year - 1 if start_year else None,
        )
    )

    for row in cashflows:
        if row.year == 0:
            balance_rows.append(
                BalanceSheetRow(
                    year=0,
                    cash=cash_balance,
                    working_capital=assumptions.working_capital,
                    net_ppe=total_capex,
                    total_assets=total_capex
                    + assumptions.working_capital
                    + cash_balance,
                    debt=total_capex * assumptions.debt_ratio,
                    equity=equity_total + cash_balance,
                    retained_earnings=0.0,
                    debt_to_equity=(
                        (total_capex * assumptions.debt_ratio) / equity_total
                        if equity_total
                        else None
                    ),
                    calendar_year=start_year - 1 if start_year else None,
                )
            )
            continue

        gross_profit = row.revenue - row.variable_costs
        ebit = row.ebitda - row.depreciation
        income_rows.append(
            IncomeStatementRow(
                year=row.year,
                revenue=row.revenue,
                cogs=row.variable_costs,
                gross_profit=gross_profit,
                operating_expenses=row.fixed_costs,
                ebitda=row.ebitda,
                depreciation=row.depreciation,
                ebit=ebit,
                interest=row.interest_expense,
                taxes=row.taxes,
                net_income=row.net_income,
                ebitda_margin=(row.ebitda / row.revenue) if row.revenue else 0.0,
                net_margin=(row.net_income / row.revenue) if row.revenue else 0.0,
                calendar_year=row.calendar_year,
            )
        )

        operating_cash = row.net_income + row.depreciation
        investing_cash = -assumptions.maintenance_capex_annual
        financing_cash = -row.principal_payment
        net_change = operating_cash + investing_cash + financing_cash
        cash_balance += net_change
        cash_statement.append(
            CashFlowStatementRow(
                year=row.year,
                operating_cash_flow=operating_cash,
                investing_cash_flow=investing_cash,
                financing_cash_flow=financing_cash,
                net_change_in_cash=net_change,
                ending_cash=cash_balance,
                calendar_year=row.calendar_year,
            )
        )

        accum_dep = min(row.year, assumptions.depreciation_years) * depreciation
        net_ppe = max(0.0, total_capex - accum_dep)
        debt_balance = row.ending_debt if row.ending_debt else 0.0
        total_assets = cash_balance + assumptions.working_capital + net_ppe
        equity = total_assets - debt_balance
        retained = equity - equity_total
        debt_to_equity = (debt_balance / equity) if equity else None
        balance_rows.append(
            BalanceSheetRow(
                year=row.year,
                cash=cash_balance,
                working_capital=assumptions.working_capital,
                net_ppe=net_ppe,
                total_assets=total_assets,
                debt=debt_balance,
                equity=equity,
                retained_earnings=retained,
                debt_to_equity=debt_to_equity,
                calendar_year=row.calendar_year,
            )
        )

    return {
        "income_statement": income_rows,
        "balance_sheet": balance_rows,
        "cash_flow_statement": cash_statement,
        "loan_schedule": loan_schedule,
    }


def npv(rate: float, cashflows: Iterable[float]) -> float:
    total = 0.0
    for year, cash in enumerate(cashflows):
        total += cash / ((1 + rate) ** year)
    return total


def irr(cashflows: Iterable[float], guess: float = 0.1) -> float:
    cashflows = list(cashflows)
    if not cashflows:
        return float("nan")

    def npv_at(rate: float) -> float:
        return sum(cf / ((1 + rate) ** idx) for idx, cf in enumerate(cashflows))

    lower, upper = -0.99, guess if guess > -0.99 else 0.1
    f_lower = npv_at(lower)
    f_upper = npv_at(upper)

    while f_lower * f_upper > 0 and upper < 10:
        upper += 0.5
        f_upper = npv_at(upper)

    if f_lower * f_upper > 0:
        return float("nan")

    for _ in range(200):
        mid = (lower + upper) / 2
        f_mid = npv_at(mid)
        if abs(f_mid) < 1e-7:
            return mid
        if f_lower * f_mid < 0:
            upper = mid
            f_upper = f_mid
        else:
            lower = mid
            f_lower = f_mid
    return mid
