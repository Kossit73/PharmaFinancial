"""Core microbrewery financial model engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional

import numpy as np
import pandas as pd

RepaymentType = Literal["linear", "annuity", "interest_only_then_linear", "specified"]


def annual_to_monthly_rate(annual_rate: float) -> float:
    """Convert an effective annual rate to an effective monthly rate."""

    return (1.0 + float(annual_rate)) ** (1.0 / 12.0) - 1.0


def safe_div(n: float, d: float, default: float = 0.0) -> float:
    """Return a safe division result with a configurable default."""

    return float(n) / float(d) if float(d) != 0.0 else float(default)


def irr(cashflows: Iterable[float], guess: float = 0.1) -> float:
    """Internal rate of return (per period) via Newton method."""

    cfs = np.array(list(cashflows), dtype=float)
    if np.allclose(cfs, 0.0):
        return np.nan

    def f(rate: float) -> float:
        return np.sum(cfs / ((1.0 + rate) ** np.arange(len(cfs))))

    def fprime(rate: float) -> float:
        t = np.arange(len(cfs))
        return np.sum(-t * cfs / ((1.0 + rate) ** (t + 1.0)))

    r = float(guess)
    for _ in range(200):
        fr = f(r)
        fpr = fprime(r)
        if abs(fpr) < 1e-12:
            break
        new_r = r - fr / fpr
        if abs(new_r - r) < 1e-10:
            return new_r
        r = new_r
    return np.nan


@dataclass(frozen=True)
class ModelConfig:
    start_date: str = "2025-01-01"
    months: int = 120
    pricing_cost_basis_month: int = 24
    price_inflation_annual: float = 0.015
    cost_inflation_annual: float = 0.015
    tax_rate: float = 0.25
    days_receivables: float = 20.0
    days_inventory: float = 15.0
    days_payables: float = 30.0
    other_current_assets_pct_revenue: float = 0.05
    other_current_liabilities_pct_direct_costs: float = 0.05
    wacc_annual: float = 0.122
    exit_month: Optional[int] = None
    exit_ev_ebitda_multiple: float = 8.0
    initial_cash: float = 0.0


@dataclass(frozen=True)
class DividendPolicy:
    enabled: bool = True
    model: Literal["cash_sweep", "share_of_profits"] = "cash_sweep"
    start_month: int = 60
    minimum_cash_position: float = 1_500_000.0
    payout_ratio: float = 0.25


@dataclass(frozen=True)
class DebtFacility:
    name: str
    principal: float
    annual_interest_rate: float
    draw_month: int = 0
    grace_months: int = 0
    term_months: int = 60
    repayment_type: RepaymentType = "linear"
    specified_principal_payments: Optional[Dict[int, float]] = None


@dataclass(frozen=True)
class CapexItem:
    name: str
    amount: float
    capex_month: int = 0
    depreciation_years: float = 0.0


@dataclass
class ModelInputs:
    skus: pd.DataFrame
    channels: pd.DataFrame
    sales_plan: pd.DataFrame
    opex_fixed_monthly: float | pd.Series = 0.0
    other_income_monthly: float | pd.Series = 0.0
    capex_items: Optional[List[CapexItem]] = None
    debt_facilities: Optional[List[DebtFacility]] = None
    equity_injections: Optional[Dict[int, float]] = None


@dataclass
class ModelRunResult:
    monthly: pd.DataFrame
    annual: pd.DataFrame
    prices: pd.DataFrame
    debt_schedules: Dict[str, pd.DataFrame]
    valuation: Dict[str, float]


class MicrobreweryFinancialModel:
    def __init__(
        self,
        config: ModelConfig,
        dividend_policy: DividendPolicy,
        inputs: ModelInputs,
    ) -> None:
        self.cfg = config
        self.div = dividend_policy
        self.inputs = inputs

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.inputs.sales_plan.empty:
            raise ValueError("Sales plan is required.")
        for col in ["sku_id", "name", "direct_cost_per_unit", "markup_pct"]:
            if col not in self.inputs.skus.columns:
                raise ValueError(f"SKU column missing: {col}")
        if "channel" not in self.inputs.channels.columns or "price_factor" not in self.inputs.channels.columns:
            raise ValueError("Channels must include 'channel' and 'price_factor'.")

    def _timeline(self) -> pd.DatetimeIndex:
        return pd.date_range(self.cfg.start_date, periods=self.cfg.months, freq="MS")

    def _inflation_index(self, annual_rate: float, idx: pd.DatetimeIndex) -> pd.Series:
        r_m = annual_to_monthly_rate(annual_rate)
        return pd.Series((1.0 + r_m) ** np.arange(len(idx)), index=idx, name="inflation_index")

    def _as_monthly_series(self, value: float | pd.Series, idx: pd.DatetimeIndex, name: str) -> pd.Series:
        if isinstance(value, pd.Series):
            series = value.reindex(idx).ffill().bfill()
            series.name = name
            return series
        series = pd.Series(float(value), index=idx, name=name)
        return series

    def _prices_matrix(self, idx: pd.DatetimeIndex, units: pd.DataFrame) -> pd.DataFrame:
        skus = self.inputs.skus.set_index("sku_id")
        base_month = int(np.clip(self.cfg.pricing_cost_basis_month, 0, len(idx) - 1))
        cost_idx = self._inflation_index(self.cfg.cost_inflation_annual, idx)
        price_idx = self._inflation_index(self.cfg.price_inflation_annual, idx)

        base_costs = skus["direct_cost_per_unit"] * cost_idx.iloc[base_month]
        markup_prices = base_costs * (1.0 + skus["markup_pct"])

        price_cols = []
        price_data = []
        for sku_id in skus.index:
            for channel, factor in self.inputs.channels.set_index("channel")["price_factor"].items():
                price_cols.append((sku_id, channel))
                base_price = float(markup_prices.loc[sku_id]) * float(factor)
                price_data.append(np.full(len(idx), base_price, dtype=float))

        prices = pd.DataFrame(
            np.array(price_data).T,
            index=idx,
            columns=pd.MultiIndex.from_tuples(price_cols, names=["sku_id", "channel"]),
        ).sort_index(axis=1)

        prices = prices.mul(price_idx, axis=0)

        if not units.empty:
            prices = prices.reindex(columns=units.columns, fill_value=0.0)

        return prices

    def _units_matrix(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        sales = self.inputs.sales_plan.copy()
        sales["date"] = pd.to_datetime(sales["date"])
        sales = sales[sales["date"].isin(idx)]

        if sales.empty:
            return pd.DataFrame(index=idx)

        sales = sales.groupby(["date", "sku_id", "channel"])["units"].sum().reset_index()
        units_wide = sales.pivot(index="date", columns=["sku_id", "channel"], values="units").fillna(0.0)
        units_wide = units_wide.reindex(idx, fill_value=0.0)
        return units_wide

    def _capex_series(self, idx: pd.DatetimeIndex) -> pd.Series:
        capex = pd.Series(0.0, index=idx, name="capex")
        for item in self.inputs.capex_items or []:
            m = int(np.clip(item.capex_month, 0, len(idx) - 1))
            capex.iloc[m] += float(item.amount)
        return capex

    def _depreciation_series(self, idx: pd.DatetimeIndex) -> pd.Series:
        dep = pd.Series(0.0, index=idx, name="depreciation")
        for item in self.inputs.capex_items or []:
            years = max(float(item.depreciation_years), 0.0)
            if years <= 0:
                continue
            months = int(np.ceil(years * 12.0))
            start = int(np.clip(item.capex_month, 0, len(idx) - 1))
            end = min(start + months, len(idx))
            monthly_dep = float(item.amount) / months
            dep.iloc[start:end] += monthly_dep
        return dep

    def _net_fixed_assets(self, capex: pd.Series, dep: pd.Series) -> pd.Series:
        net = (capex - dep).cumsum()
        return net.rename("net_fixed_assets")

    def _nwc(self, idx: pd.DatetimeIndex, *, revenue: pd.Series, direct_costs: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        def _monthly_working_cap(baseline: pd.Series, days: float) -> pd.Series:
            return baseline * float(days) / 365.0

        receivables = _monthly_working_cap(revenue, self.cfg.days_receivables)
        inventory = _monthly_working_cap(direct_costs, self.cfg.days_inventory)
        payables = _monthly_working_cap(direct_costs, self.cfg.days_payables)
        other_assets = revenue * float(self.cfg.other_current_assets_pct_revenue)
        other_liabilities = direct_costs * float(self.cfg.other_current_liabilities_pct_direct_costs)

        nwc = pd.DataFrame(
            {
                "accounts_receivable": receivables,
                "inventory": inventory,
                "accounts_payable": payables,
                "other_assets": other_assets,
                "other_liabilities": other_liabilities,
            },
            index=idx,
        )
        nwc["net_working_capital"] = nwc["accounts_receivable"] + nwc["inventory"] + nwc["other_assets"] - nwc[
            "accounts_payable"
        ] - nwc["other_liabilities"]

        change_nwc = nwc["net_working_capital"].diff().fillna(nwc["net_working_capital"]).rename("change_in_nwc")
        return nwc, change_nwc

    def _debt_schedule_one(self, idx: pd.DatetimeIndex, fac: DebtFacility) -> pd.DataFrame:
        r_m = annual_to_monthly_rate(fac.annual_interest_rate)

        beg = np.zeros(len(idx))
        draw = np.zeros(len(idx))
        interest = np.zeros(len(idx))
        principal = np.zeros(len(idx))
        end = np.zeros(len(idx))

        draw_month = int(np.clip(fac.draw_month, 0, len(idx) - 1))
        beg[draw_month] = float(fac.principal)
        draw[draw_month] = float(fac.principal)

        repay_start = int(np.clip(draw_month + fac.grace_months, 0, len(idx)))
        repay_end = int(np.clip(repay_start + fac.term_months, 0, len(idx)))

        for t in range(len(idx)):
            if t > 0:
                beg[t] = end[t - 1]

            interest[t] = beg[t] * r_m

            if fac.repayment_type == "specified":
                p = 0.0
                if fac.specified_principal_payments and t in fac.specified_principal_payments:
                    p = float(fac.specified_principal_payments[t])
                principal[t] = min(p, beg[t])

            elif t < repay_start or t >= repay_end:
                principal[t] = 0.0

            else:
                outstanding = beg[t]

                if fac.repayment_type == "linear":
                    amort_months = max(repay_end - repay_start, 1)
                    pmt_principal = float(fac.principal) / amort_months
                    principal[t] = min(pmt_principal, outstanding)

                elif fac.repayment_type == "annuity":
                    amort_months = max(repay_end - repay_start, 1)
                    principal_amount = float(fac.principal)
                    if abs(r_m) < 1e-12:
                        total_pmt = principal_amount / amort_months
                    else:
                        total_pmt = principal_amount * r_m / (1.0 - (1.0 + r_m) ** (-amort_months))
                    principal[t] = min(max(total_pmt - interest[t], 0.0), outstanding)

                elif fac.repayment_type == "interest_only_then_linear":
                    amort_months = max(repay_end - repay_start, 1)
                    pmt_principal = float(fac.principal) / amort_months
                    principal[t] = min(pmt_principal, outstanding)

                else:
                    principal[t] = 0.0

            end[t] = max(beg[t] - principal[t], 0.0)

        return pd.DataFrame(
            {
                "beginning_balance": beg,
                "draw": draw,
                "interest": interest,
                "principal_payment": principal,
                "ending_balance": end,
            },
            index=idx,
        )

    def _debt_schedules(self, idx: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        schedules: Dict[str, pd.DataFrame] = {}
        for fac in self.inputs.debt_facilities or []:
            schedules[fac.name] = self._debt_schedule_one(idx, fac)
        return schedules

    def run(self) -> ModelRunResult:
        idx = self._timeline()

        cost_idx = self._inflation_index(self.cfg.cost_inflation_annual, idx)

        units_wide = self._units_matrix(idx)
        prices_wide = self._prices_matrix(idx, units_wide)

        revenue_wide = units_wide.mul(prices_wide, fill_value=0.0)
        revenue = revenue_wide.sum(axis=1).rename("revenue")

        other_income = self._as_monthly_series(self.inputs.other_income_monthly, idx, "other_income")
        other_income = (other_income * cost_idx).rename("other_income")
        total_revenue = (revenue + other_income).rename("total_revenue")

        skus = self.inputs.skus.set_index("sku_id")
        if units_wide.empty:
            direct_costs = pd.Series(0.0, index=idx, name="direct_costs")
        else:
            cost_cols = []
            cost_data = []
            for sku_id, row in skus.iterrows():
                for channel in self.inputs.channels["channel"].tolist():
                    cost_cols.append((sku_id, channel))
                    series = float(row["direct_cost_per_unit"]) * cost_idx.values
                    cost_data.append(series)
            costs_wide = pd.DataFrame(
                np.array(cost_data).T,
                index=idx,
                columns=pd.MultiIndex.from_tuples(cost_cols, names=["sku_id", "channel"]),
            ).sort_index(axis=1)

            direct_costs_wide = units_wide.mul(costs_wide, fill_value=0.0)
            direct_costs = direct_costs_wide.sum(axis=1).rename("direct_costs")

        gross_profit = (revenue - direct_costs).rename("gross_profit")

        opex_fixed = self._as_monthly_series(self.inputs.opex_fixed_monthly, idx, "opex_fixed")
        opex = (opex_fixed * cost_idx).rename("opex")

        ebitda = (total_revenue - direct_costs - opex).rename("ebitda")

        capex = self._capex_series(idx)
        dep = self._depreciation_series(idx)
        net_fixed_assets = self._net_fixed_assets(capex, dep)

        ebit = (ebitda - dep).rename("ebit")

        nwc_comp, change_nwc = self._nwc(idx, revenue=revenue, direct_costs=direct_costs)

        debt_schedules = self._debt_schedules(idx)
        if debt_schedules:
            debt_interest = sum(df["interest"] for df in debt_schedules.values()).rename("interest_expense")
            debt_draw = sum(df["draw"] for df in debt_schedules.values()).rename("debt_draw")
            debt_principal = sum(df["principal_payment"] for df in debt_schedules.values()).rename("debt_principal_payment")
            debt_balance = sum(df["ending_balance"] for df in debt_schedules.values()).rename("debt_ending_balance")
        else:
            debt_interest = pd.Series(0.0, index=idx, name="interest_expense")
            debt_draw = pd.Series(0.0, index=idx, name="debt_draw")
            debt_principal = pd.Series(0.0, index=idx, name="debt_principal_payment")
            debt_balance = pd.Series(0.0, index=idx, name="debt_ending_balance")

        ebt = (ebit - debt_interest).rename("ebt")
        taxes = (ebt.clip(lower=0.0) * float(self.cfg.tax_rate)).rename("taxes")
        net_income = (ebt - taxes).rename("net_income")

        equity_inj = pd.Series(0.0, index=idx, name="equity_injection")
        for month, amount in (self.inputs.equity_injections or {}).items():
            month_idx = int(np.clip(int(month), 0, len(idx) - 1))
            equity_inj.iloc[month_idx] += float(amount)

        cfo = (net_income + dep - change_nwc).rename("cash_flow_from_operations")
        cfi = (-capex).rename("cash_flow_from_investing")
        cff_pre_div = (equity_inj + debt_draw - debt_principal).rename("cash_flow_from_financing_pre_div")

        dividends = pd.Series(0.0, index=idx, name="dividends")
        cash = pd.Series(0.0, index=idx, name="cash")
        cash_prev = float(self.cfg.initial_cash)

        for t, _date in enumerate(idx):
            cash_pre = cash_prev + float(cfo.iloc[t] + cfi.iloc[t] + cff_pre_div.iloc[t])

            div_t = 0.0
            if self.div.enabled and t >= int(self.div.start_month):
                if self.div.model == "cash_sweep":
                    div_t = max(cash_pre - float(self.div.minimum_cash_position), 0.0)
                else:
                    div_t = float(self.div.payout_ratio) * max(float(net_income.iloc[t]), 0.0)

            cash_end = cash_pre - div_t
            dividends.iloc[t] = div_t
            cash.iloc[t] = cash_end
            cash_prev = cash_end

        nopat = (ebit * (1.0 - float(self.cfg.tax_rate))).rename("nopat")
        fcff = (nopat + dep - capex - change_nwc).rename("fcff")

        exit_m = self.cfg.exit_month if self.cfg.exit_month is not None else (len(idx) - 1)
        exit_m = int(np.clip(exit_m, 0, len(idx) - 1))
        terminal_value = max(float(ebitda.iloc[exit_m]), 0.0) * float(self.cfg.exit_ev_ebitda_multiple)

        wacc_m = annual_to_monthly_rate(self.cfg.wacc_annual)
        discount = (1.0 + wacc_m) ** np.arange(len(idx))
        enterprise_value = float((fcff.values / discount).sum() + terminal_value / discount[exit_m])

        debt_exit = float(debt_balance.iloc[exit_m])
        cash_exit = float(cash.iloc[exit_m])
        equity_value_exit = enterprise_value - debt_exit + cash_exit

        equity_cashflows = (-equity_inj).copy()
        equity_cashflows += dividends
        equity_cashflows.iloc[exit_m] += float(equity_value_exit)

        irr_m = irr(equity_cashflows.values, guess=0.02)
        irr_annual = (1.0 + irr_m) ** 12 - 1.0 if np.isfinite(irr_m) else np.nan

        invested = float(equity_inj.sum())
        returned = float(dividends.sum() + max(equity_value_exit, 0.0))
        moic = safe_div(returned, invested, default=np.nan)

        valuation = {
            "wacc_annual": float(self.cfg.wacc_annual),
            "wacc_monthly": float(wacc_m),
            "exit_month_index": float(exit_m),
            "terminal_value": float(terminal_value),
            "enterprise_value_dcf": float(enterprise_value),
            "equity_value_exit": float(equity_value_exit),
            "equity_irr_monthly": float(irr_m) if np.isfinite(irr_m) else np.nan,
            "equity_irr_annual": float(irr_annual) if np.isfinite(irr_annual) else np.nan,
            "equity_moic": float(moic) if np.isfinite(moic) else np.nan,
        }

        monthly = pd.DataFrame(
            {
                "revenue": revenue,
                "other_income": other_income,
                "total_revenue": total_revenue,
                "direct_costs": direct_costs,
                "gross_profit": gross_profit,
                "opex": opex,
                "ebitda": ebitda,
                "depreciation": dep,
                "ebit": ebit,
                "interest_expense": debt_interest,
                "ebt": ebt,
                "taxes": taxes,
                "net_income": net_income,
                "capex": capex,
                "change_in_nwc": change_nwc,
                "debt_draw": debt_draw,
                "debt_principal_payment": debt_principal,
                "equity_injection": equity_inj,
                "dividends": dividends,
                "cash": cash,
                "debt_ending_balance": debt_balance,
                "net_fixed_assets": net_fixed_assets,
                "fcff": fcff,
            },
            index=idx,
        ).join(nwc_comp)

        annual = monthly.resample("YE").sum(numeric_only=True).rename_axis("year_end")

        return ModelRunResult(
            monthly=monthly,
            annual=annual,
            prices=prices_wide,
            debt_schedules=debt_schedules,
            valuation=valuation,
        )


def phase_growth_series(
    idx: pd.DatetimeIndex,
    start_month: int,
    start_units: float,
    monthly_growth: float,
    stop_month: Optional[int] = None,
    cap_units: Optional[float] = None,
) -> pd.Series:
    """Return a monthly series that grows from start_month with an optional cap."""

    n = len(idx)
    series = np.zeros(n, dtype=float)
    stop = stop_month if stop_month is not None else n
    stop = int(np.clip(stop, 0, n))

    units = float(start_units)
    for t in range(int(start_month), stop):
        series[t] = units
        units = units * (1.0 + float(monthly_growth))
        if cap_units is not None:
            units = min(units, float(cap_units))
    return pd.Series(series, index=idx)
