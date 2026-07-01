# Cash Flow Mapping (IFRS Indirect Method)

This project’s cash flow statement follows the IFRS indirect method.
The sections below summarise where each line item is sourced in the
codebase so reviewers can confirm the statement is pulling the intended
figures.

## Operating activities

* `Net Income`, `Taxes`, and `Interest` are retrieved from
  `FinancialModel.income_statement()` before being recombined into
  operating profit (`ni + tax + interest`).
* Non-cash charges come from `FinancialModel.depreciation_schedule()`.
* Working-capital adjustments use the deltas from
  `FinancialModel._working_capital_balances()` for inventory, receivable,
  payable, prepaid, and other asset/liability balances.
* Dividends, interest paid, and taxes paid are deducted to arrive at
  `Net Cash Generated from Operating Activities`.

## Investing activities

* Capital expenditure is assembled in `_capex_series()`, combining the
  initial capex inputs, any annual additions, and acquisition values from
  the fixed-asset schedule. These amounts flow directly into `Net Cash
  Used in Investing Activities`.

## Financing activities

* Debt drawdown/repayment uses the period-over-period change in the
  senior debt, revolver, and overdraft outstanding schedules returned by
  `amortise_entries`.
* Share capital and initial investment inputs populate the first period
  of their respective series, ensuring IFRS-compliant disclosure of
  equity and owner funding.

## Cash rollforward

* Net cash flow equals the sum of operating, investing, and financing
  subtotals.
* Beginning cash is computed as the cumulative net cash flow shifted one
  period (with an opening balance of zero), and ending cash is the sum of
  the beginning balance and current period net change.

Refer to `FinancialModel.cash_flow_statement()` for the implementation.

