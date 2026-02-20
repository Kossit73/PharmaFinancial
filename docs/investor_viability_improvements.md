# Investor Viability Improvement Plan (Analyst Review)

This note reframes the model as an investor decision tool, not just a financial statement generator.
The improvements below are prioritized by impact on fundability, valuation confidence, and downside protection.

## Executive diagnosis from the current model

The engine already computes the right headline outputs (NPV, IRR, payback, DSCR, scenario/sensitivity/Monte Carlo), but current default assumptions and output framing make it hard to rely on the results for investment committee decisions.

## 1) Assumption integrity and calibration (highest priority)

### Why this matters
Investor metrics are only as credible as the assumptions. Placeholder financing/depreciation assumptions can mathematically produce misleading returns and covenant behavior.

### What to improve
- Replace placeholder rates and debt terms with market-calibrated values by instrument and tenor.
- Separate nominal assumptions from real assumptions and enforce consistency with inflation treatment.
- Add assumption provenance metadata per key input:
  - source (benchmark/vendor/management estimate),
  - date,
  - owner,
  - confidence level.
- Introduce range checks and plausibility checks before model run (fail-fast mode).

### Suggested controls
- Hard validation for impossible combinations (for example, discount rate less than risk-free proxy, debt rates below cash rates without rationale, or outlier margins beyond product history).
- A pre-run data quality score shown on the dashboard.

## 2) Replace score-only viability with gated investment criteria

### Why this matters
A weighted viability score is useful for ranking, but investors/lenders screen with hard constraints first.

### Add hard gates before score
- Positive base-case NPV.
- IRR above explicit hurdle (equity or project hurdle).
- Minimum annual DSCR covenant in all debt years.
- Maximum discounted payback threshold.
- No liquidity shortfall year in base and downside cases.

### Reporting logic
- **Stage 1:** Pass/Fail gate card.
- **Stage 2:** If pass, show weighted investor score for relative attractiveness.

## 3) Upgrade risk analytics from deterministic scenarios to probability-based credit view

### Why this matters
Investors and lenders price downside tails, not averages.

### Extend current scenario + Monte Carlo outputs
- Probability of NPV < 0.
- Probability IRR < hurdle.
- Probability of DSCR breach by year and cumulative probability of any breach.
- P10/P50/P90 distributions for NPV, IRR, free cash flow, and min cash balance.
- Shortfall-at-risk table for covenant headroom.

### Add correlated shock design
- Tie revenue, COGS inflation, FX, and interest rates through a transparent correlation matrix.
- Keep deterministic “managerial case” share but clearly separate from stochastic draws.

## 4) Improve valuation architecture (WACC, capital structure, and terminal value discipline)

### Why this matters
Investor confidence weakens when valuation outputs can shift materially from inconsistent discounting conventions.

### Recommended upgrades
- Explicit WACC build-up page:
  - risk-free,
  - market risk premium,
  - beta/unlever-relever logic,
  - country/size/liquidity premiums,
  - after-tax cost of debt,
  - target capital structure.
- Support APV view as a cross-check when leverage changes materially over time.
- Add terminal value sanity checks (implied exit multiple and implied perpetual growth bounds).

## 5) Working capital and cash conversion deepening

### Why this matters
Many otherwise profitable pharma projects fail due to cash conversion strain.

### Improvements
- Split receivables by channel (institutional/private/export) with different DSO assumptions.
- Inventory decomposition (raw/WIP/finished goods) with policy constraints by product form.
- Payment-term stress test (supplier tightening and customer delays).
- Cash conversion cycle waterfall with year-on-year bridge to free cash flow.

## 6) Product and operational economics granularity

### Why this matters
Top-line growth alone is not investment-grade evidence; investors want margin durability proof.

### Improvements
- Per-product contribution margin waterfalls (price, volume, mix, COGS, freight, commission).
- Capacity utilization and bottleneck modeling by line, including step-fixed costs.
- Yield/scrap and batch-failure assumptions with sensitivity to GMP disruption.
- Explicit ramp curves for new SKUs with launch risk factors.

## 7) Financing strategy and covenant engineering

### Why this matters
Fundability is often constrained by debt service profile, not project NPV.

### Improvements
- Debt sculpting option to target DSCR profile rather than straight-line repayment.
- Refinancing optionality scenarios (base rate shocks and spread widening).
- Interest-rate hedge module (fixed/floating mix) and hedge effectiveness impact.
- Distribution lock-up logic when covenants are breached.

## 8) Decision-grade output pack for investors

Create a one-click **Investment Committee Pack** export with:
- KPI summary: NPV, IRR, PI, payback, DSCR min/avg, equity multiple.
- Base/upside/downside scorecard with explicit gating outcomes.
- Tornado chart on the 10 highest value drivers.
- Monte Carlo percentile dashboard and breach probabilities.
- Covenant headroom schedule and liquidity runway.
- Assumption book with provenance and changes since prior version.

## 9) Governance, model risk, and reproducibility

### Improvements
- Version every scenario set and lock seed/configuration for reproducible Monte Carlo runs.
- Add independent model check routines (balance checks, sign logic, tax integrity, debt roll-forward integrity).
- Add audit trail output (inputs hash, model version, run timestamp, user).

## 90-day implementation roadmap

### 0-30 days (quick wins)
- Add hard viability gates and gate-first reporting.
- Calibrate financing, tax, and depreciation assumptions.
- Add assumption provenance fields.

### 31-60 days
- Add probabilistic risk outputs (NPV/IRR/DSCR breach probabilities).
- Add covenant headroom and liquidity-at-risk tables.
- Build WACC/APV cross-check module.

### 61-90 days
- Add product-level contribution bridges and capacity constraints.
- Add debt sculpting/refinancing options.
- Publish full IC Pack export with governance appendix.

## Practical success criteria

The model is investor-ready when it can answer, in one run:
1. Is the project value-creating under realistic assumptions?
2. Can debt be serviced through stress periods without covenant failures?
3. Which 3-5 levers most improve value with least execution risk?
4. What is the probability distribution of returns, not just a point estimate?
