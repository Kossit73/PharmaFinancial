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


## 7) Labor model redesign — algorithm specification

### Objective
Convert labor planning from a static cost line into a role-level, shift-aware engine that produces realistic break-even behavior, stress behavior, and covenant-resilient cash flow forecasts.

### Inputs (per role `r`, year `y`)
- `role_type[r] ∈ {fixed, variable}`
- `base_headcount[r, y0]`
- `base_salary[r, y0]`
- `benefits_rate[r, y]`
- `overtime_rate[r, y]`
- `burden_rate[r, y]` (statutory + insurance + pension + bonus + training)
- `wage_escalation_direct[y]`, `wage_escalation_indirect[y]`
- `productivity_target[r, y]` (units/labor-hour)
- `utilization[y]`, `operating_hours[y]`, `shifts[y]`
- `contractor_hours[y]`, `contractor_rate[y]`
- `absenteeism[y]`, `overtime_cap[y]`, `hiring_delay_quarters[y]`

### Algorithm 1: Role-level loaded compensation
For each role and year:
1. Escalate salary with role-specific wage inflation:
   - If `r` is direct labor: `salary[r,y] = salary[r,y-1] * (1 + wage_escalation_direct[y])`
   - Else: `salary[r,y] = salary[r,y-1] * (1 + wage_escalation_indirect[y])`
2. Compute loaded annual salary:
   - `loaded_salary[r,y] = salary[r,y] * (1 + benefits_rate[r,y] + burden_rate[r,y])`
3. Compute overtime-adjusted effective salary:
   - `effective_salary[r,y] = loaded_salary[r,y] * (1 + min(overtime_rate[r,y], overtime_cap[y]))`

### Algorithm 2: Fixed vs variable headcount engine
For each role and year:
1. If `role_type[r] = fixed`:
   - `headcount[r,y] = planned_headcount[r,y]` (step changes allowed by hiring milestones).
2. If `role_type[r] = variable`:
   - Required labor hours:
     - `required_hours[r,y] = demand_units[y] / productivity_target[r,y]`
   - Shift-capacity hours:
     - `capacity_hours[r,y] = FTE_hours_per_shift * shifts[y] * utilization[y] * (1 - absenteeism[y])`
   - Raw required FTE:
     - `raw_fte[r,y] = required_hours[r,y] / capacity_hours[r,y]`
   - Apply hiring lag and step logic:
     - `headcount[r,y] = ceil_with_lag(raw_fte[r,y], hiring_delay_quarters[y], milestone_rules)`

### Algorithm 3: Shift transition cost step-up
For each year:
1. Detect shift transition:
   - If `shifts[y] > shifts[y-1]`, trigger step costs.
2. Apply discrete uplifts:
   - `step_cost[y] = transition_training_cost[y] + supervision_increment[y] + shift_allowance[y]`
3. Add to direct labor cost in transition year (no smoothing).

### Algorithm 4: Contractor/temporary labor module
For each year:
1. Calculate contractor cost:
   - `contractor_cost[y] = contractor_hours[y] * contractor_rate[y]`
2. Use contractor hours for ramp-up, shutdown, maintenance, or hiring delay coverage.
3. Apply premium multiplier vs employee hourly cost for realism.

### Algorithm 5: Total labor cost aggregation
For each year:
1. Employee labor cost:
   - `employee_cost[y] = Σ_r(headcount[r,y] * effective_salary[r,y])`
2. Add overtime, shift step-up, and contractor costs:
   - `total_labor_cost[y] = employee_cost[y] + step_cost[y] + contractor_cost[y]`
3. Split output:
   - `direct_labor_cost[y]`, `indirect_labor_cost[y]`, `fixed_labor_cost[y]`, `variable_labor_cost[y]`

### Algorithm 6: Capacity-linked KPI engine
For each year:
1. `labor_cost_per_unit[y] = total_labor_cost[y] / produced_units[y]`
2. `units_per_labor_hour[y] = produced_units[y] / total_labor_hours[y]`
3. `labor_variance[y] = labor_cost_per_unit[y] - labor_cost_per_unit[y-1]`
4. Flag deterioration if:
   - `labor_cost_per_unit` rises above threshold, or
   - `units_per_labor_hour` falls below threshold.

### Algorithm 7: Scenario/sensitivity hooks (labor-specific)
For each scenario `s`:
1. Apply shocks:
   - Wage shock: `wage_escalation += Δwage_s`
   - Absenteeism shock: `absenteeism += Δabsenteeism_s`
   - Overtime cap shock: `overtime_cap += Δotcap_s`
   - Hiring delay shock: `hiring_delay_quarters += Δdelay_s`
2. Re-run Algorithms 1–6.
3. Record impact on:
   - NPV, IRR, DSCR min/avg, payback, and liquidity headroom.

### Algorithm 8: Governance and auditability
For each labor assumption record:
1. Store metadata:
   - `source`, `owner`, `benchmark_year`, `last_updated`, `confidence_score`.
2. Run validation checks:
   - Missing metadata => warning.
   - Out-of-range salary/escalation => error.
3. Emit audit table for IC/lender pack.

### Practical v1 build order
1. Implement Algorithms 1, 2, and 5 first (core economics).
2. Add Algorithms 3 and 4 (realistic step behavior + flexibility).
3. Add Algorithms 6–8 (decision-grade analytics, stress, and governance).

## 8) Financing strategy and covenant engineering

### Why this matters
Fundability is often constrained by debt service profile, not project NPV.

### Improvements
- Debt sculpting option to target DSCR profile rather than straight-line repayment.
- Refinancing optionality scenarios (base rate shocks and spread widening).
- Interest-rate hedge module (fixed/floating mix) and hedge effectiveness impact.
- Distribution lock-up logic when covenants are breached.

## 9) Decision-grade output pack for investors

Create a one-click **Investment Committee Pack** export with:
- KPI summary: NPV, IRR, PI, payback, DSCR min/avg, equity multiple.
- Base/upside/downside scorecard with explicit gating outcomes.
- Tornado chart on the 10 highest value drivers.
- Monte Carlo percentile dashboard and breach probabilities.
- Covenant headroom schedule and liquidity runway.
- Assumption book with provenance and changes since prior version.

## 10) Governance, model risk, and reproducibility

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
