# Investor Viability Algorithms (v1)

## Algorithm A1: Assumption Integrity Scoring
1. Validate financing ranges (discount/debt/cash rates).
2. Penalize structurally inconsistent combinations (e.g., senior debt rate below cash rate).
3. Detect placeholder assumptions (e.g., repeated `1.0` values) across financing/depreciation.
4. Score labor metadata completeness when role-level labor model is used.
5. Return bounded data-quality score in `[0, 100]`.

## Algorithm A2: Hard Investment Gates
1. Compute gate thresholds from viability config and financing hurdle.
2. Evaluate base-case pass/fail conditions:
   - NPV > 0
   - IRR >= hurdle
   - DSCR >= covenant minimum
   - discounted payback <= max threshold
   - no negative ending cash period
3. Return gate pass count, pass ratio, and binary overall status.

## Algorithm A3: Probabilistic Risk Metrics from Monte Carlo
1. Read Monte Carlo output distribution (NPV/IRR).
2. Compute downside probabilities:
   - `P(NPV < 0)`
   - `P(IRR < hurdle)`
3. Compute percentile bands:
   - NPV P10/P50/P90
   - IRR P10/P50/P90
4. Expose metrics for dashboard/reporting.

## Algorithm A4: Role-Level Labor Engine (v1)
1. Parse role and settings schedules from `labor.model_v1`.
2. Compute variable-role headcount from demand, productivity, shifts, utilization, absenteeism.
3. Apply hiring-lag adjustment and step behavior.
4. Compute loaded compensation (salary + benefits + burden + overtime cap).
5. Add shift-transition step costs and contractor costs.
6. Aggregate direct/indirect labor into financial statements.

## Algorithm A5: Labor KPI Derivation
1. Compute total labor cost and total labor hours by year.
2. Derive:
   - labor cost per unit
   - units per labor hour
   - fixed labor share
3. Surface multi-year averages in summary metrics.

## Algorithm A6: Labor Scenario Hooks
1. Accept sensitivity variables:
   - wage_direct, wage_indirect, absenteeism, overtime_cap, hiring_delay
2. Apply perturbation schedules to labor model inputs.
3. Re-run model and collect NPV/IRR impact.
4. Apply equivalent Monte Carlo variable hooks in stochastic runs.
