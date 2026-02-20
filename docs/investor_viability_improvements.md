# Investor Viability Improvement Plan

This plan identifies the highest-impact actions to improve investor attractiveness in the current model implementation.

## 1) Fix input realism before optimization

The bundled assumptions use placeholder financing/depreciation values (for example `1.0` across discount rate, debt interest, and depreciation rates), which can distort return and valuation outputs. Replace placeholders with market-calibrated values and scenario-specific assumptions before using KPIs for investor decisions.

## 2) Add hard viability gates (not just a blended score)

The model already computes NPV, IRR, payback, DSCR, and profitability index. Introduce explicit pass/fail gates such as:

- NPV > 0 at base discount rate.
- IRR > hurdle rate.
- Minimum DSCR covenant in all debt years.
- Maximum discounted payback period.

Then present the blended viability score only after these gating checks pass.

## 3) Strengthen downside resilience testing

Use existing scenario and Monte Carlo tooling to report:

- Probability of NPV < 0.
- Probability of DSCR covenant breach.
- Cash runway / minimum liquidity under worst deciles.

This provides risk-aware investor evidence rather than point estimates.

## 4) Focus on value-driver levers that improve free cash flow

Prioritize assumption improvements in this order:

1. Price realization (price adjustments by product/market).
2. Gross margin expansion (production + freight + variable-cost reduction).
3. Working-capital compression (receivables/inventory days).
4. Capital intensity optimization (capex timing and asset life assumptions).
5. Debt service optimization (tenor, rates, and principal profile).

## 5) Add a “fundability pack” output for investors

Export a concise package with:

- Base/best/worst NPV + IRR + payback.
- Sensitivity tornado (price, volume, COGS, discount rate).
- Monte Carlo percentile table (P10/P50/P90 NPV).
- Covenant headroom table (DSCR and interest cover by year).
- KPI bridge showing exactly what assumptions are required to hit the target return.

## 6) Governance and model credibility

Improve investor confidence by adding:

- Assumption provenance fields (source and last-updated date).
- Validation rules for unrealistic rates and unit economics.
- Versioned scenario sets for IC memo reproducibility.
