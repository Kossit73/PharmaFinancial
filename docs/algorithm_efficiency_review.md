# Algorithm Review: Improvement & Efficiency Opportunities

## Scope
This review focuses on the core financial engine and key computational paths in:

- `src/pharma_financial/model.py` (scenario, sensitivity, Monte Carlo, metrics).
- `src/pharma_financial/debt.py` (debt amortisation schedules).

## High-Impact Improvement Opportunities

### 1) Monte Carlo simulation: reduce per-iteration overhead
The Monte Carlo loop performs per-iteration work that can be hoisted or vectorized to reduce runtime, especially for large iteration counts. Relevant sections include the simulation loop, correlated shocks, and calculations in `monte_carlo_simulation`.【F:src/pharma_financial/model.py†L1618-L2050】

**Opportunities**
- **Precompute correlation decomposition once**: `_correlated_shocks` recomputes the correlation matrix and Cholesky factor on every iteration. If correlations are enabled, build the matrix and its Cholesky factor once outside the loop and reuse it. This changes loop complexity from `O(iterations * n^3)` to `O(n^3 + iterations * n^2)` for `n` correlated variables.【F:src/pharma_financial/model.py†L1693-L1750】
- **Vectorize random draws with NumPy**: The loop calls Python `random` for every year/variable. Converting the sampling to NumPy’s RNG could cut per-iteration overhead and allow vectorized draws (e.g., normal/triangular/uniform arrays) to generate full-year growth/shift arrays at once.【F:src/pharma_financial/model.py†L1656-L1740】
- **Reuse stable arrays**: Several arrays (e.g., `base_revenue_safe`, `capital_expenditure`, `other_financing`) are already precomputed; keep this pattern for any new per-iteration derived arrays to avoid repeated list-to-array conversions.【F:src/pharma_financial/model.py†L1685-L1735】
- **Avoid repeated list conversions**: `weighted_adjusted.tolist()` and repeated list creation could be replaced with arrays where downstream functions accept arrays, or pre-allocated lists to reduce allocations.【F:src/pharma_financial/model.py†L1849-L1908】

**Expected impact**: noticeable speed-up for large iteration counts or long projection horizons.

### 2) Debt amortisation: avoid repeated linear searches
`amortise_entries` and `amortise_entry` repeatedly call `years.index(year)` to map a year to its index. This is an `O(n)` lookup inside loops, which can become `O(n^2)` across entries and periods.【F:src/pharma_financial/debt.py†L13-L95】

**Opportunities**
- **Precompute `year_to_index`**: Build a dict once (e.g., `{year: idx}`) and use direct lookups in amortisation loops. This reduces the amortisation aggregation to `O(n)` per schedule rather than `O(n^2)` when many entries are present.【F:src/pharma_financial/debt.py†L53-L95】

**Expected impact**: improves performance when there are many debt entries or a long horizon.

### 3) Sensitivity analysis: reduce deep-copy overhead
`sensitivity_analysis` deep copies the full `ModelInputs` for every scenario and recomputes the model and summary. This is straightforward but expensive for large sensitivity grids.【F:src/pharma_financial/model.py†L1568-L1617】

**Opportunities**
- **Targeted cloning**: Consider a light-copy approach where only the modified portion of inputs is copied, or use a small “patch” to override specific parameters during sensitivity runs (e.g., create a `ModelInputs` from a base plus delta).【F:src/pharma_financial/model.py†L1568-L1617】
- **Reuse shared base calculations**: If the sensitivity variables do not affect all calculations, cache invariant components (e.g., static production schedule) and reuse them across runs. The model already uses caching; exposing “seed caches” for repeated scenario runs could reduce repeated work.【F:src/pharma_financial/model.py†L73-L150】

**Expected impact**: reduces overhead in scenario grids and speeds up sensitivity analyses with many cases.

## Medium-Impact Improvement Opportunities

### 4) Goal seek, summary metrics, and payback
The `summary_metrics` calculation uses straightforward list operations. When repeated across scenarios, cost is dominated by upstream computations rather than these aggregations, but minor improvements are possible for repeated use cases.【F:src/pharma_financial/model.py†L2000-L2050】

**Opportunities**
- **Reuse discount factors**: If multiple cash flow series are discounted with the same rate, a precomputed discount factor array could be reused across scenarios (e.g., in Monte Carlo and summary metrics).【F:src/pharma_financial/model.py†L2000-L2050】

### 5) Working capital adjustments in Monte Carlo
Monte Carlo uses repeated list operations for working capital adjustments (`_difference`, list negations, etc.). These could be vectorized once `working_balances` is returned as arrays or NumPy arrays are used across the flow.【F:src/pharma_financial/model.py†L1870-L1965】

**Opportunities**
- **Vectorize `_difference` and sign transforms**: Use NumPy `np.diff` with `prepend` or manual shifts to compute deltas and adjustments as arrays, reducing Python loop overhead.【F:src/pharma_financial/model.py†L1870-L1965】

## Algorithmic Correctness/Robustness Considerations

### 6) Monte Carlo distribution bounds
The simulation clamps or restricts values to a fixed `[low, high]` range. This protects against extreme values but can distort distribution tails (e.g., normal distribution truncated). Consider exposing a configuration flag for strict truncation vs. allowing draws outside bounds if a more realistic tail behavior is desired.【F:src/pharma_financial/model.py†L1625-L1738】

### 7) Correlation matrix validation
Correlation handling does not validate positive semi-definite matrices; Cholesky decomposition sets negatives to zero. Consider validating or regularizing the matrix (e.g., nearest PSD) to avoid silent distortions in correlated draws.【F:src/pharma_financial/model.py†L1693-L1749】

## Low-Impact (Nice-to-Have) Improvements

### 8) Caching and vector alignment utilities
Several helper functions (_pad_series, _inflation_array, _risk arrays) already cache vectors. For long horizon runs, consider centralizing shared arrays and exposing them as views to avoid repeated `np.array` construction in multiple workflows.【F:src/pharma_financial/model.py†L108-L155】

---

## Summary of Recommended Next Steps
1. **Refactor Monte Carlo correlation decomposition** to precompute the Cholesky factor outside the iteration loop, and reuse it for correlated draws.【F:src/pharma_financial/model.py†L1693-L1750】
2. **Switch Monte Carlo random sampling to NumPy RNG** to vectorize sampling and reduce Python-level overhead.【F:src/pharma_financial/model.py†L1656-L1750】
3. **Precompute year indices in debt amortisation** to avoid repeated `years.index` calls.【F:src/pharma_financial/debt.py†L53-L95】
4. **Investigate a lightweight sensitivity-runner** that applies parameter patches rather than full deep-copy + full recompute cycles.【F:src/pharma_financial/model.py†L1568-L1617】

These changes should improve runtime efficiency while keeping model behavior consistent.
