# Financial Models API Contract

This file captures the HTTP contract exposed by `financial_models.api.server` so frontend developers can integrate without reading the Python code. The FastAPI app also serves an OpenAPI document at `/openapi.json` that can be fed to client generators (e.g. `npx @openapitools/openapi-generator-cli generate -g typescript-angular ...`).

## Base URL

- Local development: `http://localhost:8000`
- Production: set to your deployed FastAPI host (HTTPS recommended).

All endpoints accept and return JSON.

## Authentication

- API token: set `FINANCIAL_MODELS_API_TOKEN` on the server. Clients send `X-API-Key: <token>`. When unset, the API is open (not recommended outside local dev).
- Google ID token: set `FINANCIAL_MODELS_GOOGLE_AUDIENCE` to the allowed OAuth client ID(s). Clients send `Authorization: Bearer <id-token>`. Both API token and Google auth can be enabled; either will authorize. `/health` remains public.

## Common Schemas

- `TablePayload`
  - `index_name`: string (e.g. `"Year"` or `"Metric"`)
  - `index`: array of labels
  - `data`: object of column name → array of values (NaN/Infinity are emitted as `null`)
- `ScenarioToolResultPayload`
  - `rows`: array of objects (tabular rows)
  - `interpretation`: string summary
- `AIInsightsPayload` (optional)
  - `enabled`: boolean
  - `generative_summary`: string | null
  - `metadata`: object | null
  - `ml_forecast`: `TablePayload` | null

## Endpoints

### `GET /health`
- Auth: none
- 200 response: `{"status": "ok"}`

### `POST /model/pharma/run`
- Auth: required unless auth is disabled
- Model types: `pharma`, `microbrewery`, `biotech`, `goat_farming`, `cassava_ethanol`, and `broiler_chicken`.
- Default inputs:
  - `src/financial_models/pharma/data/default_inputs.json`
  - `src/financial_models/biotech/data/default_inputs.json`
  - `src/financial_models/microbrewery/data/default_inputs.json`
  - `src/financial_models/goat_farming/data/default_inputs.json`
  - `src/financial_models/cassava_ethanol/data/default_inputs.json`
  - `src/financial_models/broiler_chicken/data/default_inputs.json`
- Schema: `financial_models.api.schemas.pharma.PharmaInputsPayload` (validated by `/inputs/pharma/validate`)
  - Required sections: `years` (array), `production_estimate` (product → yearly units), `unit_costs`, `markup`, `raw_material_cost`, `utility_costs`, `labor`, `depreciation`, `capital_expenditure`, `financing`, `working_capital`, `tax`, `risk`, `scenarios`, `sensitivity`, `monte_carlo`.
  - Optional: `total_production_units`, `production_capacity`, `inflation_rate`/`inflation_series`, `fixed_variable_costs`, `break_even`, `distributor_commission`, `scenario_tools`, `goal_seek`, `ai`.
  - Field hints:
    - `unit_costs`: product → `{ production, price, freight }`
    - `utility_costs`: per-year rows with electricity/water/steam usage, rates, and days/hours (or scalar defaults)
    - `labor`: `{ direct: { role: cost }, indirect: { role: cost } }`
    - `depreciation.rows`: asset rows with `asset_type`, `year`, `acquisition`, `depreciation_rate`, optional `asset_life`, `method`
    - `capital_expenditure`: `{ initial, contingency, project_reserve, annual_additions }`
    - `financing`: `{ initial_investment, discount_rate, senior_debt_interest, revolver_interest, cash_interest, dividend_payout, share_capital, senior_debt[], revolver[], overdraft[] }`
    - `working_capital`: `{ days: { accounts_receivable, inventory, prepaid_expenses, other_assets, accounts_payable, other_liabilities }, calendar_days }`
    - `tax`: `{ rate, timing_adjustment, schedule[]? }`
    - `risk`: named risk series aligned to `years`
    - `scenarios`: `{ name: { inflation: [], interest: [] } }`
    - `sensitivity`: `{ variables: { name: [floats] } }`
    - `monte_carlo`: `{ iterations, revenue_growth_range, variables, metrics, seed?, distribution? }`
    - `goal_seek`: `{ metric, target, source, year? }`
    - `ai`: `{ enabled, provider, model, forecast_horizon, ml_methods, generative_features, api_key? }`
- Request body (`ModelRunRequest`):
  - `inputs`: full modelling payload (object). When omitted, the model uses its bundled defaults.
- 200 response (`ModelRunResponse` from `financial_models.api.schemas.common`): financial outputs as tables
  - `summary_metrics`, `income_statement`, `balance_sheet`, `cash_flow`, `goal_seek`, `break_even`, `payback`, `discounted_payback`, `monte_carlo`: `TablePayload`
  - `scenario_results`: object of scenario name → `TablePayload`
  - `sensitivity_results`: object of sensitivity name → `TablePayload`
  - `scenario_tool_results`: object of name → `ScenarioToolResultPayload`
  - `ai_insights`: `AIInsightsPayload` | null
  - `risk_factor_diagnostics`: `TablePayload` | null
- Example request:
  ```http
  POST /model/pharma/run
  X-API-Key: $FINANCIAL_MODELS_API_TOKEN
  Content-Type: application/json

  {
    "inputs": { ...full payload... }
  }
  ```
- Exports (CLI): income_statement.csv, balance_sheet.csv, cash_flow.csv, summary_metrics.csv, break_even.csv, payback.csv, discounted_payback.csv, scenario_*.csv, sensitivity_*.csv, monte_carlo.csv, valuation.{json,csv}
- Example response (truncated):
  ```json
  {
    "summary_metrics": {
      "index_name": "Metric",
      "index": ["NPV", "IRR"],
      "data": {"value": [1234567.89, 0.18]}
    },
    "income_statement": {
      "index_name": "Year",
      "index": [2024, 2025],
      "data": {"Revenue": [1000000, 1300000], "EBITDA": [200000, 260000]}
    },
    "scenario_results": {},
    "sensitivity_results": {},
    "ai_insights": null
  }
  ```

### `POST /inputs/pharma/validate`
- Auth: required unless auth is disabled
- Model types: `pharma` (microbrewery and biotech have equivalent validation endpoints).
- Request (`ValidationRequest`): `{ "inputs": { ...payload... } }`
- 200 response (`ValidationResponse`): `{ "valid": true|false, "message": "<detail>" }`

### `POST /model/microbrewery/run`
- Auth: required unless auth is disabled
- Request (`MicrobreweryModelRunRequest`):
  - `inputs`: object containing `config`, `dividend_policy`, `skus`, `channels`, `sales_plan`, optional `capex_items`, `debt_facilities`, `equity_injections`, `opex_fixed_monthly`, and `other_income_monthly`. When omitted, server uses defaults (`src/financial_models/microbrewery/data/default_inputs.json`).
- 200 response (`MicrobreweryModelRunResponse`):
  - `monthly`, `annual`, `prices`: `TablePayload`
  - `debt_schedules`: object of facility name → `TablePayload`
  - `valuation`: object of valuation metrics (numbers)
- Exports (CLI): monthly.csv, annual.csv, prices.csv, debt_*.csv, valuation.json, valuation.csv
- Example request:
  ```http
  POST /model/microbrewery/run
  X-API-Key: $FINANCIAL_MODELS_API_TOKEN
  Content-Type: application/json

  {
    "inputs": {
      "config": { "start_date": "2025-01-01", "months": 72, ... },
      "skus": [{ "sku_id": 1, "name": "Pale Ale 330ml", "direct_cost_per_unit": 2.05, "markup_pct": 0.65 }],
      "channels": [{ "channel": "Wholesale", "price_factor": 1.4 }],
      "sales_plan": [{ "date": "2025-02-01", "sku_id": 1, "channel": "Wholesale", "units": 2700 }]
    }
  }
  ```

### `POST /inputs/microbrewery/validate`
- Auth: required unless auth is disabled
- Request (`MicrobreweryValidationRequest`): `{ "inputs": { ...payload... } }`
- 200 response (`ValidationResponse`): `{ "valid": true|false, "message": "<detail>" }`

### `POST /model/cassava_ethanol/run`
- Auth: required unless auth is disabled
- Request (`CassavaModelRunRequest`):
  - `inputs`: optional object with `scenario` (`FARM_ONLY`, `BUY_ONLY`, `HYBRID`). When omitted, defaults are used.
- 200 response (`CassavaModelRunResponse`):
  - Monthly/annual income, balance, and cash flow statements: `TablePayload`
  - `break_even`, `payback`: `TablePayload`
  - `metrics`: object of key metrics
  - `scenario`: string
- Exports (CLI): income_statement_monthly.csv, income_statement_annual.csv, balance_sheet_monthly.csv, balance_sheet_annual.csv, cash_flow_monthly.csv, cash_flow_annual.csv, break_even.csv, payback.csv, metrics.json, metrics.csv

### `POST /inputs/cassava_ethanol/validate`
- Auth: required unless auth is disabled
- Request (`CassavaValidationRequest`): `{ "inputs": { "scenario": "FARM_ONLY" } }`
- 200 response (`ValidationResponse`): `{ "valid": true|false, "message": "<detail>" }`

### `POST /model/broiler_chicken/run`
- Auth: required unless auth is disabled
- Request (`BroilerModelRunRequest`):
  - `inputs`: optional assumption overrides (keys align to `broiler_chicken.assumptions.Assumptions`). When omitted, defaults are used.
- 200 response (`BroilerModelRunResponse`):
  - `assumptions_schedule`, `income_statement`, `balance_sheet`, `cash_flow_statement`, `cashflows`, `revenue_summary`: `TablePayload`
  - `valuation`: object with NPV/IRR and timeline metadata
  - `advanced_analytics`: object of analysis name → `TablePayload`
- Exports (CLI): assumptions_schedule.csv, income_statement.csv, balance_sheet.csv, cash_flow_statement.csv, cashflows.csv, revenue_summary.csv, valuation.json, valuation.csv, advanced_*.csv

### `POST /inputs/broiler_chicken/validate`
- Auth: required unless auth is disabled
- Request (`BroilerValidationRequest`): `{ "inputs": { ...assumption overrides... } }`
- 200 response (`ValidationResponse`): `{ "valid": true|false, "message": "<detail>" }`

Example requests:
```http
POST /model/cassava_ethanol/run
Content-Type: application/json

{ "inputs": { "scenario": "HYBRID" } }
```

```http
POST /model/broiler_chicken/run
Content-Type: application/json

{
  "inputs": {
    "discount_rate": 0.12,
    "debt_ratio": 0.55,
    "capex_housing": 1250000,
    "capex_equipment": 450000
  }
}
```

### `POST /report/{model}/generate`
- Auth: required (must include an auth-bound email via JWT/Google; API token auth is rejected)
- Available for models that define report builders (pharma, biotech, microbrewery, goat_farming, cassava_ethanol, broiler_chicken).
- Request: same as the corresponding `/model/{model}/run` body with an added `format` field (`PDF`, `Word`, `Excel`, `CSV`, `JSON`).
- Response: binary report with `Content-Disposition` set to the suggested filename.
- Subscription: requires an active subscription for the authenticated user's email (403 when inactive).

### `POST /model/goat_farming/run`
- Auth: required unless auth is disabled
- Request (`GoatModelRunRequest`):
  - `inputs`: object containing `schedule` (list of rows with a `Period` column), optional `valuation_inputs` (e.g. `WACC`, `NPV`), optional `supplementary_tables`, and optional `scenario` shocks (`milk_price_pct`, `feed_cost_pct`). When omitted, server uses the bundled default schedule generated in `src/financial_models/goat/inputs.py`.
- 200 response (`GoatModelRunResponse`):
  - `schedule`, `scenario`, `performance`, `cash_flow`, `position`, `kpis`, `break_even`: `TablePayload`
  - `advanced`: object of analysis name → `{ title, description, tables: { table_name: TablePayload } }`
  - `valuation_summary`: object with WACC/NPV/Terminal Value (numbers or null)
- Exports (CLI): schedule.csv, scenario.csv, performance.csv, cash_flow.csv, position.csv, kpis.csv, break_even.csv, advanced_*_*.csv, valuation_summary.json, valuation_summary.csv
- Example request:
  ```http
  POST /model/goat_farming/run
  X-API-Key: $FINANCIAL_MODELS_API_TOKEN
  Content-Type: application/json

  {
    "inputs": {
      "period_column": "Period",
      "valuation_inputs": { "WACC": 0.12, "NPV": 750000 },
      "schedule": [
        { "Period": "2024-01-31", "Revenue": 100000, "COGS": 45000, "Gross Margin": 55000, "EBITDA": 25000, "NPAT": 15000 }
      ]
    }
  }
  ```

### `POST /inputs/goat_farming/validate`
- Auth: required unless auth is disabled
- Request (`GoatValidationRequest`): `{ "inputs": { ...payload... } }`
- 200 response (`ValidationResponse`): `{ "valid": true|false, "message": "<detail>" }`

### `POST /model/biotech/run`
- Auth: required unless auth is disabled
- Request (`BiotechModelRunRequest`):
  - `inputs`: object with `model_config` and `products` list. When omitted, server uses defaults (`src/financial_models/biotech/data/default_inputs.json`).
- Schema: `financial_models.api.schemas.biotech.BiotechInputsPayload` (validated by `/inputs/biotech/validate`)
  - `model_config` (aliased as `config`): `{ first_year, n_years, discount_rate, country?, currency?, language?, inflation_rate?, tax_rate?, sales_tax_rate?, working_capital_days?, simulation_years?, simulation_runs? }`
  - `products`: array of drug candidates, each with common fields like `{ name, phase, success_prob, market_size, price_per_unit, units_sold, cost_per_unit, launch_year, peak_sales_year, peak_penetration, patent_life, tax_rate?, sales_tax_rate?, wacc?, rnpv? }`
  - Nested/optional product fields: `costs` (R&D/SG&A), `timeline` milestones, `geographies`, `scenarios`, `probability_adjustments`, `price_erosion`, `post_patent_sales`, `manufacturing_costs`, `capital_costs`, `funding_rounds`, `royalties`, `milestone_payments`, `orphan_drug_exclusivity`, `risk_adjustments`.
  - Validation highlights: at least one product is required; `success_prob` must be between 0 and 1; `n_years` must be positive; product names must be non-empty.
- 200 response (`BiotechModelRunResponse`):
  - `rnpv`: number
  - `consolidated`: `TablePayload`
  - `dcf_table`: `TablePayload`
  - `per_product`: object of product name → `TablePayload`
  - `per_product_prob`: object of product name → `TablePayload` (probability-weighted)
- Exports (CLI): consolidated.csv, dcf_table.csv, per_product_*.csv, per_product_prob_*.csv, valuation.json, valuation.csv, rnpv.txt
- Example request:
  ```http
  POST /model/biotech/run
  X-API-Key: $FINANCIAL_MODELS_API_TOKEN
  Content-Type: application/json

  {
    "inputs": {
      "model_config": { "first_year": 2024, "n_years": 25, ... },
      "products": [{ "name": "AgSeed-101", "success_prob": 0.35, ... }]
    }
  }
  ```

### `POST /inputs/biotech/validate`
- Auth: required unless auth is disabled
- Request (`BiotechValidationRequest`): `{ "inputs": { "model_config": {...}, "products": [...] } }`
- 200 response (`ValidationResponse`): `{ "valid": true|false, "message": "<detail>" }`

### `POST /subscriptions/check`
- Auth: required unless auth is disabled
- Request (`SubscriptionCheckRequest`): `{ "email": "user@example.com" }`
- 200 response (`SubscriptionCheckResponse`):
  - `email`: string
  - `is_active`: boolean
  - `message`: string
  - `payload`: object | null (raw Paystack response)
  - `cached`: boolean (true when served from persisted cache)
  - `cached_at`: number | null (epoch seconds when cached)

### `GET /subscriptions/status`
- Auth: required unless auth is disabled
- Query: `email=<address>`
- 200 response (`SubscriptionStatusRecord`):
  - `email`, `is_active`, `status_message`, `updated_at` (epoch seconds)
  - `source`: string | null
  - `expires_at`: number | null
  - `payload`: object | null
- 404 if missing or expired

### `POST /subscriptions/status`
- Auth: required unless auth is disabled
- Request (`SubscriptionStatusUpsert`):
  - `email`: string
  - `is_active`: boolean
  - `status_message`: string
  - `payload`: object | null
  - `source`: string | null
  - `ttl_seconds`: number | null (optional expiry)
- 200 response: `SubscriptionStatusRecord` echoing the stored record

### `DELETE /subscriptions/status`
- Auth: required unless auth is disabled
- Query: `email=<address>`
- 204 on success

## Error Shape

FastAPI default errors: `{ "detail": "<message>" }` with relevant HTTP status codes (400 for bad input, 401 for auth issues, 404/502/503 for subscription errors, 500 for unexpected conditions).

## Client Generation (optional)

Point your generator at `/openapi.json`. Example for Angular:

```bash
npx @openapitools/openapi-generator-cli generate \
  -i http://localhost:8000/openapi.json \
  -g typescript-angular \
  -o src/app/api
```
