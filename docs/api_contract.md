# Pharmaceuticals Financial Model API Contract

This file captures the HTTP contract exposed by `financial_models.api.server` so frontend developers can integrate without reading the Python code. The FastAPI app also serves an OpenAPI document at `/openapi.json` that can be fed to client generators (e.g. `npx @openapitools/openapi-generator-cli generate -g typescript-angular ...`).

## Base URL

- Local development: `http://localhost:8000`
- Production: set to your deployed FastAPI host (HTTPS recommended).

All endpoints accept and return JSON.

## Authentication

- API token: set `PHARMA_FINANCIAL_API_TOKEN` on the server. Clients send `X-API-Key: <token>`. When unset, the API is open (not recommended outside local dev).
- Google ID token: set `PHARMA_FINANCIAL_GOOGLE_AUDIENCE` to the allowed OAuth client ID(s). Clients send `Authorization: Bearer <id-token>`. Both API token and Google auth can be enabled; either will authorize. `/health` remains public.

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
- Model types: `pharma` (current). Requests to other model paths return 404 until additional models are registered.
- Request body (`ModelRunRequest`):
  - `inputs`: full modelling payload (object). When omitted, server uses bundled defaults (`src/financial_models/data/default_inputs.json`).
- 200 response (`ModelRunResponse`): financial outputs as tables
  - `summary_metrics`, `income_statement`, `balance_sheet`, `cash_flow`, `goal_seek`, `break_even`, `payback`, `discounted_payback`, `monte_carlo`: `TablePayload`
  - `scenario_results`: object of scenario name → `TablePayload`
  - `sensitivity_results`: object of sensitivity name → `TablePayload`
  - `scenario_tool_results`: object of name → `ScenarioToolResultPayload`
  - `ai_insights`: `AIInsightsPayload` | null
  - `risk_factor_diagnostics`: `TablePayload` | null
- Example request:
  ```http
  POST /model/pharma/run
  X-API-Key: $PHARMA_FINANCIAL_API_TOKEN
  Content-Type: application/json

  {
    "inputs": { ...full payload... }
  }
  ```
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
- Model types: `pharma` (current). Requests to other model paths return 404 until additional models are registered.
- Request (`ValidationRequest`): `{ "inputs": { ...payload... } }`
- 200 response (`ValidationResponse`): `{ "valid": true|false, "message": "<detail>" }`

### `POST /subscriptions/check`
- Auth: required unless auth is disabled
- Request (`SubscriptionCheckRequest`): `{ "email": "user@example.com" }`
- 200 response (`SubscriptionCheckResponse`):
  - `email`: string
  - `is_active`: boolean
  - `message`: string
  - `payload`: object | null (raw Paystack response)

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
