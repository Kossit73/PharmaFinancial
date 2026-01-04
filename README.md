# Financial Models Suite

This project provides Python implementations of the Pharmaceuticals, Biotech, Microbrewery, Goat Farming, Cassava Bioethanol, and Broiler Chicken financial models, translating comprehensive assumption sets into reproducible analytical engines and interactive dashboards.

## Features

- Structured input landing page backed by JSON assumptions.
- Production, revenue, cost, and working capital schedules covering 2024–2033.
- Core financial statements (income statement, balance sheet, cash flow statement).
- Scenario, sensitivity, and Monte Carlo simulation tooling.
- Break-even and payback analytics.
- Streamlit web application delivering dashboards and statements, complemented by a CLI that exports schedules to CSV files.
- Integrated machine-learning forecasts and optional generative AI summaries with configurable providers and API keys.
- FastAPI service with documented contract for frontend clients (see `docs/api_contract.md`).
- Microbrewery engine covering SKU/channel pricing, working capital, multi-facility debt schedules, monthly/annual statements, and equity IRR/MOIC.
- Goat farming engine for manually entered schedules (CSV/JSON/API), producing scenarios, financial statements, KPIs, break-even, and advanced analytics.
- Cassava bioethanol engine with landing-page inputs, scenarios (farm/buy/hybrid), statements, metrics, break-even/payback analytics, and report generation.
- Broiler chicken engine producing full statements, revenue schedules, discounted cash flow valuation, advanced analytics, and downloadable reports.

## Getting Started

1. **Install dependencies (optional for CLI smoke tests)**

   The core financial engine is implemented using the Python standard library, so
   the command-line workflows operate without additional packages. Installing the
   scientific stack enables richer dashboards and analytics:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the CLI**

   ```bash
   python src/financial_models/__main__.py --output outputs
   ```

   This command writes CSV schedules for all statements, scenario tables, sensitivity pivots, and the Monte Carlo simulation to the `outputs/` directory. Pass `--model microbrewery` for the microbrewery engine, `--model goat_farming` for the goat farming engine, `--model cassava_ethanol` for the cassava bioethanol engine, or `--model broiler_chicken` for the broiler model instead of the pharma default. Reports are available for every model via `--report-format` or the API.

   > **Tip:** If you prefer `python -m financial_models`, either install the
   > package in editable mode (`pip install -e .`) or export
   > `PYTHONPATH=$(pwd)/src` before running the command so that Python can locate
   > the module.

3. **Run the FastAPI service (API mode)**

   ```bash
   # set auth secrets (examples)
   export FINANCIAL_MODELS_AUTH_SECRET=dev-secret          # required for JWT auth
   export FINANCIAL_MODELS_API_TOKEN=dev-api-key           # optional API key auth
   export FINANCIAL_MODELS_GOOGLE_AUDIENCE=your-client-id  # optional Google ID token auth

   # run the API
   ./.venv/bin/uvicorn financial_models.api.server:create_app --factory --reload --port 8000
   ```

   - Base URL: `http://localhost:8000` with Swagger UI at `/docs` (use **Authorize** to add `Bearer <jwt>` or `X-API-Key`).
   - Model routes: `POST /model/pharma/run`, `POST /inputs/pharma/validate`, `POST /model/microbrewery/run`, `POST /inputs/microbrewery/validate`, `POST /model/cassava_ethanol/run`, `POST /model/broiler_chicken/run`, `POST /model/goat_farming/run`, and biotech equivalents (see `docs/api_contract.md`).
   - Auth routes: `/auth/register`, `/auth/login` (JSON or form body), `/auth/me`, `/auth/me` (PATCH), `/auth/users` (GET), `/auth/users/{email}` (DELETE).
   - User store: defaults to `~/.financial_models/users.db` (override with `FINANCIAL_MODELS_USER_DB`). Set `FINANCIAL_MODELS_AUTH_SECRET` to issue/verify JWTs.

4. **Launch the Streamlit dashboard (optional if using the API/Angular frontend)**

   ```bash
   streamlit run streamlit_app.py
   ```

   The Streamlit application includes dedicated tabs for:

   1. An input landing page summarising unit economics and labour structures.
   2. A key metrics dashboard highlighting Net Revenue, EBITDA, and investment KPIs.
   3. Statements of financial performance, position, and cash flows.
   4. Sensitivity, scenario ("IFs"), and Monte Carlo analyses.
   5. Break-even and payback visualisations.

   Use the sidebar to upload an alternative JSON assumptions file or generate a consolidated report.
  The **Report Download** dropdown exports the Key Metrics dashboard through the Monte Carlo simulation tab as a single PDF,
  Word, Excel, CSV, or JSON document. These formats rely on the `fpdf`, `python-docx`, and `openpyxl` packages, which are now
  included in `requirements.txt` so installing the project dependencies enables every export option out of the box.

### Paystack Configuration

Set the following environment variables (for example in `.env`) to enable the built-in subscription flow:

- `PAYSTACK_SECRET_KEY` – Paystack secret key used to authenticate API calls.
- `PAYSTACK_PLAN_CODE` – plan identifier returned by Paystack when you create the subscription plan.
- `PAYSTACK_PLAN_AMOUNT_KOBO` *(optional)* – fallback amount (in Kobo) used when Paystack does not return the plan amount automatically.
- `PAYSTACK_CALLBACK_URL` *(optional but recommended)* – URL Paystack redirects users to after they complete checkout. Point this at the Streamlit deployment so users return to the app automatically.
- `PAYSTACK_CANCEL_ACTION_URL` *(optional)* – URL invoked when a user cancels or Paystack declines a payment, ensuring they are taken back to the Streamlit app even if the transaction fails.

### API Authentication

When deploying the FastAPI service (`financial_models.api.server`) set `FINANCIAL_MODELS_API_TOKEN` to any strong secret. Every request (except `/health`) must then include this token via the `X-API-Key` header:

```bash
curl -H "X-API-Key: $FINANCIAL_MODELS_API_TOKEN" http://localhost:8000/model/pharma/run -d '{"inputs": ...}'
```

If `FINANCIAL_MODELS_API_TOKEN` is unset the service remains open, which is suitable only for local development.

To accept Google sign-ins instead (or in addition), configure `FINANCIAL_MODELS_GOOGLE_AUDIENCE` with the OAuth client ID(s) allowed to call the API (comma-separated when you have multiple). Requests must then include a Google ID token in the standard `Authorization: Bearer <token>` header. The server verifies the token against Google and extracts the caller’s identity before running the model. Ensure `google-auth` is available in the environment so verification succeeds.

#### Cache invalidation & webhooks

The Excel export tab now exposes a **Check subscription status** button which re-runs the Paystack lookup immediately instead of waiting for the 10‑minute session cache to expire.

For server-driven invalidation, run the lightweight webhook receiver (set `PAYSTACK_WEBHOOK_SECRET` or reuse `PAYSTACK_SECRET_KEY`) and point your Paystack dashboard at it:

```bash
python -m financial_models.webhook --host 0.0.0.0 --port 8080
```

The server validates `X-Paystack-Signature`, records events in a SQLite database (`~/.financial_models/subscriptions.db` by default, override with `SUBSCRIPTION_STORE_PATH`), and marks subscriptions as revoked when Paystack emits cancellation or failed renewal events. Every Streamlit session consults the shared store before allowing downloads, so webhook updates cut off access immediately even if the UI cache is still valid.
The implementation now lives under `financial_models.services.webhook`, but the `python -m financial_models.webhook` entry point remains unchanged for compatibility.

## Customising Assumptions

All pharma modelling assumptions are defined in [`src/financial_models/data/default_inputs.json`](src/financial_models/data/default_inputs.json) (mirrored under `financial_models/pharma/data/`). Duplicate this file and pass the new path to the CLI using `--inputs` to evaluate alternative cases. The structure mirrors the specification shared in the project brief, covering production volumes, cost inflation, labour structures, financing, and working capital. Biotech defaults live under [`src/financial_models/biotech/data/default_inputs.json`](src/financial_models/biotech/data/default_inputs.json).

Microbrewery assumptions are bundled at [`src/financial_models/microbrewery/data/default_inputs.json`](src/financial_models/microbrewery/data/default_inputs.json). They capture SKU/channel pricing, CAPEX, multi-facility debt schedules, equity injections, and dividend policy parameters. Use `--model microbrewery --inputs <path>` with the CLI or the `/model/microbrewery/run` API route to evaluate alternate scenarios.

Goat farming schedules are bundled at [`src/financial_models/goat_farming/data/default_inputs.json`](src/financial_models/goat_farming/data/default_inputs.json). Supply your own JSON/CSV via `--model goat_farming --inputs <path>` (JSON payload matching the API schema) or the `/model/goat_farming/run` endpoint.

Cassava bioethanol defaults live at [`src/financial_models/cassava_ethanol/data/default_inputs.json`](src/financial_models/cassava_ethanol/data/default_inputs.json); override the scenario via `/model/cassava_ethanol/run` or `--model cassava_ethanol --inputs <path>`.

Broiler chicken assumptions are seeded at [`src/financial_models/broiler_chicken/data/default_inputs.json`](src/financial_models/broiler_chicken/data/default_inputs.json); pass overrides as JSON to `/model/broiler_chicken/run` or via CLI `--model broiler_chicken --inputs <path>`.

### Exported outputs (CLI)

- **pharma**: income_statement.csv, balance_sheet.csv, cash_flow.csv, summary_metrics.csv, break_even.csv, payback.csv, discounted_payback.csv, scenario_*.csv, sensitivity_*.csv, monte_carlo.csv
- **biotech**: consolidated.csv, dcf_table.csv, per_product_*.csv, per_product_prob_*.csv, rnpv.txt
- **microbrewery**: monthly.csv, annual.csv, prices.csv, debt_*.csv, valuation.json, valuation.csv
- **goat_farming**: schedule.csv, scenario.csv, performance.csv, cash_flow.csv, position.csv, kpis.csv, break_even.csv, advanced_*_*.csv, valuation_summary.json, valuation_summary.csv
- **cassava_ethanol**: income_statement_monthly.csv, income_statement_annual.csv, balance_sheet_monthly.csv, balance_sheet_annual.csv, cash_flow_monthly.csv, cash_flow_annual.csv, break_even.csv, payback.csv, metrics.json
- **broiler_chicken**: assumptions_schedule.csv, income_statement.csv, balance_sheet.csv, cash_flow_statement.csv, cashflows.csv, revenue_summary.csv, valuation.json, advanced_*.csv

### Example payloads (API/CLI)

Cassava bioethanol:
```json
{
  "inputs": {
    "scenario": "HYBRID"
  }
}
```

Broiler chicken:
```json
{
  "inputs": {
    "discount_rate": 0.12,
    "debt_ratio": 0.55,
    "capex_housing": 1250000,
    "capex_equipment": 450000
  }
}
```

### AI, Machine Learning, and Generative Insights

The Input Landing Page now includes an **AI & Machine Learning Configuration** section. Use it to:

- Enable or disable AI enhancements for the current session.
- Select a provider (OpenAI, Azure OpenAI, Anthropic, Vertex AI, or a custom endpoint) and specify the deployed model name.
- Choose the machine-learning algorithms (linear regression, CAGR, moving average) used to forecast net revenue beyond the base projection horizon.
- Pick the focus areas for the generative summary (executive overview, risk review, cash-flow highlights).
- Supply an API key to call the selected provider. Keys are stored only in the active Streamlit session and are not written to disk or bundled JSON files.

When no API key is provided, the dashboard falls back to deterministic heuristic commentary so that insights remain available in offline environments. To use OpenAI models install the optional dependency and export your key before launching Streamlit:

```bash
pip install openai
export OPENAI_API_KEY="sk-your-key"
streamlit run streamlit_app.py
```

## Project Structure

```
src/financial_models/
  core/                # shared building blocks (ai/report/table) reused by every model
  pharma/              # pharma engine: inputs, model, data/default_inputs.json (+ ai/report/table shims to core)
  biotech/             # biotech valuation engine: inputs, model, data/default_inputs.json
  microbrewery/        # microbrewery engine: inputs, model, data/default_inputs.json
  goat_farming/        # goat farming engine: inputs + scenario/statements/analytics (default schedule generated in code)
  cassava_ethanol/     # cassava bioethanol engine: landing page inputs, scenarios, statements, analytics, reports
  broiler_chicken/     # broiler chicken engine: assumptions, production/financing, statements, analytics, reports
  api/                 # FastAPI app + pydantic schemas
  services/            # paystack + subscription storage + webhook receiver
  ui/                  # UI gateways used by Streamlit/Angular
  model_registry.py    # central registry used by CLI/API to expose models
  data/default_inputs.json  # legacy default inputs consumed by CLI/tests (mirrors pharma defaults)
```

### Adding additional models

1. Create `src/financial_models/<name>/` with `inputs.py`, `model.py`, and `data/default_inputs.json`. If you want shared reporting/AI, add thin wrappers that import `financial_models.core.report`/`financial_models.core.ai` and export `generate_report`/`REPORT_FORMATS` or AI helpers.
2. Register the model in `model_registry.py` (load_inputs, parse_inputs, run_core, build_response, optional report builders, CLI exporter).
3. If the API should serve it, add pydantic request/response schemas under `financial_models/api/schemas/<name>/` and wire routes in `financial_models/api/server.py`.
4. Update docs and tests with the new model key.

## Testing the Model Logic

Automated smoke tests validate that the financial engine executes end-to-end using the bundled assumptions:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

For ad-hoc experimentation you can also run the engine directly from the Python prompt. The
pharma modelling engine now lives under the `financial_models.pharma` namespace, so import from `pharma`
when scripting against the financial toolkit:

```bash
python - <<'PY'
from financial_models.pharma.inputs import load_inputs
from financial_models.pharma.model import FinancialModel

inputs = load_inputs()
model = FinancialModel(inputs)
results = model.run()
print(results.summary_metrics.as_dict())
PY
```

The printed dictionary lists NPV, IRR, and payback metrics derived from the default assumptions.

### Updating the Utility Schedule

You can revise electricity, water, and steam assumptions directly from the **Utility Schedule**
editor on the Input Landing Page:

1. **Open Streamlit** using `streamlit run streamlit_app.py` and navigate to the Input Landing
   Page.
2. Locate the **Utility Schedule** section. Each row represents a projection year with columns
   for the per-day (or per-hour) quantities and the applicable unit prices.
3. Click into any cell to adjust the quantity, rate, or operating-day/hour assumptions. The table
   updates in place and immediately syncs the new values back into the modelling payload.
4. To add an additional year, use the “Add row” control at the bottom of the table. A blank entry
   will appear that you can populate with the relevant year label and utility inputs.
5. Remove a year by selecting the corresponding row and choosing the built-in delete action.

All changes are reflected instantly across the Key Metrics dashboard, financial statements, and
analytics tabs, so you can observe the downstream impact of revised utility assumptions without
editing the underlying JSON manually.

## Notes

- The IRR calculation uses a Newton iteration approach implemented in pure Python.
- Monte Carlo simulations rely on Python's `random` module with a fixed seed for reproducibility.
- Scenario and sensitivity analyses mutate internal parameters temporarily; all baselines are restored after each computation so results remain consistent.
- For a mapping of the IFRS-style cash flow statement to its data sources, see [`docs/cash_flow_mapping.md`](docs/cash_flow_mapping.md).
