# Pharmaceuticals Model

- **Defaults:** `data/default_inputs.json` (production, pricing, costs, labour, financing, working capital, AI).
- **CLI:** `python -m financial_models --model pharma --inputs <path> --output outputs` to export income_statement.csv, balance_sheet.csv, cash_flow.csv, summary_metrics.csv, break_even.csv, payback.csv, discounted_payback.csv, scenario_*.csv, sensitivity_*.csv, monte_carlo.csv.
- **API:** `POST /model/pharma/run` and `/inputs/pharma/validate`.
- **Schema:** `financial_models.api.schemas.pharma.PharmaInputsPayload` (Pydantic).
- **Reports:** `--report-format` or `/report/pharma/generate` (PDF/Word/Excel/CSV/JSON).
