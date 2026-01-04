# Pharmaceuticals Model

- **Defaults:** `data/default_inputs.json` (production, pricing, costs, labour, financing, working capital, AI).
- **CLI:** `python -m financial_models --model pharma --inputs <path> --output outputs` to export income_statement.csv, balance_sheet.csv, cash_flow.csv, summary_metrics.csv, break_even.csv, payback.csv, discounted_payback.csv, scenario_*.csv, sensitivity_*.csv, monte_carlo.csv.
- **API:** `POST /model/pharma/run` and `/inputs/pharma/validate`.
- **Schema:** `financial_models.api.schemas.pharma.PharmaInputsPayload` (Pydantic).
- **Schema highlights:** requires `years`, `production_estimate`, `unit_costs`, `markup`, `raw_material_cost`, `utility_costs`, `labor`, `depreciation`, `capital_expenditure`, `financing`, `working_capital`, `tax`, `risk`, `scenarios`, `sensitivity`, `monte_carlo`; optional `production_capacity`, `total_production_units`, `inflation_series`, `fixed_variable_costs`, `break_even`, `distributor_commission`, `scenario_tools`, `goal_seek`, `ai`.
- **Reports:** `--report-format` or `/report/pharma/generate` (PDF/Word/Excel/CSV/JSON).
