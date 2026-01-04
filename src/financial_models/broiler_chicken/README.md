# Broiler Chicken Model

- **Defaults:** `data/default_inputs.json` (seeded from `Assumptions`).
- **CLI:** `python -m financial_models --model broiler_chicken --inputs <path>` to export `assumptions_schedule.csv`, `income_statement.csv`, `balance_sheet.csv`, `cash_flow_statement.csv`, `cashflows.csv`, `revenue_summary.csv`, `valuation.json`, and `advanced_*.csv`.
- **API:** `POST /model/broiler_chicken/run` or `/inputs/broiler_chicken/validate`.
- **Reports:** Available via `--report-format` or `/report/broiler_chicken/generate`.
