# Goat Farming Model

- **Defaults:** `data/default_inputs.json` (Period-based schedule with revenue/cost/cash-flow series).
- **CLI:** `python -m financial_models --model goat_farming --inputs <path>` (JSON payload) to write `schedule.csv`, `scenario.csv`, `performance.csv`, `cash_flow.csv`, `position.csv`, `kpis.csv`, `break_even.csv`, `advanced_*_*.csv`, `valuation_summary.{json,csv}`.
- **API:** `POST /model/goat_farming/run` or `/inputs/goat_farming/validate`.
- **Reports:** Available via `--report-format` or `/report/goat_farming/generate`.
