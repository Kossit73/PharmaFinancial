# Cassava Bioethanol Model

- **Defaults:** `data/default_inputs.json` (scenario selector; landing-page inputs generated in code).
- **CLI:** `python -m financial_models --model cassava_ethanol --inputs <path>` to export income/balance/cash-flow (monthly/annual), `break_even.csv`, `payback.csv`, and `metrics.json`.
- **API:** `POST /model/cassava_ethanol/run` or `/inputs/cassava_ethanol/validate`.
- **Reports:** Available via `--report-format` or `/report/cassava_ethanol/generate`.
