# Microbrewery Model

- **Defaults:** `data/default_inputs.json` (SKU/channel pricing, CAPEX, debt, equity, dividends).
- **CLI:** `python -m financial_models --model microbrewery --inputs <path>` to export `monthly.csv`, `annual.csv`, `prices.csv`, `debt_*.csv`, `valuation.{json,csv}`.
- **API:** `POST /model/microbrewery/run` or `/inputs/microbrewery/validate`.
- **Reports:** Available via `--report-format` or `/report/microbrewery/generate`.
