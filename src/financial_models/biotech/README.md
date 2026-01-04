# Biotech Model

- **Defaults:** `data/default_inputs.json` (model_config + products).
- **CLI:** `python -m financial_models --model biotech --inputs <path>` to export consolidated.csv, dcf_table.csv, per_product_*.csv, per_product_prob_*.csv, rnpv.txt.
- **API:** `POST /model/biotech/run` and `/inputs/biotech/validate`.
- **Reports:** `/report/biotech/generate` or `--report-format` (PDF/Word/Excel/CSV/JSON).
