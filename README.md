# Longevity Pharmaceuticals Financial Model

This project provides a Python implementation of the Longevity Pharmaceuticals financial model, translating the comprehensive set of assumptions provided into a reproducible analytical engine and interactive dashboards.

## Features

- Structured input landing page backed by JSON assumptions.
- Production, revenue, cost, and working capital schedules covering 2024–2033.
- Core financial statements (income statement, balance sheet, cash flow statement).
- Scenario, sensitivity, and Monte Carlo simulation tooling.
- Break-even and payback analytics.
- Dash web application delivering dashboards and statements, complemented by a CLI that exports schedules to CSV files.

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the CLI**

   ```bash
   python -m pharma_financial --output outputs
   ```

   This command writes CSV schedules for all statements, scenario tables, sensitivity pivots, and the Monte Carlo simulation to the `outputs/` directory.

3. **Launch the dashboard**

   ```bash
   python -m pharma_financial.app
   ```

   The Dash application serves:

   - An input landing page summarising unit economics.
   - A dashboard with Net Revenue and EBITDA trends plus key metrics.
   - Financial statements (performance, position, and cash flow).
   - Scenario/sensitivity and Monte Carlo tabs.

## Customising Assumptions

All modelling assumptions are defined in [`src/pharma_financial/data/default_inputs.json`](src/pharma_financial/data/default_inputs.json). Duplicate this file and pass the new path to the CLI using `--inputs` to evaluate alternative cases. The structure mirrors the specification shared in the project brief, covering production volumes, cost inflation, labour structures, financing, and working capital.

## Project Structure

```
src/
  pharma_financial/
    __init__.py
    __main__.py
    app.py
    cli.py
    inputs.py
    model.py
    data/
      default_inputs.json
```

## Testing the Model Logic

With dependencies installed you can execute a smoke test to ensure the model runs and emits schedules:

```bash
python - <<'PY'
from pharma_financial.inputs import load_inputs
from pharma_financial.model import FinancialModel

inputs = load_inputs()
model = FinancialModel(inputs)
results = model.run()
print(results.summary_metrics)
PY
```

The printed summary should list NPV, IRR, and payback metrics derived from the default assumptions.

## Notes

- The IRR calculation attempts to use `numpy_financial` when available and falls back to a Newton iteration otherwise.
- The Monte Carlo simulation leverages NumPy's random generator with a fixed seed (`42`) for reproducibility.
- Scenario and sensitivity analyses mutate internal parameters temporarily; all baselines are restored after each computation so results remain consistent.
