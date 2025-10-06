# Longevity Pharmaceuticals Financial Model

This project provides a Python implementation of the Longevity Pharmaceuticals financial model, translating the comprehensive set of assumptions provided into a reproducible analytical engine and interactive dashboards.

## Features

- Structured input landing page backed by JSON assumptions.
- Production, revenue, cost, and working capital schedules covering 2024–2033.
- Core financial statements (income statement, balance sheet, cash flow statement).
- Scenario, sensitivity, and Monte Carlo simulation tooling.
- Break-even and payback analytics.
- Streamlit web application delivering dashboards and statements, complemented by a CLI that exports schedules to CSV files.

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
   python src/pharma_financial/__main__.py --output outputs
   ```

   This command writes CSV schedules for all statements, scenario tables, sensitivity pivots, and the Monte Carlo simulation to the `outputs/` directory.

   > **Tip:** If you prefer `python -m pharma_financial`, either install the
   > package in editable mode (`pip install -e .`) or export
   > `PYTHONPATH=$(pwd)/src` before running the command so that Python can locate
   > the module.

3. **Launch the Streamlit dashboard**

   ```bash
   streamlit run streamlit_app.py
   ```

   The Streamlit application includes dedicated tabs for:

   1. An input landing page summarising unit economics and labour structures.
   2. A key metrics dashboard highlighting Net Revenue, EBITDA, and investment KPIs.
   3. Statements of financial performance, position, and cash flows.
   4. Sensitivity, scenario ("IFs"), and Monte Carlo analyses.
   5. Break-even and payback visualisations.

   Use the sidebar to upload an alternative JSON assumptions file or download the bundled defaults.

## Customising Assumptions

All modelling assumptions are defined in [`src/pharma_financial/data/default_inputs.json`](src/pharma_financial/data/default_inputs.json). Duplicate this file and pass the new path to the CLI using `--inputs` to evaluate alternative cases. The structure mirrors the specification shared in the project brief, covering production volumes, cost inflation, labour structures, financing, and working capital.

## Project Structure

```
streamlit_app.py
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

Automated smoke tests validate that the financial engine executes end-to-end using the bundled assumptions:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

For ad-hoc experimentation you can also run the engine directly from the Python prompt:

```bash
python - <<'PY'
from pharma_financial.inputs import load_inputs
from pharma_financial.model import FinancialModel

inputs = load_inputs()
model = FinancialModel(inputs)
results = model.run()
print(results.summary_metrics.as_dict())
PY
```

The printed dictionary lists NPV, IRR, and payback metrics derived from the default assumptions.

## Notes

- The IRR calculation uses a Newton iteration approach implemented in pure Python.
- Monte Carlo simulations rely on Python's `random` module with a fixed seed for reproducibility.
- Scenario and sensitivity analyses mutate internal parameters temporarily; all baselines are restored after each computation so results remain consistent.
