# Longevity Pharmaceuticals Financial Model

This project provides a Python implementation of the Longevity Pharmaceuticals financial model, translating the comprehensive set of assumptions provided into a reproducible analytical engine and interactive dashboards.

## Features

- Structured input landing page backed by JSON assumptions.
- Production, revenue, cost, and working capital schedules covering 2024–2033.
- Core financial statements (income statement, balance sheet, cash flow statement).
- Scenario, sensitivity, and Monte Carlo simulation tooling.
- Break-even and payback analytics.
- Streamlit web application delivering dashboards and statements, complemented by a CLI that exports schedules to CSV files.
- Integrated machine-learning forecasts and optional generative AI summaries with configurable providers and API keys.

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

   Use the sidebar to upload an alternative JSON assumptions file or generate a consolidated report.
  The **Report Download** dropdown exports the Key Metrics dashboard through the Monte Carlo simulation tab as a single PDF,
  Word, Excel, CSV, or JSON document. These formats rely on the `fpdf`, `python-docx`, and `openpyxl` packages, which are now
  included in `requirements.txt` so installing the project dependencies enables every export option out of the box.

## Customising Assumptions

All modelling assumptions are defined in [`src/pharma_financial/data/default_inputs.json`](src/pharma_financial/data/default_inputs.json). Duplicate this file and pass the new path to the CLI using `--inputs` to evaluate alternative cases. The structure mirrors the specification shared in the project brief, covering production volumes, cost inflation, labour structures, financing, and working capital.

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
