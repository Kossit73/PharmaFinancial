"""Command line entry point for running the financial models."""
from __future__ import annotations

import argparse
from pathlib import Path

from .core.inputs import load_inputs
from .core.model import FinancialModel
from .biotech import ValuationEngine as BiotechValuationEngine, build_portfolio as build_biotech_portfolio
from .biotech import load_inputs as load_biotech_inputs


def _write_table(path: Path, table) -> None:
    try:
        table.to_frame().to_csv(path)  # type: ignore[attr-defined]
    except Exception:
        table.to_csv(path)


def run_model(model: str, input_path: Path | None = None, output: Path | None = None) -> None:
    model = model.lower().strip()
    output = output or Path.cwd()
    output.mkdir(parents=True, exist_ok=True)

    if model == "pharma":
        inputs = load_inputs(input_path)
        fm = FinancialModel(inputs)
        results = fm.run()

        _write_table(
            output / "income_statement.csv",
            results.income_statement.rounded(0, exclude_keywords=("Margin", "Return")),
        )
        _write_table(output / "balance_sheet.csv", results.balance_sheet.rounded(0))
        _write_table(output / "cash_flow.csv", results.cash_flow.rounded(0))
        _write_table(output / "summary_metrics.csv", results.summary_metrics)
        _write_table(output / "break_even.csv", results.break_even)
        _write_table(output / "payback.csv", results.payback)
        _write_table(output / "discounted_payback.csv", results.discounted_payback)

        for scenario, df in results.scenario_results.items():
            _write_table(output / f"scenario_{scenario}.csv", df)

        for variable, df in results.sensitivity_results.items():
            _write_table(output / f"sensitivity_{variable}.csv", df)

        _write_table(output / "monte_carlo.csv", results.monte_carlo)
    elif model == "biotech":
        inputs = load_biotech_inputs(input_path)
        portfolio = build_biotech_portfolio(inputs)
        result = BiotechValuationEngine(portfolio).run()

        _write_table(output / "consolidated.csv", result.consolidated)
        _write_table(output / "dcf_table.csv", result.dcf_table)
        for name, df in result.per_product.items():
            _write_table(output / f"per_product_{name}.csv", df)
        for name, df in result.per_product_prob.items():
            _write_table(output / f"per_product_prob_{name}.csv", df)
        (output / "rnpv.txt").write_text(str(result.rnpv), encoding="utf-8")
    else:
        raise SystemExit(f"Unknown model '{model}'. Choose 'pharma' or 'biotech'.")



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the financial models")
    parser.add_argument(
        "--model",
        choices=["pharma", "biotech"],
        default="pharma",
        help="Which model to run.",
    )
    parser.add_argument("--inputs", type=Path, default=None, help="Path to an inputs JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory to store generated CSV schedules",
    )
    args = parser.parse_args()
    run_model(args.model, args.inputs, args.output)


if __name__ == "__main__":
    main()
