"""Command line entry point for running the Longevity Pharmaceuticals model."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .inputs import load_inputs
from .model import FinancialModel


def run_model(input_path: Path | None = None, output: Path | None = None) -> None:
    inputs = load_inputs(input_path)
    model = FinancialModel(inputs)
    results = model.run()

    output = output or Path.cwd()
    output.mkdir(parents=True, exist_ok=True)

    def _write(name: str, df: pd.DataFrame) -> None:
        df.to_csv(output / f"{name}.csv")

    _write("income_statement", results.income_statement)
    _write("balance_sheet", results.balance_sheet)
    _write("cash_flow", results.cash_flow)
    _write("summary_metrics", results.summary_metrics)
    _write("break_even", results.break_even)
    _write("payback", results.payback)
    _write("discounted_payback", results.discounted_payback)

    for scenario, df in results.scenario_results.items():
        _write(f"scenario_{scenario}", df)

    for variable, df in results.sensitivity_results.items():
        _write(f"sensitivity_{variable}", df)

    results.monte_carlo.to_csv(output / "monte_carlo.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Longevity Pharmaceuticals financial model")
    parser.add_argument("--inputs", type=Path, default=None, help="Path to an inputs JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory to store generated CSV schedules",
    )
    args = parser.parse_args()
    run_model(args.inputs, args.output)


if __name__ == "__main__":
    main()
