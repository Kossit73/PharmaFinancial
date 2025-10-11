"""Command line entry point for running the Pharmaceuticals model."""
from __future__ import annotations

import argparse
from pathlib import Path

from .inputs import load_inputs
from .model import FinancialModel


def run_model(input_path: Path | None = None, output: Path | None = None) -> None:
    inputs = load_inputs(input_path)
    model = FinancialModel(inputs)
    results = model.run()

    output = output or Path.cwd()
    output.mkdir(parents=True, exist_ok=True)

    def _write_table(name: str, table) -> None:
        path = output / f"{name}.csv"
        try:
            table.to_frame().to_csv(path)  # type: ignore[attr-defined]
        except Exception:
            table.to_csv(path)

    _write_table(
        "income_statement",
        results.income_statement.rounded(0, exclude_keywords=("Margin", "Return")),
    )
    _write_table("balance_sheet", results.balance_sheet.rounded(0))
    _write_table("cash_flow", results.cash_flow.rounded(0))
    _write_table("summary_metrics", results.summary_metrics)
    _write_table("break_even", results.break_even)
    _write_table("payback", results.payback)
    _write_table("discounted_payback", results.discounted_payback)

    for scenario, df in results.scenario_results.items():
        _write_table(f"scenario_{scenario}", df)

    for variable, df in results.sensitivity_results.items():
        _write_table(f"sensitivity_{variable}", df)

    _write_table("monte_carlo", results.monte_carlo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Pharmaceuticals financial model")
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
