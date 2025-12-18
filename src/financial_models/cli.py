"""Command line entry point for running the financial models."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .pharma.report import REPORT_FORMATS, generate_report

from .model_registry import get_model_spec, list_models


def _fallback_export(response: Any, output_dir: Path) -> None:
    """Persist the API-style response as JSON when no CSV exporter is available."""

    if hasattr(response, "model_dump_json"):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(response.model_dump_json(indent=2), encoding="utf-8")


def run_model(
    model: str,
    input_path: Path | None = None,
    output: Path | None = None,
    report_format: str | None = None,
) -> None:
    spec = get_model_spec(model)
    output_dir = output or Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = spec.load_inputs(input_path)
    result = spec.run_core(inputs)
    response = spec.build_response(result)

    if spec.cli_exporter is not None:
        spec.cli_exporter(result, output_dir)
    else:
        _fallback_export(response, output_dir)

    if report_format and spec.build_report_sections:
        fmt = report_format.strip().upper()
        if fmt not in REPORT_FORMATS:
            raise SystemExit(f"Unsupported report format '{report_format}'. Choose one of {', '.join(REPORT_FORMATS)}.")
        sections = spec.build_report_sections(inputs, result)
        data, mime, filename = generate_report(
            sections,
            fmt,
            report_name=spec.report_name or f"{model}_financial_report",
            report_title=spec.report_title or spec.name,
        )
        report_path = output_dir / filename
        report_path.write_bytes(data)
        print(f"Wrote report to {report_path} ({mime})")


def main() -> None:
    available_models = sorted(list_models().keys())
    default_model = "pharma" if "pharma" in available_models else available_models[0]

    parser = argparse.ArgumentParser(description="Run the financial models")
    parser.add_argument(
        "--model",
        choices=available_models,
        default=default_model,
        help="Which model to run.",
    )
    parser.add_argument("--inputs", type=Path, default=None, help="Path to an inputs JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory to store generated CSV schedules",
    )
    parser.add_argument(
        "--report-format",
        choices=REPORT_FORMATS,
        help="Optional: generate a consolidated report in the given format.",
    )
    args = parser.parse_args()
    run_model(args.model, args.inputs, args.output, args.report_format)


if __name__ == "__main__":
    main()
