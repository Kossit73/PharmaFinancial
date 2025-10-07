"""Utilities for assembling and exporting consolidated model reports."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO, StringIO
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .model import FinancialModel, FinancialOutputs
from .table import Table

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas optional in minimal environments
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from docx import Document  # type: ignore
except Exception:  # pragma: no cover - python-docx may not be installed
    Document = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover - FPDF may not be installed
    FPDF = None  # type: ignore

JSON_MIME = "application/json"
CSV_MIME = "text/csv"
EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
WORD_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF_MIME = "application/pdf"

REPORT_FORMATS = ["PDF", "Word", "Excel", "CSV", "JSON"]


class ReportGenerationError(RuntimeError):
    """Raised when the requested report cannot be produced."""


@dataclass
class ReportTable:
    """Container describing an individual table within a report."""

    title: str
    data: Any
    note: str | None = None


@dataclass
class ReportSection:
    """Represents a logical report section grouping multiple tables."""

    title: str
    tables: List[ReportTable]
    notes: List[str] | None = None


def collect_report_sections(model: FinancialModel, outputs: FinancialOutputs) -> List[ReportSection]:
    """Assemble report sections spanning all dashboards in the required order."""

    sections: List[ReportSection] = []

    key_tables: List[ReportTable] = [
        ReportTable("Summary Metrics", outputs.summary_metrics),
        ReportTable("Goal Seek", outputs.goal_seek),
    ]

    try:
        key_tables.append(ReportTable("Working Capital Schedule", model.working_capital_schedule()))
    except Exception as exc:  # pragma: no cover - defensive guard for runtime calculations
        key_tables.append(ReportTable("Working Capital Schedule", [], note=f"Unavailable: {exc}"))

    try:
        key_tables.append(ReportTable("Inventory Schedule", model.inventory_schedule()))
    except Exception as exc:  # pragma: no cover - defensive guard for runtime calculations
        key_tables.append(ReportTable("Inventory Schedule", [], note=f"Unavailable: {exc}"))

    ai_notes: List[str] = []
    if outputs.ai_insights is not None:
        insight = outputs.ai_insights
        if insight.ml_forecast is not None:
            key_tables.append(ReportTable("AI Revenue Forecast", insight.ml_forecast))
        if insight.generative_summary:
            ai_notes.append(insight.generative_summary)
        if insight.metadata:
            provider = insight.metadata.get("provider")
            model_name = insight.metadata.get("model")
            status = insight.metadata.get("status")
            details = ", ".join(
                str(value)
                for value in [provider, model_name, status]
                if value not in (None, "")
            )
            if details:
                ai_notes.append(f"AI configuration: {details}")

    sections.append(ReportSection("Key Metrics Dashboard", key_tables, notes=ai_notes or None))

    perf_tables: List[ReportTable] = [
        ReportTable("Statement of Financial Performance", outputs.income_statement),
    ]
    try:
        perf_tables.append(ReportTable("Gross Revenue Schedule", model.revenue_schedule()))
    except Exception as exc:  # pragma: no cover - defensive guard
        perf_tables.append(ReportTable("Gross Revenue Schedule", [], note=f"Unavailable: {exc}"))

    try:
        perf_tables.append(ReportTable("Total Expenses Schedule", model.cost_structure()))
    except Exception as exc:  # pragma: no cover - defensive guard
        perf_tables.append(ReportTable("Total Expenses Schedule", [], note=f"Unavailable: {exc}"))

    sections.append(ReportSection("Financial Performance", perf_tables))

    break_even_tables = [
        ReportTable("Break-even Analysis", outputs.break_even),
        ReportTable("Payback Schedule", outputs.payback),
        ReportTable("Discounted Payback Schedule", outputs.discounted_payback),
    ]
    sections.append(ReportSection("Break-even & Payback", break_even_tables))

    sections.append(
        ReportSection("Financial Position", [ReportTable("Statement of Financial Position", outputs.balance_sheet)])
    )

    sections.append(
        ReportSection("Cash Flow Statement", [ReportTable("Statement of Cash Flows", outputs.cash_flow)])
    )

    sensitivity_tables: List[ReportTable] = []
    for name, table in outputs.sensitivity_results.items():
        label = name.replace("_", " ").title()
        sensitivity_tables.append(ReportTable(f"Sensitivity: {label}", table))
    if not sensitivity_tables:
        sensitivity_tables.append(ReportTable("Sensitivity Analysis", [], note="No sensitivity configurations provided."))
    sections.append(ReportSection("Sensitivity Analysis", sensitivity_tables))

    scenario_tables: List[ReportTable] = []
    for name, table in outputs.scenario_results.items():
        scenario_tables.append(ReportTable(f"Scenario: {name}", table))
    for key, result in outputs.scenario_tool_results.items():
        label = key.replace("_", " ").title()
        scenario_tables.append(ReportTable(f"Scenario Tool: {label}", result.rows, note=result.interpretation))
    if not scenario_tables:
        scenario_tables.append(ReportTable("Scenario / IFs Analysis", [], note="No scenario configurations provided."))
    sections.append(ReportSection("Scenario / IFs Analysis", scenario_tables))

    sections.append(ReportSection("Monte Carlo Simulation", [ReportTable("Monte Carlo Simulation", outputs.monte_carlo)]))

    return sections


def generate_report(
    sections: Sequence[ReportSection],
    format_name: str,
    *,
    report_name: str = "longevity_financial_report",
) -> Tuple[bytes, str, str]:
    """Generate a consolidated report in the requested format."""

    if not sections:
        raise ReportGenerationError("No sections available to generate a report.")

    fmt = format_name.strip().lower()
    if fmt not in {f.lower() for f in REPORT_FORMATS}:
        raise ReportGenerationError(f"Unsupported report format: {format_name}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    if fmt == "json":
        data = _build_json(sections)
        filename = f"{report_name}_{timestamp}.json"
        return data, JSON_MIME, filename

    if fmt == "csv":
        data = _build_csv(sections)
        filename = f"{report_name}_{timestamp}.csv"
        return data, CSV_MIME, filename

    if fmt == "excel":
        data = _build_excel(sections)
        filename = f"{report_name}_{timestamp}.xlsx"
        return data, EXCEL_MIME, filename

    if fmt == "word":
        data = _build_word(sections)
        filename = f"{report_name}_{timestamp}.docx"
        return data, WORD_MIME, filename

    if fmt == "pdf":
        data = _build_pdf(sections)
        filename = f"{report_name}_{timestamp}.pdf"
        return data, PDF_MIME, filename

    raise ReportGenerationError(f"Unsupported report format: {format_name}")


# ---------------------------------------------------------------------------
# format builders
# ---------------------------------------------------------------------------

def _build_json(sections: Sequence[ReportSection]) -> bytes:
    payload: List[MutableMapping[str, Any]] = []
    for section in sections:
        entry: MutableMapping[str, Any] = {
            "title": section.title,
            "tables": [],
        }
        if section.notes:
            entry["notes"] = list(section.notes)
        for table in section.tables:
            entry["tables"].append(
                {
                    "title": table.title,
                    "rows": _table_to_rows(table.data),
                    **({"note": table.note} if table.note else {}),
                }
            )
        payload.append(entry)
    return json.dumps({"generated": datetime.now(UTC).isoformat(), "sections": payload}, indent=2).encode("utf-8")


def _build_csv(sections: Sequence[ReportSection]) -> bytes:
    buffer = StringIO()
    for section in sections:
        buffer.write(f"# Section: {section.title}\n")
        if section.notes:
            for note in section.notes:
                buffer.write(f"# Note: {note}\n")
        for table in section.tables:
            buffer.write(f"## Table: {table.title}\n")
            if table.note:
                buffer.write(f"## Note: {table.note}\n")
            rows = _table_to_rows(table.data)
            if not rows:
                buffer.write("(no data)\n\n")
                continue
            columns = list(rows[0].keys())
            buffer.write(",".join(_escape_csv(value) for value in columns) + "\n")
            for row in rows:
                buffer.write(",".join(_escape_csv(row.get(column, "")) for column in columns) + "\n")
            buffer.write("\n")
        buffer.write("\n")
    return buffer.getvalue().encode("utf-8")


def _build_excel(sections: Sequence[ReportSection]) -> bytes:
    if pd is None:
        raise ReportGenerationError("Excel export requires pandas to be installed.")

    buffer = BytesIO()
    try:
        with pd.ExcelWriter(buffer) as writer:  # type: ignore[arg-type]
            sheet_tracker: dict[str, int] = {}
            for section in sections:
                for table in section.tables:
                    frame = _table_to_dataframe(table.data)
                    if frame is None:
                        frame = pd.DataFrame()
                    frame = frame.copy()
                    if table.note:
                        note_col = "Notes"
                        if frame.empty:
                            frame = pd.DataFrame({note_col: [table.note]})
                        else:
                            if note_col not in frame.columns:
                                frame[note_col] = ""
                            frame.iloc[0, frame.columns.get_loc(note_col)] = table.note
                    sheet_name = _sheet_name(section.title, table.title, sheet_tracker)
                    frame.to_excel(writer, sheet_name=sheet_name, index=False)
                if section.notes:
                    note_sheet = _sheet_name(section.title, "Notes", sheet_tracker)
                    pd.DataFrame([[note] for note in section.notes], columns=["Notes"]).to_excel(
                        writer, sheet_name=note_sheet, index=False
                    )
    except ValueError as exc:  # pragma: no cover - raised when engine missing
        raise ReportGenerationError(
            "Excel export requires pandas with an Excel writer engine such as 'openpyxl' or 'xlsxwriter'."
        ) from exc

    return buffer.getvalue()


def _build_word(sections: Sequence[ReportSection]) -> bytes:
    if Document is None:
        raise ReportGenerationError("Word export requires the 'python-docx' package.")

    document = Document()
    document.add_heading("Longevity Pharmaceuticals Financial Report", level=1)
    document.add_paragraph(f"Generated on {datetime.now(UTC).isoformat()} UTC")

    for section in sections:
        document.add_heading(section.title, level=2)
        if section.notes:
            for note in section.notes:
                document.add_paragraph(note, style="Intense Quote")
        for table in section.tables:
            document.add_heading(table.title, level=3)
            if table.note:
                document.add_paragraph(table.note, style="Emphasis")
            rows = _table_to_rows(table.data)
            if not rows:
                document.add_paragraph("No data available.")
                continue
            columns = list(rows[0].keys())
            table_obj = document.add_table(rows=len(rows) + 1, cols=len(columns))
            hdr_cells = table_obj.rows[0].cells
            for idx, column in enumerate(columns):
                hdr_cells[idx].text = str(column)
            for row_idx, row_data in enumerate(rows, start=1):
                row_cells = table_obj.rows[row_idx].cells
                for col_idx, column in enumerate(columns):
                    row_cells[col_idx].text = _stringify(row_data.get(column, ""))

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _build_pdf(sections: Sequence[ReportSection]) -> bytes:
    if FPDF is None:
        raise ReportGenerationError("PDF export requires the 'fpdf' package.")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Longevity Pharmaceuticals Financial Report", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Generated on {datetime.now(UTC).isoformat()} UTC", ln=True)

    for section in sections:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, section.title, ln=True)
        pdf.set_font("Arial", "", 10)
        if section.notes:
            for note in section.notes:
                _pdf_multiline(pdf, f"Note: {note}")
        for table in section.tables:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 7, table.title, ln=True)
            pdf.set_font("Arial", "", 10)
            if table.note:
                _pdf_multiline(pdf, f"Note: {table.note}")
            rows = _table_to_rows(table.data)
            if not rows:
                _pdf_multiline(pdf, "No data available.")
                continue
            columns = list(rows[0].keys())
            header = " | ".join(columns)
            _pdf_multiline(pdf, header, bold=True)
            for row in rows:
                values = [
                    _stringify(row.get(column, ""))
                    for column in columns
                ]
                _pdf_multiline(pdf, " | ".join(values))

    return bytes(pdf.output(dest="S").encode("latin-1"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _table_to_rows(data: Any) -> List[Mapping[str, Any]]:
    if data is None:
        return []
    if isinstance(data, list):
        if data and isinstance(data[0], Mapping):
            return [dict(row) for row in data]  # shallow copy
        return [
            {"Value": item}
            for item in data
        ]
    if isinstance(data, Table):
        mapping = data.as_dict()
        rows: List[MutableMapping[str, Any]] = []
        for position, idx in enumerate(data.index):
            row: MutableMapping[str, Any] = {data.index_name: idx}
            for column, values in mapping.items():
                row[column] = values[position]
            rows.append(row)
        return rows
    if pd is not None and isinstance(data, pd.DataFrame):
        frame = data.reset_index()
        return frame.to_dict(orient="records")
    if isinstance(data, Mapping):
        return [
            {"Metric": key, "Value": value}
            for key, value in data.items()
        ]
    try:
        return json.loads(json.dumps(data))
    except Exception:
        return [{"Value": _stringify(data)}]


def _table_to_dataframe(data: Any):
    if pd is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, Table):
        return data.to_frame()
    rows = _table_to_rows(data)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _escape_csv(value: Any) -> str:
    text = _stringify(value)
    if any(char in text for char in [",", "\n", '"']):
        text = text.replace('"', '""')
        return f'"{text}"'
    return text


def _sheet_name(section: str, table: str, tracker: MutableMapping[str, int]) -> str:
    base = f"{section} {table}".strip() or "Sheet"
    base = re.sub(r"[^A-Za-z0-9 ]", "", base)
    base = re.sub(r"\s+", " ", base).strip()
    if not base:
        base = "Sheet"
    base = base[:31]
    count = tracker.get(base, 0)
    if count:
        suffix = f"_{count}"
        base = f"{base[:31-len(suffix)]}{suffix}"
    tracker[base] = count + 1
    return base


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"
    return str(value)


def _pdf_multiline(pdf: "FPDF", text: str, *, bold: bool = False) -> None:
    if bold:
        pdf.set_font("Arial", "B", 10)
    else:
        pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, text)
    if bold:
        pdf.set_font("Arial", "", 10)


__all__ = [
    "REPORT_FORMATS",
    "ReportGenerationError",
    "ReportSection",
    "ReportTable",
    "collect_report_sections",
    "generate_report",
]
