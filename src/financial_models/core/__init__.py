"""Shared utilities (AI, table, reporting) used across models."""

from .ai import AIInsights, MachineLearningAdvisor, GenerativeAdvisor
from .table import Table, build_table
from .report import (
    REPORT_FORMATS,
    ReportGenerationError,
    ReportSection,
    ReportTable,
    collect_report_sections,
    collect_biotech_report_sections,
    generate_report,
)

__all__ = [
    "AIInsights",
    "MachineLearningAdvisor",
    "GenerativeAdvisor",
    "Table",
    "build_table",
    "REPORT_FORMATS",
    "ReportGenerationError",
    "ReportSection",
    "ReportTable",
    "collect_report_sections",
    "collect_biotech_report_sections",
    "generate_report",
]
