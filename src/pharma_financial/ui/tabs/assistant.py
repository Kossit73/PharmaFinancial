"""Knowledge and reporting surface for the redesigned pharma workspace."""

from __future__ import annotations

from ...inputs import ModelInputs
from ...model import FinancialModel, FinancialOutputs
from .. import shell


def render_knowledge_and_reports(
    inputs: ModelInputs,
    model: FinancialModel | None,
    outputs: FinancialOutputs | None,
    digest: str,
) -> None:
    from ... import app as legacy

    del inputs
    if model is None or outputs is None:
        legacy.st.info("Run the model to generate report exports and activate the RAG assistant.")
        return

    shell.render_section_header(
        "Knowledge & Reports",
        "Export workbooks and business-plan packages, then augment the numbers with supporting documents and AI commentary.",
    )
    report_tab, assistant_tab = legacy.st.tabs(["Report Studio", "RAG Assistant"])
    with report_tab:
        legacy._render_excel_model_download(legacy.st.container(), model, outputs)
    with assistant_tab:
        legacy._render_rag_tab(model, outputs, digest)

