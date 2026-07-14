"""Knowledge and reporting surface for the redesigned pharma workspace."""

from __future__ import annotations

from ...inputs import ModelInputs
from ...model import FinancialModel, FinancialOutputs
from .. import shell


def _render_table_like(legacy, value: object) -> None:
    frame = legacy._ensure_dataframe(value)
    legacy.st.dataframe(frame, width="stretch")


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
    active_panel = legacy.st.radio(
        "Knowledge panel",
        ["Investor Pack", "Evidence Register", "RAG Assistant"],
        horizontal=True,
        key="pharma_active_knowledge_panel",
        label_visibility="collapsed",
    )
    if active_panel == "Investor Pack":
        legacy.st.markdown("### Investor Pack Preview")
        _render_table_like(legacy, outputs.bankability_gate)
        legacy.st.markdown("### Sources & Uses")
        _render_table_like(legacy, outputs.sources_and_uses)
        legacy._render_excel_model_download(legacy.st.container(), model, outputs)
    if active_panel == "Evidence Register":
        legacy.st.markdown("### Evidence Register")
        _render_table_like(legacy, outputs.evidence_register or [])
        legacy._render_data_quality_exceptions_dashboard(
            outputs.data_quality_exceptions or [],
            caption="Resolve flagged input or financing issues before sharing the exported workbook with investors.",
        )
    if active_panel == "RAG Assistant":
        legacy._render_rag_tab(model, outputs, digest)
