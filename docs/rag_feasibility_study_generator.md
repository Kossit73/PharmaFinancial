<?md
# RAG Feasibility Study Generator

A production-ready blueprint (plus reference code) for a Retrieval-Augmented Generation (RAG) system that ingests up to 1 GB of project materials and automatically drafts a comprehensive feasibility study grounded in your financial model outputs and accompanying documents.

This document includes:
1) Architecture & design choices
2) Data schema & parsing of financial model (Excel)
3) Prompt strategy & section templates
4) Reference implementation (FastAPI + FAISS + Sentence-Transformers, optional reranker)
5) Quality, auditing & reproducibility
6) Deployment notes (including handling 1 GB uploads)

---

## 0) RAC: Model-Integrated Design (RAG inside the Financial Model)

### What changed
We now treat the Excel workbook as the system of record and orchestrator. The Retrieval–Aggregation–Composer (RAC) service is triggered from the model (button/macros/Office Script/xlwings) and does two things:
1) Collects results directly from defined cells/ranges in the model (NPV/IRR/DSCR, capex/opex, scenarios, sensitivities, flags) and writes a structured `financial_snapshot`.
2) Ingests and retrieves external evidence from up to 1 GB of uploaded sources (PDF/DOCX/PPTX/CSV/TXT) to ground narrative sections with citations.

### Why this pattern
- Ensures single source of truth for numbers (no drift between the model and the report).
- Reduces user steps: analysts stay in Excel; one Generate Feasibility Study action runs end-to-end.
- Enables repeatable runs: report always tied to a workbook hash + timestamp.

### RAC Components
1) **Workbook Adapter (in‑workbook)**
   - Implementation options: Office Scripts (Excel on web), VBA macro, or xlwings (Python bridge).
   - Provides a mapping config of named ranges/cell addresses → semantic keys (npv, irr, dscr_min, etc.).
   - Exposes a Generate Report control that calls the RAC API with the extracted snapshot.
2) **Results Collector (API)**
   - Validates workbook hash; normalizes units/currency; runs sanity checks (e.g., DSCR > 0, IRR ∈ [−1, 1]).
   - Saves `projects/<id>/financial/snapshot.json` with lineage: `{workbook_hash, as_of, model_version, cell_map}`.
3) **Evidence Ingestor (API)**
   - Streams large uploads to disk; parses & chunks; embeds; stores in FAISS with rich metadata.
   - Optional OCR pass for scanned PDFs (add Tesseract layer if needed).
4) **Retriever + Reranker**
   - Section‑specific queries (Market, Technical, Legal, etc.) pull top‑k passages; cross‑encoder reranks.
5) **Composer**
   - Section templates that inject the `financial_snapshot` + retrieved passages; forces inline citations; flags gaps.
6) **Auditor**
   - Produces a provenance table (file → hash → cited pages/sheets) and reprints key model outputs in the Appendix.

### End‑to‑end flow (in workbook)
1. Analyst updates assumptions → workbook computes.
2. Click Generate Feasibility Study.
3. Workbook Adapter reads named ranges/rules → posts to `/collect` with `financial_snapshot + cell_map`.
4. Evidence Ingestor already holds uploads (or accepts them in the same action up to 1 GB).
5. Retriever + Composer generate each section → returns `report.md` & `report.json` to the workbook (or a download link).

### API shape (revised)
- `POST /collect` – accepts `{project_id, financial_snapshot, cell_map, workbook_hash}`.
- `POST /ingest` – streamed uploads for external sources.
- `POST /generate` – accepts `{project_id, section_outline?}`; uses provided `financial_snapshot` (does not guess).

---

## 1) High-level Architecture

### Goals
- Upload up to 1 GB of files (PDF, DOCX, XLSX/CSV, PPTX, text) without exhausting memory.
- Parse the financial model (Excel-based) to extract key outputs (NPV, IRR, DSCR, payback, capex/opex tables, sensitivities, scenarios).
- Build a local vector store (FAISS) for RAG retrieval with metadata filters (file, section, page).
- Assemble a full feasibility study (Executive Summary → Market → Technical → Implementation → Financial → Risk/ESG → Conclusion) with verbatim citations and appendices.

### Pipeline
1. **Upload & Ingest:** Stream large files to disk → parse text (per type) → chunk (token-aware) → embed → store in FAISS; persist metadata (SQLite/JSONL).
2. **Financial Model Extraction:** Load Excel → extract standardized metrics/tables via a schema (configurable sheet/label mapping).
3. **Retrieval:** Hybrid (BM25 + dense) optional; here we ship dense + MMR + cross‑encoder rerank.
4. **Planning:** Build section-by-section plan informed by model metrics.
5. **Generation:** For each section, craft grounded prompts with citations from retrieved chunks + structured financial bullets.
6. **Audit:** Attach a sources table and financial snapshot (as-of timestamp, workbook hash) for reproducibility.

### Core tech (reference)
- FastAPI for API server
- Sentence-Transformers for embeddings (e.g., `all-MiniLM-L6-v2` or `bge-base`)
- FAISS vector store (local)
- Cross-Encoder for reranking (optional)
- pandas / openpyxl for Excel parsing
- pypdf, python-docx, pptx for document extraction
- tiktoken or a simple tokenizer for chunk sizing

You can swap in Pinecone/Qdrant for the vector DB, or Azure/OpenAI/Anthropic for the LLM. The design below abstracts the LLM client.

---

## 2) Data Model & Financial Schema

### Project store
- `projects/<project_id>/uploads/` – raw files (streamed)
- `projects/<project_id>/parsed/` – extracted text, JSON for tables
- `projects/<project_id>/index/` – FAISS index + meta.jsonl
- `projects/<project_id>/financial/` – snapshot.json

### Metadata per chunk
```json
{
  "project_id": "string",
  "file_path": "string",
  "file_type": "pdf|docx|xlsx|csv|pptx|txt",
  "page_or_sheet": "number|string",
  "section": "optional heading",
  "char_start": 0,
  "char_end": 1024,
  "hash": "sha256 of source file"
}
```

### Financial snapshot (`projects/<id>/financial/snapshot.json`)
```json
{
  "as_of": "2025-11-17T08:00:00Z",
  "workbook_path": ".../model.xlsx",
  "workbook_hash": "sha256",
  "currency": "USD",
  "assumptions": {
    "discount_rate": 0.12,
    "inflation": 0.03,
    "tax_rate": 0.28
  },
  "capex_total": 120000000,
  "opex_annual": 8500000,
  "revenue_annual": 23500000,
  "npv": 54000000,
  "irr": 0.19,
  "payback_years": 5.6,
  "dscr_min": 1.35,
  "sensitivities": [
    {"variable": "price", "delta": 0.1, "npv": 60000000, "irr": 0.205},
    {"variable": "price", "delta": -0.1, "npv": 48000000, "irr": 0.175}
  ],
  "scenarios": [
    {"name": "Base", "npv": 54000000, "irr": 0.19},
    {"name": "Downside", "npv": 30000000, "irr": 0.14},
    {"name": "Upside", "npv": 78000000, "irr": 0.24}
  ]
}
```

### Excel parsing strategy
- Config-driven mapping of sheet names and anchor labels (e.g., lookup tables for NPV/IRR/DSCR).
- Fallback heuristics: scan for keywords in first column, regex for %, IRR, NPV, DSCR.
- Preserve units & currency, record workbook hash and sheet cell addresses for traceability.

---

## 3) Prompt Strategy & Section Templates

### System prompt (global)
> You are a financial analyst producing a feasibility study. Only use facts from provided CONTEXT and FINANCIAL_SNAPSHOT. Cite sources inline like [Source: filename p.12] or [Sheet: Assumptions!B7]. If a claim is unsupported, say so. Keep each section structured, concise, and decision-oriented.

### Section outline
1. Executive Summary
2. Project Description & Scope
3. Market & Demand Analysis
4. Technical & Operations
5. Legal, Permitting & Environmental
6. Implementation Plan (Schedule, Procurement, Organization)
7. Financial Analysis (NPV/IRR/DSCR, sensitivities, scenarios)
8. Risk Assessment & Mitigations (incl. ESG)
9. Conclusion & Recommendation
10. Appendices (Detailed assumptions, tables, source list)

### Section prompt template (per section)
```
[GOAL]
Draft the <SECTION_NAME> for the feasibility study.

[GUIDANCE]
- Use FINANCIAL_SNAPSHOT metrics explicitly where relevant.
- Use CONTEXT passages with inline citations [Source: <file> p.<n>] or [Sheet: <name>!<cell>].
- State uncertainties and missing data.
- Avoid boilerplate; keep it specific to the project.

[FINANCIAL_SNAPSHOT]
{{structured bullets}}

[CONTEXT]
{{top_k passages with metadata}}

[OUTPUT]
- 3–7 well-structured paragraphs
- Subheadings
- Bullet lists for key metrics & risks
- Citations inline
```

### Citation style
- Text sources: `[Source: filename p.12]`
- Excel: `[Sheet: SheetName!CellRef]`

---

## 4) Reference Implementation (FastAPI + FAISS + Sentence-Transformers)

See the reference implementation at:
- `tools/rag/app.py` (single-file API example)

---

## 5) Quality, Auditing & Reproducibility

**Grounding & Citations**
- Retrieval → rerank → limit to top 3–5 rich chunks per section.
- Enforce inline citations in prompts; reject hallucinations (tell the model to say “unknown”).

**Determinism (as much as possible)**
- Fix EMBED_MODEL, RERANK_MODEL versions.
- Record workbook_hash and as_of timestamp.

**Validation checks**
- Numeric sanity: Ensure IRR ∈ [−100%, +100%], NPV units consistent with currency, DSCR > 0.
- Cross-check: If DSCR < 1, require red flag in the Financial Analysis section.

**Human-in-the-loop**
- Return report.md for redlining.
- Include report.json for structured review or BI export.

---

## 6) Handling 1 GB Uploads (Important)

**FastAPI/Starlette**
- The code uses streamed writes (UploadFile.read in chunks) to avoid large memory spikes.
- Starlette itself does not enforce a body-size limit; your reverse proxy typically does.

**Recommended Nginx front**
```
client_max_body_size 1024m;
proxy_request_buffering off;
proxy_buffering off;
```

**Uvicorn/Workers**
- Run multiple workers for concurrency: `uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4`.
- Use a fast disk (NVMe) for projects/ storage.

**Persistence & Backups**
- Persist FAISS index and meta logs in `projects/<id>/index/`.
- Regularly snapshot to object storage if desired.

---

## 7) Extending the Blueprint

- Hybrid retrieval: Add BM25 (e.g., rank_bm25) and fuse scores with dense.
- Structured table extraction: Parse key tables to JSON (capex breakdown, ramp-up curves) for richer prompts.
- Section-specific retrievers: Use specialized queries per section to fetch more targeted context.
- Guardrails: Regex check generated claims for numeric mismatches vs snapshot.
- Model choice: Plug your preferred LLM in LLMClient.complete().
- Report styling: Convert report.md to PDF via weasyprint or pandoc.

---

## 8) Section-specific Retrieval Queries (ready-made)

- Executive Summary: “materiality of results, decision drivers, showstoppers”
- Market: “market size, demand forecast, price assumptions, offtake”
- Technical: “process design, throughput, yield, utilities, site layout”
- Legal/Env: “permits, EIA/ESIA, land rights, community”
- Implementation: “schedule, capex phasing, procurement strategy, org”
- Financial: “NPV, IRR, DSCR, payback, sensitivities”
- Risk/ESG: “risk register, mitigation, ESG metrics”

Implement by reusing `vi.search(<specialized-query>)` per section and merging top passages.

---

## 9) Report Composition & Appendices

**Financial Statements and Schedules**
- The generated feasibility study will embed all major financial statements from the model:
  - Income Statement (projected)
  - Balance Sheet
  - Cash Flow Statement (including financing flows)
- Each is drawn directly from the Excel model via the defined cell_map and sheet references.
- Figures are refreshed on every generation cycle and verified against the workbook hash.

**Schedules and Supporting Tables**
- Detailed schedules are auto-exported from model sheets (e.g., Capex, Opex, Revenue, Debt, Depreciation, Sensitivity, Scenario Analysis).
- Each schedule is reproduced in the Appendices of the report for transparency.
- Graphs derived from these schedules (NPV curve, DSCR timeline, cash-flow waterfall, cost breakdown pie) are generated via matplotlib or Excel’s native charts and embedded in the final report.

**Appendices**
- Full reproduction of key model tables and charts.
- Sensitivity matrices (e.g., Price vs IRR, Cost vs NPV).
- Scenario summaries (Base, Downside, Upside) with comparative metrics.
- Audit trail: workbook hash, timestamp, and cell reference map.

---

## 10) Charts & Visualizations — Matplotlib Examples (Drop‑in)

The snippets below are copy‑paste additions to `tools/rag/app.py`. They render charts to `projects/<id>/charts/` and embed them in `report.md`.

### 10.1 Utilities
```python
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def _ensure_chart_dir(project_dir: str) -> str:
    chart_dir = os.path.join(project_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    return chart_dir

def _fmt_currency(curr: str):
    def _fmt(x, pos):
        try:
            return f"{curr} {x:,.0f}"
        except Exception:
            return f"{x:,.0f}"
    return FuncFormatter(_fmt)
```

### 10.2 NPV Curve (by Scenario or Sensitivity)
```python
def plot_npv_curve(financial: dict, out_path: str):
    # Uses scenarios if present; otherwise derives from sensitivities with variable=="price"
    xs, ys = [], []
    if financial.get("scenarios"):
        xs = [s.get("name", f"S{i+1}") for i, s in enumerate(financial["scenarios"])]
        ys = [float(s.get("npv", 0)) for s in financial["scenarios"]]
    elif financial.get("sensitivities"):
        sens = [s for s in financial["sensitivities"] if s.get("variable") == "price" and s.get("npv") is not None]
        sens = sorted(sens, key=lambda s: s.get("delta", 0))
        xs = [f"{int(s.get('delta', 0) * 100)}%" for s in sens]
        ys = [float(s["npv"]) for s in sens]
    else:
        return None

    fig = plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.title("NPV Curve")
    plt.xlabel("Scenario / Delta")
    plt.ylabel("NPV")
    curr = financial.get("currency", "") or ""
    plt.gca().yaxis.set_major_formatter(_fmt_currency(curr))
    plt.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
```

### 10.3 DSCR Trend (time series read from Excel)
```python
def plot_dscr_trend_from_excel(xlsx_path: str, sheet: str, date_col: str, dscr_col: str, out_path: str):
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if date_col not in df.columns or dscr_col not in df.columns:
        return None
    df = df[[date_col, dscr_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, dscr_col])

    fig = plt.figure()
    plt.plot(df[date_col], df[dscr_col])
    plt.title("DSCR Trend")
    plt.xlabel("Date")
    plt.ylabel("DSCR")
    plt.axhline(1.0, linestyle='--', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
```

### 10.4 Integrate into `/generate`
Add inside the `/generate` handler after loading financial and before stitching markdown:
```python
# === Charts ===
chart_dir = _ensure_chart_dir(base)
npv_png = os.path.join(chart_dir, "npv_curve.png")
dscr_png = os.path.join(chart_dir, "dscr_trend.png")

try:
    plot_npv_curve(financial, npv_png)
except Exception:
    npv_png = None

xlsx_path = financial.get("workbook_path")
try:
    plot_dscr_trend_from_excel(
        xlsx_path, sheet="Debt", date_col="Date", dscr_col="DSCR", out_path=dscr_png
    )
except Exception:
    dscr_png = None
```

Then, when writing `report.md`, embed the images if present.

### 10.5 Optional: Cash Flow Waterfall
```python
def plot_cashflow_waterfall(cf_series: pd.Series, out_path: str, title: str = "Cash Flow Waterfall"):
    running = cf_series.cumsum()
    fig = plt.figure()
    plt.bar(cf_series.index, cf_series.values)
    plt.plot(running.index, running.values, marker='o')
    plt.title(title)
    plt.ylabel("Amount")
    plt.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
```

---

## 11) Security & Privacy

- All data stays local by default. No outbound calls unless you wire an external LLM.
- Hash all source files; keep a manifest for chain‑of‑custody.
- Optional: encrypt projects/ at rest; restrict file permissions.

---

## 12) What you’ll customize for your project

- Excel sheet & label mappings to your specific Pharmaceuticals Financial Model outputs.
- The section outline and tone.
- The LLM provider and model.
- Any industry-specific compliance/ESG frameworks.
```
