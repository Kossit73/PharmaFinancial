"""Input-loading and RAG text processing helpers."""

from __future__ import annotations

import csv
import io
import json
from typing import List, Mapping


def load_payload_from_bytes(data: bytes, suffix: str) -> Mapping[str, object]:
    suffix = suffix or ".json"
    if suffix in {".json", ""}:
        return load_payload_from_text(data.decode("utf-8"))
    if suffix == ".csv":
        return load_payload_from_csv(data)
    if suffix in {".xlsx", ".xls"}:
        return load_payload_from_excel(data)
    if suffix == ".docx":
        return load_payload_from_docx(data)
    if suffix == ".pdf":
        return load_payload_from_pdf(data)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_payload_from_text(text: str) -> Mapping[str, object]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Uploaded file was empty.")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("Uploaded document does not contain valid JSON assumptions.") from exc


def load_payload_from_csv(data: bytes) -> Mapping[str, object]:
    from .. import app as legacy

    text = data.decode("utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fragment = legacy._extract_json_fragment(text)
        if fragment and fragment != text:
            return load_payload_from_text(fragment)

        reader = csv.reader(io.StringIO(text))
        cells: list[str] = []
        for row in reader:
            cells.extend(cell for cell in row if cell is not None)
        joined = legacy._extract_json_fragment("".join(cells).strip())
        if not joined:
            raise ValueError("CSV file did not contain any usable JSON text.")
        return load_payload_from_text(joined)


def load_payload_from_excel(data: bytes) -> Mapping[str, object]:
    from .. import app as legacy

    if legacy.load_workbook is None:
        raise ValueError("Excel support requires the 'openpyxl' package to be installed.")

    workbook = legacy.load_workbook(filename=io.BytesIO(data), data_only=True)
    text_parts: list[str] = []
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                value = cell.value
                if value is None:
                    continue
                text_parts.append(str(value))
    combined = legacy._extract_json_fragment("\n".join(text_parts).strip())
    if not combined:
        raise ValueError("Excel file did not contain any readable text.")
    return load_payload_from_text(combined)


def load_payload_from_docx(data: bytes) -> Mapping[str, object]:
    from .. import app as legacy

    if legacy.Document is None:
        raise ValueError("Word support requires the 'python-docx' package to be installed.")

    document = legacy.Document(io.BytesIO(data))
    text = legacy._extract_json_fragment(
        "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
    )
    if not text:
        raise ValueError("Word document did not contain any readable text.")
    return load_payload_from_text(text)


def load_payload_from_pdf(data: bytes) -> Mapping[str, object]:
    from .. import app as legacy

    if legacy.PdfReader is None:
        raise ValueError("PDF support requires the 'PyPDF2' package to be installed.")

    reader = legacy.PdfReader(io.BytesIO(data))
    text_parts: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        text_parts.append(extracted)
    combined = legacy._extract_json_fragment("\n".join(text_parts).strip())
    if not combined:
        raise ValueError("PDF file did not contain any readable text.")
    return load_payload_from_text(combined)


def extract_text_from_upload(filename: str, data: bytes) -> str:
    from .. import app as legacy

    suffix = legacy.Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return data.decode("utf-8", errors="replace")
    if suffix == ".docx":
        if legacy.Document is None:
            raise ValueError("Word support requires the 'python-docx' package to be installed.")
        document = legacy.Document(io.BytesIO(data))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
        if not text:
            raise ValueError("Word document did not contain any readable text.")
        return text
    if suffix == ".pdf":
        if legacy.PdfReader is None:
            raise ValueError("PDF support requires the 'PyPDF2' package to be installed.")
        reader = legacy.PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            raise ValueError("PDF file did not contain any readable text.")
        return text
    raise ValueError("Unsupported file type for RAG. Upload TXT, MD, CSV, DOCX, or PDF files.")


def build_rag_chunks(documents: List[Mapping[str, str]]) -> List[Mapping[str, object]]:
    from .. import app as legacy

    chunks: List[Mapping[str, object]] = []
    for doc in documents:
        name = str(doc.get("name", "Document"))
        text = str(doc.get("text", "") or "").strip()
        for chunk in legacy._chunk_text(
            text,
            size=legacy.RAG_CHUNK_SIZE,
            overlap=legacy.RAG_CHUNK_OVERLAP,
        ):
            tokens = legacy._tokenize(chunk)
            chunks.append({"source": name, "text": chunk, "tokens": tokens})
    return chunks


def score_chunks(query: str, chunks: List[Mapping[str, object]]) -> List[Mapping[str, object]]:
    from .. import app as legacy

    query_tokens = set(legacy._tokenize(query))
    if not query_tokens:
        return []
    scored: List[Mapping[str, object]] = []
    for chunk in chunks:
        tokens = chunk.get("tokens", [])
        overlap = sum(1 for token in tokens if token in query_tokens)
        if overlap == 0:
            continue
        score = overlap / max(len(tokens), 1)
        scored.append({**chunk, "score": score})
    scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return scored[: legacy.RAG_TOP_RESULTS]

