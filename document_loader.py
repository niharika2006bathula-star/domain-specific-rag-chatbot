from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

MAX_PDF_SIZE_MB = 20
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120


@dataclass
class DocumentChunk:
    text: str
    source: str
    page_number: int
    chunk_index: int = 0


def validate_pdf_upload(uploaded_file: Any, max_size_mb: int = MAX_PDF_SIZE_MB) -> None:
    """Validate a PDF upload before processing."""
    name = getattr(uploaded_file, "name", "") or ""
    if not name.lower().endswith(".pdf"):
        raise ValueError(f"Unsupported file type for '{name}'. Please upload a PDF file.")

    size = getattr(uploaded_file, "size", 0) or 0
    if size == 0:
        raise ValueError(f"The file '{name}' is empty.")

    if size > max_size_mb * 1024 * 1024:
        raise ValueError(f"The file '{name}' exceeds the {max_size_mb} MB limit.")


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    """Extract page text from a PDF while preserving per-page metadata."""
    if not file_bytes:
        raise ValueError(f"The file '{filename}' is empty and cannot be processed.")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not read '{filename}'. The PDF may be corrupted or unreadable.") from exc

    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(f"The PDF '{filename}' is encrypted and cannot be read without a password.") from exc
        if decrypted == 0:
            raise ValueError(f"The PDF '{filename}' is encrypted and cannot be read without a password.")

    extracted_pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        cleaned_text = " ".join(page_text.split())
        if not cleaned_text:
            continue

        extracted_pages.append(
            {
                "source": filename,
                "page_number": page_number,
                "text": page_text.strip(),
            }
        )

    if not extracted_pages:
        raise ValueError(
            f"No extractable text was found in '{filename}'. OCR or a text-based PDF may be required."
        )

    return extracted_pages


def split_text_into_chunks(
    pages: Iterable[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split page text into chunks while preserving source and page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    document_chunks: list[DocumentChunk] = []
    for page in pages:
        page_text = page.get("text", "")
        if not page_text or not page_text.strip():
            continue

        chunks = splitter.split_text(page_text)
        for chunk_index, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            document_chunks.append(
                DocumentChunk(
                    text=chunk.strip(),
                    source=page.get("source", "unknown.pdf"),
                    page_number=int(page.get("page_number", 1)),
                    chunk_index=chunk_index,
                )
            )

    return document_chunks


def load_documents_from_uploads(uploaded_files: List[Any], max_size_mb: int = MAX_PDF_SIZE_MB) -> tuple[list[DocumentChunk], dict[str, Any]]:
    """Process a list of uploaded PDF files and return chunks plus summary metadata."""
    all_chunks: list[DocumentChunk] = []
    summary = {
        "processed_pdfs": 0,
        "processed_pages": 0,
        "chunks": 0,
        "errors": [],
    }

    if not uploaded_files:
        return all_chunks, summary

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "") or "uploaded.pdf"
        try:
            validate_pdf_upload(uploaded_file, max_size_mb=max_size_mb)
            file_bytes = uploaded_file.read()
            pages = extract_text_from_pdf(file_bytes, filename)
            summary["processed_pdfs"] += 1
            summary["processed_pages"] += len(pages)
            chunks = split_text_into_chunks(pages)
            all_chunks.extend(chunks)
            summary["chunks"] = len(all_chunks)
        except Exception as exc:  # noqa: BLE001 - defensive UI handling
            summary["errors"].append(f"{filename}: {exc}")

    return all_chunks, summary
