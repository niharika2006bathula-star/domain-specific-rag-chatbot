from io import BytesIO

from reportlab.pdfgen import canvas

from document_loader import DocumentChunk, load_documents_from_uploads, split_text_into_chunks


class FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self.size = len(data)
        self._data = data

    def read(self) -> bytes:
        return self._data


def make_pdf_bytes(page_texts):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    for index, text in enumerate(page_texts, start=1):
        pdf.setTitle(f"Page {index}")
        pdf.drawString(72, 720, text)
        if index < len(page_texts):
            pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_extraction_of_text_from_valid_pdf():
    pdf_bytes = make_pdf_bytes(["Alpha page one text.", "Beta page two text."])
    uploaded = FakeUpload("sample.pdf", pdf_bytes)

    chunks, summary = load_documents_from_uploads([uploaded])

    assert summary["processed_pdfs"] == 1
    assert summary["processed_pages"] == 2
    assert len(chunks) > 0
    assert chunks[0].source == "sample.pdf"


def test_preserves_source_filename_and_page_number():
    pdf_bytes = make_pdf_bytes(["First page about policies.", "Second page about procedures."])
    uploaded = FakeUpload("policy_document.pdf", pdf_bytes)

    chunks, _ = load_documents_from_uploads([uploaded])

    sources = [chunk.source for chunk in chunks]
    page_numbers = [chunk.page_number for chunk in chunks]
    assert all(source == "policy_document.pdf" for source in sources)
    assert 1 in page_numbers and 2 in page_numbers


def test_skips_empty_pages():
    pdf_bytes = make_pdf_bytes(["", "Visible content on page two."])
    uploaded = FakeUpload("empty_page.pdf", pdf_bytes)

    chunks, summary = load_documents_from_uploads([uploaded])

    assert summary["processed_pages"] == 1
    assert all(chunk.page_number == 2 for chunk in chunks)


def test_handles_pdf_with_no_extractable_text():
    pdf_bytes = make_pdf_bytes(["", ""])
    uploaded = FakeUpload("blank.pdf", pdf_bytes)

    chunks, summary = load_documents_from_uploads([uploaded])

    assert chunks == []
    assert summary["processed_pdfs"] == 0
    assert summary["errors"]


def test_correct_text_chunking():
    long_text = "This is a long paragraph. " * 50
    pages = [{"source": "chunk_test.pdf", "page_number": 1, "text": long_text}]

    chunks = split_text_into_chunks(pages, chunk_size=900, chunk_overlap=120)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 900 for chunk in chunks)
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 1
