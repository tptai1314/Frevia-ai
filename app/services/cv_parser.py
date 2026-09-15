from io import BytesIO

from docx import Document
from pypdf import PdfReader


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages).strip()


def extract_docx_text(file_bytes: bytes) -> str:
    document = Document(BytesIO(file_bytes))

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


def extract_cv_text(
    file_bytes: bytes,
    filename: str,
) -> str:
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        return extract_pdf_text(file_bytes)

    if filename_lower.endswith(".docx"):
        return extract_docx_text(file_bytes)

    raise ValueError("Only PDF and DOCX files are supported.")