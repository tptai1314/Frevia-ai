import re
import unicodedata
from io import BytesIO

from docx import Document
from pypdf import PdfReader


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)

    text = text.replace('\u00a0', ' ')

    text = re.sub(r'[ \t]+', ' ', text)

    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ''

        if text.strip():
            pages.append(text)

    return normalize_text('\n'.join(pages))


def extract_docx_text(file_bytes: bytes) -> str:
    document = Document(BytesIO(file_bytes))

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return normalize_text('\n'.join(paragraphs))


def extract_cv_text(file_bytes: bytes, filename: str) -> str:
    filename_lower = filename.lower()

    if filename_lower.endswith('.pdf'):
        return extract_pdf_text(file_bytes)

    if filename_lower.endswith('.docx'):
        return extract_docx_text(file_bytes)

    raise ValueError('Only PDF and DOCX files are supported.')