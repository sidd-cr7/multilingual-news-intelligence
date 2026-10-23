import pymupdf as fitz
from app.config import Settings
from app.services.errors import IngestionError


def extract_pdf_text(content: bytes, settings: Settings) -> tuple[str, list[str]]:
    if not content:
        raise IngestionError("The uploaded PDF is empty.")
    if len(content) > settings.max_upload_bytes:
        raise IngestionError(f"PDF exceeds the {settings.max_upload_bytes // 1024 // 1024} MB upload limit.", 413)
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except (fitz.FileDataError, RuntimeError, ValueError) as error:
        raise IngestionError("This file is corrupted or is not a readable PDF.") from error
    try:
        if document.needs_pass:
            raise IngestionError("This PDF is encrypted and cannot be processed in Phase 1.")
        text = "\n\n".join(page.get_text("text") for page in document)
        if not text.strip():
            raise IngestionError("No selectable text was found. This may be a scanned PDF; OCR is not included in Phase 1.")
        return text, []
    finally:
        document.close()
