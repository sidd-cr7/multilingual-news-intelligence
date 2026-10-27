import re
from app.config import Settings
from app.services.errors import IngestionError


def normalize_text(raw_text: str, settings: Settings) -> str:
    """Collapse incidental whitespace while retaining paragraph boundaries."""
    paragraphs = [re.sub(r"[ \t]+", " ", paragraph).strip() for paragraph in re.split(r"\n\s*\n", raw_text)]
    text = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
    if not text:
        raise IngestionError("Article text cannot be empty.")
    if len(text) < settings.min_text_chars:
        raise IngestionError(f"Article text must contain at least {settings.min_text_chars} characters.")
    if len(text) > settings.max_text_chars:
        raise IngestionError(f"Article text exceeds the {settings.max_text_chars:,}-character limit.")
    return text


def text_counts(text: str) -> tuple[int, int, int]:
    # A word is a contiguous non-whitespace token; this avoids Latin-script assumptions.
    return len(text), len(re.findall(r"\S+", text, flags=re.UNICODE)), len([p for p in text.split("\n\n") if p.strip()])


def infer_title(text: str, max_length: int = 120) -> str:
    """Return a clearly fallible opening-text title candidate for pasted content."""
    opening = next((part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()), "Untitled article")
    opening = re.sub(r"\s+", " ", opening).strip(" -–—:;")
    # Keep this short enough to read as a candidate while avoiding sentence
    # splitting that mistakes honorifics (for example, "Dr.") for a headline.
    words = opening.split()
    candidate = " ".join(words[:16])
    return candidate[:max_length].rstrip(".,;: ") or "Untitled article"
