from datetime import datetime, timezone
from uuid import uuid4
from app.config import Settings
from app.schemas.article import ArticleMetadata, InputMethod
from app.services.language_service import LanguageDetector
from app.services.text_service import normalize_text, text_counts


def create_analysis(text: str, input_method: InputMethod, settings: Settings, detector: LanguageDetector, title: str | None = None, original_url: str | None = None, warnings: list[str] | None = None) -> ArticleMetadata:
    normalized = normalize_text(text, settings)
    characters, words, paragraphs = text_counts(normalized)
    now = datetime.now(timezone.utc)
    return ArticleMetadata(analysis_id=str(uuid4()), input_method=input_method, original_url=original_url, title=title, language=detector.detect(normalized), extracted_text=normalized, character_count=characters, word_count=words, paragraph_count=paragraphs, warnings=warnings or [], created_at=now, processed_at=now)
