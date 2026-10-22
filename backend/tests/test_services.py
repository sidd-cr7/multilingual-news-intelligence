import fitz
import pytest
from app.config import Settings
from app.services.article_service import create_analysis
from app.services.errors import IngestionError
from app.services.language_service import LanguageDetector
from app.services.pdf_service import extract_pdf_text
from app.services.text_service import normalize_text
from app.services.url_service import validate_public_url
from app.schemas.article import InputMethod


@pytest.fixture
def settings() -> Settings:
    return Settings(min_text_chars=10, max_text_chars=1000)


def test_normalizes_paragraphs(settings: Settings) -> None:
    assert normalize_text("  This   is a valid paragraph.\n\n Another paragraph here. ", settings) == "This is a valid paragraph.\n\nAnother paragraph here."


def test_rejects_short_text(settings: Settings) -> None:
    with pytest.raises(IngestionError, match="at least"):
        normalize_text("short", settings)


@pytest.mark.parametrize(("text", "code"), [
    ("This is a sufficiently long English news report about an event and its impact.", "en"),
    ("यह एक पर्याप्त लंबा हिंदी समाचार लेख है जिसमें घटना और उसके प्रभाव का वर्णन किया गया है।", "hi"),
    ("ಇದು ಘಟನೆ ಮತ್ತು ಅದರ ಪರಿಣಾಮವನ್ನು ವಿವರಿಸುವ ಸಾಕಷ್ಟು ಉದ್ದವಾದ ಕನ್ನಡ ಸುದ್ದಿ ಲೇಖನವಾಗಿದೆ.", "kn"),
    ("ఇది సంఘటన మరియు దాని ప్రభావాన్ని వివరించే తగినంత పొడవైన తెలుగు వార్తా కథనం.", "te"),
    ("இது நிகழ்வு மற்றும் அதன் தாக்கத்தை விவரிக்கும் போதுமான நீளமான தமிழ் செய்திக் கட்டுரை.", "ta"),
])
def test_language_detection(settings: Settings, text: str, code: str) -> None:
    assert LanguageDetector(settings).detect(text).code == code


def test_extracts_text_pdf(settings: Settings) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A real text PDF fixture for extraction testing.")
    content = document.tobytes()
    document.close()
    text, warnings = extract_pdf_text(content, settings)
    assert "real text PDF" in text and warnings == []


def test_rejects_non_pdf(settings: Settings) -> None:
    with pytest.raises(IngestionError):
        extract_pdf_text(b"not a pdf", settings)


def test_blocks_local_url() -> None:
    with pytest.raises(IngestionError, match="Private"):
        validate_public_url("http://127.0.0.1/private")


def test_metadata_generation(settings: Settings) -> None:
    result = create_analysis("This is a sufficiently long English article for metadata generation.", InputMethod.text, settings, LanguageDetector(settings))
    assert result.analysis_id and result.word_count > 0 and result.input_method == InputMethod.text
