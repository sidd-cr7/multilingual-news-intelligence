import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.services.analysis_service import analyse_article
from app.services.language_service import LanguageDetector

client = TestClient(app)


def test_analysis_evidence_is_an_exact_original_span() -> None:
    text = "Dr. Maya Rao said the shocking crisis was caused by poor planning on 12 March 2025."
    settings = Settings(min_text_chars=10)
    result = analyse_article(text, settings, LanguageDetector(settings))
    assert result.indicators
    assert all(result.original_text[item.start:item.end] == item.quote for item in result.indicators)
    assert result.language.code == "en"


@pytest.mark.parametrize(("text", "code"), [
    ("यह एक संकट पर पर्याप्त लंबा समाचार लेख है जिसमें जिम्मेदार लोगों का उल्लेख है।", "hi"),
    ("ಇದು ಸಂಕಟದ ಬಗ್ಗೆ ಸಾಕಷ್ಟು ಉದ್ದದ ಸುದ್ದಿ ಲೇಖನವಾಗಿದೆ ಮತ್ತು ಕಾರಣವನ್ನು ವಿವರಿಸುತ್ತದೆ.", "kn"),
    ("ఇది సంక్షోభం గురించి తగినంత పొడవైన వార్తా కథనం మరియు కారణాన్ని వివరిస్తుంది.", "te"),
    ("இது நெருக்கடி பற்றிய போதுமான நீளமான செய்திக் கட்டுரை மற்றும் காரணத்தை விளக்குகிறது.", "ta"),
])
def test_indic_original_text_and_offsets_are_preserved(text: str, code: str) -> None:
    settings = Settings(min_text_chars=10)
    result = analyse_article(text, settings, LanguageDetector(settings))
    assert result.language.code == code
    assert result.original_text == text
    assert all(result.original_text[x.start:x.end] == x.quote for x in result.indicators)


def test_api_analysis_contract_and_empty_input() -> None:
    response = client.post("/api/analyze/article", json={"text": "The official said the shocking crisis caused harm to residents."})
    assert response.status_code == 200
    body = response.json()
    assert body["original_text"] and body["task_availability"]
    assert all(body["original_text"][x["start"]:x["end"]] == x["quote"] for x in body["indicators"])
    assert client.post("/api/analyze/article", json={"text": ""}).status_code == 422


def test_analysis_respects_text_limit() -> None:
    settings = Settings(min_text_chars=10, max_text_chars=30)
    with pytest.raises(Exception):
        analyse_article("This deliberately exceeds the configured character limit.", settings, LanguageDetector(settings))
