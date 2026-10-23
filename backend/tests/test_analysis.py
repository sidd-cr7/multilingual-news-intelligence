import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.schemas.article import EvidenceSpan
from app.services.analysis_service import _validated_evidence, analyse_article
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


def test_phase_three_categories_and_normalised_report() -> None:
    settings = Settings(min_text_chars=10)
    text = "Everyone said the shocking plan may fail. The plan was caused by poor planning."
    result = analyse_article(text, settings, LanguageDetector(settings))
    categories = {item.category for item in result.indicators}
    assert {"attributed_claim", "hedging_or_qualification", "absolutist_or_generalising_language", "unattributed_assertion_candidate"} <= categories
    assert result.statistics["evidence_density_per_1000_tokens"] is not None
    assert result.statistics["indicator_counts"]["attributed_claim"] == 1


def test_repeated_and_overlapping_evidence_remains_valid() -> None:
    settings = Settings(min_text_chars=10)
    text = "The shocking claim was shocking, and everyone may disagree."
    result = analyse_article(text, settings, LanguageDetector(settings))
    assert sum(item.quote == "shocking" for item in result.indicators) == 2
    assert all(0 <= item.start < item.end <= len(result.original_text) for item in result.indicators)
    assert all(result.original_text[item.start:item.end] == item.quote for item in result.indicators)


def test_phase_three_api_keeps_existing_response_sections() -> None:
    response = client.post("/api/analyze/article", json={"text": "Officials said the proposal may affect everyone next year."})
    assert response.status_code == 200
    body = response.json()
    assert {"statistics", "indicators", "sentiment", "task_availability"} <= body.keys()
    assert "evidence_density_per_1000_tokens" in body["statistics"]


@pytest.mark.parametrize(("text", "code"), [
    ("Everyone may disagree with this sufficiently long English report.", "en"),
    ("\u092f\u0939 \u092a\u0930\u094d\u092f\u093e\u092a\u094d\u0924 \u0932\u0902\u092c\u093e \u0939\u093f\u0902\u0926\u0940 \u0938\u092e\u093e\u091a\u093e\u0930 \u0939\u0948 \u0914\u0930 \u0936\u093e\u092f\u0926 \u0938\u092d\u0940 \u0938\u0939\u092e\u0924 \u0939\u094b\u0902\u0964", "hi"),
    ("\u0c87\u0ca6\u0cc1 \u0cb8\u0cbe\u0c95\u0cb7\u0ccd\u0c9f\u0cc1 \u0c89\u0ca6\u0ccd\u0ca6\u0cb5\u0cbe\u0ca6 \u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0cb8\u0cc1\u0ca6\u0ccd\u0ca6\u0cbf \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0c86\u0c97\u0cb2\u0cb9\u0cc1\u0ca6\u0cc1.", "kn"),
    ("\u0c07\u0c26\u0c3f \u0c24\u0c17\u0c3f\u0c28\u0c02\u0c24 \u0c2a\u0c4a\u0c21\u0c35\u0c48\u0c28 \u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 \u0c35\u0c3e\u0c30\u0c4d\u0c24 \u0c15\u0c25\u0c28\u0c02 \u0c2e\u0c30\u0c3f\u0c2f\u0c41 \u0c05\u0c35\u0c15\u0c3e\u0c36\u0c02 \u0c05\u0c02\u0c26\u0c30\u0c3f\u0c15\u0c40 \u0c05\u0c2a\u0c4d\u0c2a\u0c41\u0ca1\u0c42.", "te"),
    ("\u0b87\u0ba4\u0bc1 \u0baa\u0bcb\u0ba4\u0bc1\u0bae\u0bbe\u0ba9 \u0ba8\u0bc0\u0bb3\u0bae\u0bbe\u0ba9 \u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0b9a\u0bc6\u0baf\u0bcd\u0ba4\u0bbf \u0b85\u0bb1\u0bbf\u0b95\u0bcd\u0b95\u0bc8 \u0bb5\u0bbf\u0bb5\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bbf\u0bb1\u0ba4\u0bc1.", "ta"),
])
def test_phase_three_availability_covers_target_languages(text: str, code: str) -> None:
    settings = Settings(min_text_chars=10)
    result = analyse_article(text, settings, LanguageDetector(settings))
    assert result.language.code == code
    assert any(task.task == "bias_indicator_engine" and task.status == "available" for task in result.task_availability)


def test_invalid_evidence_is_rejected() -> None:
    invalid = EvidenceSpan(category="test", quote="wrong", start=0, end=5, context="", explanation="", method="rule-based")
    with pytest.raises(ValueError, match="invalid evidence"):
        _validated_evidence("right", [invalid])
