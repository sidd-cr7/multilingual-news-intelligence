from app.config import Settings
from app.schemas.article import ComparisonSettings
from app.services.analysis_service import analyse_article
from app.services.comparison_service import build_queries, parse_gdelt, parse_google_rss, verify_event
from app.services.language_service import LanguageDetector


def original():
    settings = Settings(min_text_chars=10)
    return analyse_article("Minister Maya Rao said flooding caused a crisis in Delhi on 12 March 2025.", settings, LanguageDetector(settings))


def test_query_generation_omits_original_language() -> None:
    queries = build_queries(original(), ComparisonSettings(target_languages=["en", "hi", "ta"]))
    assert "en" not in queries and {"hi", "ta"} <= queries.keys()
    assert "Maya Rao" in queries["hi"]


def test_provider_parsers_and_duplicate_safe_fields() -> None:
    assert parse_gdelt({"articles": [{"url": "https://example.com/a", "title": "Flood report", "domain": "example.com"}]})[0]["title"] == "Flood report"
    xml = "<rss><channel><item><title>Flood report</title><link>https://example.com/b</link><source>Example</source></item></channel></rss>"
    assert parse_google_rss(xml)[0]["source_name"] == "Example"
    assert parse_google_rss("<invalid") == []


def test_same_event_verification_is_transparent() -> None:
    status, score, reasons, uncertainty = verify_event(original(), "Maya Rao on Delhi flooding", "Flooding caused a crisis in Delhi.")
    assert status in {"verified_match", "possible_match"} and score > 0 and reasons and not uncertainty
    status, _, _, uncertainty = verify_event(original(), "Unrelated sports update", None)
    assert status == "insufficient_evidence" and uncertainty
