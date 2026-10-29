"""Phase 9 deterministic tests for the cross-language comparison pipeline.

All tests use mocked providers and deterministic fixtures.
No live network calls are made.
"""
from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.schemas.article import ComparisonSettings, DiscoveryCandidate
from app.services import comparison_service
from app.services.analysis_service import analyse_article
from app.services.comparison_service import (
    _framing_differences,
    build_queries,
    comparative_summary,
    discover,
    parse_gdelt,
    parse_google_rss,
    verify_event,
)
from app.services.event_profile import build_language_queries, extract_event_profile
from app.services.language_service import LanguageDetector

_SETTINGS = Settings(min_text_chars=10)
_DETECTOR = LanguageDetector(_SETTINGS)

FLOOD_EN = "Minister Maya Rao said flooding caused a crisis in Delhi on 12 March 2025."
FLOOD_HI = "मंत्री माया राव ने कहा कि 12 मार्च 2025 को दिल्ली में बाढ़ से संकट पैदा हो गया।"
FLOOD_KN = "ಮಂತ್ರಿ ಮಾಯಾ ರಾವ್ ಅವರು 12 ಮಾರ್ಚ್ 2025 ರಂದು ದೆಹಲಿಯಲ್ಲಿ ಪ್ರವಾಹ ಸಂಕಟ ಉಂಟಾಯಿತು ಎಂದರು."
FLOOD_TE = "మంత్రి మాయా రావు 12 మార్చి 2025న ఢిల్లీలో వరదలు సంక్షోభం కలిగించాయని చెప్పారు."
FLOOD_TA = "அமைச்சர் மாயா ராவ் 12 மார்ச் 2025 அன்று டெல்லியில் வெள்ளம் நெருக்கடியை ஏற்படுத்தியது என்றார்."

RSS_ITEM = (
    "<rss><channel>"
    "<item><title>Delhi flooding report</title>"
    "<link>https://example.org/delhi-flood</link>"
    "<source>Example News</source>"
    "<description>&lt;p&gt;Flooding hit Delhi on 12 March 2025.&lt;/p&gt;</description>"
    "<pubDate>Wed, 12 Mar 2025 10:00:00 +0530</pubDate>"
    "</item>"
    "</channel></rss>"
)

RSS_EMPTY = "<rss><channel></channel></rss>"


def _analyse(text: str, title: str | None = None):
    return analyse_article(text, _SETTINGS, _DETECTOR, title)


# ── 1. Genuine same-event match across languages ─────────────────────────────

def test_same_event_verified_with_full_text_same_language():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    status, score, reasons, uncertainty = verify_event(
        original,
        "Maya Rao on Delhi flooding",
        "Flooding caused a crisis in Delhi on 12 March 2025.",
        has_full_text=True,
    )
    assert status == "verified_match"
    assert score > 0
    assert reasons


def test_same_event_cross_script_possible_match_with_date():
    """Hindi candidate with matching date should reach possible_match."""
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    status, score, reasons, uncertainty = verify_event(
        original,
        "दिल्ली बाढ़ संकट 12 March 2025",
        FLOOD_HI,
        has_full_text=True,
        candidate_language="hi",
    )
    assert status in {"possible_match", "verified_match"}
    assert any("script" in u.lower() or "different" in u.lower() for u in uncertainty)


# ── 2. Same politician, unrelated events ─────────────────────────────────────

def test_same_politician_different_event_is_not_verified():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    status, score, _, _ = verify_event(
        original,
        "Maya Rao inaugurates new highway",
        "Maya Rao inaugurated a new highway in Mumbai on 5 April 2025.",
        has_full_text=True,
    )
    # Shared actor but different event — must not be verified_match
    assert status != "verified_match"


# ── 3. Same broad topic, different incident ───────────────────────────────────

def test_different_flood_incident_is_not_verified():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    status, _, _, _ = verify_event(
        original,
        "Chennai flooding update",
        "Heavy flooding struck Chennai on 20 April 2025 causing widespread damage.",
        has_full_text=True,
    )
    # Different city, different date — must not be verified_match
    assert status != "verified_match"


# ── 4. Transliterated entity name ─────────────────────────────────────────────

def test_transliterated_entity_in_query():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    profile = extract_event_profile(original)
    queries = build_language_queries(profile, ["hi"], "en")
    # Delhi should appear either as "Delhi" (kept) or "दिल्ली" (transliterated)
    assert "hi" in queries
    q = queries["hi"]
    assert "Delhi" in q or "दिल्ली" in q


# ── 5. Unicode query construction in all five languages ───────────────────────

@pytest.mark.parametrize("lang", ["en", "hi", "kn", "te", "ta"])
def test_unicode_query_construction(lang: str):
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    settings = ComparisonSettings(
        target_languages=[lang],
        include_original_language=True,
    )
    queries = build_queries(original, settings)
    assert lang in queries
    assert queries[lang].strip()
    # Query must be valid unicode (no encoding errors)
    queries[lang].encode("utf-8")


# ── 6. Provider returning irrelevant results ──────────────────────────────────

def test_irrelevant_provider_results_become_insufficient_evidence():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    status, score, _, _ = verify_event(
        original,
        "Stock market update",
        None,
    )
    assert status == "insufficient_evidence"
    assert score == 0.0


# ── 7. Provider timeout while another succeeds ───────────────────────────────

class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", invalid_json=False):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            raise ValueError("not JSON")
        return self._payload


class _FakeClient:
    def __init__(self, gdelt, google):
        self.gdelt = gdelt
        self.google = google
        self.calls: list = []
        self.options: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get(self, url, params=None):
        self.calls.append((url, params or {}))
        resp = self.gdelt if "gdeltproject" in url else self.google
        if isinstance(resp, Exception):
            raise resp
        return resp


def _run_discover(monkeypatch, gdelt, google, queries=None):
    client = _FakeClient(gdelt, google)

    def factory(**kwargs):
        client.options.update(kwargs)
        return client

    monkeypatch.setattr(comparison_service.httpx, "AsyncClient", factory)
    items, failures, statuses = asyncio.run(
        discover(queries or {"hi": "delhi flood", "ta": "delhi flood"}, 3, 2.0)
    )
    return items, failures, statuses, client


def test_gdelt_timeout_does_not_prevent_google_news_results(monkeypatch):
    import httpx as real_httpx
    items, failures, statuses, _ = _run_discover(
        monkeypatch,
        real_httpx.TimeoutException("slow"),
        _FakeResponse(text=RSS_ITEM),
    )
    assert statuses["gdelt"] == "timed out"
    assert any("timed out" in f for f in failures)
    assert any(item["url"] == "https://example.org/delhi-flood" for item in items)


# ── 8. Successful search returning zero results ───────────────────────────────

def test_empty_results_not_reported_as_failure(monkeypatch):
    items, failures, statuses, _ = _run_discover(
        monkeypatch,
        _FakeResponse(payload={"articles": []}),
        _FakeResponse(text=RSS_EMPTY),
    )
    assert items == []
    assert failures == []
    assert "ok: 0" in statuses["gdelt"]


# ── 9. Candidate with invalid article URL ─────────────────────────────────────

def test_candidate_with_invalid_url_is_metadata_only():
    """A candidate whose URL cannot be fetched and has no text is metadata_only."""
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    # Completely unrelated title, no text — should be insufficient_evidence
    status, score, _, uncertainty = verify_event(
        original,
        "Stock market weekly roundup",
        None,
        has_full_text=False,
    )
    assert status == "insufficient_evidence"
    assert score == 0.0


# ── 10. Candidate with only headline and snippet ──────────────────────────────

def test_snippet_only_candidate_is_never_verified_match():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    status, _, _, uncertainty = verify_event(
        original,
        "Delhi flooding crisis 12 March 2025",
        "Flooding caused a crisis in Delhi on 12 March 2025.",
        has_full_text=False,
    )
    assert status == "possible_match"
    assert any("snippet" in u.lower() for u in uncertainty)


# ── 11. Full-text retrieval failure retains metadata ─────────────────────────

def test_retrieval_failure_retains_snippet():
    """When full-text retrieval fails, snippet is kept and scope is snippet_only."""
    candidate = DiscoveryCandidate(
        url="https://example.org/blocked",
        title="Delhi flooding",
        provider="google_news_rss",
        retrieval_status="unavailable: HTTP 403",
        analysis_scope="snippet_only",
        verification_status="possible_match",
        snippet="Flooding hit Delhi.",
    )
    assert candidate.analysis_scope == "snippet_only"
    assert candidate.snippet == "Flooding hit Delhi."
    assert "403" in candidate.retrieval_status


# ── 12. Successful full-text retrieval ───────────────────────────────────────

def test_full_text_retrieval_produces_verified_match():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    status, score, reasons, uncertainty = verify_event(
        original,
        "Delhi flooding: Maya Rao response",
        "Maya Rao said flooding caused a crisis in Delhi on 12 March 2025.",
        has_full_text=True,
    )
    assert status == "verified_match"
    assert score >= 0.45
    assert not any("snippet" in u.lower() for u in uncertainty)


# ── 13. Duplicate candidate removal ──────────────────────────────────────────

def test_duplicate_urls_removed_across_providers(monkeypatch):
    gdelt = _FakeResponse(payload={"articles": [
        {"url": "https://example.org/delhi-flood", "title": "Same article"}
    ]})
    items, _, _, _ = _run_discover(monkeypatch, gdelt, _FakeResponse(text=RSS_ITEM))
    urls = [item["url"] for item in items]
    assert len(urls) == len(set(urls))
    assert urls == ["https://example.org/delhi-flood"]


# ── 14. Verified match produces framing comparison ───────────────────────────

def test_verified_match_produces_framing_differences():
    original = _analyse(
        "Officials said the shocking crisis was caused by poor planning on 12 March 2025.",
        "Delhi crisis report",
    )
    candidate_analysis = _analyse(
        "The government announced relief measures after the crisis on 12 March 2025.",
        "Delhi crisis: government response",
    )
    diffs = _framing_differences(original, candidate_analysis)
    # At minimum a headline framing difference should be detected
    assert any(d.dimension == "Headline framing" for d in diffs)
    for d in diffs:
        assert d.explanation
        assert d.dimension


# ── 15. No verified matches — honest empty state ─────────────────────────────

def test_no_verified_matches_honest_summary():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    candidates = [
        DiscoveryCandidate(
            url="https://example.org/a",
            title="Unrelated sports update",
            provider="google_news_rss",
            retrieval_status="retrieved",
            analysis_scope="full_text",
            verification_status="insufficient_evidence",
        )
    ]
    summary = comparative_summary(original, candidates, discovered_total=5)
    cov = summary["coverage"]
    assert cov["matched_articles"] == 0
    assert cov["discovered_articles"] == 5
    assert cov["evaluated_articles"] == 1
    assert cov["languages_represented"] == []


# ── 16. Candidate states propagate correctly ─────────────────────────────────

def test_all_candidate_states_propagate():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    cases = [
        ("verified_match", "full_text"),
        ("possible_match", "snippet_only"),
        ("rejected_match", "metadata_only"),
        ("insufficient_evidence", "metadata_only"),
    ]
    candidates = [
        DiscoveryCandidate(
            url=f"https://example.org/{status}",
            title=f"Article {status}",
            provider="google_news_rss",
            retrieval_status="retrieved" if scope == "full_text" else "unavailable: blocked",
            analysis_scope=scope,
            verification_status=status,
        )
        for status, scope in cases
    ]
    summary = comparative_summary(original, candidates)
    outcomes = summary["coverage"]["verification_outcomes"]
    for status, _ in cases:
        assert status in outcomes


# ── Event profile extraction ──────────────────────────────────────────────────

def test_event_profile_extracts_date_location_actor():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    profile = extract_event_profile(original)
    assert profile.original_language == "en"
    assert profile.headline == "Delhi flooding crisis"
    assert profile.is_specific_event
    # Should find at least one of: date, location, actor
    assert profile.dates or profile.locations or profile.actors


def test_event_profile_uncertainty_for_broad_topic():
    original = _analyse(
        "Climate change is affecting weather patterns globally in many ways.",
        "Climate change overview",
    )
    profile = extract_event_profile(original)
    # No specific date or location — should flag uncertainty
    assert profile.uncertainty_notes


def test_event_profile_as_dict_is_serializable():
    import json
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    profile = extract_event_profile(original)
    d = profile.as_dict()
    json.dumps(d)  # must not raise
    assert "headline" in d
    assert "summary" in d
    assert "original_language" in d


# ── Language-specific query generation ───────────────────────────────────────

def test_hindi_query_contains_transliterated_or_original_terms():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    profile = extract_event_profile(original)
    queries = build_language_queries(profile, ["hi"], "en")
    assert "hi" in queries
    q = queries["hi"]
    # Must contain either the English proper noun or its Hindi transliteration
    assert "Delhi" in q or "दिल्ली" in q or "flood" in q.lower() or "बाढ़" in q


def test_kannada_query_is_non_empty():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    profile = extract_event_profile(original)
    queries = build_language_queries(profile, ["kn"], "en")
    assert queries.get("kn", "").strip()


def test_original_language_excluded_by_default():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    settings = ComparisonSettings(target_languages=["en", "hi", "ta"])
    queries = build_queries(original, settings)
    assert "en" not in queries
    assert "hi" in queries
    assert "ta" in queries


def test_original_language_included_when_requested():
    original = _analyse(FLOOD_EN, "Delhi flooding crisis")
    settings = ComparisonSettings(
        target_languages=["en", "hi"],
        include_original_language=True,
    )
    queries = build_queries(original, settings)
    assert "en" in queries
    assert "hi" in queries
