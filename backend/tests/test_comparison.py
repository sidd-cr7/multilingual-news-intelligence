from app.config import Settings
from app.schemas.article import ComparisonSettings, DiscoveryCandidate
from app.services.analysis_service import analyse_article
from app.services.comparison_service import (
    build_queries,
    comparative_summary,
    discovery_terms,
    parse_gdelt,
    parse_google_rss,
    verify_event,
)
from app.services.language_service import LanguageDetector


def original():
    settings = Settings(min_text_chars=10)
    return analyse_article("Minister Maya Rao said flooding caused a crisis in Delhi on 12 March 2025.", settings, LanguageDetector(settings))


def test_query_generation_omits_original_language() -> None:
    queries = build_queries(original(), ComparisonSettings(target_languages=["en", "hi", "ta"]))
    assert "en" not in queries and {"hi", "ta"} <= queries.keys()
    assert 1 <= len(queries["hi"].split()) <= 5
    assert "said" not in queries["hi"].casefold()


def test_discovery_terms_stay_short_and_drop_rare_person_names() -> None:
    titled = original().model_copy(update={"title": "Delhi flooding crisis"})
    assert discovery_terms(titled) == ["Delhi", "flooding", "crisis"]
    terms = [term.casefold() for term in discovery_terms(original())]
    assert "maya" not in terms and "rao" not in terms
    assert "said" not in terms and "caused" not in terms
    assert 1 <= len(terms) <= 3


def test_provider_parsers_and_duplicate_safe_fields() -> None:
    assert parse_gdelt({"articles": [{"url": "https://example.com/a", "title": "Flood report", "domain": "example.com"}]})[0]["title"] == "Flood report"
    xml = "<rss><channel><item><title>Flood report</title><link>https://example.com/b</link><source>Example</source></item></channel></rss>"
    assert parse_google_rss(xml)[0]["source_name"] == "Example"
    assert parse_google_rss("<invalid") == []


def test_same_event_verification_is_transparent() -> None:
    status, score, reasons, uncertainty = verify_event(
        original(), "Maya Rao on Delhi flooding", "Flooding caused a crisis in Delhi.", has_full_text=True
    )
    assert status in {"verified_match", "possible_match"} and score > 0 and reasons and not uncertainty
    status, _, _, uncertainty = verify_event(original(), "Unrelated sports update", None)
    assert status == "insufficient_evidence" and uncertainty


def test_snippet_only_is_never_verified_match() -> None:
    status, _, _, uncertainty = verify_event(
        original(),
        "Delhi flooding",
        "Flooding caused a crisis in Delhi on 12 March 2025",
        has_full_text=False,
        candidate_language="en",
    )
    assert status == "possible_match"
    assert any("snippet" in item.lower() for item in uncertainty)


def test_verified_match_requires_body_event_terms_not_just_a_matching_title() -> None:
    status, _, reasons, uncertainty = verify_event(
        original(),
        "Delhi flooding update",
        "Maya Rao appeared at an event.",
        has_full_text=True,
        candidate_language="en",
    )
    assert status != "verified_match"
    assert any("fewer than two" in item.lower() for item in uncertainty)
    assert not any("body event terms" in item.lower() for item in reasons)


def test_snippet_and_cross_language_candidates_report_their_limits() -> None:
    status, score, _, uncertainty = verify_event(
        original(), "Delhi flooding update", "Flooding in Delhi", has_full_text=False
    )
    assert status in {"possible_match", "verified_match"}
    assert any("snippet" in item.lower() for item in uncertainty)
    _, _, _, uncertainty = verify_event(
        original(), "दिल्ली बाढ़", "दिल्ली में बाढ़", has_full_text=False, candidate_language="hi"
    )
    assert any("different" in item.lower() for item in uncertainty)


def test_coverage_keeps_discovered_and_evaluated_denominators_separate() -> None:
    one = DiscoveryCandidate(
        url="https://example.org/a",
        title="Delhi flooding",
        provider="google_news_rss",
        retrieval_status="retrieved",
        analysis_scope="full_text",
        verification_status="possible_match",
    )
    summary = comparative_summary(original(), [one], discovered_total=71)
    coverage = summary["coverage"]
    assert coverage["discovered_articles"] == 71
    assert coverage["evaluated_articles"] == 1
    assert coverage["matched_articles"] == 1


def test_wording_counts_use_only_verified_full_text_matches() -> None:
    analysis = original()
    snippet_possible = DiscoveryCandidate(
        url="https://example.org/snippet",
        title="Delhi flooding snippet",
        provider="google_news_rss",
        retrieval_status="unavailable: blocked",
        analysis_scope="snippet_only",
        verification_status="possible_match",
        analysis=analysis,
    )
    verified_full = DiscoveryCandidate(
        url="https://example.org/full",
        title="Delhi flooding full text",
        provider="google_news_rss",
        retrieval_status="retrieved",
        analysis_scope="full_text",
        verification_status="verified_match",
        analysis=analysis,
    )
    only_snippet = comparative_summary(analysis, [snippet_possible])
    assert only_snippet["indicator_counts_across_matches"] == {}

    summary = comparative_summary(analysis, [snippet_possible, verified_full])
    expected = sum(analysis.statistics["indicator_counts"].values())
    assert sum(summary["indicator_counts_across_matches"].values()) == expected
    assert summary["wording_comparison_basis"].startswith(
        "Indicator counts include only verified full-text matches (1"
    )


def test_date_conflict_is_an_explicit_rejected_match() -> None:
    status, _, _, uncertainty = verify_event(
        original(), "Delhi flooding update", "Flooding caused a crisis in Delhi on 13 March 2025."
    )
    assert status == "rejected_match"
    assert any("date conflict" in item.lower() for item in uncertainty)


# ── Provider behaviour (deterministic fakes, no network) ─────────────────────

import asyncio

from app.services import comparison_service

RSS_ONE_ITEM = (
    "<rss><channel><item><title>Delhi flooding report</title>"
    "<link>https://example.org/delhi-flood</link>"
    "<source>Example News</source>"
    "<description>&lt;p&gt;Flooding hit Delhi.&lt;/p&gt;</description>"
    "</item></channel></rss>"
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", invalid_json=False):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            raise ValueError("Response is not JSON")
        return self._payload


class FakeClient:
    def __init__(self, gdelt, google):
        self.gdelt = gdelt
        self.google = google
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get(self, url, params=None):
        self.calls.append((url, params or {}))
        response = self.gdelt if "gdeltproject" in url else self.google
        if isinstance(response, Exception):
            raise response
        return response


def run_discover(monkeypatch, gdelt, google, queries=None):
    client = FakeClient(gdelt, google)
    client.options = {}

    def factory(**kwargs):
        client.options.update(kwargs)
        return client

    monkeypatch.setattr(comparison_service.httpx, "AsyncClient", factory)
    items, failures, statuses = asyncio.run(
        comparison_service.discover(queries or {"hi": "delhi flood", "ta": "delhi flood"}, 3, 2.0)
    )
    return items, failures, statuses, client


def test_discovery_follows_provider_redirects(monkeypatch):
    """Google News redirects to a normalised locale URL; without following it the
    feed body is empty and the provider looks like a legitimate zero-result."""
    items, failures, _, client = run_discover(
        monkeypatch,
        FakeResponse(payload={"articles": []}),
        FakeResponse(text=RSS_ONE_ITEM),
    )
    assert client.options.get("follow_redirects") is True
    assert items and failures == []


def test_discovery_uses_one_gdelt_request_for_all_requested_languages(monkeypatch):
    gdelt = FakeResponse(payload={"articles": [{"url": "https://example.org/a", "title": "Delhi flooding"}]})
    items, failures, statuses, client = run_discover(monkeypatch, gdelt, FakeResponse(text=RSS_ONE_ITEM))
    gdelt_calls = [params for url, params in client.calls if "gdeltproject" in url]
    assert len(gdelt_calls) == 1
    assert "sourcelang:hindi" in gdelt_calls[0]["query"]
    assert "sourcelang:tamil" in gdelt_calls[0]["query"]
    assert failures == []
    assert statuses["gdelt"] == "ok: 1 result(s)"
    assert len(items) == 2


def test_gdelt_rate_limit_is_reported_without_disabling_google_news(monkeypatch):
    items, failures, statuses, _ = run_discover(
        monkeypatch,
        FakeResponse(status_code=429, text="Please limit requests"),
        FakeResponse(text=RSS_ONE_ITEM),
    )
    assert statuses["gdelt"] == "rate limited (HTTP 429)"
    assert any("rate limited" in failure for failure in failures)
    assert any("Google News" not in failure for failure in failures)
    assert [item["url"] for item in items] == ["https://example.org/delhi-flood"]


def test_empty_provider_results_are_not_reported_as_failures(monkeypatch):
    items, failures, statuses, _ = run_discover(
        monkeypatch,
        FakeResponse(payload={"articles": []}),
        FakeResponse(text="<rss><channel></channel></rss>"),
    )
    assert items == []
    assert failures == []
    assert statuses["gdelt"] == "ok: 0 result(s)"
    assert statuses["google_news_rss"] == "hi: ok: 0 result(s); ta: ok: 0 result(s)"


def test_gdelt_timeout_and_invalid_json_are_distinct_failures(monkeypatch):
    import httpx as real_httpx

    _, failures, statuses, _ = run_discover(
        monkeypatch,
        real_httpx.TimeoutException("slow"),
        FakeResponse(text=RSS_ONE_ITEM),
    )
    assert statuses["gdelt"] == "timed out"
    assert any("timed out" in failure for failure in failures)

    _, failures, statuses, _ = run_discover(
        monkeypatch,
        FakeResponse(invalid_json=True),
        FakeResponse(text=RSS_ONE_ITEM),
    )
    assert statuses["gdelt"] == "invalid response"
    assert any("not valid JSON" in failure for failure in failures)


def test_duplicate_candidate_urls_are_removed_across_providers(monkeypatch):
    gdelt = FakeResponse(payload={"articles": [{"url": "https://example.org/delhi-flood", "title": "Same"}]})
    items, _, _, _ = run_discover(monkeypatch, gdelt, FakeResponse(text=RSS_ONE_ITEM))
    assert [item["url"] for item in items] == ["https://example.org/delhi-flood"]
    assert items[0]["provider"] == "gdelt"
