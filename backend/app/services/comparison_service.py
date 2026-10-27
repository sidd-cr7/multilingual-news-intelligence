"""Bounded public-news discovery and transparent same-event comparison."""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from collections import Counter
from html import unescape
from urllib.parse import urlparse

import httpx

from app.config import Settings
from app.schemas.article import ArticleAnalysisResponse, ComparisonSettings, ComparisonResponse, DiscoveryCandidate
from app.services.analysis_service import analyse_article
from app.services.errors import IngestionError
from app.services.language_service import LanguageDetector
from app.services.url_service import fetch_article

WORD_RE = re.compile(r"\w+", re.UNICODE)
STOP = {"the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "has", "have"}
# Search providers treat space-separated terms as conjunctive. Live checks showed
# a six-term query returning no results where a three-term query returned broad
# coverage, so discovery queries stay short and exclude reporting verbs.
DISCOVERY_STOP = STOP | {
    "said", "says", "told", "reported", "according", "claimed", "announced", "added", "stated",
    "will", "would", "could", "should", "after", "before", "amid", "also", "more", "than",
    "when", "where", "which", "while", "about", "into", "over", "under", "have", "has", "had",
    "been", "were", "was", "are", "this", "that", "these", "those", "their", "there", "they",
    "caused", "cause", "causing", "during", "since", "other", "just", "like", "some", "such",
    "only", "officials", "government", "people",
}
DATE_TOKEN_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan(?:uary)?|feb(?:ruary)?|march|april|may|june|july|august|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,?\s*\d{4})?|\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|march|april|may|june|july|august|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?)\b",
    re.IGNORECASE,
)
LANGUAGE_NAMES = {
    "en": "english",
    "hi": "hindi",
    "kn": "kannada",
    "te": "telugu",
    "ta": "tamil",
}
GOOGLE_NEWS_LOCALES = {
    "en": ("en-IN", "IN:en"),
    "hi": ("hi-IN", "IN:hi"),
    "kn": ("kn-IN", "IN:kn"),
    "te": ("te-IN", "IN:te"),
    "ta": ("ta-IN", "IN:ta"),
}


def keywords(analysis: ArticleAnalysisResponse, limit: int = 6) -> list[str]:
    entity_terms = [entity.text for entity in analysis.entities]
    tokens = [word.casefold() for word in WORD_RE.findall(analysis.original_text) if len(word) > 3 and word.casefold() not in STOP]
    frequent = [word for word, _ in Counter(tokens).most_common(limit)]
    return list(dict.fromkeys(entity_terms + frequent))[:limit]


def discovery_terms(analysis: ArticleAnalysisResponse, limit: int = 3) -> list[str]:
    """Short, high-recall query terms for discovery providers.

    A title, when present, is the most compact summary of the event. Otherwise
    the most frequent content words of the body are used. Proper-name candidates
    are excluded from body-derived queries: a rare name alone is a poor event
    query and returned no coverage in live checks.
    """
    if analysis.title:
        title_terms: list[str] = []
        for word in WORD_RE.findall(analysis.title):
            if len(word) > 2 and word.casefold() not in DISCOVERY_STOP:
                title_terms.append(word)
        unique_title_terms: list[str] = []
        for word in title_terms:
            if word.casefold() in (term.casefold() for term in unique_title_terms):
                continue
            unique_title_terms.append(word)
        if unique_title_terms:
            return unique_title_terms[:limit]
    excluded = {
        token.casefold()
        for entity in analysis.entities
        if entity.label == "PERSON_CANDIDATE"
        for token in WORD_RE.findall(entity.text)
    }
    counts = Counter(
        word.casefold()
        for word in WORD_RE.findall(analysis.original_text)
        if len(word) > 3
        and word.casefold() not in DISCOVERY_STOP
        and word.casefold() not in excluded
    )
    return [word for word, _ in counts.most_common(limit)]


def event_profile(analysis: ArticleAnalysisResponse) -> dict:
    """A transparent, bounded event profile used before provider discovery.

    This is not entity recognition or event verification.  It records the
    distinctive terms, dates, and candidate entities actually available so a
    generic topical search is not presented as same-event discovery.
    """
    dates = [entity.text for entity in analysis.entities if entity.label == "DATE"]
    people = [entity.text for entity in analysis.entities if entity.label == "PERSON_CANDIDATE"]
    terms = discovery_terms(analysis)
    return {
        "headline": analysis.title,
        "language": analysis.language.code,
        "distinctive_terms": terms,
        "dates": dates,
        "people_candidates": people,
        "summary": analysis.summary.get("text"),
        "sufficient_for_targeted_search": bool(terms and (dates or people or analysis.title)),
        "limitation": "The profile is rule-derived from the supplied article. It does not translate event names or prove that a retrieved candidate covers the same event.",
    }


def build_queries(analysis: ArticleAnalysisResponse, settings: ComparisonSettings) -> dict[str, str]:
    base = " ".join(discovery_terms(analysis)) or analysis.original_text[:120]
    languages = [code for code in settings.target_languages if settings.include_original_language or code != analysis.language.code]
    return {code: base for code in languages if code in {"en", "hi", "kn", "te", "ta"}}


def parse_gdelt(payload: dict, provider: str = "gdelt") -> list[dict]:
    return [{"url": item["url"], "title": item.get("title") or item["url"], "source_name": item.get("domain"), "published_at": item.get("seendate"), "snippet": item.get("snippet"), "provider": provider} for item in payload.get("articles", []) if isinstance(item, dict) and item.get("url")]


def parse_google_rss(xml: str, provider: str = "google_news_rss") -> list[dict]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    items = []
    for item in root.findall(".//item"):
        link, title = item.findtext("link"), item.findtext("title")
        if link and title:
            source = item.findtext("source")
            description = item.findtext("description")
            snippet = re.sub(r"<[^>]+>", " ", unescape(description or "")).strip() or None
            items.append({"url": link, "title": title, "source_name": source, "published_at": item.findtext("pubDate"), "snippet": snippet, "provider": provider})
    return items


async def _gdelt_request(client: httpx.AsyncClient, query: str, languages: list[str], limit: int) -> tuple[list[dict], str | None, str]:
    """One bounded GDELT request per comparison.

    GDELT documents a limit of one request every five seconds, so querying it once
    per language (five requests in quick succession) reliably produced rate-limit
    failures. The requested source languages are expressed as a single filter.
    """
    filters = " OR ".join(f"sourcelang:{LANGUAGE_NAMES[code]}" for code in languages)
    try:
        response = await client.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": f"{query} ({filters})" if filters else query, "mode": "artlist", "format": "json", "maxrecords": limit},
        )
    except httpx.TimeoutException:
        return [], "GDELT: timed out at the configured limit.", "timed out"
    except httpx.HTTPError as exc:
        return [], f"GDELT: request failed ({type(exc).__name__}).", f"failed ({type(exc).__name__})"
    if response.status_code == 429:
        return [], "GDELT: rate limited (HTTP 429); retry later. No other provider was affected.", "rate limited (HTTP 429)"
    if response.status_code >= 400:
        return [], f"GDELT: HTTP {response.status_code}.", f"failed (HTTP {response.status_code})"
    try:
        payload = response.json()
    except ValueError:
        return [], "GDELT: the response was not valid JSON.", "invalid response"
    items = parse_gdelt(payload)
    return items, None, f"ok: {len(items)} result(s)"


async def _google_news_request(client: httpx.AsyncClient, query: str, language: str) -> tuple[list[dict], str | None, str]:
    hl, ceid = GOOGLE_NEWS_LOCALES[language]
    label = f"Google News RSS ({language})"
    try:
        response = await client.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": hl, "gl": "IN", "ceid": ceid},
        )
    except httpx.TimeoutException:
        return [], f"{label}: timed out at the configured limit.", "timed out"
    except httpx.HTTPError as exc:
        return [], f"{label}: request failed ({type(exc).__name__}).", f"failed ({type(exc).__name__})"
    if response.status_code >= 400:
        return [], f"{label}: HTTP {response.status_code}.", f"failed (HTTP {response.status_code})"
    items = parse_google_rss(response.text)
    return items, None, f"ok: {len(items)} result(s)"


async def discover(queries: dict[str, str], limit: int, timeout: float) -> tuple[list[dict], list[str], dict[str, str]]:
    """Provider failures are data, not fatal errors. No credentials are used.

    A failure in one provider never disables the other, and an empty result is
    reported as an empty result rather than as a failure.
    """
    failures: list[str] = []
    candidates: list[dict] = []
    statuses: dict[str, str] = {}
    # News providers redirect (Google News normalises the locale parameters), so
    # redirects must be followed or the adapter silently sees an empty body and
    # reports zero results. Only fixed public provider URLs use this client;
    # article fetching keeps its own SSRF-validated redirect handling.
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "MultilingualNewsIntelligence/0.1"},
    ) as client:
        languages = list(queries)
        if languages:
            gdelt_items, gdelt_failure, gdelt_status = await _gdelt_request(client, next(iter(queries.values())), languages, limit)
            statuses["gdelt"] = gdelt_status
            if gdelt_failure:
                failures.append(gdelt_failure)
            candidates.extend(gdelt_items)
        per_language_status: list[str] = []
        for language, query in queries.items():
            found, failure, status = await _google_news_request(client, query, language)
            per_language_status.append(f"{language}: {status}")
            if failure:
                failures.append(failure)
            for item in found:
                item["requested_language"] = language
            candidates.extend(found)
        if per_language_status:
            statuses["google_news_rss"] = "; ".join(per_language_status)
    unique: dict[str, dict] = {}
    for item in candidates:
        url = item["url"]
        if urlparse(url).scheme in {"http", "https"} and url not in unique:
            unique[url] = item
    return list(unique.values()), failures, statuses


def verify_event(
    original: ArticleAnalysisResponse,
    title: str,
    available_text: str | None,
    *,
    has_full_text: bool = False,
    candidate_language: str | None = None,
) -> tuple[str, float, list[str], list[str]]:
    haystack = f"{title} {available_text or ''}".casefold()
    original_terms = set(keywords(original))
    shared = [term for term in original_terms if term.casefold() in haystack]
    candidate_terms = {word.casefold() for word in WORD_RE.findall(haystack) if len(word) > 3}
    source_terms = {word.casefold() for word in keywords(original)}
    overlap = len(candidate_terms & source_terms)
    score = min(1.0, (len(shared) * 0.2) + (overlap * 0.08))
    reasons = ([f"Shared event terms: {', '.join(shared[:3])}."] if shared else []) + ([f"Keyword overlap: {overlap}."] if overlap else [])
    original_dates = {value.casefold() for value in DATE_TOKEN_RE.findall(original.original_text)}
    candidate_dates = {value.casefold() for value in DATE_TOKEN_RE.findall(f"{title} {available_text or ''}")}
    date_conflict = bool(original_dates and candidate_dates and original_dates.isdisjoint(candidate_dates))
    uncertainty: list[str] = []
    if not has_full_text:
        uncertainty.append("Only title/snippet was available; no full-text event verification.")
    if candidate_language and candidate_language != original.language.code:
        uncertainty.append(
            "Candidate text is in a different language; no translation is performed, so "
            "cross-script lexical same-event verification is unavailable."
        )
    if date_conflict:
        status = "rejected_match"
        uncertainty.append("Explicit date conflict between the original and candidate text.")
    elif score >= 0.45 and available_text:
        status = "verified_match"
    elif score >= 0.2:
        status = "possible_match"
    elif score > 0 and available_text:
        status = "rejected_match"
    else:
        status = "insufficient_evidence"
    return status, round(score, 2), reasons, uncertainty


def comparative_summary(original: ArticleAnalysisResponse, candidates: list[DiscoveryCandidate], discovered_total: int | None = None) -> dict:
    matched = [candidate for candidate in candidates if candidate.verification_status in {"verified_match", "possible_match"}]
    analyses = [candidate.analysis for candidate in matched if candidate.analysis]
    category_counts = Counter()
    for analysis in analyses:
        category_counts.update(analysis.statistics.get("indicator_counts", {}))
    languages = sorted({candidate.language.code for candidate in matched if candidate.language})
    evaluated = len(candidates)
    evidence_comparisons: list[dict] = []
    # A comparison is allowed only after the configured same-event check and
    # when both sides have full text.  It cites the exact evidence on each side
    # and reports a wording difference, never an inference about bias.
    original_by_category = {item.category: item for item in original.indicators}
    for candidate in matched:
        if candidate.verification_status != "verified_match" or candidate.analysis_scope != "full_text" or not candidate.analysis:
            continue
        candidate_by_category = {item.category: item for item in candidate.analysis.indicators}
        for category in sorted(set(original_by_category) | set(candidate_by_category)):
            left, right = original_by_category.get(category), candidate_by_category.get(category)
            if not left or not right or left.occurrences == right.occurrences:
                continue
            evidence_comparisons.append({
                "candidate_url": candidate.url,
                "candidate_title": candidate.title,
                "category": category,
                "original_evidence": {"quote": left.quote, "start": left.start, "end": left.end, "context": left.context, "occurrences": left.occurrences},
                "candidate_evidence": {"quote": right.quote, "start": right.start, "end": right.end, "context": right.context, "occurrences": right.occurrences},
                "interpretation": "Available full texts contain different counts of this observable cue. This may reflect editorial choices, article length, reporting scope, or context; it is not a bias finding.",
            })
    return {"coverage": {"discovered_articles": evaluated if discovered_total is None else discovered_total, "evaluated_articles": evaluated, "matched_articles": len(matched), "languages_represented": languages, "sources_represented": sorted({item.source_name for item in matched if item.source_name}), "full_text_articles": sum(item.analysis_scope == "full_text" for item in matched), "snippet_only_articles": sum(item.analysis_scope == "snippet_only" for item in matched), "verification_outcomes": dict(Counter(item.verification_status for item in candidates)), "retrieval_outcomes": dict(Counter(item.analysis_scope for item in candidates))}, "indicator_counts_across_matches": dict(category_counts), "evidence_comparisons": evidence_comparisons, "observation": "Counts describe observable cues in available text only. Discovered articles include candidates not shortlisted for retrieval; verification outcomes use the evaluated denominator. They are not bias scores or source-quality judgments."}


async def compare_article(text: str, title: str | None, original_url: str | None, options: ComparisonSettings, settings: Settings, detector: LanguageDetector) -> ComparisonResponse:
    original = analyse_article(text, settings, detector, title, original_url)
    profile = event_profile(original)
    queries = build_queries(original, options)
    raw, failures, provider_status = await discover(queries, options.result_limit, settings.discovery_timeout_seconds)
    deduped: dict[str, dict] = {item["url"]: item for item in raw}
    # The per-language result limit applies per requested language, and full-text
    # retrieval is bounded so a slow publisher cannot stall the whole comparison.
    candidate_limit = min(len(deduped), max(options.result_limit, options.result_limit * max(1, len(queries))))
    shortlisted = list(deduped.values())[:candidate_limit]

    async def retrieve(item: dict) -> tuple[dict, str | None, str]:
        """Return the candidate, the text available for verification, and its retrieval status."""
        fallback = item.get("snippet")
        try:
            extracted, fetched_title, final_url, _ = await fetch_article(item["url"], settings)
            item["url"] = final_url
            item["title"] = fetched_title or item["title"]
            return item, extracted, "retrieved"
        except IngestionError as exc:
            return item, fallback, f"unavailable: {exc.message}"

    throttle = asyncio.Semaphore(4)

    async def bounded_retrieve(item: dict) -> tuple[dict, str | None, str]:
        async with throttle:
            return await retrieve(item)

    retrieved = await asyncio.gather(*(bounded_retrieve(item) for item in shortlisted)) if shortlisted else []
    candidates: list[DiscoveryCandidate] = []
    for item, available, retrieval in retrieved:
        scope = "full_text" if retrieval == "retrieved" else "snippet_only" if available else "metadata_only"
        language = detector.detect(available or item["title"])
        status, score, reasons, uncertainty = verify_event(
            original,
            item["title"],
            available,
            has_full_text=retrieval == "retrieved",
            candidate_language=language.code,
        )
        analysis = analyse_article(available, settings, detector, item["title"], item["url"]) if available and len(available) >= settings.min_text_chars else None
        candidates.append(DiscoveryCandidate(url=item["url"], title=item["title"], source_name=item.get("source_name"), published_at=item.get("published_at"), snippet=item.get("snippet"), provider=item["provider"], language=language, retrieval_status=retrieval, analysis_scope=scope, verification_status=status, relevance_score=score, match_reasons=reasons, uncertainty=uncertainty, analysis=analysis))
    summary = comparative_summary(original, candidates, len(deduped))
    summary["event_profile"] = profile
    return ComparisonResponse(original_analysis=original, discovery_queries=queries, candidates=candidates, comparison=summary, provider_failures=list(dict.fromkeys(failures)), provider_status=provider_status, limitations=["Discovery providers can be unavailable, rate limited, or return incomplete metadata. Similarity scores are rule-based relevance signals, not calibrated probabilities.", "Only retrieved full text is represented as full-article analysis; snippets are explicitly scoped.", "An empty candidate list can mean the providers returned nothing for these queries; it is not evidence that no other coverage exists."])
