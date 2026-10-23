"""Bounded public-news discovery and transparent same-event comparison."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from urllib.parse import quote_plus, urlparse

import httpx

from app.config import Settings
from app.schemas.article import ArticleAnalysisResponse, ComparisonSettings, ComparisonResponse, DiscoveryCandidate
from app.services.analysis_service import analyse_article
from app.services.errors import IngestionError
from app.services.language_service import LanguageDetector
from app.services.url_service import fetch_article

WORD_RE = re.compile(r"[w']+", re.UNICODE)
STOP = {"the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "has", "have"}


def keywords(analysis: ArticleAnalysisResponse, limit: int = 6) -> list[str]:
    entity_terms = [entity.text for entity in analysis.entities]
    tokens = [word.casefold() for word in WORD_RE.findall(analysis.original_text) if len(word) > 3 and word.casefold() not in STOP]
    frequent = [word for word, _ in Counter(tokens).most_common(limit)]
    return list(dict.fromkeys(entity_terms + frequent))[:limit]


def build_queries(analysis: ArticleAnalysisResponse, settings: ComparisonSettings) -> dict[str, str]:
    terms = keywords(analysis)
    base = " ".join(terms) or analysis.original_text[:120]
    languages = [code for code in settings.target_languages if settings.include_original_language or code != analysis.language.code]
    return {code: base for code in languages if code in {"en", "hi", "kn", "te", "ta"}}


def parse_gdelt(payload: dict, provider: str = "gdelt") -> list[dict]:
    return [{"url": item["url"], "title": item.get("title") or item["url"], "source_name": item.get("domain"), "published_at": item.get("seendate"), "snippet": item.get("socialimage") and None, "provider": provider} for item in payload.get("articles", []) if isinstance(item, dict) and item.get("url")]


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
            items.append({"url": link, "title": title, "source_name": source, "published_at": item.findtext("pubDate"), "snippet": item.findtext("description"), "provider": provider})
    return items


async def discover(query: str, limit: int, timeout: float) -> tuple[list[dict], list[str]]:
    """Provider failures are data, not fatal errors. No credentials are used."""
    failures: list[str] = []; candidates: list[dict] = []
    async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "MultilingualNewsIntelligence/0.1"}) as client:
        try:
            response = await client.get(f"https://api.gdeltproject.org/api/v2/doc/doc?query={quote_plus(query)}&mode=artlist&format=json&maxrecords={limit}")
            response.raise_for_status(); candidates.extend(parse_gdelt(response.json()))
        except (httpx.HTTPError, ValueError) as exc:
            failures.append(f"GDELT unavailable: {type(exc).__name__}")
        try:
            response = await client.get(f"https://news.google.com/rss/search?q={quote_plus(query)}")
            response.raise_for_status(); candidates.extend(parse_google_rss(response.text))
        except httpx.HTTPError as exc:
            failures.append(f"Google News RSS unavailable: {type(exc).__name__}")
    unique: dict[str, dict] = {}
    for item in candidates:
        url = item["url"]
        if urlparse(url).scheme in {"http", "https"} and url not in unique:
            unique[url] = item
    return list(unique.values())[:limit], failures


def verify_event(original: ArticleAnalysisResponse, title: str, available_text: str | None) -> tuple[str, float, list[str], list[str]]:
    haystack = f"{title} {available_text or ''}".casefold()
    original_terms = set(keywords(original))
    shared = [term for term in original_terms if term.casefold() in haystack]
    candidate_terms = {word.casefold() for word in WORD_RE.findall(haystack) if len(word) > 3}
    source_terms = {word.casefold() for word in keywords(original)}
    overlap = len(candidate_terms & source_terms)
    score = min(1.0, (len(shared) * 0.2) + (overlap * 0.08))
    reasons = ([f"Shared event terms: {', '.join(shared[:3])}."] if shared else []) + ([f"Keyword overlap: {overlap}."] if overlap else [])
    uncertainty = [] if available_text else ["Only title/snippet was available; no full-text event verification."]
    status = "verified_match" if score >= 0.45 and available_text else "possible_match" if score >= 0.2 else "insufficient_evidence"
    return status, round(score, 2), reasons, uncertainty


def comparative_summary(original: ArticleAnalysisResponse, candidates: list[DiscoveryCandidate]) -> dict:
    matched = [candidate for candidate in candidates if candidate.verification_status in {"verified_match", "possible_match"}]
    analyses = [candidate.analysis for candidate in matched if candidate.analysis]
    category_counts = Counter()
    for analysis in analyses:
        category_counts.update(analysis.statistics.get("indicator_counts", {}))
    languages = sorted({candidate.language.code for candidate in matched if candidate.language})
    return {"coverage": {"discovered_articles": len(candidates), "matched_articles": len(matched), "languages_represented": languages, "sources_represented": sorted({item.source_name for item in matched if item.source_name}), "full_text_articles": sum(item.analysis_scope == "full_text" for item in matched), "snippet_only_articles": sum(item.analysis_scope == "snippet_only" for item in matched)}, "indicator_counts_across_matches": dict(category_counts), "observation": "Counts compare observable cues in available text only; they are not bias scores or source-quality judgments."}


async def compare_article(text: str, title: str | None, original_url: str | None, options: ComparisonSettings, settings: Settings, detector: LanguageDetector) -> ComparisonResponse:
    original = analyse_article(text, settings, detector, title, original_url)
    queries = build_queries(original, options)
    raw: list[dict] = []; failures: list[str] = []
    for language, query in queries.items():
        found, provider_failures = await discover(query, options.result_limit, settings.discovery_timeout_seconds)
        for item in found: item["requested_language"] = language
        raw.extend(found); failures.extend(provider_failures)
    deduped: dict[str, dict] = {item["url"]: item for item in raw}
    candidates: list[DiscoveryCandidate] = []
    for item in list(deduped.values())[:options.result_limit]:
        available = item.get("snippet")
        scope, retrieval = ("snippet_only", "snippet_available") if available else ("metadata_only", "not_retrieved")
        # Full text is fetched only after deduplication and uses Phase 1 safeguards.
        try:
            extracted, fetched_title, final_url, _ = await fetch_article(item["url"], settings)
            item["url"] = final_url; item["title"] = fetched_title or item["title"]; available = extracted; scope, retrieval = "full_text", "retrieved"
        except IngestionError as exc:
            retrieval = f"unavailable: {exc.message}"
        status, score, reasons, uncertainty = verify_event(original, item["title"], available)
        language = detector.detect(available or item["title"])
        analysis = analyse_article(available, settings, detector, item["title"], item["url"]) if available and len(available) >= settings.min_text_chars else None
        candidates.append(DiscoveryCandidate(url=item["url"], title=item["title"], source_name=item.get("source_name"), published_at=item.get("published_at"), snippet=item.get("snippet"), provider=item["provider"], language=language, retrieval_status=retrieval, analysis_scope=scope, verification_status=status, relevance_score=score, match_reasons=reasons, uncertainty=uncertainty, analysis=analysis))
    return ComparisonResponse(original_analysis=original, discovery_queries=queries, candidates=candidates, comparison=comparative_summary(original, candidates), provider_failures=list(dict.fromkeys(failures)), limitations=["Discovery providers can be unavailable or return incomplete metadata. Similarity scores are rule-based relevance signals, not calibrated probabilities.", "Only retrieved full text is represented as full-article analysis; snippets are explicitly scoped."])
