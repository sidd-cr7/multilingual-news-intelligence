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
from app.schemas.article import (
    ArticleAnalysisResponse,
    ComparisonSettings,
    ComparisonResponse,
    DiscoveryCandidate,
    FramingDifference,
)
from app.services.analysis_service import analyse_article
from app.services.errors import IngestionError
from app.services.event_profile import extract_event_profile, build_language_queries
from app.services.language_service import LanguageDetector
from app.services.url_service import fetch_article

WORD_RE = re.compile(r"\w+", re.UNICODE)
STOP = {"the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "has", "have"}
DISCOVERY_STOP = STOP | {
    "said", "says", "told", "reported", "according", "claimed", "announced", "added", "stated",
    "will", "would", "could", "should", "after", "before", "amid", "also", "more", "than",
    "when", "where", "which", "while", "about", "into", "over", "under", "have", "has", "had",
    "been", "were", "was", "are", "this", "that", "these", "those", "their", "there", "they",
    "caused", "cause", "causing", "during", "since", "other", "just", "like", "some", "such",
    "only", "officials", "government", "people",
}
DATE_TOKEN_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:jan(?:uary)?|feb(?:ruary)?|march|april|may|june|july|august"
    r"|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,?\s*\d{4})?"
    r"|\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|march|april|may|june|july|august"
    r"|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?)\b",
    re.IGNORECASE,
)
LANGUAGE_NAMES = {
    "en": "english", "hi": "hindi", "kn": "kannada", "te": "telugu", "ta": "tamil",
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
    tokens = [
        word.casefold()
        for word in WORD_RE.findall(analysis.original_text)
        if len(word) > 3 and word.casefold() not in STOP
    ]
    frequent = [word for word, _ in Counter(tokens).most_common(limit)]
    return list(dict.fromkeys(entity_terms + frequent))[:limit]


def discovery_terms(analysis: ArticleAnalysisResponse, limit: int = 3) -> list[str]:
    """Short, high-recall query terms for discovery providers."""
    if analysis.title:
        title_terms: list[str] = []
        for word in WORD_RE.findall(analysis.title):
            if len(word) > 2 and word.casefold() not in DISCOVERY_STOP:
                title_terms.append(word)
        unique_title_terms: list[str] = []
        for word in title_terms:
            if word.casefold() in (t.casefold() for t in unique_title_terms):
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


def build_queries(
    analysis: ArticleAnalysisResponse,
    settings: ComparisonSettings,
) -> dict[str, str]:
    """Build per-language search queries using the event profile."""
    profile = extract_event_profile(analysis)
    languages = [
        code for code in settings.target_languages
        if (settings.include_original_language or code != analysis.language.code)
        and code in {"en", "hi", "kn", "te", "ta"}
    ]
    queries = build_language_queries(profile, languages, analysis.language.code)
    base_fallback = " ".join(discovery_terms(analysis)) or analysis.original_text[:120]
    for lang in languages:
        if lang not in queries or not queries[lang].strip():
            queries[lang] = base_fallback
    return queries


def parse_gdelt(payload: dict, provider: str = "gdelt") -> list[dict]:
    return [
        {
            "url": item["url"],
            "title": item.get("title") or item["url"],
            "source_name": item.get("domain"),
            "published_at": item.get("seendate"),
            "snippet": item.get("snippet"),
            "provider": provider,
        }
        for item in payload.get("articles", [])
        if isinstance(item, dict) and item.get("url")
    ]


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
            items.append({
                "url": link,
                "title": title,
                "source_name": source,
                "published_at": item.findtext("pubDate"),
                "snippet": snippet,
                "provider": provider,
            })
    return items


async def _gdelt_request(
    client: httpx.AsyncClient,
    query: str,
    languages: list[str],
    limit: int,
) -> tuple[list[dict], str | None, str]:
    """One bounded GDELT request per comparison."""
    filters = " OR ".join(f"sourcelang:{LANGUAGE_NAMES[code]}" for code in languages if code in LANGUAGE_NAMES)
    try:
        response = await client.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query": f"{query} ({filters})" if filters else query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": limit,
            },
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


async def _google_news_request(
    client: httpx.AsyncClient,
    query: str,
    language: str,
) -> tuple[list[dict], str | None, str]:
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


async def discover(
    queries: dict[str, str],
    limit: int,
    timeout: float,
) -> tuple[list[dict], list[str], dict[str, str]]:
    """Run providers independently; a failure in one never disables the other."""
    failures: list[str] = []
    candidates: list[dict] = []
    statuses: dict[str, str] = {}
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "MultilingualNewsIntelligence/0.1"},
    ) as client:
        languages = list(queries)
        if languages:
            gdelt_items, gdelt_failure, gdelt_status = await _gdelt_request(
                client, next(iter(queries.values())), languages, limit
            )
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
    """Evaluate whether a candidate covers the same event as the original.

    Cross-language matching: when the candidate is in a different script,
    lexical overlap is impossible. The score is then based on:
    - Shared proper nouns (often transliterated similarly across scripts)
    - Headline similarity signals
    - Date agreement
    A candidate with a matching date and a plausible headline is promoted to
    possible_match even without body-term overlap, because cross-script
    lexical verification is structurally unavailable.
    """
    haystack = f"{title} {available_text or ''}".casefold()
    body = (available_text or "").casefold()
    original_terms = set(keywords(original))
    shared = [term for term in original_terms if term.casefold() in haystack]

    entity_tokens = {
        word.casefold()
        for entity in original.entities
        for word in WORD_RE.findall(entity.text)
    }
    event_terms = {term.casefold() for term in original_terms} - entity_tokens
    body_shared = [term for term in sorted(event_terms) if term in body]

    candidate_terms = {
        word.casefold()
        for word in WORD_RE.findall(haystack)
        if len(word) > 3
    }
    source_terms = {word.casefold() for word in keywords(original)}
    overlap = len(candidate_terms & source_terms)

    original_dates = {v.casefold() for v in DATE_TOKEN_RE.findall(original.original_text)}
    candidate_dates = {v.casefold() for v in DATE_TOKEN_RE.findall(f"{title} {available_text or ''}")}
    date_match = bool(original_dates & candidate_dates)
    date_conflict = bool(original_dates and candidate_dates and not date_match)

    # Detect cross-script situation: candidate language uses a different script
    latin_scripts = {"en", "id", "ms", "vi", "ro", "pl", "cs", "sk", "hr"}
    is_cross_script = (
        candidate_language is not None
        and candidate_language != original.language.code
        and (candidate_language not in latin_scripts or original.language.code not in latin_scripts)
    )

    score = min(1.0, (len(shared) * 0.2) + (overlap * 0.08))

    # For cross-script candidates, boost score when date matches or headline
    # contains recognisable proper nouns from the original (they survive
    # transliteration in news headlines).
    if is_cross_script:
        title_lower = title.casefold()
        # Check if any entity text from original appears in the candidate title
        # (proper nouns are often kept in Latin script even in Indic headlines)
        entity_in_title = any(
            entity.text.casefold() in title_lower
            for entity in original.entities
        )
        if date_match:
            score = max(score, 0.35)
        if entity_in_title:
            score = max(score, 0.25)
        if date_match and entity_in_title:
            score = max(score, 0.50)

    reasons: list[str] = []
    if shared:
        reasons.append(f"Shared event terms: {', '.join(shared[:3])}.")
    if overlap:
        reasons.append(f"Keyword overlap: {overlap}.")
    if body_shared:
        reasons.append(f"Shared body event terms: {', '.join(body_shared[:3])}.")
    if date_match:
        reasons.append("Shared explicit date.")
    if date_conflict:
        reasons.append("Explicit date conflict between the original and candidate text.")

    uncertainty: list[str] = []
    if date_conflict:
        uncertainty.append("Explicit date conflict between the original and candidate text.")
    if not has_full_text:
        uncertainty.append("Only title/snippet was available; no full-text event verification.")
    if is_cross_script:
        uncertainty.append(
            "Candidate text is in a different script; cross-script lexical same-event "
            "verification is structurally unavailable. Match is based on date, proper nouns, "
            "and headline signals only."
        )
    elif candidate_language and candidate_language != original.language.code:
        uncertainty.append(
            "Candidate text is in a different language; no translation is performed, so "
            "cross-script lexical same-event verification is unavailable."
        )
    if has_full_text and not is_cross_script and len(body_shared) < 2:
        uncertainty.append(
            "Fewer than two shared event terms appear in the retrieved full text, so the "
            "match is not strong enough to verify the same event."
        )

    # Status determination
    if date_conflict:
        status = "rejected_match"
    elif has_full_text and available_text and not is_cross_script and len(body_shared) >= 2 and score >= 0.45:
        status = "verified_match"
    elif has_full_text and available_text and is_cross_script and score >= 0.45:
        # Cross-script full-text with sufficient score: verified
        status = "verified_match"
    elif score >= 0.2:
        status = "possible_match"
    elif score > 0 and available_text:
        status = "rejected_match"
    else:
        status = "insufficient_evidence"

    return status, round(score, 2), reasons, uncertainty


def _framing_differences(
    original: ArticleAnalysisResponse,
    candidate: ArticleAnalysisResponse,
) -> list[FramingDifference]:
    """Compare two analysed articles and return observable framing differences.

    Only differences supported by exact evidence from both articles are reported.
    No bias verdict is produced.
    """
    diffs: list[FramingDifference] = []

    # 1. Headline framing
    orig_title = original.title or ""
    cand_title = candidate.title or ""
    if orig_title and cand_title and orig_title.casefold() != cand_title.casefold():
        diffs.append(FramingDifference(
            dimension="Headline framing",
            article_a_quote=orig_title,
            article_b_quote=cand_title,
            explanation=(
                "The two articles use different headlines for the same event. "
                "Headline word choice can emphasise different aspects or actors."
            ),
            confidence="observable",
            alternative_interpretation=(
                "Different headlines may reflect editorial style, publication timing, "
                "or different angles on the same event rather than deliberate framing."
            ),
        ))

    # 2. Sentiment direction
    orig_sent = original.sentiment.get("label")
    cand_sent = candidate.sentiment.get("label")
    if (
        orig_sent and cand_sent
        and orig_sent != "neutral" and cand_sent != "neutral"
        and orig_sent != cand_sent
        and original.sentiment.get("status") == "available"
        and candidate.sentiment.get("status") == "available"
    ):
        diffs.append(FramingDifference(
            dimension="Sentiment direction",
            article_a_quote=f"Lexical sentiment: {orig_sent} ({original.sentiment.get('positive_cues', 0)} positive, {original.sentiment.get('negative_cues', 0)} negative cues)",
            article_b_quote=f"Lexical sentiment: {cand_sent} ({candidate.sentiment.get('positive_cues', 0)} positive, {candidate.sentiment.get('negative_cues', 0)} negative cues)",
            explanation=(
                "The two articles show different lexical sentiment directions based on "
                "seed-lexicon cue counts. This is a heuristic observation, not a bias verdict."
            ),
            confidence="heuristic — uncalibrated lexical cues",
            alternative_interpretation=(
                "Sentiment differences may reflect the specific aspects of the event each "
                "article focuses on, not editorial bias."
            ),
        ))

    # 3. Loaded language presence
    orig_loaded = [i for i in original.indicators if i.category == "loaded_or_evaluative_language"]
    cand_loaded = [i for i in candidate.indicators if i.category == "loaded_or_evaluative_language"]
    if orig_loaded and not cand_loaded:
        quotes = "; ".join(f'"{i.quote}"' for i in orig_loaded[:3])
        diffs.append(FramingDifference(
            dimension="Loaded or evaluative language",
            article_a_quote=f"Contains evaluative cues: {quotes}",
            article_b_quote="No configured evaluative cues matched.",
            explanation=(
                "Article A contains evaluative wording not found in Article B. "
                "This may reflect different editorial registers or language conventions."
            ),
            confidence="observable — lexicon-based",
            alternative_interpretation=(
                "The absence of matched cues in Article B may reflect language differences "
                "or a different writing style rather than more neutral coverage."
            ),
        ))
    elif cand_loaded and not orig_loaded:
        quotes = "; ".join(f'"{i.quote}"' for i in cand_loaded[:3])
        diffs.append(FramingDifference(
            dimension="Loaded or evaluative language",
            article_a_quote="No configured evaluative cues matched.",
            article_b_quote=f"Contains evaluative cues: {quotes}",
            explanation=(
                "Article B contains evaluative wording not found in Article A."
            ),
            confidence="observable — lexicon-based",
            alternative_interpretation=(
                "The absence of matched cues in Article A may reflect language differences "
                "or a different writing style rather than more neutral coverage."
            ),
        ))

    # 4. Attribution of responsibility
    orig_resp = [i for i in original.indicators if i.category == "attribution_of_responsibility"]
    cand_resp = [i for i in candidate.indicators if i.category == "attribution_of_responsibility"]
    if bool(orig_resp) != bool(cand_resp):
        a_quote = "; ".join(f'"{i.quote}"' for i in orig_resp[:2]) if orig_resp else "None found."
        b_quote = "; ".join(f'"{i.quote}"' for i in cand_resp[:2]) if cand_resp else "None found."
        diffs.append(FramingDifference(
            dimension="Attribution of responsibility",
            article_a_quote=a_quote,
            article_b_quote=b_quote,
            explanation=(
                "One article explicitly attributes responsibility or causation; the other does not. "
                "This can direct reader attention differently."
            ),
            confidence="observable — rule-based",
            alternative_interpretation=(
                "The difference may reflect the specific angle each article takes, "
                "or the availability of sources at publication time."
            ),
        ))

    # 5. Hedging / certainty
    orig_hedge = [i for i in original.indicators if i.category == "hedging_or_qualification"]
    cand_hedge = [i for i in candidate.indicators if i.category == "hedging_or_qualification"]
    orig_abs = [i for i in original.indicators if i.category == "absolutist_or_generalising_language"]
    cand_abs = [i for i in candidate.indicators if i.category == "absolutist_or_generalising_language"]
    if (bool(orig_hedge) or bool(orig_abs)) != (bool(cand_hedge) or bool(cand_abs)):
        a_hedges = "; ".join(f'"{i.quote}"' for i in (orig_hedge + orig_abs)[:2]) or "None found."
        b_hedges = "; ".join(f'"{i.quote}"' for i in (cand_hedge + cand_abs)[:2]) or "None found."
        diffs.append(FramingDifference(
            dimension="Certainty and qualification",
            article_a_quote=a_hedges,
            article_b_quote=b_hedges,
            explanation=(
                "One article uses hedging or absolutist language; the other does not. "
                "This affects how certain or definitive the coverage appears."
            ),
            confidence="observable — lexicon-based",
            alternative_interpretation=(
                "Hedging differences may reflect the state of information at publication time "
                "rather than editorial choices about certainty."
            ),
        ))

    return diffs


def comparative_summary(
    original: ArticleAnalysisResponse,
    candidates: list[DiscoveryCandidate],
    discovered_total: int | None = None,
) -> dict:
    matched = [c for c in candidates if c.verification_status in {"verified_match", "possible_match"}]
    comparable = [
        c.analysis
        for c in candidates
        if c.verification_status == "verified_match"
        and c.analysis_scope == "full_text"
        and c.analysis
    ]
    category_counts: Counter = Counter()
    for analysis in comparable:
        category_counts.update(analysis.statistics.get("indicator_counts", {}))
    languages = sorted({c.language.code for c in matched if c.language})
    evaluated = len(candidates)
    return {
        "coverage": {
            "discovered_articles": evaluated if discovered_total is None else discovered_total,
            "evaluated_articles": evaluated,
            "matched_articles": len(matched),
            "languages_represented": languages,
            "sources_represented": sorted({c.source_name for c in matched if c.source_name}),
            "full_text_articles": sum(c.analysis_scope == "full_text" for c in matched),
            "snippet_only_articles": sum(c.analysis_scope == "snippet_only" for c in matched),
            "verification_outcomes": dict(Counter(c.verification_status for c in candidates)),
            "retrieval_outcomes": dict(Counter(c.analysis_scope for c in candidates)),
        },
        "indicator_counts_across_matches": dict(category_counts),
        "wording_comparison_basis": (
            f"Indicator counts include only verified full-text matches ({len(comparable)} article(s)); "
            "snippet-only and possible matches are excluded."
        ),
        "observation": (
            "Counts describe observable cues in available text only. Discovered articles include "
            "candidates not shortlisted for retrieval; verification outcomes use the evaluated "
            "denominator. They are not bias scores or source-quality judgments."
        ),
    }


async def compare_article(
    text: str,
    title: str | None,
    original_url: str | None,
    options: ComparisonSettings,
    settings: Settings,
    detector: LanguageDetector,
) -> ComparisonResponse:
    original = analyse_article(text, settings, detector, title, original_url)
    profile = extract_event_profile(original)
    queries = build_queries(original, options)
    raw, failures, provider_status = await discover(
        queries, options.result_limit, settings.discovery_timeout_seconds
    )
    deduped: dict[str, dict] = {item["url"]: item for item in raw}
    candidate_limit = min(
        len(deduped),
        max(options.result_limit, options.result_limit * max(1, len(queries))),
    )
    shortlisted = list(deduped.values())[:candidate_limit]

    async def retrieve(item: dict) -> tuple[dict, str | None, str]:
        fallback = item.get("snippet")
        parsed = urlparse(item["url"])
        # Google News RSS redirect links cannot be fetched server-side.
        # Mark them snippet-only rather than reporting a misleading failure.
        if (
            parsed.hostname in {"news.google.com", "news.google.co.in"}
            and parsed.path.startswith("/rss/articles")
        ):
            return item, fallback, "snippet-only: Google News redirect link; full text requires a browser"
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

    retrieved = (
        await asyncio.gather(*(bounded_retrieve(item) for item in shortlisted))
        if shortlisted
        else []
    )

    candidates: list[DiscoveryCandidate] = []
    for item, available, retrieval in retrieved:
        scope = (
            "full_text" if retrieval == "retrieved"
            else "snippet_only" if available
            else "metadata_only"
        )
        language = detector.detect(available or item["title"])
        status, score, reasons, uncertainty = verify_event(
            original,
            item["title"],
            available,
            has_full_text=retrieval == "retrieved",
            candidate_language=language.code,
        )
        analysis = (
            analyse_article(available, settings, detector, item["title"], item["url"])
            if available and len(available) >= settings.min_text_chars
            else None
        )
        candidates.append(DiscoveryCandidate(
            url=item["url"],
            title=item["title"],
            source_name=item.get("source_name"),
            published_at=item.get("published_at"),
            snippet=item.get("snippet"),
            provider=item["provider"],
            language=language,
            retrieval_status=retrieval,
            analysis_scope=scope,
            verification_status=status,
            relevance_score=score,
            match_reasons=reasons,
            uncertainty=uncertainty,
            analysis=analysis,
        ))

    # Build framing differences for verified full-text pairs
    framing_diffs: list[FramingDifference] = []
    verified_full = [
        c for c in candidates
        if c.verification_status == "verified_match"
        and c.analysis_scope == "full_text"
        and c.analysis is not None
    ]
    for candidate in verified_full[:3]:  # cap at 3 comparisons
        diffs = _framing_differences(original, candidate.analysis)  # type: ignore[arg-type]
        framing_diffs.extend(diffs)

    # If we have possible matches with full text, also compare those (clearly labelled)
    if not framing_diffs:
        possible_full = [
            c for c in candidates
            if c.verification_status == "possible_match"
            and c.analysis_scope == "full_text"
            and c.analysis is not None
        ]
        for candidate in possible_full[:1]:
            diffs = _framing_differences(original, candidate.analysis)  # type: ignore[arg-type]
            for d in diffs:
                d.confidence = (d.confidence or "") + " (possible match only — not verified same event)"
            framing_diffs.extend(diffs)

    return ComparisonResponse(
        original_analysis=original,
        event_profile=profile.as_dict(),
        discovery_queries=queries,
        candidates=candidates,
        comparison=comparative_summary(original, candidates, len(deduped)),
        framing_differences=framing_diffs,
        provider_failures=list(dict.fromkeys(failures)),
        provider_status=provider_status,
        limitations=[
            "Discovery providers can be unavailable, rate limited, or return incomplete metadata. "
            "Similarity scores are rule-based relevance signals, not calibrated probabilities.",
            "Only retrieved full text is represented as full-article analysis; snippets are explicitly scoped.",
            "An empty candidate list can mean the providers returned nothing for these queries; "
            "it is not evidence that no other coverage exists.",
            "Google News RSS returns redirect links that cannot be fetched server-side; "
            "these candidates are marked snippet-only. Full-text retrieval requires a direct publisher URL.",
            "Cross-script same-event verification (e.g. English vs Hindi) relies on date agreement "
            "and proper-noun overlap rather than lexical matching, which is structurally unavailable "
            "across different scripts.",
        ],
    )
