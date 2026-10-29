"""Extract a structured event profile from an article for cross-language search.

The profile is used to generate language-specific search queries and to verify
that a candidate article covers the same event. No translation is performed;
the profile captures observable structure from the original text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.article import ArticleAnalysisResponse

# Transliteration tables for common Indian proper nouns.
# These are one-directional (English → script) and cover only the most
# frequent patterns. They are used to build native-script query variants
# when the original article is in English.
_HINDI_TRANSLIT: dict[str, str] = {
    "delhi": "दिल्ली",
    "mumbai": "मुंबई",
    "india": "भारत",
    "government": "सरकार",
    "minister": "मंत्री",
    "flood": "बाढ़",
    "flooding": "बाढ़",
    "crisis": "संकट",
    "attack": "हमला",
    "court": "अदालत",
    "election": "चुनाव",
    "protest": "विरोध",
    "police": "पुलिस",
    "army": "सेना",
    "fire": "आग",
    "accident": "दुर्घटना",
    "death": "मौत",
    "killed": "मारे गए",
    "arrested": "गिरफ्तार",
    "parliament": "संसद",
    "supreme court": "सुप्रीम कोर्ट",
    "high court": "हाई कोर्ट",
    "karnataka": "कर्नाटक",
    "telangana": "तेलंगाना",
    "tamilnadu": "तमिलनाडु",
    "tamil nadu": "तमिलनाडु",
    "bengaluru": "बेंगलुरु",
    "bangalore": "बेंगलुरु",
    "hyderabad": "हैदराबाद",
    "chennai": "चेन्नई",
}

_KANNADA_TRANSLIT: dict[str, str] = {
    "delhi": "ದೆಹಲಿ",
    "india": "ಭಾರತ",
    "government": "ಸರ್ಕಾರ",
    "flood": "ಪ್ರವಾಹ",
    "crisis": "ಸಂಕಟ",
    "karnataka": "ಕರ್ನಾಟಕ",
    "bengaluru": "ಬೆಂಗಳೂರು",
    "bangalore": "ಬೆಂಗಳೂರು",
    "election": "ಚುನಾವಣೆ",
    "protest": "ಪ್ರತಿಭಟನೆ",
    "court": "ನ್ಯಾಯಾಲಯ",
    "minister": "ಮಂತ್ರಿ",
    "police": "ಪೊಲೀಸ್",
    "fire": "ಬೆಂಕಿ",
    "accident": "ಅಪಘಾತ",
}

_TELUGU_TRANSLIT: dict[str, str] = {
    "delhi": "ఢిల్లీ",
    "india": "భారత్",
    "government": "ప్రభుత్వం",
    "flood": "వరదలు",
    "crisis": "సంక్షోభం",
    "telangana": "తెలంగాణ",
    "hyderabad": "హైదరాబాద్",
    "andhra": "ఆంధ్ర",
    "election": "ఎన్నికలు",
    "protest": "నిరసన",
    "court": "న్యాయస్థానం",
    "minister": "మంత్రి",
    "police": "పోలీసు",
    "fire": "అగ్ని",
    "accident": "ప్రమాదం",
}

_TAMIL_TRANSLIT: dict[str, str] = {
    "delhi": "டெல்லி",
    "india": "இந்தியா",
    "government": "அரசு",
    "flood": "வெள்ளம்",
    "crisis": "நெருக்கடி",
    "tamilnadu": "தமிழ்நாடு",
    "tamil nadu": "தமிழ்நாடு",
    "chennai": "சென்னை",
    "election": "தேர்தல்",
    "protest": "போராட்டம்",
    "court": "நீதிமன்றம்",
    "minister": "அமைச்சர்",
    "police": "காவல்துறை",
    "fire": "தீ",
    "accident": "விபத்து",
}

_TRANSLIT_BY_LANG: dict[str, dict[str, str]] = {
    "hi": _HINDI_TRANSLIT,
    "kn": _KANNADA_TRANSLIT,
    "te": _TELUGU_TRANSLIT,
    "ta": _TAMIL_TRANSLIT,
}

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:jan(?:uary)?|feb(?:ruary)?|march|april|may|june|july|august"
    r"|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,?\s*\d{4})?"
    r"|\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|march|april|may|june|july|august"
    r"|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?)\b",
    re.IGNORECASE,
)
_LOCATION_HINTS = re.compile(
    r"\b(in|at|from|near|across|throughout)\s+([A-Z][a-zA-Z\s]{2,30}?)(?=[,.\s]|$)",
    re.MULTILINE,
)
_ACTION_VERBS = {
    "flood", "flooding", "flooded", "attack", "attacked", "killed", "arrested",
    "protest", "protested", "protesting", "fire", "fired", "collapse", "collapsed",
    "crash", "crashed", "announce", "announced", "ruling", "ruled", "verdict",
    "election", "elected", "resign", "resigned", "strike", "struck", "riot",
    "rioted", "explosion", "exploded", "earthquake", "landslide", "cyclone",
    "storm", "drought", "famine", "outbreak", "epidemic", "pandemic",
}
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "are",
    "has", "have", "had", "been", "said", "says", "told", "reported", "according",
    "claimed", "announced", "added", "stated", "will", "would", "could", "should",
    "after", "before", "amid", "also", "more", "than", "when", "where", "which",
    "while", "about", "into", "over", "under", "their", "there", "they", "caused",
    "cause", "causing", "during", "since", "other", "just", "like", "some", "such",
    "only", "officials", "government", "people", "its", "his", "her", "our",
    "their", "who", "what", "how", "but", "not", "all", "one", "two", "three",
    "new", "year", "day", "time", "week", "month",
}


@dataclass
class EventProfile:
    """Structured summary of the event described in an article."""
    headline: str | None
    summary: str
    actors: list[str]
    organizations: list[str]
    locations: list[str]
    dates: list[str]
    main_action: str | None
    distinctive_phrases: list[str]
    original_language: str
    is_specific_event: bool
    uncertainty_notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "actors": self.actors,
            "organizations": self.organizations,
            "locations": self.locations,
            "dates": self.dates,
            "main_action": self.main_action,
            "distinctive_phrases": self.distinctive_phrases,
            "original_language": self.original_language,
            "is_specific_event": self.is_specific_event,
            "uncertainty_notes": self.uncertainty_notes,
        }


def extract_event_profile(analysis: ArticleAnalysisResponse) -> EventProfile:
    """Extract a structured event profile from an already-analysed article."""
    text = analysis.original_text
    title = analysis.title or ""
    lang = analysis.language.code

    # Dates
    dates = list(dict.fromkeys(
        m.group() for m in _DATE_RE.finditer(f"{title} {text}")
    ))[:4]

    # Actors (PERSON_CANDIDATE entities)
    actors = list(dict.fromkeys(
        e.text for e in analysis.entities if e.label == "PERSON_CANDIDATE"
    ))[:5]

    # Organizations — capitalized multi-word sequences that are not person names
    orgs: list[str] = []
    for m in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4})\b", text):
        candidate = m.group(1)
        if candidate not in actors and len(candidate) > 4:
            orgs.append(candidate)
    orgs = list(dict.fromkeys(orgs))[:5]

    # Locations — heuristic: words following location prepositions
    locations: list[str] = []
    for m in _LOCATION_HINTS.finditer(f"{title} {text}"):
        loc = m.group(2).strip().rstrip(",.")
        if 3 < len(loc) < 40:
            locations.append(loc)
    locations = list(dict.fromkeys(locations))[:5]

    # Main action — first action verb found in title then body
    main_action: str | None = None
    for word in _WORD_RE.findall(f"{title} {text}"):
        if word.lower() in _ACTION_VERBS:
            main_action = word.lower()
            break

    # Distinctive phrases — 2-3 word sequences from title that are not stopwords
    distinctive: list[str] = []
    if title:
        title_words = [w for w in _WORD_RE.findall(title) if w.lower() not in _STOP and len(w) > 2]
        distinctive = title_words[:6]

    # Summary — first two sentences of the article
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentences[:2])[:300]

    # Specificity check — a specific event has at least one of: date, location, named actor
    is_specific = bool(dates or locations or actors)

    uncertainty: list[str] = []
    if not dates:
        uncertainty.append("No explicit date found; temporal matching will be approximate.")
    if not locations:
        uncertainty.append("No location extracted; geographic matching unavailable.")
    if not actors and not orgs:
        uncertainty.append("No named actors or organizations extracted.")
    if not is_specific:
        uncertainty.append(
            "The article may describe a broad topic rather than a specific incident; "
            "search results may be less precise."
        )

    return EventProfile(
        headline=title or None,
        summary=summary,
        actors=actors,
        organizations=orgs,
        locations=locations,
        dates=dates,
        main_action=main_action,
        distinctive_phrases=distinctive,
        original_language=lang,
        is_specific_event=is_specific,
        uncertainty_notes=uncertainty,
    )


def build_language_queries(
    profile: EventProfile,
    target_languages: list[str],
    original_language: str,
) -> dict[str, str]:
    """Build one search query per target language using the event profile.

    Strategy:
    1. Proper nouns (actors, locations) are preserved as-is — they are often
       recognisable across scripts.
    2. For languages with a transliteration table, key action/topic words are
       replaced with their native-script equivalents.
    3. The query is kept short (≤ 5 terms) to maximise recall.
    4. The original language is excluded unless explicitly requested.
    """
    queries: dict[str, str] = {}

    # Core English terms: actors + locations + main action + distinctive phrases
    # Prefer proper nouns because they survive cross-script searches better.
    anchor_terms: list[str] = []
    anchor_terms.extend(profile.actors[:2])
    anchor_terms.extend(profile.locations[:2])
    if profile.main_action:
        anchor_terms.append(profile.main_action)
    anchor_terms.extend(profile.distinctive_phrases[:3])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_anchors: list[str] = []
    for t in anchor_terms:
        key = t.casefold()
        if key not in seen and key not in _STOP:
            seen.add(key)
            unique_anchors.append(t)

    # Fallback: use headline words if no anchors extracted
    if not unique_anchors and profile.headline:
        unique_anchors = [
            w for w in _WORD_RE.findall(profile.headline)
            if w.lower() not in _STOP and len(w) > 2
        ][:5]

    base_query = " ".join(unique_anchors[:5]) or (profile.summary[:80] if profile.summary else "")

    for lang in target_languages:
        if lang == original_language:
            continue
        if lang not in {"en", "hi", "kn", "te", "ta"}:
            continue

        translit = _TRANSLIT_BY_LANG.get(lang, {})
        if not translit or original_language != "en":
            # For non-English originals or languages without a table,
            # use the base query (proper nouns are often shared).
            queries[lang] = base_query
            continue

        # Replace known English words with native-script equivalents.
        # Proper nouns (actors, locations) are kept in English because
        # they are often indexed in both scripts by news providers.
        native_terms: list[str] = []
        for term in unique_anchors[:5]:
            lower = term.lower()
            native = translit.get(lower)
            if native:
                native_terms.append(native)
            else:
                # Keep the original — proper nouns survive cross-script search
                native_terms.append(term)

        queries[lang] = " ".join(native_terms[:5]) or base_query

    return queries
