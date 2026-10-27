"""Explainable, dependency-light multilingual article analysis.

Heuristics expose observable wording only.  They do not diagnose author intent or
prove bias.  Optional statistical NLP models can be added behind this interface.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from uuid import uuid4

from app.config import Settings
from app.schemas.article import ArticleAnalysisResponse, Entity, EvidenceSpan, SentenceSentiment, TaskAvailability
from app.services.language_service import LanguageDetector
from app.services.text_service import infer_title, normalize_text, text_counts

SENTENCE_RE = re.compile(r"[^.!?\u0964\u0965]+[.!?\u0964\u0965]*", re.UNICODE)
TOKEN_RE = re.compile(r"\S+", re.UNICODE)

# Compact, inspectable seed lexicons. They are intentionally not presented as
# calibrated multilingual sentiment or emotion models.
SENTIMENT = {
    "en": {"positive": {"good", "benefit", "success", "improve", "improved", "support", "supported", "welcome", "welcomed", "progress", "relief", "peace", "effective"}, "negative": {"bad", "failure", "failed", "harm", "harmed", "crisis", "attack", "attacked", "corrupt", "corruption", "danger", "dangerous", "death", "deaths", "loss", "losses", "damage", "damaged", "violence", "severe"}},
    "hi": {"positive": {"सफल", "लाभ", "समर्थन", "बेहतर"}, "negative": {"संकट", "हमला", "हानि", "भ्रष्ट"}},
    "kn": {"positive": {"ಯಶಸ್ಸು", "ಬೆಂಬಲ", "ಲಾಭ"}, "negative": {"ಸಂಕಟ", "ದಾಳಿ", "ಹಾನಿ"}},
    "te": {"positive": {"విజయం", "మద్దతు", "లాభం"}, "negative": {"సంక్షోభం", "దాడి", "నష్టం"}},
    "ta": {"positive": {"வெற்றி", "ஆதரவு", "நன்மை"}, "negative": {"நெருக்கடி", "தாக்குதல்", "சேதம்"}},
}
# Small, reviewed seed lexicons. Counts are evidence occurrences, not emotion
# probabilities, and every match remains traceable to the original text.
EMOTIONS = {
    "en": {"anger": {"outrage", "furious", "anger", "angry"}, "fear": {"fear", "danger", "threat", "afraid"}, "sadness": {"tragic", "tragedy", "grief", "sad", "mourning"}, "joy": {"joy", "celebrate", "celebrated", "hope", "relief"}, "disgust": {"disgusting", "disgust", "revulsion"}, "surprise": {"surprising", "shocking", "unexpected"}},
    "hi": {"anger": {"गुस्सा", "क्रोध"}, "fear": {"डर", "खतरा"}, "sadness": {"दुख", "शोक"}, "joy": {"खुशी", "उत्सव"}, "disgust": {"घृणा"}, "surprise": {"चौंकाने वाला"}},
    "kn": {"anger": {"ಕೋಪ"}, "fear": {"ಭಯ"}, "sadness": {"ದುಃಖ"}, "joy": {"ಸಂತೋಷ"}, "disgust": {"ಅಸಹ್ಯ"}, "surprise": {"ಆಘಾತಕಾರಿ"}},
    "te": {"anger": {"కోపం"}, "fear": {"భయం"}, "sadness": {"విషాదం"}, "joy": {"సంతోషం"}, "disgust": {"అసహ్యం"}, "surprise": {"సంచలనాత్మక"}},
    "ta": {"anger": {"கோபம்"}, "fear": {"பயம்"}, "sadness": {"துக்கம்"}, "joy": {"மகிழ்ச்சி"}, "disgust": {"அருவருப்பு"}, "surprise": {"அதிர்ச்சியளிக்கும்"}},
}
LOADED = {"en": {"shocking", "disgraceful", "radical", "so-called", "brutal", "heroic", "outrageous"}, "hi": {"चौंकाने वाला", "क्रूर", "शर्मनाक"}, "kn": {"ಕ್ರೂರ", "ಆಘಾತಕಾರಿ"}, "te": {"క్రూరమైన", "సంచలనాత్మక"}, "ta": {"கொடூரமான", "அதிர்ச்சியளிக்கும்"}}
RESPONSIBILITY = {"en": {"because of", "caused by", "responsible for", "blamed", "failed to"}, "hi": {"के कारण", "जिम्मेदार", "दोषी"}, "kn": {"ಕಾರಣ", "ಜವಾಬ್ದಾರ"}, "te": {"కారణంగా", "బాధ్యత"}, "ta": {"காரணமாக", "பொறுப்பு"}}
# Whole-cue matching intentionally avoids stems inside unrelated words. Keep
# attested inflected forms explicitly where a supported language commonly joins
# the grammatical ending to the responsibility noun.
RESPONSIBILITY["kn"].add("\u0c95\u0cbe\u0cb0\u0ca3\u0cb5\u0ca8\u0ccd\u0ca8\u0cc1")

SOURCE_RE = re.compile(r"\b(?:said|says|reported|according to|told|claimed|announced)\b", re.IGNORECASE)
SOURCE_TERMS = {
    "en": {"said", "says", "reported", "according to", "told", "claimed", "announced"},
    "hi": {"\u0915\u0939\u093e", "\u092c\u0924\u093e\u092f\u093e", "\u0915\u0947 \u0905\u0928\u0941\u0938\u093e\u0930"},
    "kn": {"\u0939\u0947\u0933\u0cbf\u0ca6\u0cb0\u0cc1", "\u0cb5\u0cb0\u0ca6\u0cbf"},
    "te": {"\u0c05\u0c28\u0c4d\u0c28\u0c3e\u0c30\u0c41", "\u0c1a\u0c46\u0c2a\u0c4d\u0c2a\u0c3e\u0c30\u0c41"},
    "ta": {"\u0b95\u0bc2\u0bb1\u0bbf\u0ba9\u0bbe\u0bb0\u0bcd", "\u0ba4\u0bc6\u0bb0\u0bbf\u0bb5\u0bbf\u0ba4\u0bcd\u0ba4\u0bbe\u0bb0\u0bcd"},
}
HEDGES = {
    "en": {"may", "might", "could", "possibly", "reportedly", "appears", "suggests"},
    "hi": {"\u0936\u093e\u092f\u0926", "\u0938\u0902\u092d\u0935\u0924\u0903", "\u0939\u094b \u0938\u0915\u0924\u093e \u0939\u0948"},
    "kn": {"\u0cb8\u0cbe\u0ca7\u0ccd\u0caf", "\u0c86\u0c97\u0cac\u0cb9\u0cc1\u0ca6\u0cc1"},
    "te": {"\u0c2c\u0c39\u0c41\u0c36\u0c3e", "\u0c05\u0c35\u0c15\u0c3e\u0c36\u0c02"},
    "ta": {"\u0b87\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bb2\u0bbe\u0bae\u0bcd", "\u0b8e\u0ba9\u0ba4\u0bcd \u0ba4\u0bc6\u0bb0\u0bbf\u0b95\u0bbf\u0bb1\u0ba4\u0bc1"},
}
ABSOLUTIST = {
    "en": {"always", "never", "everyone", "no one", "all", "none", "undeniably"},
    "hi": {"\u0939\u092e\u0947\u0936\u093e", "\u0915\u092d\u0940 \u0928\u0939\u0940\u0902", "\u0938\u092d\u0940", "\u0915\u094b\u0908 \u0928\u0939\u0940\u0902"},
    "kn": {"\u0caf\u0cbe\u0cb5\u0cbe\u0c97\u0cb2\u0cc2", "\u0caf\u0cbe\u0cb0\u0cc2"},
    "te": {"\u0c0e\u0c32\u0c4d\u0c32\u0c2a\u0c4d\u0c2a\u0c41\u0ca1\u0c42", "\u0c0e\u0c2a\u0c4d\u0c2a\u0c41\u0ca1\u0c42"},
    "ta": {"\u0b8e\u0baa\u0bcd\u0baa\u0bcb\u0ba4\u0bc1\u0bae\u0bcd", "\u0b92\u0bb0\u0bc1\u0baa\u0bcb\u0ba4\u0bc1\u0bae\u0bcd"},
}
# These English-only patterns label the *status of wording*, rather than the
# truth of a claim.  They are deliberately not translated into the other
# supported languages without reviewed resources for those languages.
ALLEGATION = {"alleged", "allegedly", "accused", "accusation", "allegation"}
DISPUTE = {"denied", "denies", "disputed", "disputes", "contested", "rejected"}
OPINION = {"believes", "believed", "opinion", "view", "argued", "argues"}
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|March|April|May|June|July|Aug(?:ust)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?)\b", re.IGNORECASE)
PERSON_RE = re.compile(r"\b(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
NAME_NEAR_SPEECH_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+(?:said|says|told|announced|claimed)\b")
ENTITY_STOP = {"friendship day", "education policy", "prime minister", "chief minister", "united states", "new delhi"}

@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int

def segment(text: str, limit: int) -> list[Sentence]:
    return [Sentence(m.group(), m.start(), m.end()) for m in SENTENCE_RE.finditer(text) if m.group().strip()][:limit]

ASSERTION_RE = re.compile(r"\b(?:is|are|was|were|has|have|will)\b|\d", re.IGNORECASE | re.UNICODE)


@dataclass(frozen=True)
class CueHit:
    """One matched observable cue and its exact position in the original text."""

    cue: str
    start: int
    end: int


_CUE_PATTERNS: dict[str, re.Pattern[str]] = {}


def _cue_pattern(term: str) -> re.Pattern[str]:
    """A lexicon entry must match as a complete lexical item, not as a substring.

    ``\b`` is avoided because it is unreliable at the edges of Indic scripts;
    the lookarounds only require that the entry is not embedded in a longer word.
    """
    pattern = _CUE_PATTERNS.get(term)
    if pattern is None:
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        _CUE_PATTERNS[term] = pattern
    return pattern


def _hits(text: str, terms: set[str]) -> list[CueHit]:
    found: list[CueHit] = []
    for term in sorted(terms):
        found.extend(CueHit(term, match.start(), match.end()) for match in _cue_pattern(term).finditer(text))
    return found


def _matches(text: str, terms: set[str]) -> list[tuple[int, int]]:
    return [(hit.start, hit.end) for hit in _hits(text, terms)]


def _regex_hits(pattern: re.Pattern[str], text: str) -> list[CueHit]:
    return [CueHit(match.group(), match.start(), match.end()) for match in pattern.finditer(text)]


def _sentence_at(sentences: list[Sentence], start: int) -> Sentence | None:
    return next((sentence for sentence in sentences if sentence.start <= start < sentence.end), None)


def _context(sentences: list[Sentence], start: int) -> str:
    sentence = _sentence_at(sentences, start)
    return sentence.text.strip() if sentence else ""


def _evidence(category: str, text: str, hits: list[CueHit], sentences: list[Sentence], explanation: str, method: str, uncertainty: str, strength: str, *, group: str = "cue") -> list[EvidenceSpan]:
    """Collapse repeated wording into one finding per distinct cue.

    Repeated occurrences are counted rather than duplicated as separate findings,
    but they are never erased: the anchor keeps its exact quote and offsets and
    ``occurrence_spans`` records every occurrence that contributed to the count.
    Sentence-level rules use ``group="span"`` so distinct sentences stay distinct.
    """
    grouped: dict[tuple, list[CueHit]] = {}
    for hit in sorted(hits, key=lambda item: (item.start, item.end)):
        key = (hit.start, hit.end) if group == "span" else (hit.cue.casefold(),)
        grouped.setdefault(key, []).append(hit)
    evidence: list[EvidenceSpan] = []
    for members in grouped.values():
        spans = sorted({(hit.start, hit.end) for hit in members})
        anchor_start, anchor_end = spans[0]
        contexts = list(dict.fromkeys(
            context for context in (_context(sentences, start) for start, _ in spans) if context
        ))[:5]
        occurrences = len(spans)
        detail = explanation if occurrences == 1 else f"{explanation} This cue appears {occurrences} times in the analysed text."
        evidence.append(
            EvidenceSpan(
                category=category,
                quote=text[anchor_start:anchor_end],
                start=anchor_start,
                end=anchor_end,
                context=_context(sentences, anchor_start),
                explanation=detail,
                method=method,
                uncertainty=uncertainty,
                cue=members[0].cue,
                occurrences=occurrences,
                occurrence_spans=spans,
                contexts=contexts,
                evidence_strength=strength,
            )
        )
    return sorted(evidence, key=lambda item: item.start)


def _validated_evidence(text: str, evidence: list[EvidenceSpan]) -> list[EvidenceSpan]:
    """Fail closed: no evidence with altered text or invalid Unicode offsets leaves the service."""
    for item in evidence:
        if not (0 <= item.start < item.end <= len(text)) or text[item.start:item.end] != item.quote:
            raise ValueError("Analysis generated an invalid evidence span.")
        occurrence_spans = item.occurrence_spans or [(item.start, item.end)]
        if item.occurrences != len(occurrence_spans):
            raise ValueError("Analysis generated inconsistent evidence occurrence counts.")
        for start, end in occurrence_spans:
            if not (0 <= start < end <= len(text)) or text[start:end].casefold() != item.quote.casefold():
                raise ValueError("Analysis generated an invalid evidence occurrence span.")
    return evidence


def _unattributed_assertions(sentences: list[Sentence], source_hits: list[CueHit], framing_hits: list[CueHit]) -> list[CueHit]:
    """Framing-relevant declarative sentences with no configured attribution marker.

    The cue is anchored on the sentence's first predicate token rather than on the
    whole sentence, so the returned span stays a precise, inspectable slice. This
    is a coverage observation, not a claim that the statement is false or biased.
    """
    hits: list[CueHit] = []
    if not framing_hits:
        return hits
    for sentence in sentences:
        if any(sentence.start <= source.start < sentence.end for source in source_hits):
            continue
        if not any(sentence.start <= hit.start < sentence.end for hit in framing_hits):
            continue
        match = ASSERTION_RE.search(sentence.text)
        if match:
            hits.append(CueHit(match.group(), sentence.start + match.start(), sentence.start + match.end()))
    return hits


def _one_sided_hits(loaded_hits: list[CueHit], source_hits: list[CueHit], sentences: list[Sentence]) -> list[CueHit]:
    """Evaluative cues inside sentences that carry no configured attribution."""
    hits: list[CueHit] = []
    for hit in loaded_hits:
        sentence = _sentence_at(sentences, hit.start)
        if sentence and not any(sentence.start <= source.start < sentence.end for source in source_hits):
            hits.append(hit)
    return hits


def _source_representation(sentences: list[Sentence], source_hits: list[CueHit]) -> dict[str, int | str]:
    """Return conservative, descriptive attribution counts.

    A marker is not a speaker identity: this avoids inventing named sources
    from syntax or claiming that unmarked sentences are unsupported.  Counts
    are sentence-based so repeated ``said`` tokens in one sentence do not
    inflate the apparent diversity of sources.
    """
    attributed_sentences = sum(
        any(sentence.start <= hit.start < sentence.end for hit in source_hits)
        for sentence in sentences
    )
    return {
        "attribution_marker_occurrences": len({(hit.start, hit.end) for hit in source_hits}),
        "sentences_with_attribution_markers": attributed_sentences,
        "sentences_without_attribution_markers": len(sentences) - attributed_sentences,
        "interpretation": "Descriptive marker counts only; they do not identify unique sources, prove balance, or show that unmarked reporting lacks support.",
    }


def _framing_analysis(indicators: list[EvidenceSpan]) -> dict:
    """Translate observable cues into cautious, evidence-linked framing notes."""
    labels = {
        "loaded_or_evaluative_language": ("Potentially evaluative framing", "The wording may colour how an event or actor is perceived."),
        "potentially_one_sided_framing_cue": ("Tentative evaluative framing", "The cue appears without a configured attribution marker."),
        "absolutist_or_generalising_language": ("Potential generalisation", "Broad wording can make a narrower claim sound universal."),
        "hedging_or_qualification": ("Qualified claim", "Uncertainty wording can appropriately limit certainty."),
        "attribution_of_responsibility": ("Responsibility framing", "Causal or blame wording can focus attention on a particular actor or cause."),
        "allegation_or_accusation": ("Allegation framing", "The wording presents a claim as an allegation, not an established fact."),
        "explicitly_disputed_claim": ("Disputed claim", "The wording records disagreement rather than resolving it."),
    }
    findings = []
    for item in indicators:
        if item.category not in labels:
            continue
        label, effect = labels[item.category]
        findings.append({"category": item.category, "label": label, "evidence_text": item.quote, "start_offset": item.start, "end_offset": item.end, "explanation": item.explanation, "potential_effect": effect, "interpretation": "tentative" if "potential" in item.category or item.category in {"loaded_or_evaluative_language", "absolutist_or_generalising_language"} else "contextually supported", "alternative_interpretation": item.uncertainty or "The wording may be appropriate and does not establish bias or factual inaccuracy.", "method": item.method})
    if not findings:
        assessment = "No reliable overall framing conclusion can be drawn because the configured indicators did not identify a supported pattern."
    elif any(item["category"] in {"loaded_or_evaluative_language", "potentially_one_sided_framing_cue"} for item in findings):
        assessment = "The article contains some evaluative wording. This is a tentative framing observation, not a finding of bias, intent, or factual inaccuracy."
    else:
        assessment = "The article contains observable attribution, certainty, or responsibility patterns, but the available evidence does not support an overall bias conclusion."
    return {"overall_assessment": assessment, "findings": findings, "limitation": "This rule-based analysis identifies wording patterns only. It cannot establish factual accuracy, political orientation, omitted perspectives, or author intent."}

def _entities(text: str) -> list[Entity]:
    entities = [Entity(text=m.group(), label="DATE", start=m.start(), end=m.end(), method="rule-based") for m in DATE_RE.finditer(text)]
    # A capitalised phrase alone is not a person.  Require an honorific or a
    # speech attribution construction and retain the candidate label.
    for pattern in (PERSON_RE, NAME_NEAR_SPEECH_RE):
        for match in pattern.finditer(text):
            name = match.group(1)
            if name.casefold() not in ENTITY_STOP:
                entities.append(Entity(text=name, label="PERSON_CANDIDATE", start=match.start(1), end=match.end(1), method="conservative rule-based candidate", confidence=None))
    unique: dict[tuple[int, int], Entity] = {(e.start, e.end): e for e in entities}
    return list(unique.values())

def analyse_article(raw_text: str, settings: Settings, detector: LanguageDetector, title: str | None = None, original_url: str | None = None) -> ArticleAnalysisResponse:
    text = normalize_text(raw_text, settings)
    language = detector.detect(text)
    sentences = segment(text, settings.max_analysis_sentences)
    resolved_title = title.strip() if title and title.strip() else infer_title(text)
    title_source = "source-provided" if title and title.strip() else "inferred-opening-text"
    words = [m.group().casefold().strip(".,!?;:\"'()[]{}") for m in TOKEN_RE.finditer(text)]
    lexicon = SENTIMENT.get(language.code)
    sentence_results: list[SentenceSentiment] = []
    total_positive = total_negative = 0
    if lexicon:
        for sentence in sentences:
            tokens = {match.group().casefold().strip(".,!?;:\"'()[]{}") for match in TOKEN_RE.finditer(sentence.text)}
            positive, negative = len(tokens & lexicon["positive"]), len(tokens & lexicon["negative"])
            total_positive += positive; total_negative += negative
            label = "positive" if positive > negative else "negative" if negative > positive else "neutral"
            sentence_results.append(SentenceSentiment(text=sentence.text, start=sentence.start, end=sentence.end, label=label, score=None, method="lexicon-based heuristic"))
    indicators: list[EvidenceSpan] = []
    source_terms = SOURCE_TERMS.get(language.code, set())
    source_hits = _hits(text, source_terms) if source_terms else _regex_hits(SOURCE_RE, text)
    loaded_hits = _hits(text, LOADED.get(language.code, set()))
    responsibility_hits = _hits(text, RESPONSIBILITY.get(language.code, set()))
    hedge_hits = _hits(text, HEDGES.get(language.code, set()))
    absolutist_hits = _hits(text, ABSOLUTIST.get(language.code, set()))
    allegation_hits = _hits(text, ALLEGATION) if language.code == "en" else []
    dispute_hits = _hits(text, DISPUTE) if language.code == "en" else []
    opinion_hits = _hits(text, OPINION) if language.code == "en" else []
    if loaded_hits:
        indicators += _evidence("loaded_or_evaluative_language", text, loaded_hits, sentences, "Evaluative wording can colour how an event or actor is perceived.", "lexicon-based", "A small lexicon can miss context, quotation, irony, and language variation; presence alone does not establish bias or intent.", "direct lexical match")
    if responsibility_hits:
        indicators += _evidence("attribution_of_responsibility", text, responsibility_hits, sentences, "This phrase links an actor or event to responsibility or causation, which can direct attention to a blamed party.", "rule-based", "The rule identifies wording, not whether the attributed responsibility is justified.", "direct phrasal match")
    if source_hits:
        indicators += _evidence("attributed_claim", text, source_hits, sentences, "This wording attributes information to a reported source rather than stating it as the reporter's own assertion.", "rule-based", "Attribution does not itself establish source quality, accuracy, or balance.", "attribution marker match")
    assertion_hits = _unattributed_assertions(sentences, source_hits, loaded_hits + responsibility_hits + hedge_hits + absolutist_hits)
    if assertion_hits:
        indicators += _evidence("unattributed_assertion_candidate", text, assertion_hits, sentences, "A sentence with framing-relevant wording carries no configured attribution marker; the predicate is highlighted as the assertion anchor.", "rule-based", "This is a coverage cue only; it does not establish that the statement is false, biased, or improperly sourced.", "sentence-level rule", group="span")
    if hedge_hits:
        indicators += _evidence("hedging_or_qualification", text, hedge_hits, sentences, "This wording expresses uncertainty, qualification, or limited certainty.", "lexicon-based", "Hedging can be appropriate reporting practice and is not evidence of bias.", "direct lexical match")
    if absolutist_hits:
        indicators += _evidence("absolutist_or_generalising_language", text, absolutist_hits, sentences, "This wording makes a broad or absolute-sounding claim, which can overstate a pattern if the context is narrower.", "lexicon-based", "The lexicon cannot determine whether a generalisation is factually warranted or rhetorical.", "direct lexical match")
    if allegation_hits:
        indicators += _evidence("allegation_or_accusation", text, allegation_hits, sentences, "This wording signals that the surrounding claim is presented as an allegation or accusation rather than an established fact.", "English lexical rule", "The rule does not determine whether the allegation is true, false, sourced, or fairly represented.", "direct lexical match")
    if dispute_hits:
        indicators += _evidence("explicitly_disputed_claim", text, dispute_hits, sentences, "This wording signals that a claim, account, or decision is explicitly disputed or denied.", "English lexical rule", "The rule does not identify every party or resolve the dispute.", "direct lexical match")
    if opinion_hits:
        indicators += _evidence("opinion_or_interpretation_marker", text, opinion_hits, sentences, "This wording marks an opinion, belief, argument, or interpretation in the surrounding sentence.", "English lexical rule", "An opinion marker does not establish the speaker, evidence quality, or whether readers should accept the interpretation.", "direct lexical match")
    # Potentially one-sided framing is deliberately a narrow, sentence-level co-occurrence cue.
    # It never produces an article-level label; the anchor is the evaluative cue itself.
    one_sided_hits = _one_sided_hits(loaded_hits, source_hits, sentences)
    if one_sided_hits:
        indicators += _evidence("potentially_one_sided_framing_cue", text, one_sided_hits, sentences, "Evaluative wording appears in a sentence with no configured attribution marker; the highlighted cue is the co-occurrence anchor.", "rule-based composition", "Insufficient evidence to infer one-sided framing, bias, intent, or deception; quotations and context can change interpretation.", "sentence co-occurrence rule")
    indicators = _validated_evidence(text, indicators)
    emotion_counts: Counter[str] = Counter()
    emotion_evidence: list[EvidenceSpan] = []
    emotion_lexicon = EMOTIONS.get(language.code)
    if emotion_lexicon:
        for emotion, terms in emotion_lexicon.items():
            hits = _hits(text, terms)
            emotion_counts[emotion] = len(hits)
            emotion_evidence += _evidence(f"emotion_{emotion}", text, hits, sentences, f"This wording is a seed-lexicon cue associated with {emotion}.", "language-specific lexicon", "A cue may be quoted, descriptive, ironic, or contextually neutral; counts are not confidence scores.", "direct lexical match")
    chars, word_count, paragraphs = text_counts(text)
    entities = _entities(text)
    summary_sentences = [sentence.text.strip() for sentence in sentences if len(sentence.text.split()) >= 5]
    summary = {
        "status": "available" if summary_sentences else "unavailable",
        "method": "extractive opening-sentence baseline" if summary_sentences else None,
        "text": " ".join(summary_sentences[:2]) if summary_sentences else None,
        "limitation": "Extractive summary retains original sentences and may omit later context; it does not generate or verify facts.",
    }
    indicator_counts = dict(Counter(item.category for item in indicators))
    occurrence_counts: Counter[str] = Counter()
    for item in indicators:
        occurrence_counts[item.category] += item.occurrences
    total_occurrences = sum(item.occurrences for item in indicators)
    evidence_density = round((total_occurrences / word_count) * 1000, 1) if word_count else None
    availability = [
        TaskAvailability(task="sentence_segmentation", status="available", method="rule-based", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Punctuation-based segmentation is approximate."),
        TaskAvailability(task="tokenization", status="available", method="rule-based", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Whitespace tokenization is approximate for all scripts."),
        TaskAvailability(task="pos_tagging", status="unavailable", limitation="No bundled pretrained multilingual POS model; install and configure one explicitly to enable this task."),
        TaskAvailability(task="named_entity_recognition", status="partial", method="rule-based", supported_languages=["en"], limitation="Dates and conservative English person candidates only; not a substitute for a pretrained multilingual NER model."),
        TaskAvailability(task="sentiment", status="available" if lexicon else "unsupported", method="lexicon-based heuristic" if lexicon else None, supported_languages=list(SENTIMENT), limitation="Uncalibrated lexical cues are not a model prediction and do not measure bias."),
        TaskAvailability(task="emotion", status="available" if emotion_lexicon else "unsupported", method="language-specific seed lexicon" if emotion_lexicon else None, supported_languages=list(EMOTIONS), limitation="Seed lexicons identify explicit emotion wording; they are not calibrated multilingual emotion models and neutral means no configured cue matched."),
        TaskAvailability(task="bias_indicator_engine", status="available", method="rule-based and lexicon-based", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Observable cues are heuristics, not conclusions about author intent, source quality, or political bias."),
        TaskAvailability(task="claim_status_markers", status="available" if language.code == "en" else "unavailable", method="English lexical rules" if language.code == "en" else None, supported_languages=["en"], limitation="Allegation, dispute, and opinion status markers are currently reviewed only for English; no cross-language equivalence is claimed."),
        TaskAvailability(task="source_representation", status="partial", method="sentence-level attribution-marker counts", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Counts configured attribution markers, not named or unique quoted sources; they are descriptive statistics only."),
    ]
    limitations = ["No article-level bias score is produced. Indicators identify observable wording and require human interpretation.", "Original text is retained and analysed without translation.", "Repeated occurrences of the same cue are reported once per category with an occurrence count; counts describe observable wording, not independent evidence of bias.", "The unattributed-assertion rule requires framing-relevant wording plus a copular or numeric predicate, and its absence does not show that a sentence is properly attributed.", "Allegation, dispute, and opinion-status markers are supported only for English. Other language-specific capabilities are limited to the explicitly listed cue resources."]
    if len(SENTENCE_RE.findall(text)) > len(sentences): limitations.append(f"Only the first {settings.max_analysis_sentences} sentences were analysed due to the configured limit.")
    emotion_distribution = {key: value for key, value in emotion_counts.items() if value}
    emotion = {"status": "available" if emotion_lexicon else "unsupported", "method": "language-specific seed lexicon" if emotion_lexicon else None, "distribution": emotion_distribution, "dominant_emotion": max(emotion_distribution, key=emotion_distribution.get) if emotion_distribution else "neutral", "evidence": [item.model_dump() for item in emotion_evidence], "limitation": "Neutral means no configured emotion cue matched; cue counts are not probabilities and do not establish the writer's or reader's emotion."}
    return ArticleAnalysisResponse(analysis_id=str(uuid4()), title=resolved_title, title_source=title_source, original_url=original_url, original_text=text, language=language, linguistic={"sentence_count": len(sentences), "token_count": len(words), "token_count_convention": "Whitespace-delimited non-empty tokens in the original normalized article text.", "pos_distribution": None}, entities=entities, sentiment={"status": "available" if lexicon else "unsupported", "method": "lexicon-based heuristic" if lexicon else None, "label": "positive" if total_positive > total_negative else "negative" if total_negative > total_positive else "neutral", "positive_cues": total_positive, "negative_cues": total_negative, "sentences": sentence_results}, emotion=emotion, framing_analysis=_framing_analysis(indicators), summary=summary, indicators=indicators, statistics={"character_count": chars, "word_count": word_count, "paragraph_count": paragraphs, "entity_count": len(entities), "indicator_counts": indicator_counts, "indicator_occurrences": dict(occurrence_counts), "source_representation": _source_representation(sentences, source_hits), "total_evidence_findings": len(indicators), "total_cue_occurrences": total_occurrences, "evidence_density_per_1000_tokens": evidence_density, "sentiment_distribution": dict(Counter(item.label for item in sentence_results)) if lexicon else None, "evidence_coverage": "Repeated wording is reported once per category with an occurrence count. Every anchor quote and every occurrence span is validated as an exact Unicode slice of original_text."}, task_availability=availability, limitations=limitations)
