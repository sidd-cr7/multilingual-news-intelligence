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
from app.services.text_service import normalize_text, text_counts

SENTENCE_RE = re.compile(r"[^.!?\u0964\u0965]+[.!?\u0964\u0965]*", re.UNICODE)
TOKEN_RE = re.compile(r"\S+", re.UNICODE)

# Compact, inspectable seed lexicons. They are intentionally not presented as
# calibrated multilingual sentiment or emotion models.
SENTIMENT = {
    "en": {"positive": {"good", "benefit", "success", "improve", "support", "welcome"}, "negative": {"bad", "failure", "harm", "crisis", "attack", "corrupt", "danger"}},
    "hi": {"positive": {"सफल", "लाभ", "समर्थन", "बेहतर"}, "negative": {"संकट", "हमला", "हानि", "भ्रष्ट"}},
    "kn": {"positive": {"ಯಶಸ್ಸು", "ಬೆಂಬಲ", "ಲಾಭ"}, "negative": {"ಸಂಕಟ", "ದಾಳಿ", "ಹಾನಿ"}},
    "te": {"positive": {"విజయం", "మద్దతు", "లాభం"}, "negative": {"సంక్షోభం", "దాడి", "నష్టం"}},
    "ta": {"positive": {"வெற்றி", "ஆதரவு", "நன்மை"}, "negative": {"நெருக்கடி", "தாக்குதல்", "சேதம்"}},
}
EMOTIONS = {"en": {"anger": {"outrage", "furious", "anger"}, "fear": {"fear", "danger", "threat"}, "sadness": {"tragic", "grief", "sad"}, "joy": {"joy", "celebrate", "hope"}}}
LOADED = {"en": {"shocking", "disgraceful", "radical", "so-called", "brutal", "heroic", "outrageous"}, "hi": {"चौंकाने वाला", "क्रूर", "शर्मनाक"}, "kn": {"ಕ್ರೂರ", "ಆಘಾತಕಾರಿ"}, "te": {"క్రూరమైన", "సంచలనాత్మక"}, "ta": {"கொடூரமான", "அதிர்ச்சியளிக்கும்"}}
RESPONSIBILITY = {"en": {"because of", "caused by", "responsible for", "blamed", "failed to"}, "hi": {"के कारण", "जिम्मेदार", "दोषी"}, "kn": {"ಕಾರಣ", "ಜವಾಬ್ದಾರ"}, "te": {"కారణంగా", "బాధ్యత"}, "ta": {"காரணமாக", "பொறுப்பு"}}
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
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|March|April|May|June|July|Aug(?:ust)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?)\b", re.IGNORECASE)
PERSON_RE = re.compile(r"\b(?:Mr\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")

@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int

def segment(text: str, limit: int) -> list[Sentence]:
    return [Sentence(m.group(), m.start(), m.end()) for m in SENTENCE_RE.finditer(text) if m.group().strip()][:limit]

def _matches(text: str, terms: set[str]) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    lowered = text.casefold()
    for term in terms:
        start = 0
        while (index := lowered.find(term.casefold(), start)) != -1:
            found.append((index, index + len(term))); start = index + len(term)
    return found

def _context(sentences: list[Sentence], start: int) -> str:
    return next((s.text.strip() for s in sentences if s.start <= start < s.end), "")

def _evidence(category: str, text: str, spans: list[tuple[int, int]], sentences: list[Sentence], explanation: str, method: str, uncertainty: str) -> list[EvidenceSpan]:
    return [EvidenceSpan(category=category, quote=text[a:b], start=a, end=b, context=_context(sentences, a), explanation=explanation, method=method, uncertainty=uncertainty) for a, b in sorted(set(spans))]

def _validated_evidence(text: str, evidence: list[EvidenceSpan]) -> list[EvidenceSpan]:
    """Fail closed: no evidence with altered text or invalid Unicode offsets leaves the service."""
    for item in evidence:
        if not (0 <= item.start < item.end <= len(text)) or text[item.start:item.end] != item.quote:
            raise ValueError("Analysis generated an invalid evidence span.")
    return evidence

def _unattributed_assertions(text: str, sentences: list[Sentence], source_terms: set[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for sentence in sentences:
        compact = sentence.text.strip()
        lowered = compact.casefold()
        # A conservative candidate: declarative content with a number or common copula,
        # and no configured source marker. It is not a claim that the statement is false.
        has_assertion_shape = bool(re.search(r"\d|\b(?:is|are|was|were|has|have|will)\b", compact, re.IGNORECASE))
        if has_assertion_shape and not any(term.casefold() in lowered for term in source_terms):
            start = sentence.start + len(sentence.text) - len(sentence.text.lstrip())
            spans.append((start, start + len(compact)))
    return spans

def _entities(text: str) -> list[Entity]:
    entities = [Entity(text=m.group(), label="DATE", start=m.start(), end=m.end(), method="rule-based") for m in DATE_RE.finditer(text)]
    # Conservative English-only proper-name heuristic; it deliberately avoids pretending to be multilingual NER.
    entities.extend(Entity(text=m.group(1), label="PERSON_CANDIDATE", start=m.start(1), end=m.end(1), method="rule-based", confidence=None) for m in PERSON_RE.finditer(text))
    unique: dict[tuple[int, int], Entity] = {(e.start, e.end): e for e in entities}
    return list(unique.values())

def analyse_article(raw_text: str, settings: Settings, detector: LanguageDetector, title: str | None = None, original_url: str | None = None) -> ArticleAnalysisResponse:
    text = normalize_text(raw_text, settings)
    language = detector.detect(text)
    sentences = segment(text, settings.max_analysis_sentences)
    words = [m.group().casefold().strip(".,!?;:\"'()[]{}") for m in TOKEN_RE.finditer(text)]
    lexicon = SENTIMENT.get(language.code)
    sentence_results: list[SentenceSentiment] = []
    total_positive = total_negative = 0
    if lexicon:
        for sentence in sentences:
            tokens = {w.casefold().strip(".,!?;:\"'()[]{}") for w in TOKEN_RE.finditer(sentence.text)}
            positive, negative = len(tokens & lexicon["positive"]), len(tokens & lexicon["negative"])
            total_positive += positive; total_negative += negative
            label = "positive" if positive > negative else "negative" if negative > positive else "neutral"
            sentence_results.append(SentenceSentiment(text=sentence.text, start=sentence.start, end=sentence.end, label=label, score=None, method="lexicon-based heuristic"))
    indicators: list[EvidenceSpan] = []
    source_terms = SOURCE_TERMS.get(language.code, set())
    if language.code in LOADED:
        indicators += _evidence("loaded_or_evaluative_language", text, _matches(text, LOADED[language.code]), sentences, "Evaluative wording is present; its presence alone does not establish bias or intent.", "lexicon-based", "A small lexicon can miss context, quotation, irony, and language variation.")
    if language.code in RESPONSIBILITY:
        indicators += _evidence("attribution_of_responsibility", text, _matches(text, RESPONSIBILITY[language.code]), sentences, "This phrase links an actor or event to responsibility or causation.", "rule-based", "The rule identifies wording, not whether responsibility is justified.")
    source_spans = _matches(text, source_terms) if source_terms else [(m.start(), m.end()) for m in SOURCE_RE.finditer(text)]
    indicators += _evidence("attributed_claim", text, source_spans, sentences, "This wording attributes information to a reported source or claim.", "rule-based", "Attribution does not itself establish source quality, accuracy, or balance.")
    indicators += _evidence("unattributed_assertion_candidate", text, _unattributed_assertions(text, sentences, source_terms), sentences, "This declarative sentence has no configured attribution marker.", "rule-based", "This is a coverage cue only; it does not establish that the statement is false, biased, or improperly sourced.")
    if language.code in HEDGES:
        indicators += _evidence("hedging_or_qualification", text, _matches(text, HEDGES[language.code]), sentences, "This wording expresses uncertainty, qualification, or limited certainty.", "lexicon-based", "Hedging can be appropriate reporting practice and is not evidence of bias.")
    if language.code in ABSOLUTIST:
        indicators += _evidence("absolutist_or_generalising_language", text, _matches(text, ABSOLUTIST[language.code]), sentences, "This wording makes a broad or absolute-sounding claim.", "lexicon-based", "The lexicon cannot determine whether a generalisation is factually warranted or rhetorical.")
    # Potentially one-sided framing is deliberately a narrow, sentence-level co-occurrence cue.
    # It never produces an article-level label.
    loaded_spans = _matches(text, LOADED.get(language.code, set()))
    loaded_sentence_starts = {s.start for s in sentences for a, _ in loaded_spans if s.start <= a < s.end}
    one_sided = [(s.start, s.end) for s in sentences if s.start in loaded_sentence_starts and not any(a >= s.start and a < s.end for a, _ in source_spans)]
    indicators += _evidence("potentially_one_sided_framing_cue", text, one_sided, sentences, "Evaluative wording appears in a sentence without a configured attribution marker.", "rule-based composition", "Insufficient evidence to infer one-sided framing, bias, intent, or deception; quotations and context can change interpretation.")
    indicators = _validated_evidence(text, indicators)
    emotion_counts: Counter[str] = Counter()
    if language.code in EMOTIONS:
        for emotion, terms in EMOTIONS[language.code].items(): emotion_counts[emotion] = len(_matches(text, terms))
    chars, word_count, paragraphs = text_counts(text)
    indicator_counts = dict(Counter(i.category for i in indicators))
    evidence_density = round((len(indicators) / word_count) * 1000, 1) if word_count else None
    availability = [
        TaskAvailability(task="sentence_segmentation", status="available", method="rule-based", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Punctuation-based segmentation is approximate."),
        TaskAvailability(task="tokenization", status="available", method="rule-based", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Whitespace tokenization is approximate for all scripts."),
        TaskAvailability(task="pos_tagging", status="unavailable", limitation="No bundled pretrained multilingual POS model; install and configure one explicitly to enable this task."),
        TaskAvailability(task="named_entity_recognition", status="partial", method="rule-based", supported_languages=["en"], limitation="Dates and conservative English person candidates only; not a substitute for a pretrained multilingual NER model."),
        TaskAvailability(task="sentiment", status="available" if lexicon else "unsupported", method="lexicon-based heuristic" if lexicon else None, supported_languages=list(SENTIMENT), limitation="Uncalibrated lexical cues are not a model prediction and do not measure bias."),
        TaskAvailability(task="emotion", status="partial" if language.code == "en" else "unavailable", method="lexicon-based heuristic" if language.code == "en" else None, supported_languages=["en"], limitation="English seed lexicon only; no emotion claim is made when unavailable."),
        TaskAvailability(task="bias_indicator_engine", status="available", method="rule-based and lexicon-based", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Observable cues are heuristics, not conclusions about author intent, source quality, or political bias."),
    ]
    limitations = ["No article-level bias score is produced. Indicators identify observable wording and require human interpretation.", "Original text is retained and analysed without translation."]
    if len(SENTENCE_RE.findall(text)) > len(sentences): limitations.append(f"Only the first {settings.max_analysis_sentences} sentences were analysed due to the configured limit.")
    return ArticleAnalysisResponse(analysis_id=str(uuid4()), title=title, original_url=original_url, original_text=text, language=language, linguistic={"sentence_count": len(sentences), "token_count": len(words), "token_count_convention": "Whitespace-delimited non-empty tokens in the original normalized article text.", "pos_distribution": None}, entities=_entities(text), sentiment={"status": "available" if lexicon else "unsupported", "method": "lexicon-based heuristic" if lexicon else None, "label": "positive" if total_positive > total_negative else "negative" if total_negative > total_positive else "neutral", "positive_cues": total_positive, "negative_cues": total_negative, "sentences": sentence_results}, emotion={"status": "partial" if language.code == "en" else "unavailable", "method": "lexicon-based heuristic" if language.code == "en" else None, "distribution": {key: value for key, value in emotion_counts.items() if value}}, indicators=indicators, statistics={"character_count": chars, "word_count": word_count, "paragraph_count": paragraphs, "entity_count": len(_entities(text)), "indicator_counts": indicator_counts, "evidence_density_per_1000_tokens": evidence_density, "evidence_density_convention": "Raw indicator evidence spans divided by whitespace-token count, multiplied by 1,000. This is not a bias score.", "sentiment_distribution": dict(Counter(item.label for item in sentence_results)) if lexicon else None, "evidence_coverage": "Every returned evidence span is validated as an exact Unicode slice of original_text."}, task_availability=availability, limitations=limitations)
