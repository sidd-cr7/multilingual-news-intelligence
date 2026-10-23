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
    if language.code in LOADED:
        indicators += _evidence("loaded_or_evaluative_language", text, _matches(text, LOADED[language.code]), sentences, "Evaluative wording is present; its presence alone does not establish bias or intent.", "lexicon-based", "A small lexicon can miss context, quotation, irony, and language variation.")
    if language.code in RESPONSIBILITY:
        indicators += _evidence("attribution_of_responsibility", text, _matches(text, RESPONSIBILITY[language.code]), sentences, "This phrase links an actor or event to responsibility or causation.", "rule-based", "The rule identifies wording, not whether responsibility is justified.")
    indicators += _evidence("source_attribution", text, [(m.start(), m.end()) for m in SOURCE_RE.finditer(text)], sentences, "This reporting verb signals attributed information or a source claim.", "rule-based", "Source presence does not by itself indicate balance, accuracy, or bias.")
    emotion_counts: Counter[str] = Counter()
    if language.code in EMOTIONS:
        for emotion, terms in EMOTIONS[language.code].items(): emotion_counts[emotion] = len(_matches(text, terms))
    chars, word_count, paragraphs = text_counts(text)
    availability = [
        TaskAvailability(task="sentence_segmentation", status="available", method="rule-based", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Punctuation-based segmentation is approximate."),
        TaskAvailability(task="tokenization", status="available", method="rule-based", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Whitespace tokenization is approximate for all scripts."),
        TaskAvailability(task="pos_tagging", status="unavailable", limitation="No bundled pretrained multilingual POS model; install and configure one explicitly to enable this task."),
        TaskAvailability(task="named_entity_recognition", status="partial", method="rule-based", supported_languages=["en"], limitation="Dates and conservative English person candidates only; not a substitute for a pretrained multilingual NER model."),
        TaskAvailability(task="sentiment", status="available" if lexicon else "unsupported", method="lexicon-based heuristic" if lexicon else None, supported_languages=list(SENTIMENT), limitation="Uncalibrated lexical cues are not a model prediction and do not measure bias."),
        TaskAvailability(task="emotion", status="partial" if language.code == "en" else "unavailable", method="lexicon-based heuristic" if language.code == "en" else None, supported_languages=["en"], limitation="English seed lexicon only; no emotion claim is made when unavailable."),
        TaskAvailability(task="framing_indicators", status="available", method="rule-based and lexicon-based", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Observable cues are heuristics, not conclusions about author intent or political bias."),
    ]
    limitations = ["No article-level bias score is produced. Indicators identify observable wording and require human interpretation.", "Original text is retained and analysed without translation."]
    if len(SENTENCE_RE.findall(text)) > len(sentences): limitations.append(f"Only the first {settings.max_analysis_sentences} sentences were analysed due to the configured limit.")
    return ArticleAnalysisResponse(analysis_id=str(uuid4()), title=title, original_url=original_url, original_text=text, language=language, linguistic={"sentence_count": len(sentences), "token_count": len(words), "pos_distribution": None}, entities=_entities(text), sentiment={"status": "available" if lexicon else "unsupported", "method": "lexicon-based heuristic" if lexicon else None, "label": "positive" if total_positive > total_negative else "negative" if total_negative > total_positive else "neutral", "positive_cues": total_positive, "negative_cues": total_negative, "sentences": sentence_results}, emotion={"status": "partial" if language.code == "en" else "unavailable", "method": "lexicon-based heuristic" if language.code == "en" else None, "distribution": {key: value for key, value in emotion_counts.items() if value}}, indicators=indicators, statistics={"character_count": chars, "word_count": word_count, "paragraph_count": paragraphs, "entity_count": len(_entities(text)), "indicator_counts": dict(Counter(i.category for i in indicators)), "evidence_coverage": "Indicator evidence includes exact original-text offsets."}, task_availability=availability, limitations=limitations)
