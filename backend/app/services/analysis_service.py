"""Explainable, dependency-light multilingual article analysis.

Heuristics expose observable wording only.  They do not diagnose author intent or
prove bias.  Optional statistical NLP models can be added behind this interface.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from uuid import uuid4

from app.config import Settings
from app.schemas.article import ArticleAnalysisResponse, Entity, EvidenceSpan, SentenceSentiment, TaskAvailability
from app.services.language_service import LanguageDetector
from app.services.bias_judgment_service import build_bias_judgment
from app.services.groq_framing_service import GroqFramingService
from app.services.model_manager import MODEL_CATALOG, ModelInference, MultilingualModelManager
from app.services.text_service import normalize_text, text_counts

SENTENCE_RE = re.compile(r"[^.!?\u0964\u0965]+[.!?\u0964\u0965]*", re.UNICODE)
TOKEN_RE = re.compile(r"\S+", re.UNICODE)

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
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|March|April|May|June|July|Aug(?:ust)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?)\b", re.IGNORECASE)
PERSON_RE = re.compile(r"\b(?:Mr\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")

@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


@lru_cache(maxsize=4)
def _shared_model_manager(
    sentiment_model_id: str | None,
    emotion_model_id: str | None,
    ner_model_id: str | None,
    sentiment_model_path: str | None,
    emotion_model_path: str | None,
    ner_model_path: str | None,
    ner_model_id_te: str | None,
) -> MultilingualModelManager:
    return MultilingualModelManager(
        configured_ids={
            "sentiment_model_id": sentiment_model_id,
            "emotion_model_id": emotion_model_id,
            "ner_model_id": ner_model_id,
            "ner_model_id_te": ner_model_id_te,
        },
        configured_paths={
            "sentiment_model_path": sentiment_model_path,
            "emotion_model_path": emotion_model_path,
            "ner_model_path": ner_model_path,
        },
    )


def _run_model_task(
    manager: MultilingualModelManager,
    task: str,
    language: str,
    sentences: list[Sentence],
    max_sentences: int,
    max_input_chars: int,
) -> tuple[str, str | None, str | None, list[dict], str | None]:
    spec = manager.resolve(task, language=language)
    if spec.status != "inconclusive" or not spec.model_id:
        return spec.status, spec.model_id, None, [], spec.notes
    selected = sentences[:max(0, max_sentences)]
    if not selected:
        return "inconclusive", spec.model_id, None, [], "No sentence was available for model inference."

    predictions: list[dict] = []
    method: str | None = None
    limitation: str | None = None
    status = "available_with_limitations"
    for sentence in selected:
        input_text = sentence.text[:max(1, max_input_chars)]
        result: ModelInference = manager.infer(task, input_text, language=language)
        if result.status != "available_with_limitations":
            status = result.status
            limitation = result.limitation
            break
        method = result.method
        limitation = result.limitation
        if task == "named_entity_recognition":
            for entity in result.predictions:
                start, end = entity.get("start"), entity.get("end")
                if not (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and 0 <= start < end <= len(sentence.text)
                ):
                    status = "inference_failed"
                    limitation = "The NER model returned an invalid original-text offset."
                    break
                while start < end and sentence.text[start].isspace():
                    start += 1
                while start < end and sentence.text[end - 1].isspace():
                    end -= 1
                if start >= end:
                    status = "inference_failed"
                    limitation = "The NER model returned an entity span containing only whitespace."
                    break
                predictions.append({
                    "text": sentence.text[start:end],
                    "label": entity.get("entity_group", entity.get("entity", "UNKNOWN")),
                    "start": sentence.start + start,
                    "end": sentence.start + end,
                    "score": entity.get("score"),
                })
            if status == "inference_failed":
                break
            continue

        scores: dict[str, float] = {}
        for item in result.predictions:
            label = str(item.get("label", "")).strip().casefold()
            score = item.get("score")
            if label and isinstance(score, (int, float)) and isfinite(float(score)):
                scores[label] = float(score)
        if not scores:
            status = "inference_failed"
            limitation = "The configured classifier returned no usable label scores."
            break
        label = max(scores, key=scores.get)
        predictions.append({
            "text": sentence.text,
            "start": sentence.start,
            "end": sentence.end,
            "label": label,
            "score": scores[label],
            "class_scores": scores,
        })

    notes = [limitation] if limitation else []
    if len(selected) < len(sentences):
        notes.append(f"Model inference was limited to the first {len(selected)} of {len(sentences)} sentences.")
    if any(len(sentence.text) > max(1, max_input_chars) for sentence in selected):
        notes.append(f"Sentences longer than {max_input_chars} characters were truncated before model inference.")
    if notes:
        limitation = " ".join(notes)
    return status, spec.model_id, method, predictions, limitation

def segment(text: str, limit: int) -> list[Sentence]:
    """Split on sentence terminators while handling Indic script end punctuation.

    This is deliberately light-weight: it keeps the original text untouched,
    preserves offsets, and avoids splitting on common abbreviations such as
    "Dr." or "Mr." when the trailing period is followed by a lowercase letter.
    """
    if not text:
        return []
    matches: list[Sentence] = []
    start = 0
    for index, character in enumerate(text):
        if character not in ".!?\u0964\u0965":
            continue
        if character in ".!?" and re.match(
            r'["”]\s*,?\s+(?:(?:the|a|an)\s+)?'
            r'(?:minister|official|spokesperson|leader|opposition leader|activist|'
            r'judge|police|government|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+'
            r'(?:said|says|claimed|argued|reported|told)\b',
            text[index + 1:],
            re.IGNORECASE,
        ):
            continue
        if character == "." and re.search(
            r"(?:\bMr|\bMrs|\bMs|\bDr|\bProf|\bSt|\bvs|\betc)\.$",
            text[start:index + 1],
            re.IGNORECASE,
        ):
            continue
        end = index + 1
        while end < len(text) and text[end] in ".!?\u0964\u0965":
            end += 1
        while end < len(text) and text[end] in "\"'”’)]}":
            end += 1
        while end < len(text) and text[end].isspace():
            end += 1
        content_start = start
        while content_start < index + 1 and text[content_start].isspace():
            content_start += 1
        content_end = end
        while content_end > content_start and text[content_end - 1].isspace():
            content_end -= 1
        if content_start < content_end:
            matches.append(Sentence(text[content_start:content_end], content_start, content_end))
        start = end
        # The for-loop index has already advanced; skip whitespace and preserve
        # the true next sentence start on the next iteration.
        if start > index + 1:
            continue
    tail_start = start
    while tail_start < len(text) and text[tail_start].isspace():
        tail_start += 1
    if tail_start < len(text):
        matches.append(Sentence(text[tail_start:], tail_start, len(text)))
    return matches[:limit]

CATEGORY_DEFINITIONS = {
    "loaded_or_evaluative_language": "Words that carry an evaluative or emotive judgment rather than a neutral description.",
    "attribution_of_responsibility": "Phrases that explicitly link an actor or event to responsibility, cause, or blame.",
    "attributed_claim": "A reporting verb or attribution marker that presents information as coming from a source.",
    "hedging_or_qualification": "Wording that limits certainty or explicitly marks a statement as provisional.",
    "absolutist_or_generalising_language": "Broad or absolute quantifiers that can overstate a pattern.",
}
CLAIM_STATUS_DEFINITIONS = {
    "attributed": "The anchor sentence contains a configured attribution marker; the wording is presented as reported rather than as the reporter's own assertion.",
    "unattributed": "No configured attribution marker appears in the anchor sentence. This is a coverage observation, not proof that the statement is unsourced or false.",
    "attribution_marker": "The finding is the attribution marker itself; it describes how information is sourced, not whether the source is reliable.",
    "mixed": "Repeated occurrences of this cue appear in sentences with different attribution contexts.",
}


def _methodology() -> dict:
    return {
        "approach": "Framing observations are evidence-first and rule-based for selected English and limited Hindi patterns; Kannada, Telugu, and Tamil framing is not run. Sentiment, emotion, or NER model predictions appear only after configured local inference succeeds. No translation or numerical bias score is produced.",
        "categories": CATEGORY_DEFINITIONS,
        "claim_status": CLAIM_STATUS_DEFINITIONS,
        "offsets": "Zero-based Unicode character offsets with an exclusive end. Every quote is an exact slice of original_text.",
        "interpretation": "Findings are review cues. Presence of a cue does not establish bias, intent, deception, or source quality.",
    }


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


def _claim_status(sentences: list[Sentence], source_hits: list[CueHit], start: int, override: str | None) -> str | None:
    """Describe attribution around the anchor sentence without judging truth or source quality."""
    if override:
        return override
    sentence = _sentence_at(sentences, start)
    if sentence is None:
        return None
    if any(sentence.start <= source.start < sentence.end for source in source_hits):
        return "attributed"
    return "unattributed"


def _evidence(category: str, text: str, hits: list[CueHit], sentences: list[Sentence], explanation: str, method: str, uncertainty: str, strength: str, *, group: str = "cue", source_hits: list[CueHit] | None = None, claim_status_override: str | None = None) -> list[EvidenceSpan]:
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
        statuses = {
            status
            for start, _ in spans
            if (status := _claim_status(sentences, source_hits or [], start, claim_status_override))
        }
        claim_status = next(iter(statuses)) if len(statuses) == 1 else ("mixed" if statuses else None)
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
                claim_status=claim_status,
                claim_status_note=CLAIM_STATUS_DEFINITIONS.get(claim_status) if claim_status else None,
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
    model_manager = _shared_model_manager(
        settings.sentiment_model_id,
        settings.emotion_model_id,
        settings.ner_model_id,
        settings.sentiment_model_path,
        settings.emotion_model_path,
        settings.ner_model_path,
        settings.ner_model_id_te,
    )
    max_model_sentences = max(0, min(settings.max_model_sentences, 32, settings.max_analysis_sentences))
    max_model_input_chars = max(1, min(settings.max_model_input_chars, 4_000))
    sentiment_status, sentiment_model_id, sentiment_method, sentiment_predictions, sentiment_limitation = _run_model_task(
        model_manager, "sentiment", language.code, sentences,
        max_model_sentences, max_model_input_chars,
    )
    emotion_status, emotion_model_id, emotion_method, emotion_predictions, emotion_limitation = _run_model_task(
        model_manager, "emotion", language.code, sentences,
        max_model_sentences, max_model_input_chars,
    )
    ner_status, ner_model_id, ner_method, ner_predictions, ner_limitation = _run_model_task(
        model_manager, "named_entity_recognition", language.code, sentences,
        max_model_sentences, max_model_input_chars,
    )
    sentence_results = [
        SentenceSentiment(
            text=item["text"],
            start=item["start"],
            end=item["end"],
            label=item["label"],
            score=item["score"],
            method=sentiment_method or "configured classifier",
            model_identifier=sentiment_model_id,
            class_scores=item["class_scores"],
            uncertainty="Raw classifier scores are not calibrated for news articles; quotation, sarcasm, and speaker attribution are not resolved.",
        )
        for item in sentiment_predictions
    ]
    indicators: list[EvidenceSpan] = []
    source_terms = SOURCE_TERMS.get(language.code, set())
    source_hits = _hits(text, source_terms) if source_terms else _regex_hits(SOURCE_RE, text)
    loaded_hits = _hits(text, LOADED.get(language.code, set()))
    responsibility_hits = _hits(text, RESPONSIBILITY.get(language.code, set()))
    hedge_hits = _hits(text, HEDGES.get(language.code, set()))
    absolutist_hits = _hits(text, ABSOLUTIST.get(language.code, set()))
    if loaded_hits:
        indicators += _evidence("loaded_or_evaluative_language", text, loaded_hits, sentences, "Evaluative wording can colour how an event or actor is perceived.", "lexicon-based", "A small lexicon can miss context, quotation, irony, and language variation; presence alone does not establish bias or intent.", "direct lexical match", source_hits=source_hits)
    if responsibility_hits:
        indicators += _evidence("attribution_of_responsibility", text, responsibility_hits, sentences, "This phrase links an actor or event to responsibility or causation, which can direct attention to a blamed party.", "rule-based", "The rule identifies wording, not whether the attributed responsibility is justified.", "direct phrasal match", source_hits=source_hits)
    if source_hits:
        indicators += _evidence("attributed_claim", text, source_hits, sentences, "This wording attributes information to a reported source rather than stating it as the reporter's own assertion.", "rule-based", "Attribution does not itself establish source quality, accuracy, or balance.", "attribution marker match", source_hits=source_hits, claim_status_override="attribution_marker")
    if hedge_hits:
        indicators += _evidence("hedging_or_qualification", text, hedge_hits, sentences, "This wording expresses uncertainty, qualification, or limited certainty.", "lexicon-based", "Hedging can be appropriate reporting practice and is not evidence of bias.", "direct lexical match", source_hits=source_hits)
    if absolutist_hits:
        indicators += _evidence("absolutist_or_generalising_language", text, absolutist_hits, sentences, "This wording makes a broad or absolute-sounding claim, which can overstate a pattern if the context is narrower.", "lexicon-based", "The lexicon cannot determine whether a generalisation is factually warranted or rhetorical.", "direct lexical match", source_hits=source_hits)
    indicators = _validated_evidence(text, indicators)
    chars, word_count, paragraphs = text_counts(text)
    entities = _entities(text)
    if ner_status == "available_with_limitations":
        model_person_spans = [
            (item["start"], item["end"])
            for item in ner_predictions
            if str(item["label"]).upper() in {"PER", "PERSON"}
        ]
        entities = [
            entity
            for entity in entities
            if entity.label != "PERSON_CANDIDATE"
            or not any(
                entity.start < model_end and model_start < entity.end
                for model_start, model_end in model_person_spans
            )
        ]
        entities.extend(
            Entity(
                text=item["text"],
                label=str(item["label"]),
                start=item["start"],
                end=item["end"],
                method=ner_method or "configured named-entity model",
                confidence=item["score"] if isinstance(item["score"], (int, float)) else None,
            )
            for item in ner_predictions
        )
    entities = list({(entity.start, entity.end): entity for entity in entities}.values())
    bias_assessment = GroqFramingService(settings).analyze(
        text,
        language.code,
        include_emotion_fallback=emotion_status != "available_with_limitations",
        include_sentiment_fallback=sentiment_status != "available_with_limitations",
    )
    indicator_counts = dict(Counter(item.category for item in indicators))
    occurrence_counts: Counter[str] = Counter()
    for item in indicators:
        occurrence_counts[item.category] += item.occurrences
    total_occurrences = sum(item.occurrences for item in indicators)
    evidence_density = round((total_occurrences / word_count) * 1000, 1) if word_count else None
    claim_counts = Counter(item.claim_status for item in indicators if item.claim_status)
    attribution_summary = {
        "attributed_findings": claim_counts.get("attributed", 0),
        "unattributed_findings": claim_counts.get("unattributed", 0),
        "mixed_status_findings": claim_counts.get("mixed", 0),
        "attribution_marker_findings": claim_counts.get("attribution_marker", 0),
        "denominator": "Distinct evidence findings with a sentence context.",
        "note": "Counts describe where configured attribution markers appear. They are not counts of unique sources, and they do not measure fairness or balance.",
    }
    ner_availability = ner_status
    ner_limitation = ner_limitation or "No configured NER model was run."
    if ner_status == "not_configured" and language.code == "en":
        ner_method = "English date/person candidate rules"
        ner_limitation = (
            "The English candidate rules are not pretrained NER and do not support "
            "organization recognition. No NER model is configured."
        )
    embedding_status = "inconclusive" if settings.embedding_model_id else "not_configured"
    embedding_limitation = (
        "An embedding model ID is configured, but this application does not load or run embedding models."
        if settings.embedding_model_id
        else "No cross-language embedding model is integrated; similarity would not constitute bias analysis."
    )
    sentiment_task_status = sentiment_status
    sentiment_task_method = sentiment_method
    sentiment_task_model_id = sentiment_model_id
    sentiment_task_limitation = sentiment_limitation or MODEL_CATALOG["sentiment"]["limitations"]
    if sentiment_status != "available_with_limitations" and bias_assessment.sentiment_fallback:
        sentiment_task_status = "available_with_limitations"
        sentiment_task_method = "Groq contextual fallback"
        sentiment_task_model_id = bias_assessment.sentiment_fallback["model_identifier"]
        sentiment_task_limitation = (
            f"{bias_assessment.sentiment_fallback['explanation']} "
            f"{bias_assessment.sentiment_fallback['uncertainty']}"
        )
    emotion_task_status = emotion_status
    emotion_task_method = emotion_method
    emotion_task_model_id = emotion_model_id
    emotion_task_limitation = emotion_limitation or MODEL_CATALOG["emotion"]["limitations"]
    if emotion_status != "available_with_limitations" and bias_assessment.emotion_fallback:
        emotion_task_status = "available_with_limitations"
        emotion_task_method = "Groq contextual fallback"
        emotion_task_model_id = bias_assessment.emotion_fallback["model_identifier"]
        emotion_task_limitation = (
            f"{bias_assessment.emotion_fallback['explanation']} "
            f"{bias_assessment.emotion_fallback['uncertainty']}"
        )
    availability = [
        TaskAvailability(task="language_identification", status="available_with_limitations" if language.supported else "unsupported", method="Unicode script ranges with langdetect fallback", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Verified on native-script fixtures; short, mixed-language, and Latin-script transliterations can be inconclusive."),
        TaskAvailability(task="sentence_segmentation", status="available_with_limitations", method="Unicode-aware rules", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Approximate punctuation-based sentence boundaries; original Unicode offsets are preserved."),
        TaskAvailability(task="tokenization", status="available_with_limitations", method="Unicode whitespace tokenization", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="Whitespace tokens are counts, not language-specific morphological tokens."),
        TaskAvailability(task="sentiment", status=sentiment_task_status, method=sentiment_task_method, supported_languages=MODEL_CATALOG["sentiment"]["supported_languages"] if sentiment_model_id == MODEL_CATALOG["sentiment"]["model_id"] and sentiment_task_method != "Groq contextual fallback" else [], limitation=sentiment_task_limitation, model_identifier=sentiment_task_model_id),
        TaskAvailability(task="emotion", status=emotion_task_status, method=emotion_task_method, supported_languages=MODEL_CATALOG["emotion"]["supported_languages"] if emotion_model_id == MODEL_CATALOG["emotion"]["model_id"] and emotion_task_method != "Groq contextual fallback" else [], limitation=emotion_task_limitation, model_identifier=emotion_task_model_id),
        TaskAvailability(task="named_entity_recognition", status=ner_availability, method=ner_method, supported_languages=(
            MODEL_CATALOG["named_entity_recognition"]["language_models"]["te"]["supported_languages"]
            if ner_model_id == MODEL_CATALOG["named_entity_recognition"]["language_models"]["te"]["model_id"]
            else MODEL_CATALOG["named_entity_recognition"]["supported_languages"]
            if ner_model_id == MODEL_CATALOG["named_entity_recognition"]["model_id"]
            else []
        ), limitation=ner_limitation, model_identifier=ner_model_id, fallback="English date/person candidate rules (not pretrained NER)" if language.code == "en" and ner_status == "not_configured" else None),
        TaskAvailability(task="pos_tagging", status="unsupported", limitation="No POS model or tagging implementation is configured."),
        TaskAvailability(task="auxiliary_keyword_indicators", status="available_with_limitations", method="rule-based cues", supported_languages=["en", "hi", "kn", "te", "ta"], limitation="These sparse lexical/phrase cues are auxiliary only. They do not determine the contextual assessment or establish bias."),
        TaskAvailability(task="contextual_framing", status=bias_assessment.status, method="Groq chat completions" if bias_assessment.status == "available" else None, supported_languages=["en", "hi", "kn", "te", "ta"] if bias_assessment.status == "available" else [], limitation=("Model-generated contextual analysis; not independently fact-checked and not validated for real-world accuracy." if bias_assessment.status == "available" else bias_assessment.summary), model_identifier=bias_assessment.model),
        TaskAvailability(task="sarcasm_analysis", status="unsupported", limitation="No dedicated sarcasm/irony model with verified support for the detected language is configured."),
        TaskAvailability(task="cross_language_semantic_matching", status=embedding_status, limitation=embedding_limitation, model_identifier=None),
        TaskAvailability(task="code_switching_detection", status="unsupported", limitation="Mixed-language spans are not reliably detected; dominant-language detection is not code-switch analysis."),
        TaskAvailability(task="transliteration_understanding", status="unsupported", limitation="Latin-script transliteration is not verified for the configured classifiers."),
    ]
    limitations = [
        "A provisional directional-framing indicator is returned only when distinct validated authorial evidence meets the documented sufficiency threshold. It is not a scientific bias score.",
        "Original text is retained and analysed without translation.",
        "Groq credentials are server-side only. If contextual inference is not configured or unavailable, the response exposes that state and does not substitute keyword findings.",
        "Raw model scores are not calibrated for news. Classifiers do not resolve quotation, speaker attribution, sarcasm, or author intent.",
        "Repeated cue counts describe observable wording, not independent evidence of bias.",
        "Attribution markers do not verify a source or establish that a claim is true.",
    ]
    if len(SENTENCE_RE.findall(text)) > len(sentences):
        limitations.append(f"Only the first {settings.max_analysis_sentences} sentences were analysed due to the configured limit.")
    sentiment_scores = {
        label: sum(item["class_scores"].get(label, 0.0) for item in sentiment_predictions) / len(sentiment_predictions)
        for label in sorted({label for item in sentiment_predictions for label in item["class_scores"]})
    } if sentiment_status == "available_with_limitations" and sentiment_predictions else {}
    sentiment_labels = {item["label"] for item in sentiment_predictions} if sentiment_scores else set()
    if {"positive", "negative"} <= sentiment_labels:
        sentiment_label = "mixed"
    elif sentiment_predictions:
        sentiment_counts = Counter(item["label"] for item in sentiment_predictions)
        sentiment_label = max(sentiment_counts, key=sentiment_counts.get)
    else:
        sentiment_label = None
    emotion_scores = {
        label: sum(item["class_scores"].get(label, 0.0) for item in emotion_predictions) / len(emotion_predictions)
        for label in sorted({label for item in emotion_predictions for label in item["class_scores"]})
    } if emotion_status == "available_with_limitations" and emotion_predictions else {}
    emotion_label = max(emotion_scores, key=emotion_scores.get) if emotion_scores else None
    emotion_sentences = [
        {
            "text": item["text"],
            "start": item["start"],
            "end": item["end"],
            "label": item["label"],
            "score": item["score"],
            "class_scores": item["class_scores"],
        }
        for item in emotion_predictions
    ]
    sentiment_result = {
        "status": sentiment_status,
        "method": sentiment_method,
        "model_identifier": sentiment_model_id,
        "analysis_method": sentiment_method or "not run",
        "label": sentiment_label,
        "class_scores": sentiment_scores,
        "score_note": "Mean raw model outputs over analysed sentences; not calibrated probabilities." if sentiment_scores else None,
        "sentences": sentence_results,
        "limitation": sentiment_limitation or "No model prediction is available; no sentiment label was inferred from keyword counts.",
        "limitations": ["Quoted language is not separated from journalist narration; slang, sarcasm, and news-domain reliability are not validated."],
    }
    if sentiment_status != "available_with_limitations" and bias_assessment.sentiment_fallback:
        fallback = bias_assessment.sentiment_fallback
        sentiment_result.update(
            {
                "status": "available_with_limitations",
                "method": "Groq contextual fallback",
                "analysis_method": "Groq contextual fallback",
                "model_identifier": fallback["model_identifier"],
                "label": fallback["label"],
                "score_note": (
                    "Optional model-generated contextual fallback; confidence is not "
                    "calibrated and sentiment is not evidence of bias."
                ),
                "sentences": (
                    [
                        {
                            "text": text[fallback["start_offset"]:fallback["end_offset"]],
                            "start": fallback["start_offset"],
                            "end": fallback["end_offset"],
                            "label": fallback["label"],
                            "score": fallback["confidence"],
                            "class_scores": {},
                        }
                    ]
                    if fallback["start_offset"] is not None
                    and fallback["end_offset"] is not None
                    else []
                ),
                "limitation": (
                    f"{fallback['explanation']} {fallback['uncertainty']}"
                ),
            }
        )
    emotion_result = {
        "status": emotion_status,
        "method": emotion_method,
        "model_identifier": emotion_model_id,
        "analysis_method": emotion_method or "not run",
        "label": emotion_label,
        "distribution": emotion_scores,
        "sentences": emotion_sentences,
        "score_note": "Mean raw model outputs over analysed sentences; not calibrated probabilities." if emotion_scores else None,
        "limitation": emotion_limitation or "No emotion prediction is available; sentiment is not used as an emotion substitute.",
        "supported_labels": ["anger", "fear", "sadness", "joy", "disgust", "surprise", "neutral"],
        "limitations": ["Quoted reactions are not attributed to a speaker; sarcasm and news-domain reliability are not validated."],
    }
    if (
        emotion_status != "available_with_limitations"
        and bias_assessment.emotion_fallback is not None
    ):
        fallback = bias_assessment.emotion_fallback
        emotion_result.update(
            {
                "status": "available_with_limitations",
                "method": "Groq contextual fallback",
                "analysis_method": "Groq contextual fallback",
                "model_identifier": fallback["model_identifier"],
                "label": fallback["label"],
                "supported_labels": [
                    "anger",
                    "fear",
                    "sadness",
                    "joy",
                    "disgust",
                    "surprise",
                    "neutral",
                    "mixed",
                    "uncertain",
                    "not_detectable",
                ],
                "score_note": (
                    "Optional model-generated contextual fallback; confidence is not "
                    "calibrated and does not indicate author intent."
                ),
                "sentences": (
                    [
                        {
                            "text": text[fallback["start_offset"]:fallback["end_offset"]],
                            "start": fallback["start_offset"],
                            "end": fallback["end_offset"],
                            "label": fallback["label"],
                            "score": fallback["confidence"],
                        }
                    ]
                    if fallback["start_offset"] is not None
                    and fallback["end_offset"] is not None
                    else []
                ),
                "limitation": (
                    f"{fallback['explanation']} {fallback['uncertainty']} "
                    f"Source classification: {fallback['source'].replace('_', ' ')}."
                ),
            }
        )
    overall_judgment = build_bias_judgment(
        original_text=text,
        assessment=bias_assessment,
        indicators=indicators,
        sentiment=sentiment_result,
        emotion=emotion_result,
        entities=entities,
        word_count=word_count,
    )
    return ArticleAnalysisResponse(
        analysis_id=str(uuid4()),
        title=title,
        original_url=original_url,
        original_text=text,
        language=language,
        linguistic={
            "sentence_count": len(sentences),
            "token_count": len(words),
            "token_count_convention": "Whitespace-delimited non-empty tokens in the original normalized article text.",
            "pos_distribution": None,
        },
        entities=entities,
        sentiment=sentiment_result,
        emotion=emotion_result,
        indicators=indicators,
        bias_assessment=bias_assessment,
        overall_judgment=overall_judgment,
        statistics={
            "character_count": chars,
            "word_count": word_count,
            "paragraph_count": paragraphs,
            "entity_count": len(entities),
            "indicator_counts": indicator_counts,
            "indicator_occurrences": dict(occurrence_counts),
            "total_evidence_findings": len(indicators),
            "total_cue_occurrences": total_occurrences,
            "evidence_density_per_1000_tokens": evidence_density,
            "evidence_density_convention": "Total cue occurrences divided by whitespace-token count, multiplied by 1,000; this is not a bias score.",
            "sentiment_distribution": dict(Counter(item.label for item in sentence_results)) if sentiment_scores else None,
            "attribution_summary": attribution_summary,
            "evidence_coverage": "Every finding and occurrence span is validated as an exact Unicode slice of original_text.",
        },
        methodology=_methodology(),
        task_availability=availability,
        limitations=limitations,
    )
