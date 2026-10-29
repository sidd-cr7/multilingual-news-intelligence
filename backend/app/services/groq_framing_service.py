"""Groq-backed contextual analysis with fail-closed evidence validation."""
from __future__ import annotations

import json
import hashlib
import logging
import time
from collections import OrderedDict
from collections import defaultdict
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Literal

from groq import APIConnectionError, APIStatusError, Groq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings
from app.schemas.article import ActorPortrayal, BiasAssessment, FramingFinding

logger = logging.getLogger(__name__)
SUPPORTED_LANGUAGES = {"en", "hi", "kn", "te", "ta"}
PROMPT_SCHEMA_VERSION = "framing-v1"
_CACHE_LOCK = RLock()
_RESPONSE_CACHE: OrderedDict[str, tuple[float, dict]] = OrderedDict()


class FindingOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: Literal[
        "loaded_language",
        "contextual_framing",
        "attribution",
        "sarcasm_irony",
        "actor_portrayal",
        "emotional_positioning",
        "factual_claim_interpretation",
        "selective_emphasis",
        "generalization",
    ]
    label: str = Field(min_length=1, max_length=160)
    evidence_quote: str = Field(min_length=1, max_length=1_200)
    actor: str | None = Field(default=None, max_length=240)
    speaker: str | None = Field(default=None, max_length=240)
    attribution_type: Literal[
        "journalist_narration",
        "direct_quote",
        "attributed_claim",
        "unattributed_claim",
        "unclear",
    ] = "unclear"
    framing_direction: Literal["positive", "negative", "mixed", "neutral", "ambiguous"] = "ambiguous"
    explanation: str = Field(min_length=1, max_length=1_200)
    contextual_interpretation: str = Field(min_length=1, max_length=1_200)
    potential_reader_effect: str = Field(min_length=1, max_length=800)
    alternative_interpretation: str = Field(min_length=1, max_length=800)
    evidence_strength: str = Field(min_length=1, max_length=400)
    uncertainty: str = Field(min_length=1, max_length=600)
    confidence: float = Field(ge=0, le=1)


class EmotionFallbackOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: Literal[
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
    ]
    source: Literal["article_tone", "quoted_source", "mixed", "uncertain"]
    evidence_quote: str | None = Field(default=None, max_length=600)
    explanation: str = Field(min_length=1, max_length=600)
    uncertainty: str = Field(min_length=1, max_length=600)
    confidence: float | None = Field(default=None, ge=0, le=1)


class SentimentFallbackOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: Literal["positive", "negative", "neutral", "mixed", "uncertain", "not_detectable"]
    evidence_quote: str | None = Field(default=None, max_length=600)
    explanation: str = Field(min_length=1, max_length=600)
    uncertainty: str = Field(min_length=1, max_length=600)
    confidence: float | None = Field(default=None, ge=0, le=1)


class GroqResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    overall_assessment: str = Field(min_length=1, max_length=1_200)
    summary: str = Field(min_length=1, max_length=1_200)
    sufficient_evidence_for_overall_assessment: bool
    representation_comparison: str = Field(min_length=1, max_length=1_200)
    selective_presentation: str = Field(min_length=1, max_length=1_200)
    competing_attributed_viewpoints: bool | None = None
    findings: list[FindingOutput] = Field(default_factory=list, max_length=30)
    neutral_rewrite: str | None = Field(default=None, max_length=12_000)
    limitations: list[str] = Field(default_factory=list, max_length=20)
    emotion_fallback: EmotionFallbackOutput | None = None
    sentiment_fallback: SentimentFallbackOutput | None = None


@dataclass(frozen=True)
class ProviderFailure:
    code: str
    recoverable: bool
    retry_same_key: bool = False


ClientFactory = Callable[[str, float], Groq]


class ProviderUnavailable(Exception):
    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(code)


class GroqFramingService:
    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: ClientFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory or self._make_client
        self._sleep = sleep

    @staticmethod
    def _make_client(api_key: str, timeout: float) -> Groq:
        return Groq(api_key=api_key, timeout=timeout, max_retries=0)

    def analyze(
        self,
        text: str,
        language: str,
        *,
        include_emotion_fallback: bool = False,
        include_sentiment_fallback: bool = False,
    ) -> BiasAssessment:
        if not isinstance(text, str) or not text.strip():
            return self._unavailable("invalid_input", "Article text is required for contextual framing.")
        if language not in SUPPORTED_LANGUAGES:
            return self._unavailable("unsupported_language", f"Contextual framing is not configured for language '{language}'.")
        primary_key = (self._settings.groq_api_key or "").strip()
        if not primary_key:
            return self._unavailable("not_configured", "Groq contextual analysis is not configured because GROQ_API_KEY is missing.")

        model = self._settings.groq_model.strip()
        if not model:
            return self._unavailable("invalid_model", "GROQ_MODEL is empty; configure a model supported by Groq.")
        selected, partial = _select_article_context(text, self._settings.groq_max_input_chars)
        if not selected.strip():
            return self._unavailable("invalid_input", "The article contains no analyzable text.")

        cache_key = self._cache_key(
            text,
            language,
            model,
            include_emotion_fallback,
            include_sentiment_fallback,
        )
        payload = self._get_cached_response(cache_key)
        cache_hit = payload is not None
        keys = [primary_key]
        backup_key = (self._settings.groq_api_key_backup or "").strip()
        if backup_key and backup_key != primary_key:
            keys.append(backup_key)
        try:
            if payload is None:
                payload = self._request_with_failover(
                    keys,
                    model,
                    selected,
                    language,
                    include_emotion_fallback=include_emotion_fallback,
                    include_sentiment_fallback=include_sentiment_fallback,
                )
            parsed = GroqResponse.model_validate(payload)
        except ProviderUnavailable as exc:
            return self._unavailable(exc.code, exc.public_message, model=model)
        except (ValidationError, json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Groq contextual response failed validation (%s)", type(exc).__name__)
            return self._unavailable(
                "invalid_response",
                "Groq returned a response that did not match the required analysis format.",
                model=model,
            )
        limitations = list(parsed.limitations)
        findings: list[FramingFinding] = []
        rejected_quotes = False
        for item in parsed.findings:
            start = text.find(item.evidence_quote)
            if start < 0:
                rejected_quotes = True
                continue
            end = start + len(item.evidence_quote)
            findings.append(
                FramingFinding(
                    id=f"groq-{start}-{end}-{len(findings)}",
                    category=item.category,
                    label=item.label,
                    actor=item.actor,
                    speaker=item.speaker,
                    attribution_type=item.attribution_type,
                    framing_direction=item.framing_direction,
                    evidence_text=text[start:end],
                    start_offset=start,
                    end_offset=end,
                    explanation=item.explanation,
                    contextual_interpretation=item.contextual_interpretation,
                    potential_reader_effect=item.potential_reader_effect,
                    evidence_strength=item.evidence_strength,
                    alternative_interpretation=item.alternative_interpretation,
                    method="Groq contextual analysis",
                    uncertainty=item.uncertainty,
                    confidence=item.confidence,
                )
            )
        if parsed.findings and rejected_quotes and not findings:
            return self._unavailable(
                "invalid_evidence",
                "Groq findings did not contain verifiable quotations from the original article.",
                model=model,
            )
        emotion_fallback = None
        if include_emotion_fallback and parsed.emotion_fallback:
            emotion = parsed.emotion_fallback
            emotion_quote_start = (
                text.find(emotion.evidence_quote) if emotion.evidence_quote else -1
            )
            quote_required = emotion.label not in {"uncertain", "not_detectable"}
            if emotion_quote_start < 0 and quote_required:
                limitations.append(
                    "The optional emotion fallback was excluded because its evidence quote could not be verified."
                )
            else:
                emotion_fallback = {
                    **emotion.model_dump(),
                    "start_offset": emotion_quote_start if emotion_quote_start >= 0 else None,
                    "end_offset": (
                        emotion_quote_start + len(emotion.evidence_quote)
                        if emotion_quote_start >= 0 and emotion.evidence_quote
                        else None
                    ),
                    "model_identifier": model,
                    "status": "available_with_limitations",
                }
        sentiment_fallback = None
        if include_sentiment_fallback and parsed.sentiment_fallback:
            sentiment = parsed.sentiment_fallback
            sentiment_quote_start = (
                text.find(sentiment.evidence_quote) if sentiment.evidence_quote else -1
            )
            quote_required = sentiment.label not in {"uncertain", "not_detectable"}
            if sentiment_quote_start < 0 and quote_required:
                limitations.append(
                    "The optional sentiment fallback was excluded because its evidence quote could not be verified."
                )
            else:
                sentiment_fallback = {
                    **sentiment.model_dump(),
                    "start_offset": sentiment_quote_start if sentiment_quote_start >= 0 else None,
                    "end_offset": (
                        sentiment_quote_start + len(sentiment.evidence_quote)
                        if sentiment_quote_start >= 0 and sentiment.evidence_quote
                        else None
                    ),
                    "model_identifier": model,
                    "status": "available_with_limitations",
                }
        if not cache_hit:
            self._store_cached_response(
                cache_key,
                payload,
                ttl=max(0, self._settings.groq_cache_ttl_seconds),
                max_entries=max(1, self._settings.groq_cache_max_entries),
            )

        limitations.append(
            "This is model-generated textual analysis, not independent fact-checking or proof of author intent."
        )
        if partial:
            limitations.append(
                "Only representative excerpts were analyzed because the article exceeded GROQ_MAX_INPUT_CHARS; this is not a full-article assessment."
            )
        if rejected_quotes:
            limitations.append(
                "Some model findings were excluded because their quotations were not exact spans in the original article."
            )
        if parsed.neutral_rewrite:
            limitations.append(
                "The neutral rewrite is model-generated and has not been independently checked for complete claim preservation."
            )

        actor_findings: dict[str, list[FramingFinding]] = defaultdict(list)
        for finding in findings:
            if finding.actor and finding.framing_direction in {"positive", "negative", "mixed"}:
                actor_findings[finding.actor].append(finding)
        portrayals = [
            ActorPortrayal(
                actor=actor,
                direction=(
                    next(iter({finding.framing_direction for finding in items}))
                    if len({finding.framing_direction for finding in items}) == 1
                    else "mixed"
                ),
                finding_ids=[finding.id for finding in items],
                explanation="Model-generated summary of text-linked findings; not a judgment about the actor.",
            )
            for actor, items in actor_findings.items()
        ]
        return BiasAssessment(
            status="available",
            provider="groq",
            model=model,
            groq_used=not cache_hit,
            cache_hit=cache_hit,
            emotion_fallback=emotion_fallback,
            sentiment_fallback=sentiment_fallback,
            analysis_scope="sampled_excerpts" if partial else "full_article",
            conclusion=parsed.overall_assessment,
            summary=parsed.summary,
            strongest_patterns=[f"{finding.label}: {finding.evidence_text}" for finding in findings[:5]],
            actor_portrayals=portrayals,
            competing_attributed_viewpoints=parsed.competing_attributed_viewpoints,
            representation_comparison=parsed.representation_comparison,
            selective_presentation=parsed.selective_presentation,
            sufficient_evidence_for_overall_assessment=(
                parsed.sufficient_evidence_for_overall_assessment
            ),
            findings=findings,
            limitations=limitations,
            neutral_rewrite=parsed.neutral_rewrite,
        )

    def _cache_key(
        self,
        text: str,
        language: str,
        model: str,
        include_emotion_fallback: bool,
        include_sentiment_fallback: bool,
    ) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        material = "\0".join(
            (
                digest,
                language,
                model,
                PROMPT_SCHEMA_VERSION,
                str(self._settings.groq_max_input_chars),
                str(include_emotion_fallback),
                str(include_sentiment_fallback),
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _get_cached_response(cache_key: str) -> dict | None:
        now = time.monotonic()
        with _CACHE_LOCK:
            cached = _RESPONSE_CACHE.get(cache_key)
            if cached is None:
                return None
            expires_at, payload = cached
            if expires_at <= now:
                del _RESPONSE_CACHE[cache_key]
                return None
            _RESPONSE_CACHE.move_to_end(cache_key)
            return payload.copy()

    @staticmethod
    def _store_cached_response(
        cache_key: str,
        payload: dict,
        *,
        ttl: int,
        max_entries: int,
    ) -> None:
        if ttl <= 0:
            return
        with _CACHE_LOCK:
            _RESPONSE_CACHE[cache_key] = (time.monotonic() + ttl, payload.copy())
            _RESPONSE_CACHE.move_to_end(cache_key)
            while len(_RESPONSE_CACHE) > max_entries:
                _RESPONSE_CACHE.popitem(last=False)

    def _request_with_failover(
        self,
        keys: list[str],
        model: str,
        article: str,
        language: str,
        *,
        include_emotion_fallback: bool,
        include_sentiment_fallback: bool,
    ) -> dict:
        retries = max(0, min(self._settings.groq_max_retries, 2))
        last_failure = ProviderFailure("provider_unavailable", recoverable=True)
        for key_index, key in enumerate(keys):
            client = self._client_factory(
                key,
                max(1.0, min(self._settings.groq_timeout_seconds, 120.0)),
            )
            for attempt in range(retries + 1):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": _user_prompt(
                                    article,
                                    language,
                                    include_emotion_fallback=include_emotion_fallback,
                                    include_sentiment_fallback=include_sentiment_fallback,
                                ),
                            },
                        ],
                        temperature=0,
                        max_completion_tokens=3_000,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content
                    if not isinstance(content, str) or not content.strip():
                        raise ProviderUnavailable(
                            "invalid_response",
                            "Groq returned an empty contextual-analysis response.",
                        )
                    parsed = json.loads(content)
                    if not isinstance(parsed, dict):
                        raise ProviderUnavailable(
                            "invalid_response",
                            "Groq returned a response that was not a JSON object.",
                        )
                    return parsed
                except ProviderUnavailable:
                    raise
                except (APIStatusError, APIConnectionError) as exc:
                    failure = _provider_failure(exc)
                    last_failure = failure
                    status_code = getattr(exc, "status_code", None)
                    logger.warning(
                        "Groq request failed (type=%s, status=%s, key_slot=%s)",
                        type(exc).__name__,
                        status_code if isinstance(status_code, int) else "network",
                        "primary" if key_index == 0 else "backup",
                    )
                    if failure.retry_same_key and attempt < retries:
                        self._sleep(min(0.25 * (2**attempt), 1.0))
                        continue
                    if not failure.recoverable:
                        raise ProviderUnavailable(
                            failure.code,
                            "Groq rejected the configured model or request; verify GROQ_MODEL and account access.",
                        ) from None
                    break
                except ImportError:
                    raise ProviderUnavailable(
                        "sdk_unavailable",
                        "The Groq Python SDK is not installed. Install backend requirements.",
                    ) from None
            else:
                continue
        raise ProviderUnavailable(
            last_failure.code,
            "Groq contextual analysis is currently unavailable. Check provider status and account limits, then try again.",
        )

    @staticmethod
    def _unavailable(code: str, message: str, *, model: str | None = None) -> BiasAssessment:
        return BiasAssessment(
            status="not_configured" if code == "not_configured" else "unavailable",
            provider="groq",
            model=model,
            error_code=code,
            conclusion="Insufficient evidence for an overall assessment.",
            summary=message,
            representation_comparison="Not assessed because contextual analysis did not complete.",
            selective_presentation="Not assessed because contextual analysis did not complete.",
            sufficient_evidence_for_overall_assessment=False,
            findings=[],
            limitations=[message],
        )


def _provider_failure(exc: APIStatusError | APIConnectionError) -> ProviderFailure:
    if isinstance(exc, APIConnectionError):
        return ProviderFailure("provider_unavailable", recoverable=True, retry_same_key=True)
    status_code = exc.status_code
    if status_code in {401, 403, 429}:
        return ProviderFailure("provider_unavailable", recoverable=True)
    if status_code >= 500:
        return ProviderFailure("provider_unavailable", recoverable=True, retry_same_key=True)
    if status_code in {400, 404, 422}:
        return ProviderFailure("invalid_model", recoverable=False)
    return ProviderFailure("provider_error", recoverable=False)


def _select_article_context(text: str, max_chars: int) -> tuple[str, bool]:
    budget = max(1_000, min(max_chars, 100_000))
    if len(text) <= budget:
        return text, False

    marker = "\n\n[... middle article content omitted ...]\n\n"
    remaining = max(3, budget - len(marker))
    allocations = [int(remaining * 0.4), int(remaining * 0.2)]
    allocations.append(remaining - sum(allocations))
    starts = [0, max(0, (len(text) - allocations[1]) // 2), max(0, len(text) - allocations[2])]
    spans: list[tuple[int, int]] = []
    for index, (start, length) in enumerate(zip(starts, allocations, strict=True)):
        end = min(len(text), start + length)
        if index > 0:
            paragraph_start = text.find("\n", start, min(len(text), start + 200))
            if paragraph_start >= 0:
                start = paragraph_start + 1
        if index < 2:
            paragraph_end = text.rfind("\n", max(start, end - 200), end)
            if paragraph_end > start:
                end = paragraph_end
        if spans and start <= spans[-1][1]:
            continue
        spans.append((start, end))
    if len(spans) < 2:
        spans = [(0, budget // 2), (len(text) - budget // 2, len(text))]
    selected = marker.join(text[start:end] for start, end in spans)
    if len(selected) > budget:
        raise ValueError("Excerpt selection exceeded its configured character budget.")
    return selected, True


_SYSTEM_PROMPT = """You are a cautious multilingual news-framing analyst. Analyze the supplied article in its original language, including English, Hindi, Kannada, Telugu, or Tamil. Do not translate the evidence. Return only a JSON object matching the requested fields.

Assess contextual framing, evaluative wording, speaker attribution, actor portrayal, emotional positioning, claims versus interpretations, and sarcasm/irony only where context supports it. Do not infer journalist intent or an author's psychological state from quoted language. Do not infer factual truth, deliberate omission, political alignment, or unsupported cultural meanings. An absent perspective is not an omission unless the article itself provides enough context to support that observation. Negative sentiment alone is not bias. Empty findings and an insufficient-evidence assessment are valid. Every evidence_quote must be copied exactly and contiguously from the provided article text, preserving original Unicode and punctuation. Findings must distinguish journalist narration, direct quotation, attributed claim, unattributed claim, or unclear attribution. Confidence must be a number from 0 to 1 and reflect textual support, not factual truth. Keep explanations concise and in English. A neutral_rewrite is optional; if supplied, preserve the claims and speakers, add no facts, and write in the article's original language. Do not independently verify claims."""


def _user_prompt(
    article: str,
    language: str,
    *,
    include_emotion_fallback: bool = False,
    include_sentiment_fallback: bool = False,
) -> str:
    emotion_instruction = (
        "- emotion_fallback: optional object only if requested; label must be anger, fear, sadness, joy, disgust, surprise, neutral, mixed, uncertain, or not_detectable; source must be article_tone, quoted_source, mixed, or uncertain; include an exact evidence_quote for a detected emotion, never infer an author's psychological state, distinguish quoted emotion from article tone, and explain uncertainty"
        if include_emotion_fallback
        else ""
    )
    sentiment_instruction = (
        "- sentiment_fallback: optional object only if requested; label must be positive, negative, neutral, mixed, uncertain, or not_detectable; include an exact evidence_quote for a supported prediction and concise explanation/uncertainty; assess textual sentiment only, not bias or truth"
        if include_sentiment_fallback
        else ""
    )
    return f"""Analyze this article. Its detected language code is {language}. Text between omission markers is not supplied and must not be inferred.

Return JSON with:
- overall_assessment: concise qualitative conclusion, or insufficient evidence
- summary: concise evidence-based assessment
- sufficient_evidence_for_overall_assessment: boolean
- representation_comparison: only what this single article supports
- selective_presentation: explain when comparison evidence is insufficient
- competing_attributed_viewpoints: boolean or null
- findings: at most 30 objects with category (one of loaded_language, contextual_framing, attribution, sarcasm_irony, actor_portrayal, emotional_positioning, factual_claim_interpretation, selective_emphasis, generalization), label, evidence_quote, actor, speaker, attribution_type (journalist_narration, direct_quote, attributed_claim, unattributed_claim, unclear), framing_direction (positive, negative, mixed, neutral, ambiguous), explanation, contextual_interpretation, potential_reader_effect, alternative_interpretation, evidence_strength, uncertainty, confidence
- neutral_rewrite: string or null
- limitations: list of concise limitations
{emotion_instruction}
{sentiment_instruction}

ARTICLE:
{article}"""
