"""Build a transparent, provisional directionality summary from validated evidence."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.schemas.article import (
    ActorPortrayal,
    BiasAssessment,
    BiasJudgment,
    BiasJudgmentEvidence,
    EvidenceSpan,
    Entity,
)

_RELEVANT_CATEGORIES = {
    "loaded_language",
    "contextual_framing",
    "actor_portrayal",
    "emotional_positioning",
    "factual_claim_interpretation",
    "selective_emphasis",
    "generalization",
}
_AUTHORIAL_ATTRIBUTION = {"journalist_narration", "unattributed_claim"}


def build_bias_judgment(
    *,
    original_text: str,
    assessment: BiasAssessment,
    indicators: list[EvidenceSpan],
    sentiment: dict[str, Any],
    emotion: dict[str, Any],
    entities: list[Entity],
    word_count: int,
) -> BiasJudgment:
    """Summarize evidence direction without treating classifier polarity as bias."""
    validated = _validated_findings(original_text, assessment)
    reasons = [
        BiasJudgmentEvidence(
            evidence_text=item.evidence_text,
            start_offset=item.start_offset,
            end_offset=item.end_offset,
            category=item.category,
            label=item.label,
            actor=item.actor,
            speaker=item.speaker,
            attribution_type=item.attribution_type,
            framing_direction=item.framing_direction,
            explanation=item.contextual_interpretation or item.explanation,
            confidence=item.confidence,
            method=item.method,
        )
        for item in validated
    ]
    distinct: dict[tuple[int, int], Any] = {}
    for item in validated:
        distinct.setdefault((item.start_offset, item.end_offset), item)
    authorial = [
        item
        for item in distinct.values()
        if item.attribution_type in _AUTHORIAL_ATTRIBUTION
        and item.category in _RELEVANT_CATEGORIES
        and item.framing_direction in {"positive", "negative"}
    ]
    positive = sum(_finding_weight(item) for item in authorial if item.framing_direction == "positive")
    negative = sum(_finding_weight(item) for item in authorial if item.framing_direction == "negative")
    total = positive + negative
    effective_count = sum(_finding_weight(item) for item in authorial)
    enough_evidence = (
        len(distinct) >= 4
        and len(authorial) >= 3
        and effective_count >= 1.5
        and word_count >= 80
        and assessment.status == "available"
    )

    if assessment.status != "available":
        status = "unavailable"
        direction = "insufficient_evidence"
        indicator = None
        confidence = "low"
        summary = (
            "The contextual analysis did not complete, so no article-level directional "
            "judgment or indicator is available."
        )
    elif not enough_evidence:
        status = "insufficient_evidence"
        direction = "insufficient_evidence"
        indicator = None
        confidence = "low"
        summary = (
            "There is not enough distinct, validated authorial framing evidence to estimate "
            "a directional indicator. Quoted or attributed statements are reported separately "
            "and are not treated as the article's endorsement."
        )
    else:
        status = "available"
        consistency = abs(positive - negative) / total if total else 0.0
        coverage = min(1.0, len(authorial) / 5)
        quality = sum(_finding_confidence(item) for item in authorial) / len(authorial)
        indicator = round(100 * consistency * coverage * quality)
        direction = (
            "mixed"
            if consistency < 0.25
            else "favors"
            if positive > negative
            else "criticizes"
        )
        confidence = "high" if len(authorial) >= 5 and quality >= 0.75 else "moderate"
        summary = (
            f"Validated authorial or unattributed findings lean {direction}; this is a "
            "provisional estimate of directional framing evidence, not an objective bias, "
            "truth, or quality score."
        )

    targets = _targets(validated)
    contextual_cues = [
        {
            "category": item.category,
            "quote": original_text[item.start:item.end],
            "start_offset": item.start,
            "end_offset": item.end,
            "explanation": item.explanation,
            "claim_status": item.claim_status,
        }
        for item in indicators
        if 0 <= item.start < item.end <= len(original_text)
        and original_text[item.start:item.end] == item.quote
    ]
    return BiasJudgment(
        status=status,
        summary=summary,
        indicator=indicator,
        indicator_label="Provisional directional-framing evidence indicator",
        direction=direction,
        targets=targets,
        reasons=reasons,
        validated_auxiliary_cues=contextual_cues,
        sentiment_signal={
            "status": sentiment.get("status", "unknown"),
            "label": sentiment.get("label"),
            "model_identifier": sentiment.get("model_identifier"),
            "limitation": sentiment.get("limitation"),
            "role": "Supporting description only; sentiment polarity is not used as a bias score.",
        },
        emotion_signal={
            "status": emotion.get("status", "unknown"),
            "label": emotion.get("label"),
            "model_identifier": emotion.get("model_identifier"),
            "limitation": emotion.get("limitation"),
            "role": "Supporting description only; predicted emotion does not establish author intent or bias.",
        },
        entity_mentions=[
            {
                "text": entity.text,
                "label": entity.label,
                "start": entity.start,
                "end": entity.end,
                "method": entity.method,
                "role": next(
                    (
                        target.direction
                        for target in targets
                        if target.actor.casefold() == entity.text.casefold()
                    ),
                    None,
                ),
            }
            for entity in entities
        ],
        evidence_sufficiency=confidence,
        scoring_method=(
            "Provisional method: require at least 4 distinct validated findings, including "
            "3 authorial/unattributed directional findings, at least 1.5 effective evidence "
            "units, and 80 whitespace-delimited tokens. Each eligible finding contributes "
            "its confidence (0.6 when absent), multiplied by 0.8 for journalist narration "
            "or 0.5 for an unattributed claim. The 0–100 indicator is the positive/negative "
            "weight imbalance multiplied by evidence coverage (eligible finding count / 5, "
            "capped at 1) and mean confidence. Attributed speech is excluded from scoring; "
            "duplicate spans are counted once. This method is not empirically calibrated."
        ),
        limitations=list(dict.fromkeys([
            *assessment.limitations,
            "Sentiment and emotion are descriptive supporting signals and are excluded from the indicator formula.",
            "Named entities identify mentions, not an actor's role or the truth of claims; target roles are linked only when a validated contextual finding names the same actor.",
            "The indicator is provisional and has not been calibrated against human-reviewed news-framing judgments.",
        ])),
    )


def _validated_findings(text: str, assessment: BiasAssessment) -> list[Any]:
    return [
        finding
        for finding in assessment.findings
        if 0 <= finding.start_offset < finding.end_offset <= len(text)
        and text[finding.start_offset:finding.end_offset] == finding.evidence_text
    ]


def _finding_confidence(finding: Any) -> float:
    return finding.confidence if finding.confidence is not None else 0.6


def _finding_weight(finding: Any) -> float:
    attribution_weight = 0.8 if finding.attribution_type == "journalist_narration" else 0.5
    return _finding_confidence(finding) * attribution_weight


def _targets(findings: list[Any]) -> list[ActorPortrayal]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for finding in findings:
        if finding.actor and finding.framing_direction in {"positive", "negative"}:
            grouped[finding.actor].append(finding)

    result: list[ActorPortrayal] = []
    for actor, items in grouped.items():
        directions = {item.framing_direction for item in items}
        direction = (
            "mixed"
            if len(directions) > 1
            else "favored"
            if "positive" in directions
            else "criticized"
        )
        result.append(
            ActorPortrayal(
                actor=actor,
                direction=direction,
                finding_ids=[item.id for item in items],
                explanation=(
                    "This describes validated text-linked framing observations. "
                    "Attributed statements are not treated as the article's endorsement."
                ),
            )
        )
    return result
