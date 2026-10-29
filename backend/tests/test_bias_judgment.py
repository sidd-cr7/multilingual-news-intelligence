from app.schemas.article import BiasAssessment
from app.services.bias_judgment_service import build_bias_judgment


ARTICLE = " ".join(
    [
        "The council approved its public transit plan after months of debate.",
        "The mayor described the proposal as a responsible investment in local services.",
        "The administration presented cost estimates and implementation details.",
        "Independent analysts reviewed projected ridership and future maintenance.",
        "The report also noted concerns about the project's long-term funding.",
        "Several residents said the plan could improve access to work and school.",
        "The council published responses from supporters and critics.",
        "The article summarizes the meeting and the competing views expressed there.",
        "Officials said more information would be released before construction begins.",
        "The proposal remains subject to further review and public comment.",
    ]
)


def assessment(findings, *, status="available"):
    return BiasAssessment(
        status=status,
        provider="groq",
        model="openai/gpt-oss-120b",
        conclusion="Text-linked framing observations.",
        summary="Summary of validated observations.",
        representation_comparison="Not compared.",
        selective_presentation="Cannot be assessed from one article.",
        sufficient_evidence_for_overall_assessment=False,
        findings=findings,
        limitations=[],
    )


def finding(quote, direction="negative", *, attribution="journalist_narration", actor="council", start=None):
    offset = ARTICLE.find(quote) if start is None else start
    return {
        "id": f"finding-{offset}-{direction}",
        "category": "contextual_framing",
        "label": "Contextual observation",
        "actor": actor,
        "speaker": None,
        "attribution_type": attribution,
        "framing_direction": direction,
        "evidence_text": quote,
        "start_offset": offset,
        "end_offset": offset + len(quote),
        "explanation": "Contextual explanation.",
        "contextual_interpretation": "Contextual interpretation.",
        "potential_reader_effect": "Potential effect.",
        "evidence_strength": "Exact phrase.",
        "alternative_interpretation": "Alternative context.",
        "method": "Groq contextual analysis",
        "uncertainty": "Not independently verified.",
        "confidence": 0.8,
    }


def build(findings, *, sentiment=None):
    return build_bias_judgment(
        original_text=ARTICLE,
        assessment=assessment(findings),
        indicators=[],
        sentiment=sentiment or {"status": "not_configured", "label": None},
        emotion={"status": "not_configured", "label": None},
        entities=[],
        word_count=len(ARTICLE.split()),
    )


def test_insufficient_evidence_does_not_get_a_score():
    result = build([finding("The article summarizes", start=0)])

    assert result.status == "insufficient_evidence"
    assert result.indicator is None
    assert result.direction == "insufficient_evidence"


def test_quoted_accusations_are_not_scored_as_authorial_endorsement():
    quotes = [
        finding(
            "The mayor described",
            attribution="direct_quote",
            start=ARTICLE.index("The mayor described"),
        ),
        finding(
            "Several residents said",
            attribution="attributed_claim",
            start=ARTICLE.index("Several residents said"),
        ),
        finding(
            "The council published",
            attribution="direct_quote",
            start=ARTICLE.index("The council published"),
        ),
    ]

    result = build(quotes)

    assert result.status == "insufficient_evidence"
    assert result.indicator is None
    assert result.reasons


def test_distinct_directional_evidence_gets_a_bounded_explainable_score():
    spans = [
        "The administration presented",
        "Independent analysts reviewed",
        "The report also noted",
        "The council published",
        "Officials said more information",
    ]
    findings = [
        finding(phrase, start=ARTICLE.index(phrase))
        for phrase in spans
    ]

    result = build(findings)

    assert result.status == "available"
    assert result.direction == "criticizes"
    assert 0 < result.indicator <= 100
    assert len(result.reasons) == len(spans)
    assert "not empirically calibrated" in result.scoring_method


def test_repeated_same_span_is_not_counted_as_independent_evidence():
    repeated = finding("The report also noted", start=ARTICLE.index("The report also noted"))
    findings = [
        repeated,
        repeated,
        finding("The administration presented", start=ARTICLE.index("The administration presented")),
        finding("The council published", start=ARTICLE.index("The council published")),
    ]

    result = build(findings)

    assert result.status == "insufficient_evidence"
    assert result.indicator is None


def test_counter_framing_changes_direction_and_sentiment_is_not_in_the_formula():
    quotes = [
        finding("The administration presented", "negative", start=ARTICLE.index("The administration presented")),
        finding("Independent analysts reviewed", "negative", start=ARTICLE.index("Independent analysts reviewed")),
        finding("The report also noted", "positive", start=ARTICLE.index("The report also noted")),
        finding("Several residents said", "positive", start=ARTICLE.index("Several residents said")),
    ]

    first = build(quotes, sentiment={"status": "available", "label": "negative"})
    second = build(quotes, sentiment={"status": "available", "label": "positive"})

    assert first.direction == "mixed"
    assert first.indicator == second.indicator


def test_unverifiable_model_quotes_are_excluded_from_evidence():
    invalid = finding("The article summarizes", start=len(ARTICLE) + 10)

    result = build([invalid])

    assert result.status == "insufficient_evidence"
    assert result.reasons == []
