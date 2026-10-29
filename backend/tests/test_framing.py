import json
from types import SimpleNamespace

import httpx
import pytest
from groq import APIConnectionError, APIStatusError, APITimeoutError
from fastapi.testclient import TestClient

from app.api import routes
from app.config import Settings
from app.main import app
from app.schemas.article import BiasAssessment
from app.services import analysis_service
from app.services.groq_framing_service import GroqFramingService


def finding(
    quote: str,
    *,
    category: str = "contextual_framing",
    confidence: float = 0.82,
    speaker: str | None = "official",
    actor: str | None = "council",
    attribution_type: str = "direct_quote",
) -> dict:
    return {
        "category": category,
        "label": "Attributed characterization",
        "evidence_quote": quote,
        "actor": actor,
        "speaker": speaker,
        "attribution_type": attribution_type,
        "framing_direction": "negative",
        "explanation": "The wording criticizes the named decision.",
        "contextual_interpretation": "The speaker, not the reporter, makes the evaluation.",
        "potential_reader_effect": "Readers may view the decision negatively.",
        "alternative_interpretation": "The quotation may accurately reflect the speaker's opinion.",
        "evidence_strength": "Direct quotation in the supplied text.",
        "uncertainty": "The claim is not independently verified.",
        "confidence": confidence,
    }


def payload(findings: list[dict] | None = None, **overrides) -> dict:
    result = {
        "overall_assessment": "The article includes an attributed critical viewpoint.",
        "summary": "A quoted source criticizes the council's decision.",
        "sufficient_evidence_for_overall_assessment": False,
        "representation_comparison": "No comparison with other coverage was performed.",
        "selective_presentation": "Cannot be established from this article alone.",
        "competing_attributed_viewpoints": None,
        "findings": findings or [],
        "neutral_rewrite": None,
        "limitations": [],
    }
    result.update(overrides)
    return result


def success_response(data: dict):
    content = json.dumps(data, ensure_ascii=False)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class FakeClient:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return success_response(self.outcome)


def provider_error(status: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return APIStatusError("redacted provider failure", response=response, body={})


def configured_settings(**changes) -> Settings:
    values = {
        "min_text_chars": 10,
        "groq_api_key": "primary-test-secret",
        "groq_api_key_backup": None,
        "groq_max_retries": 0,
    }
    values.update(changes)
    return Settings(**values)


def test_primary_key_success_returns_validated_contextual_finding_and_exact_offsets():
    text = "मंत्री ने कहा, “यह एक शर्मनाक निर्णय है।”"
    quote = "शर्मनाक निर्णय"
    primary = FakeClient(payload([finding(quote)]))
    keys = []
    service = GroqFramingService(
        configured_settings(),
        client_factory=lambda key, timeout: (keys.append(key) or primary),
    )

    result = service.analyze(text, "hi")

    assert result.status == "available"
    assert result.provider == "groq"
    assert result.model == "openai/gpt-oss-120b"
    assert result.findings[0].evidence_text == quote
    assert text[result.findings[0].start_offset:result.findings[0].end_offset] == quote
    assert result.findings[0].contextual_interpretation
    assert result.findings[0].confidence == 0.82
    assert keys == ["primary-test-secret"]
    assert primary.calls[0]["response_format"] == {"type": "json_object"}


def test_identical_request_uses_bounded_cached_groq_response():
    client = FakeClient(payload([finding("dangerous decision")]))
    calls = []
    settings = configured_settings()
    service = GroqFramingService(
        settings,
        client_factory=lambda key, timeout: (calls.append(key) or client),
    )
    text = "Critics called it a dangerous decision."

    first = service.analyze(text, "en")
    second = service.analyze(text, "en")

    assert first.status == second.status == "available"
    assert first.groq_used is True and first.cache_hit is False
    assert second.groq_used is False and second.cache_hit is True
    assert len(calls) == 1
    assert len(client.calls) == 1


def test_optional_emotion_fallback_requires_an_exact_original_quote():
    text = "The minister described the delay as deeply worrying."
    output = payload(
        emotion_fallback={
            "label": "fear",
            "source": "quoted_source",
            "evidence_quote": "deeply worrying",
            "explanation": "A quoted source expresses concern.",
            "uncertainty": "This describes the source, not the journalist.",
            "confidence": 0.72,
        }
    )
    service = GroqFramingService(
        configured_settings(),
        client_factory=lambda *_: FakeClient(output),
    )

    result = service.analyze(text, "en", include_emotion_fallback=True)

    assert result.status == "available"
    assert result.emotion_fallback["label"] == "fear"
    start = result.emotion_fallback["start_offset"]
    end = result.emotion_fallback["end_offset"]
    assert text[start:end] == "deeply worrying"


def test_invalid_emotion_fallback_quote_is_dropped_without_faking_a_label():
    output = payload(
        emotion_fallback={
            "label": "anger",
            "source": "article_tone",
            "evidence_quote": "invented phrase",
            "explanation": "Unsupported phrase.",
            "uncertainty": "Uncertain.",
            "confidence": 0.8,
        }
    )
    result = GroqFramingService(
        configured_settings(),
        client_factory=lambda *_: FakeClient(output),
    ).analyze("The council met on Tuesday.", "en", include_emotion_fallback=True)

    assert result.status == "available"
    assert result.emotion_fallback is None
    assert any("emotion fallback was excluded" in item for item in result.limitations)


def test_optional_sentiment_fallback_requires_an_exact_original_quote():
    text = "The mayor called the proposal a reckless decision."
    output = payload(
        sentiment_fallback={
            "label": "negative",
            "evidence_quote": "reckless decision",
            "explanation": "The phrase expresses a negative evaluation.",
            "uncertainty": "This does not indicate article bias.",
            "confidence": 0.74,
        }
    )
    result = GroqFramingService(
        configured_settings(),
        client_factory=lambda *_: FakeClient(output),
    ).analyze(text, "te", include_sentiment_fallback=True)

    assert result.status == "available"
    assert result.sentiment_fallback["label"] == "negative"
    start = result.sentiment_fallback["start_offset"]
    end = result.sentiment_fallback["end_offset"]
    assert text[start:end] == "reckless decision"


@pytest.mark.parametrize("status", [401, 403, 429])
def test_primary_auth_or_rate_limit_failure_falls_back_to_backup(status):
    backup = FakeClient(payload([finding("dangerous decision")]))
    keys = []

    def factory(key, _timeout):
        keys.append(key)
        return FakeClient(provider_error(status)) if key == "primary-test-secret" else backup

    settings = configured_settings(groq_api_key_backup="backup-test-secret")
    result = GroqFramingService(settings, client_factory=factory).analyze(
        "Critics called it a dangerous decision.", "en"
    )

    assert result.status == "available"
    assert keys == ["primary-test-secret", "backup-test-secret"]
    assert len(backup.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        APIConnectionError(
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        ),
        APITimeoutError(
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        ),
    ],
)
def test_network_and_timeout_failures_try_backup(error):
    backup = FakeClient(payload())
    keys = []

    def factory(key, _timeout):
        keys.append(key)
        return FakeClient(error) if key == "primary-test-secret" else backup

    settings = configured_settings(groq_api_key_backup="backup-test-secret")
    result = GroqFramingService(settings, client_factory=factory, sleep=lambda _: None).analyze(
        "The council met on Tuesday.", "en"
    )

    assert result.status == "available"
    assert keys == ["primary-test-secret", "backup-test-secret"]


def test_temporary_server_failure_retries_once_then_uses_backup():
    backup = FakeClient(payload())
    keys = []

    def factory(key, _timeout):
        keys.append(key)
        return FakeClient(provider_error(503)) if key == "primary-test-secret" else backup

    settings = configured_settings(
        groq_api_key_backup="backup-test-secret",
        groq_max_retries=1,
    )
    delays = []
    result = GroqFramingService(
        settings,
        client_factory=factory,
        sleep=delays.append,
    ).analyze("The council met on Tuesday.", "en")

    assert result.status == "available"
    assert keys == ["primary-test-secret", "backup-test-secret"]
    assert delays == [0.25]


def test_both_keys_failing_return_safe_unavailable_state_without_secrets(caplog):
    keys = []

    def factory(key, _timeout):
        keys.append(key)
        return FakeClient(provider_error(401 if len(keys) == 1 else 429))

    settings = configured_settings(groq_api_key_backup="backup-test-secret")
    result = GroqFramingService(settings, client_factory=factory).analyze(
        "The council met on Tuesday.", "en"
    )
    serialized = result.model_dump_json()

    assert result.status == "unavailable"
    assert result.error_code == "provider_unavailable"
    assert result.findings == []
    assert "primary-test-secret" not in serialized
    assert "backup-test-secret" not in serialized
    assert "primary-test-secret" not in caplog.text
    assert "backup-test-secret" not in caplog.text
    assert keys == ["primary-test-secret", "backup-test-secret"]


def test_missing_primary_key_does_not_attempt_backup():
    calls = []
    service = GroqFramingService(
        configured_settings(groq_api_key=None, groq_api_key_backup="backup-test-secret"),
        client_factory=lambda *_: calls.append(True),
    )

    result = service.analyze("The council met on Tuesday.", "en")

    assert result.status == "not_configured"
    assert result.error_code == "not_configured"
    assert calls == []


def test_missing_backup_key_still_uses_primary_normally():
    primary = FakeClient(payload())
    keys = []

    result = GroqFramingService(
        configured_settings(),
        client_factory=lambda key, _timeout: (keys.append(key) or primary),
    ).analyze("The council met on Tuesday.", "en")

    assert result.status == "available"
    assert keys == ["primary-test-secret"]


def test_bad_model_configuration_is_not_retried_or_hidden_by_key_failover():
    calls = []
    settings = configured_settings(groq_api_key_backup="backup-test-secret")
    result = GroqFramingService(
        settings,
        client_factory=lambda key, _timeout: (
            calls.append(key) or FakeClient(provider_error(400))
        ),
    ).analyze("The council met on Tuesday.", "en")

    assert result.status == "unavailable"
    assert result.error_code == "invalid_model"
    assert calls == ["primary-test-secret"]


def test_malformed_model_output_returns_error_not_placeholder_success():
    service = GroqFramingService(
        configured_settings(),
        client_factory=lambda *_: FakeClient({"unexpected": "shape"}),
    )

    result = service.analyze("The council met on Tuesday.", "en")

    assert result.status == "unavailable"
    assert result.error_code == "invalid_response"
    assert result.findings == []
    assert result.summary.startswith("Groq returned")


def test_fabricated_evidence_quote_is_rejected():
    service = GroqFramingService(
        configured_settings(),
        client_factory=lambda *_: FakeClient(payload([finding("invented quotation")])),
    )

    result = service.analyze("The council met on Tuesday.", "en")

    assert result.status == "unavailable"
    assert result.error_code == "invalid_evidence"
    assert result.findings == []


def test_invalid_quote_is_removed_when_other_findings_are_verifiable():
    text = "The minister said the delay was disgraceful."
    service = GroqFramingService(
        configured_settings(),
        client_factory=lambda *_: FakeClient(
            payload([finding("invented phrase"), finding("disgraceful")])
        ),
    )

    result = service.analyze(text, "en")

    assert result.status == "available"
    assert [item.evidence_text for item in result.findings] == ["disgraceful"]
    assert any("Some model findings were excluded" in item for item in result.limitations)


@pytest.mark.parametrize(
    ("language", "text", "quote"),
    [
        ("en", 'The minister called it "a disgraceful decision."', "disgraceful decision"),
        ("hi", "मंत्री ने कहा, “यह शर्मनाक निर्णय है।”", "शर्मनाक निर्णय"),
        ("kn", "ಸಚಿವರು ಇದನ್ನು ಅನ್ಯಾಯದ ನಿರ್ಧಾರ ಎಂದರು.", "ಅನ್ಯಾಯದ ನಿರ್ಧಾರ"),
        ("te", "మంత్రి ఇది అన్యాయమైన నిర్ణయం అని అన్నారు.", "అన్యాయమైన నిర్ణయం"),
        ("ta", "அமைச்சர் இது அநியாயமான முடிவு என்றார்.", "அநியாயமான முடிவு"),
    ],
)
def test_all_target_language_outputs_preserve_exact_original_evidence(language, text, quote):
    output = payload([finding(quote)])
    result = GroqFramingService(
        configured_settings(),
        client_factory=lambda *_: FakeClient(output),
    ).analyze(text, language)

    assert result.status == "available"
    assert result.findings[0].evidence_text == quote
    assert text[result.findings[0].start_offset:result.findings[0].end_offset] == quote


def test_empty_findings_are_a_valid_successful_assessment():
    service = GroqFramingService(
        configured_settings(),
        client_factory=lambda *_: FakeClient(
            payload(
                overall_assessment="Insufficient evidence for an overall assessment.",
                summary="The supplied text does not support a meaningful framing finding.",
                sufficient_evidence_for_overall_assessment=False,
            )
        ),
    )

    result = service.analyze("The council met on Tuesday.", "en")

    assert result.status == "available"
    assert result.findings == []
    assert not result.sufficient_evidence_for_overall_assessment


def test_sarcasm_and_implicit_criticism_retain_uncertainty():
    quote = "Another flawless rollout, residents waited six hours."
    output = payload(
        [
            finding(
                quote,
                category="sarcasm_irony",
                confidence=0.61,
                speaker=None,
                actor="service rollout",
                attribution_type="journalist_narration",
            )
        ]
    )
    result = GroqFramingService(
        configured_settings(),
        client_factory=lambda *_: FakeClient(output),
    ).analyze(quote, "en")

    assert result.findings[0].category == "sarcasm_irony"
    assert result.findings[0].confidence == 0.61
    assert result.findings[0].uncertainty


def test_neutral_rewrite_is_returned_with_claim_preservation_limitation():
    output = payload(
        [],
        neutral_rewrite="The minister said the decision was difficult.",
    )
    result = GroqFramingService(
        configured_settings(),
        client_factory=lambda *_: FakeClient(output),
    ).analyze("The minister said the decision was terrible.", "en")

    assert result.neutral_rewrite == "The minister said the decision was difficult."
    assert any("not been independently checked" in item for item in result.limitations)


def test_long_article_uses_partial_sample_and_reports_scope():
    text = "\n\n".join(f"Paragraph {index}: " + ("council report " * 100) for index in range(20))
    captured = []
    client = FakeClient(payload())

    def factory(_key, _timeout):
        return client

    original_create = client.create

    def capture(**kwargs):
        captured.append(kwargs["messages"][1]["content"])
        return original_create(**kwargs)

    client.chat.completions.create = capture
    settings = configured_settings(groq_max_input_chars=2_000)
    result = GroqFramingService(settings, client_factory=factory).analyze(text, "en")

    assert result.status == "available"
    assert result.analysis_scope == "sampled_excerpts"
    article_context = captured[0].split("\nARTICLE:\n", maxsplit=1)[1]
    assert len(article_context) <= settings.groq_max_input_chars
    assert "middle article content omitted" in captured[0]
    assert any("not a full-article assessment" in item for item in result.limitations)


def test_analysis_api_exposes_provider_status_and_findings(monkeypatch):
    quote = "a dangerous decision"

    class StubFramingService:
        def __init__(self, _settings):
            pass

        def analyze(self, _text, _language, **_options):
            return BiasAssessment(
                status="available",
                provider="groq",
                model="openai/gpt-oss-120b",
                analysis_scope="full_article",
                conclusion="An attributed critical description is present.",
                summary="The quoted evidence attributes criticism to the speaker.",
                representation_comparison="Not compared.",
                selective_presentation="Not assessed from one article.",
                sufficient_evidence_for_overall_assessment=False,
                findings=[
                    {
                        "id": "groq-22-44-0",
                        "category": "contextual_framing",
                        "label": "Attributed criticism",
                        "actor": "decision",
                        "speaker": "minister",
                        "attribution_type": "direct_quote",
                        "framing_direction": "negative",
                        "evidence_text": quote,
                        "start_offset": 22,
                        "end_offset": 22 + len(quote),
                        "explanation": "The quote is critical.",
                        "contextual_interpretation": "The criticism is attributed.",
                        "potential_reader_effect": "Readers may view the decision negatively.",
                        "evidence_strength": "Direct quote.",
                        "alternative_interpretation": "It may reflect the speaker's view.",
                        "method": "Groq contextual analysis",
                        "uncertainty": "Not independently verified.",
                        "confidence": 0.8,
                    }
                ],
            )

    monkeypatch.setattr(analysis_service, "GroqFramingService", StubFramingService)
    text = 'The minister said, "a dangerous decision" after the meeting.'
    response = TestClient(app).post("/api/analyze/article", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert body["bias_assessment"]["provider"] == "groq"
    assert body["bias_assessment"]["findings"][0]["evidence_text"] == quote
    task = next(item for item in body["task_availability"] if item["task"] == "contextual_framing")
    assert task["status"] == "available"


def test_groq_service_response_flows_through_analysis_api(monkeypatch):
    text = "The mayor called the proposal a reckless decision during the public meeting."
    quote = "reckless decision"
    fake_client = FakeClient(payload(
        [finding(quote)],
        sentiment_fallback={
            "label": "negative",
            "evidence_quote": quote,
            "explanation": "The phrase is evaluative.",
            "uncertainty": "It does not establish bias.",
            "confidence": 0.7,
        },
        emotion_fallback={
            "label": "anger",
            "source": "quoted_source",
            "evidence_quote": quote,
            "explanation": "The quoted characterization expresses criticism.",
            "uncertainty": "The quote does not establish the journalist's tone.",
            "confidence": 0.7,
        },
    ))
    settings = configured_settings()
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(
        GroqFramingService,
        "_make_client",
        staticmethod(lambda _key, _timeout: fake_client),
    )

    response = TestClient(app).post("/api/analyze/article", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assessment = body["bias_assessment"]
    assert assessment["provider"] == "groq"
    assert assessment["model"] == settings.groq_model
    assert assessment["findings"][0]["evidence_text"] == quote
    start = assessment["findings"][0]["start_offset"]
    end = assessment["findings"][0]["end_offset"]
    assert text[start:end] == quote
    assert assessment["findings"][0]["speaker"] == "official"
    assert assessment["findings"][0]["contextual_interpretation"]
    assert "primary-test-secret" not in response.text
    contextual_task = next(
        task for task in body["task_availability"] if task["task"] == "contextual_framing"
    )
    assert contextual_task["status"] == "available"
    assert contextual_task["model_identifier"] == settings.groq_model
    assert "emotion_fallback" in fake_client.calls[0]["messages"][1]["content"]
    assert body["emotion"]["status"] == "available_with_limitations"
    assert body["emotion"]["label"] == "anger"
    emotion_span = body["emotion"]["sentences"][0]
    assert text[emotion_span["start"]:emotion_span["end"]] == quote
    assert body["sentiment"]["status"] == "available_with_limitations"
    assert body["sentiment"]["label"] == "negative"
    sentiment_span = body["sentiment"]["sentences"][0]
    assert text[sentiment_span["start"]:sentiment_span["end"]] == quote
    sentiment_task = next(
        task for task in body["task_availability"] if task["task"] == "sentiment"
    )
    assert sentiment_task["status"] == "available_with_limitations"
