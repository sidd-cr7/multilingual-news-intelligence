import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.schemas.article import EvidenceSpan
from app.services.analysis_service import _matches, _validated_evidence, analyse_article
from app.services.language_service import LanguageDetector

client = TestClient(app)


def test_analysis_evidence_is_an_exact_original_span() -> None:
    text = "Dr. Maya Rao said the shocking crisis was caused by poor planning on 12 March 2025."
    settings = Settings(min_text_chars=10)
    result = analyse_article(text, settings, LanguageDetector(settings))
    assert result.indicators
    assert all(result.original_text[item.start:item.end] == item.quote for item in result.indicators)
    assert result.language.code == "en"


@pytest.mark.parametrize(("text", "code"), [
    ("यह एक संकट पर पर्याप्त लंबा समाचार लेख है जिसमें जिम्मेदार लोगों का उल्लेख है।", "hi"),
    ("ಇದು ಸಂಕಟದ ಬಗ್ಗೆ ಸಾಕಷ್ಟು ಉದ್ದದ ಸುದ್ದಿ ಲೇಖನವಾಗಿದೆ ಮತ್ತು ಕಾರಣವನ್ನು ವಿವರಿಸುತ್ತದೆ.", "kn"),
    ("ఇది సంక్షోభం గురించి తగినంత పొడవైన వార్తా కథనం మరియు కారణాన్ని వివరిస్తుంది.", "te"),
    ("இது நெருக்கடி பற்றிய போதுமான நீளமான செய்திக் கட்டுரை மற்றும் காரணத்தை விளக்குகிறது.", "ta"),
])
def test_indic_original_text_and_offsets_are_preserved(text: str, code: str) -> None:
    settings = Settings(min_text_chars=10)
    result = analyse_article(text, settings, LanguageDetector(settings))
    assert result.language.code == code
    assert result.original_text == text
    assert all(result.original_text[x.start:x.end] == x.quote for x in result.indicators)


def test_api_analysis_contract_and_empty_input() -> None:
    response = client.post("/api/analyze/article", json={"text": "The official said the shocking crisis caused harm to residents."})
    assert response.status_code == 200
    body = response.json()
    assert body["original_text"] and body["task_availability"]
    assert all(body["original_text"][x["start"]:x["end"]] == x["quote"] for x in body["indicators"])
    assert client.post("/api/analyze/article", json={"text": ""}).status_code == 422


def test_analysis_respects_text_limit() -> None:
    settings = Settings(min_text_chars=10, max_text_chars=30)
    with pytest.raises(Exception):
        analyse_article("This deliberately exceeds the configured character limit.", settings, LanguageDetector(settings))


def test_phase_three_categories_and_normalised_report() -> None:
    settings = Settings(min_text_chars=10)
    text = "Everyone said the shocking plan may fail. The plan was caused by poor planning."
    result = analyse_article(text, settings, LanguageDetector(settings))
    categories = {item.category for item in result.indicators}
    assert {"attributed_claim", "hedging_or_qualification", "absolutist_or_generalising_language", "unattributed_assertion_candidate"} <= categories
    assert result.statistics["evidence_density_per_1000_tokens"] is not None
    assert result.statistics["indicator_counts"]["attributed_claim"] == 1


def test_repeated_and_overlapping_evidence_remains_valid() -> None:
    settings = Settings(min_text_chars=10)
    text = "The shocking claim was shocking, and everyone may disagree."
    result = analyse_article(text, settings, LanguageDetector(settings))
    assert all(0 <= item.start < item.end <= len(result.original_text) for item in result.indicators)
    assert all(result.original_text[item.start:item.end] == item.quote for item in result.indicators)
    assert all(
        result.original_text[start:end] == item.quote
        for item in result.indicators
        for start, end in item.occurrence_spans
    )


def test_repeated_cue_is_one_finding_with_occurrence_counts() -> None:
    settings = Settings(min_text_chars=10)
    text = "The shocking claim was shocking, and the shocking result was shocking."
    result = analyse_article(text, settings, LanguageDetector(settings))
    loaded = [item for item in result.indicators if item.category == "loaded_or_evaluative_language"]
    assert len(loaded) == 1
    finding = loaded[0]
    assert finding.cue == "shocking"
    assert finding.quote == "shocking"
    assert finding.occurrences == 4
    assert len(finding.occurrence_spans) == 4
    assert len({start for start, _ in finding.occurrence_spans}) == 4
    assert result.statistics["indicator_occurrences"]["loaded_or_evaluative_language"] == 4


def test_evidence_spans_are_specific_rather_than_whole_sentences() -> None:
    settings = Settings(min_text_chars=10)
    text = "The shocking failure was caused by corruption. Everyone knew the brutal attack was shocking."
    result = analyse_article(text, settings, LanguageDetector(settings))
    assert result.indicators
    # Whole-sentence spans previously overlapped every word-level cue and were
    # rendered as repeated highlights. Sentence rules now anchor on a predicate.
    assert all(item.quote != item.context for item in result.indicators)
    assert all(len(item.quote) <= 40 for item in result.indicators)
    pairs = [(item.category, item.start, item.end) for item in result.indicators]
    assert len(pairs) == len(set(pairs))


def test_sentence_level_rule_anchors_on_the_predicate() -> None:
    settings = Settings(min_text_chars=10)
    result = analyse_article("The shocking failure was caused by corruption.", settings, LanguageDetector(settings))
    assertions = [item for item in result.indicators if item.category == "unattributed_assertion_candidate"]
    assert assertions
    assert all(item.quote == "was" for item in assertions)


def test_evidence_carries_context_and_strength() -> None:
    settings = Settings(min_text_chars=10)
    result = analyse_article("Officials said the shocking crisis was caused by poor planning.", settings, LanguageDetector(settings))
    assert result.indicators
    for item in result.indicators:
        assert item.context and item.context in result.original_text
        assert item.contexts and item.contexts[0] in result.original_text
        assert item.evidence_strength
        assert item.occurrences >= 1


def test_title_summary_sentiment_emotion_and_entities_are_meaningful_but_scoped() -> None:
    settings = Settings(min_text_chars=10)
    text = "Dr. Maya Rao said the severe attack caused damage. The community welcomed relief and celebrated the successful response."
    result = analyse_article(text, settings, LanguageDetector(settings))
    assert result.title.startswith("Dr. Maya Rao said")
    assert result.title_source == "inferred-opening-text"
    assert result.summary["status"] == "available" and "Maya Rao" in result.summary["text"]
    assert result.sentiment["positive_cues"] > 0 and result.sentiment["negative_cues"] > 0
    assert result.emotion["status"] == "available" and result.emotion["distribution"]["joy"] > 0
    assert any(entity.text == "Maya Rao" and entity.label == "PERSON_CANDIDATE" for entity in result.entities)


def test_capitalised_events_are_not_person_candidates() -> None:
    settings = Settings(min_text_chars=10)
    result = analyse_article("Friendship Day and Education Policy were discussed in the public meeting.", settings, LanguageDetector(settings))
    assert not any(entity.label == "PERSON_CANDIDATE" for entity in result.entities)


@pytest.mark.parametrize(("text", "language", "emotion"), [
    ("The angry crowd expressed outrage after the decision.", "en", "anger"),
    ("लोगों में डर और गुस्सा था, लेकिन यह केवल एक रिपोर्ट है।", "hi", "fear"),
    ("ಜನರಲ್ಲಿ ಭಯ ಮತ್ತು ಕೋಪ ಇತ್ತು ಎಂದು ವರದಿ ತಿಳಿಸಿದೆ.", "kn", "fear"),
    ("ప్రజల్లో భయం మరియు కోపం ఉందని నివేదిక తెలిపింది.", "te", "fear"),
    ("மக்களிடம் பயம் மற்றும் கோபம் இருந்ததாக அறிக்கை தெரிவித்தது.", "ta", "fear"),
])
def test_emotion_evidence_is_language_scoped_and_offset_valid(text: str, language: str, emotion: str) -> None:
    result = analyse_article(text, Settings(min_text_chars=10), LanguageDetector(Settings(min_text_chars=10)))
    assert result.language.code == language
    assert result.emotion["status"] == "available"
    assert result.emotion["distribution"][emotion] >= 1
    assert all(result.original_text[item["start"]:item["end"]] == item["quote"] for item in result.emotion["evidence"])


def test_emotion_neutral_and_framing_assessment_do_not_fabricate_conclusions() -> None:
    result = analyse_article("The council met on Tuesday and published the minutes on its website.", Settings(min_text_chars=10), LanguageDetector(Settings(min_text_chars=10)))
    assert result.emotion["dominant_emotion"] == "neutral"
    assert result.emotion["evidence"] == []
    assert result.framing_analysis["findings"] == []
    assert "No reliable overall framing conclusion" in result.framing_analysis["overall_assessment"]


def test_framing_analysis_cites_loaded_wording_with_an_alternative_interpretation() -> None:
    result = analyse_article("The shocking decision was a brutal failure, officials said.", Settings(min_text_chars=10), LanguageDetector(Settings(min_text_chars=10)))
    findings = result.framing_analysis["findings"]
    assert findings
    assert all(result.original_text[item["start_offset"]:item["end_offset"]] == item["evidence_text"] for item in findings)
    assert all(item["alternative_interpretation"] for item in findings)


def test_claim_status_markers_preserve_the_difference_between_allegation_and_fact() -> None:
    settings = Settings(min_text_chars=10)
    text = "Police alleged that Maya was responsible. Maya denied the allegation and argued the account was incomplete."
    result = analyse_article(text, settings, LanguageDetector(settings))
    categories = {item.category for item in result.indicators}
    assert {"allegation_or_accusation", "explicitly_disputed_claim", "opinion_or_interpretation_marker"} <= categories
    assert all(result.original_text[item.start:item.end] == item.quote for item in result.indicators)
    allegation = next(item for item in result.indicators if item.category == "allegation_or_accusation")
    assert "rather than an established fact" in allegation.explanation


def test_source_representation_is_sentence_descriptive_not_a_source_count() -> None:
    settings = Settings(min_text_chars=10)
    result = analyse_article("Maya said the proposal may change. Ravi said the plan was shocking.", settings, LanguageDetector(settings))
    source = result.statistics["source_representation"]
    assert source["attribution_marker_occurrences"] == 2
    assert source["sentences_with_attribution_markers"] == 2
    assert "do not identify unique sources" in source["interpretation"]
    availability = next(task for task in result.task_availability if task.task == "source_representation")
    assert availability.status == "partial"


def test_statistics_separate_findings_from_occurrences() -> None:
    settings = Settings(min_text_chars=10)
    text = "The shocking plan was shocking. Officials said the shocking plan failed."
    result = analyse_article(text, settings, LanguageDetector(settings))
    stats = result.statistics
    assert stats["total_evidence_findings"] == len(result.indicators)
    assert sum(stats["indicator_counts"].values()) == len(result.indicators)
    assert sum(stats["indicator_occurrences"].values()) == stats["total_cue_occurrences"]
    assert stats["total_cue_occurrences"] == sum(item.occurrences for item in result.indicators)
    expected_density = round(
        stats["total_cue_occurrences"] / result.linguistic["token_count"] * 1000, 1
    )
    assert stats["evidence_density_per_1000_tokens"] == expected_density


def test_indic_repeated_cue_keeps_exact_unicode_offsets() -> None:
    settings = Settings(min_text_chars=10)
    text = "यह क्रूर हमला है। दूसरा क्रूर हमला भी क्रूर था।"
    result = analyse_article(text, settings, LanguageDetector(settings))
    assert result.language.code == "hi"
    loaded = [item for item in result.indicators if item.category == "loaded_or_evaluative_language"]
    assert len(loaded) == 1
    assert loaded[0].occurrences == 3
    for start, end in loaded[0].occurrence_spans:
        assert result.original_text[start:end] == "क्रूर"


def test_lexicon_matches_do_not_use_substrings_inside_other_words() -> None:
    assert _matches("The team called for a review.", {"all"}) == []
    assert _matches("Everyone called for a review.", {"all", "everyone"}) == [(0, 8)]


def test_phase_three_api_keeps_existing_response_sections() -> None:
    response = client.post("/api/analyze/article", json={"text": "Officials said the proposal may affect everyone next year."})
    assert response.status_code == 200
    body = response.json()
    assert {"statistics", "indicators", "sentiment", "task_availability"} <= body.keys()
    assert "evidence_density_per_1000_tokens" in body["statistics"]


@pytest.mark.parametrize(("text", "code"), [
    ("Everyone may disagree with this sufficiently long English report.", "en"),
    ("\u092f\u0939 \u092a\u0930\u094d\u092f\u093e\u092a\u094d\u0924 \u0932\u0902\u092c\u093e \u0939\u093f\u0902\u0926\u0940 \u0938\u092e\u093e\u091a\u093e\u0930 \u0939\u0948 \u0914\u0930 \u0936\u093e\u092f\u0926 \u0938\u092d\u0940 \u0938\u0939\u092e\u0924 \u0939\u094b\u0902\u0964", "hi"),
    ("\u0c87\u0ca6\u0cc1 \u0cb8\u0cbe\u0c95\u0cb7\u0ccd\u0c9f\u0cc1 \u0c89\u0ca6\u0ccd\u0ca6\u0cb5\u0cbe\u0ca6 \u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0cb8\u0cc1\u0ca6\u0ccd\u0ca6\u0cbf \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0c86\u0c97\u0cb2\u0cb9\u0cc1\u0ca6\u0cc1.", "kn"),
    ("\u0c07\u0c26\u0c3f \u0c24\u0c17\u0c3f\u0c28\u0c02\u0c24 \u0c2a\u0c4a\u0c21\u0c35\u0c48\u0c28 \u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 \u0c35\u0c3e\u0c30\u0c4d\u0c24 \u0c15\u0c25\u0c28\u0c02 \u0c2e\u0c30\u0c3f\u0c2f\u0c41 \u0c05\u0c35\u0c15\u0c3e\u0c36\u0c02 \u0c05\u0c02\u0c26\u0c30\u0c3f\u0c15\u0c40 \u0c05\u0c2a\u0c4d\u0c2a\u0c41\u0ca1\u0c42.", "te"),
    ("\u0b87\u0ba4\u0bc1 \u0baa\u0bcb\u0ba4\u0bc1\u0bae\u0bbe\u0ba9 \u0ba8\u0bc0\u0bb3\u0bae\u0bbe\u0ba9 \u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0b9a\u0bc6\u0baf\u0bcd\u0ba4\u0bbf \u0b85\u0bb1\u0bbf\u0b95\u0bcd\u0b95\u0bc8 \u0bb5\u0bbf\u0bb5\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bbf\u0bb1\u0ba4\u0bc1.", "ta"),
])
def test_phase_three_availability_covers_target_languages(text: str, code: str) -> None:
    settings = Settings(min_text_chars=10)
    result = analyse_article(text, settings, LanguageDetector(settings))
    assert result.language.code == code
    assert any(task.task == "bias_indicator_engine" and task.status == "available" for task in result.task_availability)


def test_invalid_evidence_is_rejected() -> None:
    invalid = EvidenceSpan(category="test", quote="wrong", start=0, end=5, context="", explanation="", method="rule-based")
    with pytest.raises(ValueError, match="invalid evidence"):
        _validated_evidence("right", [invalid])
