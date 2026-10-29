import sys
from types import SimpleNamespace

from app.config import Settings
from app.services import analysis_service
from app.services.analysis_service import analyse_article
from app.services.language_service import LanguageDetector
from app.services.model_manager import MODEL_CATALOG, ModelInference, MultilingualModelManager


def test_catalog_does_not_conflate_sentiment_emotion_ner_or_framing() -> None:
    assert MODEL_CATALOG["sentiment"]["task"] == "text-classification"
    assert MODEL_CATALOG["emotion"]["task"] == "text-classification"
    assert MODEL_CATALOG["named_entity_recognition"]["task"] == "token-classification"
    assert "bias" not in MODEL_CATALOG
    assert {"en", "hi"} == set(MODEL_CATALOG["sentiment"]["supported_languages"])
    assert "te" not in MODEL_CATALOG["named_entity_recognition"]["supported_languages"]


def test_unconfigured_model_is_not_reported_as_loaded_or_available() -> None:
    manager = MultilingualModelManager()
    result = manager.infer("sentiment", "The report is positive.", language="en")

    assert result.status == "not_configured"
    assert result.model_id is None
    assert not result.predictions


def test_configured_model_language_scope_is_enforced() -> None:
    manager = MultilingualModelManager(configured_ids={
        "sentiment_model_id": MODEL_CATALOG["sentiment"]["model_id"],
    })
    result = manager.infer("sentiment", "A sufficiently long Telugu sentence.", language="te")

    assert result.status == "unsupported"
    assert result.model_id == MODEL_CATALOG["sentiment"]["model_id"]


def test_telugu_indic_ner_checkpoint_is_language_routed_and_keeps_unicode_spans(monkeypatch) -> None:
    model_id = MODEL_CATALOG["named_entity_recognition"]["language_models"]["te"]["model_id"]
    text = "హైదరాబాద్‌లో రేపు సమావేశం జరుగుతుంది."

    def fake_loader(task: str, configured_id: str):
        assert task == "named_entity_recognition"
        assert configured_id == model_id
        return lambda _text, **_kwargs: [
            {"entity_group": "LOC", "start": 0, "end": 9, "score": 0.91},
        ]

    manager = MultilingualModelManager(
        configured_ids={"ner_model_id_te": model_id},
        pipeline_loader=fake_loader,
    )
    monkeypatch.setattr(analysis_service, "_shared_model_manager", lambda *_args: manager)
    settings = Settings(
        min_text_chars=10,
        sentiment_model_id=None,
        emotion_model_id=None,
        ner_model_id=None,
        ner_model_id_te=model_id,
    )

    result = analyse_article(text, settings, LanguageDetector(settings))

    assert result.language.code == "te"
    assert any(entity.method == "Transformers local CPU inference" for entity in result.entities)
    entity = next(entity for entity in result.entities if entity.method == "Transformers local CPU inference")
    assert entity.text == text[entity.start:entity.end]
    assert entity.label == "LOC"
    task = next(item for item in result.task_availability if item.task == "named_entity_recognition")
    assert task.status == "available_with_limitations"
    assert task.supported_languages == ["te"]


def test_unreviewed_custom_model_is_not_loaded() -> None:
    calls = []
    manager = MultilingualModelManager(
        configured_ids={"sentiment_model_id": "organization/custom-model"},
        pipeline_loader=lambda *args: calls.append(args),
    )

    result = manager.infer("sentiment", "A sufficiently long English article.", language="en")

    assert result.status == "inconclusive"
    assert not calls


def test_local_inference_is_called_and_class_scores_are_returned() -> None:
    calls = []

    def fake_loader(task: str, model_id: str):
        calls.append((task, model_id))

        def fake_pipeline(text: str, **kwargs):
            assert kwargs["top_k"] is None
            return [
                {"label": "negative", "score": 0.1},
                {"label": "neutral", "score": 0.2},
                {"label": "positive", "score": 0.7},
            ]

        return fake_pipeline

    manager = MultilingualModelManager(
        configured_ids={
            "sentiment_model_id": MODEL_CATALOG["sentiment"]["model_id"],
        },
        pipeline_loader=fake_loader,
    )
    one = manager.infer("sentiment", "The council approved the measure.", language="en")
    two = manager.infer("sentiment", "The proposal was welcomed.", language="en")

    assert one.status == "available_with_limitations"
    assert one.predictions[-1]["label"] == "positive"
    assert calls == [("sentiment", MODEL_CATALOG["sentiment"]["model_id"])]
    assert two.status == "available_with_limitations"


def test_ner_inference_uses_pipeline_supported_arguments() -> None:
    calls = []

    def fake_loader(task: str, model_id: str):
        assert task == "named_entity_recognition"
        assert model_id == MODEL_CATALOG["named_entity_recognition"]["model_id"]

        def fake_pipeline(text: str, **kwargs):
            calls.append((text, kwargs))
            return [{"entity_group": "PER", "start": 0, "end": 5, "score": 0.99}]

        return fake_pipeline

    manager = MultilingualModelManager(configured_ids={
        "ner_model_id": MODEL_CATALOG["named_entity_recognition"]["model_id"],
    }, pipeline_loader=fake_loader)
    result = manager.infer("named_entity_recognition", "Maya Rao attended the meeting.", language="en")

    assert result.status == "available_with_limitations"
    assert result.predictions[0]["entity_group"] == "PER"
    assert calls == [("Maya Rao attended the meeting.", {})]


def test_configured_local_model_path_is_used_without_exposing_it_as_model_id() -> None:
    model_id = MODEL_CATALOG["sentiment"]["model_id"]
    local_path = r"C:\models\sentiment"
    loaded_locations = []

    def fake_loader(task: str, model_location: str):
        loaded_locations.append((task, model_location))
        return lambda _text, **_kwargs: [
            {"label": "neutral", "score": 1.0},
        ]

    manager = MultilingualModelManager(
        configured_ids={"sentiment_model_id": model_id},
        configured_paths={"sentiment_model_path": local_path},
        pipeline_loader=fake_loader,
    )
    result = manager.infer("sentiment", "The council met on Tuesday.", language="en")

    assert result.status == "available_with_limitations"
    assert result.model_id == model_id
    assert loaded_locations == [("sentiment", local_path)]


def test_article_analysis_uses_configured_model_scores_and_original_passages(monkeypatch) -> None:
    model_id = MODEL_CATALOG["sentiment"]["model_id"]

    def fake_loader(task: str, configured_id: str):
        assert task == "sentiment"
        assert configured_id == model_id
        return lambda text, **kwargs: [
            {"label": "negative", "score": 0.1},
            {"label": "neutral", "score": 0.2},
            {"label": "positive", "score": 0.7},
        ]

    manager = MultilingualModelManager(
        configured_ids={"sentiment_model_id": model_id},
        pipeline_loader=fake_loader,
    )
    monkeypatch.setattr(analysis_service, "_shared_model_manager", lambda *_args: manager)
    text = "The council approved the proposal after a public meeting."
    settings = Settings(
        min_text_chars=10,
        sentiment_model_id=model_id,
        emotion_model_id=None,
        ner_model_id=None,
    )

    result = analyse_article(text, settings, LanguageDetector(settings))

    assert result.sentiment["status"] == "available_with_limitations"
    assert result.sentiment["model_identifier"] == model_id
    assert result.sentiment["class_scores"]["positive"] == 0.7
    assert result.sentiment["sentences"][0].text == text
    assert result.original_text[
        result.sentiment["sentences"][0].start:result.sentiment["sentences"][0].end
    ] == text
    assert any(
        task.task == "sentiment" and task.status == "available_with_limitations"
        for task in result.task_availability
    )


def test_model_ner_suppresses_overlapping_english_person_candidates(monkeypatch) -> None:
    model_id = MODEL_CATALOG["named_entity_recognition"]["model_id"]

    def fake_loader(task: str, configured_id: str):
        assert task == "named_entity_recognition"
        assert configured_id == model_id
        return lambda _text, **_kwargs: [
            {"entity_group": "PER", "start": 8, "end": 17, "score": 0.99},
        ]

    manager = MultilingualModelManager(configured_ids={"ner_model_id": model_id}, pipeline_loader=fake_loader)
    monkeypatch.setattr(analysis_service, "_shared_model_manager", lambda *_args: manager)
    text = "Minister Maya Rao met the council on Tuesday."
    settings = Settings(
        min_text_chars=10,
        sentiment_model_id=None,
        emotion_model_id=None,
        ner_model_id=model_id,
    )

    result = analyse_article(text, settings, LanguageDetector(settings))

    assert [(entity.text, entity.label) for entity in result.entities] == [("Maya Rao", "PER")]
    assert result.original_text[result.entities[0].start:result.entities[0].end] == "Maya Rao"
    assert result.entities[0].start == 9


def test_model_loading_failure_is_reported() -> None:
    def missing_files(_task: str, _model_id: str):
        raise OSError("not cached")

    manager = MultilingualModelManager(
        configured_ids={
            "sentiment_model_id": MODEL_CATALOG["sentiment"]["model_id"],
        },
        pipeline_loader=missing_files,
    )

    result = manager.infer("sentiment", "The council approved the proposal.", language="en")

    assert result.status == "model_unavailable"
    assert "local cache" in result.limitation
    assert not result.predictions


def test_model_loading_is_skipped_when_available_ram_is_below_budget(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=100 * 1024 * 1024)),
    )
    manager = MultilingualModelManager(configured_ids={
        "sentiment_model_id": MODEL_CATALOG["sentiment"]["model_id"],
    })

    result = manager.infer("sentiment", "The council approved its proposal.", language="en")

    assert result.status == "model_unavailable"
    assert "free RAM" in result.limitation
