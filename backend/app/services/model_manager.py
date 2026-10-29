"""Optional, local-only model inference with explicit task and language scope."""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)
MODEL_SETTING_KEYS = {
    "sentiment": "sentiment_model_id",
    "emotion": "emotion_model_id",
    "named_entity_recognition": "ner_model_id",
}
MODEL_PATH_SETTING_KEYS = {
    "sentiment": "sentiment_model_path",
    "emotion": "emotion_model_path",
    "named_entity_recognition": "ner_model_path",
}

MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "sentiment": {
        "model_id": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
        "supported_languages": ["en", "hi"],
        "task": "text-classification",
        "parameters_millions": 278,
        "weights_mb": 1061,
        "minimum_free_memory_mb": 2122,
        "license": "Model-weight license is not specified in the model metadata; review before redistribution.",
        "training_data": "CardiffNLP Tweet Sentiment Multilingual: English, Hindi, Arabic, French, German, Italian, Portuguese, and Spanish tweets.",
        "informal_text": "Trained on tweets; informal social-media text is represented. Indian-language code-switching and Latin-script transliteration are not established by the documented task data.",
        "limitations": "Three-way tweet sentiment, not emotion, sarcasm, framing, or bias detection; reported test macro-F1 is about 0.693 on its dataset and is not an estimate of news-article accuracy.",
    },
    "emotion": {
        "model_id": "j-hartmann/emotion-english-distilroberta-base",
        "supported_languages": ["en"],
        "task": "text-classification",
        "parameters_millions": 82,
        "weights_mb": 313,
        "minimum_free_memory_mb": 626,
        "license": "No model-weight license is declared in the inspected model metadata; review before redistribution.",
        "training_data": "A balanced subset from six English emotion datasets, including Twitter, Reddit, self-reports, and TV dialogue.",
        "informal_text": "Some informal English social-media and conversational text is represented; slang and sarcasm are not separate validated tasks.",
        "limitations": "Seven English emotion labels; not sentiment, sarcasm, or a multilingual emotion model. Reported accuracy is about 66% on its documented evaluation subset, not news-domain accuracy.",
    },
    "named_entity_recognition": {
        "model_id": "Davlan/xlm-roberta-base-ner-hrl",
        "supported_languages": ["ar", "de", "en", "es", "fr", "it", "lv", "nl", "pt", "zh"],
        "language_models": {
            "te": {
                "model_id": "vaibhav006/indic-ner-muril-naamapadam",
                "supported_languages": ["te"],
                "parameters_millions": 278,
                "weights_mb": 1_060,
                "minimum_free_memory_mb": 2_120,
                "license": "Apache-2.0",
                "limitations": "Fine-tuned MuRIL NER; model card reports Telugu F1 0.8328 on its evaluation set. This is not a news-domain accuracy estimate.",
            },
        },
        "task": "token-classification",
        "parameters_millions": 278,
        "weights_mb": 1059,
        "minimum_free_memory_mb": 2118,
        "license": "AFL-3.0",
        "training_data": "News/entity datasets for Arabic, German, English, Spanish, French, Italian, Latvian, Dutch, Portuguese, and Chinese.",
        "informal_text": "Trained on entity-annotated news, not documented as slang or code-switching coverage.",
        "limitations": "Person, organization, and location labels; Hindi, Kannada, Telugu, and Tamil are not supported by its documented training languages.",
    },
}


@dataclass(slots=True)
class ModelSpec:
    task: str
    model_id: str | None
    status: str
    supported_languages: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    parameters_millions: int | None = None
    weights_mb: int | None = None
    minimum_free_memory_mb: int | None = None
    license: str | None = None
    notes: str | None = None
    loaded: bool = False
    fallback: str | None = None


@dataclass(slots=True)
class ModelInference:
    task: str
    status: str
    model_id: str | None = None
    predictions: list[dict[str, Any]] = field(default_factory=list)
    method: str | None = None
    limitation: str | None = None


PipelineLoader = Callable[[str, str], Any]
MAX_MODEL_TOKENS = 512


class ModelResourceError(RuntimeError):
    pass


def _load_local_pipeline(task: str, model_location: str) -> Any:
    import psutil

    required_mb = MODEL_CATALOG[task]["minimum_free_memory_mb"]
    available_mb = psutil.virtual_memory().available // (1024 * 1024)
    if available_mb < required_mb:
        raise ModelResourceError(
            f"Model requires approximately {required_mb} MB free RAM; {available_mb} MB is available."
        )

    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )

    tokenizer_options = {}
    if task == "named_entity_recognition" and int(transformers.__version__.split(".", 1)[0]) >= 5:
        tokenizer_options["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_location,
        local_files_only=True,
        **tokenizer_options,
    )
    if task == "named_entity_recognition":
        model_token_limit = tokenizer.model_max_length
        if not isinstance(model_token_limit, int) or model_token_limit <= 0:
            model_token_limit = MAX_MODEL_TOKENS
        tokenizer.model_max_length = min(model_token_limit, MAX_MODEL_TOKENS)
        model = AutoModelForTokenClassification.from_pretrained(
            model_location,
            local_files_only=True,
        )
        return pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=-1,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_location,
        local_files_only=True,
    )
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
    )


class MultilingualModelManager:
    """Load only explicitly configured, catalogued models from the local cache."""

    def __init__(
        self,
        *,
        configured_ids: dict[str, str | None] | None = None,
        configured_paths: dict[str, str | None] | None = None,
        pipeline_loader: PipelineLoader = _load_local_pipeline,
    ) -> None:
        self._lock = RLock()
        self._config = configured_ids or {}
        self._paths = configured_paths or {}
        self._pipeline_loader = pipeline_loader
        self._active_key: tuple[str, str] | None = None
        self._active_pipeline: Any = None

    def resolve(self, task: str, *, language: str | None = None) -> ModelSpec:
        config = MODEL_CATALOG.get(task)
        if config is None:
            return ModelSpec(
                task=task,
                model_id=None,
                status="unsupported",
                notes="No model is catalogued for this task.",
            )

        model_id = self._config.get(MODEL_SETTING_KEYS.get(task, f"{task}_model_id"))
        language_config = config.get("language_models", {}).get(language, {})
        language_setting_key = f"{MODEL_SETTING_KEYS.get(task, task + '_model_id')}_{language}"
        language_model_id = self._config.get(language_setting_key)
        if language_config:
            model_id = language_model_id or language_config["model_id"]
        languages = list(language_config.get("supported_languages", config["supported_languages"]))
        selected_config = (
            language_config
            if language_config and (
                language_model_id or language not in config["supported_languages"]
            )
            else config
        )
        catalogued_ids = {
            config["model_id"],
            *(model["model_id"] for model in config.get("language_models", {}).values()),
        }
        common = {
            "task": task,
            "model_id": model_id,
            "supported_languages": languages,
            "requires": ["transformers", "torch"],
            "parameters_millions": selected_config["parameters_millions"],
            "weights_mb": selected_config["weights_mb"],
            "minimum_free_memory_mb": selected_config["minimum_free_memory_mb"],
            "license": selected_config["license"],
            "fallback": None,
        }
        if model_id and model_id not in catalogued_ids:
            common["supported_languages"] = []
        if model_id is None and language and language not in languages:
            return ModelSpec(
                **common,
                status="unsupported",
                notes=f"'{language}' is not in this model's documented training languages.",
            )
        if model_id is None:
            return ModelSpec(
                **common,
                status="not_configured",
                notes=(
                    f"No model is configured. Candidate: "
                    f"{selected_config.get('model_id', config['model_id'])}. "
                    f"{selected_config['limitations']}"
                ),
            )
        if model_id not in catalogued_ids:
            return ModelSpec(
                **common,
                status="inconclusive",
                notes="Custom model is not in the reviewed catalog; its task and language support are unverified.",
            )
        if language and language not in languages:
            return ModelSpec(
                **common,
                status="unsupported",
                notes=f"'{language}' is not in this model's documented training languages.",
            )
        return ModelSpec(
            **common,
            status="inconclusive",
            notes="Configured but not yet loaded or inference-tested in this process.",
        )

    def _pipeline(self, task: str, model_id: str, model_path: str | None) -> Any:
        model_location = model_path or model_id
        key = (task, model_location)
        with self._lock:
            if self._active_key == key:
                return self._active_pipeline
            self._active_pipeline = None
            self._active_key = None
            gc.collect()
            self._active_pipeline = self._pipeline_loader(task, model_location)
            self._active_key = key
            return self._active_pipeline

    def infer(self, task: str, text: str, *, language: str) -> ModelInference:
        with self._lock:
            return self._infer_locked(task, text, language=language)

    def _infer_locked(self, task: str, text: str, *, language: str) -> ModelInference:
        spec = self.resolve(task, language=language)
        if spec.status != "inconclusive" or spec.model_id is None:
            return ModelInference(
                task=task,
                status=spec.status,
                model_id=spec.model_id,
                limitation=spec.notes,
            )
        catalogued_ids = {
            MODEL_CATALOG.get(task, {}).get("model_id"),
            *(
                item["model_id"]
                for item in MODEL_CATALOG.get(task, {}).get("language_models", {}).values()
            ),
        }
        if spec.model_id not in catalogued_ids:
            return ModelInference(
                task=task,
                status="inconclusive",
                model_id=spec.model_id,
                limitation="Custom model is not in the reviewed catalog; inference was not attempted.",
            )

        try:
            classifier = self._pipeline(
                task,
                spec.model_id,
                self._paths.get(MODEL_PATH_SETTING_KEYS[task]),
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("Model runtime unavailable for task %s: %s", task, exc)
            return ModelInference(
                task=task,
                status="model_unavailable",
                model_id=spec.model_id,
                limitation=f"Optional inference dependency unavailable: {exc}",
            )
        except ModelResourceError as exc:
            logger.warning("Insufficient memory for task %s: %s", task, exc)
            return ModelInference(
                task=task,
                status="model_unavailable",
                model_id=spec.model_id,
                limitation=str(exc),
            )
        except OSError as exc:
            logger.warning("Local model files unavailable for task %s: %s", task, exc)
            return ModelInference(
                task=task,
                status="model_unavailable",
                model_id=spec.model_id,
                limitation=f"Model files were not found in the local cache: {exc}",
            )
        except Exception as exc:
            logger.exception("Could not load local model for task %s", task)
            return ModelInference(
                task=task,
                status="inference_failed",
                model_id=spec.model_id,
                limitation=f"Model loading failed: {type(exc).__name__}: {exc}",
            )

        try:
            if task == "named_entity_recognition":
                raw = classifier(text)
                predictions = raw if isinstance(raw, list) else []
            else:
                raw = classifier(
                    text,
                    truncation=True,
                    max_length=512,
                    top_k=None,
                )
                if isinstance(raw, list) and raw and isinstance(raw[0], list):
                    raw = raw[0]
                predictions = raw if isinstance(raw, list) else []
            return ModelInference(
                task=task,
                status="available_with_limitations",
                model_id=spec.model_id,
                predictions=[item for item in predictions if isinstance(item, dict)],
                method="Transformers local CPU inference",
                limitation=(
                    MODEL_CATALOG[task].get("language_models", {}).get(language, MODEL_CATALOG[task])["limitations"]
                ),
            )
        except Exception as exc:
            logger.exception("Inference failed for task %s", task)
            return ModelInference(
                task=task,
                status="inference_failed",
                model_id=spec.model_id,
                limitation=f"Inference failed: {type(exc).__name__}: {exc}",
            )

    def get_catalog(self) -> dict[str, dict[str, Any]]:
        return MODEL_CATALOG
