from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_env: str = "development"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_text_chars: int = 100_000
    min_text_chars: int = 40
    http_timeout_seconds: float = 10.0
    max_url_response_bytes: int = 5 * 1024 * 1024
    max_redirects: int = 3
    cors_origins: str = "http://localhost:3000"
    fasttext_model_path: str | None = None
    sentiment_model_id: str | None = None
    sentiment_model_path: str | None = None
    emotion_model_id: str | None = None
    emotion_model_path: str | None = None
    ner_model_id: str | None = None
    ner_model_path: str | None = None
    ner_model_id_te: str | None = None
    embedding_model_id: str | None = None
    max_analysis_sentences: int = 500
    max_model_sentences: int = 32
    max_model_input_chars: int = 4_000
    groq_api_key: str | None = None
    groq_api_key_backup: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_timeout_seconds: float = 45.0
    groq_max_retries: int = 1
    groq_max_input_chars: int = 30_000
    groq_cache_ttl_seconds: int = 1_800
    groq_cache_max_entries: int = 128
    discovery_result_limit: int = 5
    discovery_timeout_seconds: float = 8.0

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
