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
    max_analysis_sentences: int = 500

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
