import pytest

from app.config import get_settings
from app.services.groq_framing_service import _CACHE_LOCK, _RESPONSE_CACHE


@pytest.fixture(autouse=True)
def isolate_optional_provider_calls(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY_BACKUP", "")
    get_settings.cache_clear()
    with _CACHE_LOCK:
        _RESPONSE_CACHE.clear()
    yield
    get_settings.cache_clear()
    with _CACHE_LOCK:
        _RESPONSE_CACHE.clear()
