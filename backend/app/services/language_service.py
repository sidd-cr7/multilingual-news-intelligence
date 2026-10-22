from pathlib import Path
from app.config import Settings
from app.schemas.article import LanguageResult

LANGUAGES = {"en": "English", "hi": "Hindi", "kn": "Kannada", "te": "Telugu", "ta": "Tamil"}
SCRIPT_RANGES = [("hi", 0x0900, 0x097F), ("kn", 0x0C80, 0x0CFF), ("te", 0x0C00, 0x0C7F), ("ta", 0x0B80, 0x0BFF)]


class LanguageDetector:
    def __init__(self, settings: Settings) -> None:
        self._model = None
        if settings.fasttext_model_path and Path(settings.fasttext_model_path).is_file():
            try:
                import fasttext
                self._model = fasttext.load_model(settings.fasttext_model_path)
            except (ImportError, ValueError, OSError):
                pass

    def detect(self, text: str) -> LanguageResult:
        counts = {code: sum(start <= ord(char) <= end for char in text) for code, start, end in SCRIPT_RANGES}
        code, script_chars = max(counts.items(), key=lambda item: item[1])
        letters = sum(char.isalpha() for char in text)
        if script_chars and script_chars / max(letters, 1) > 0.45:
            return LanguageResult(name=LANGUAGES[code], code=code, confidence=None, supported=True)
        if self._model:
            labels, values = self._model.predict(text.replace("\n", " "), k=1)
            code = labels[0].replace("__label__", "")
            confidence = float(values[0])
            return LanguageResult(name=LANGUAGES.get(code, code), code=code, confidence=confidence, supported=code in LANGUAGES, needs_confirmation=confidence < 0.70)
        try:
            from langdetect import DetectorFactory, detect_langs
            DetectorFactory.seed = 0
            candidate = detect_langs(text)[0]
            code, confidence = candidate.lang, float(candidate.prob)
            return LanguageResult(name=LANGUAGES.get(code, code), code=code, confidence=confidence, supported=code in LANGUAGES, needs_confirmation=confidence < 0.70)
        except Exception:
            return LanguageResult(name="Unknown", code="und", confidence=None, supported=False, needs_confirmation=True)
