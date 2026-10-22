# Phase 1 Architecture

The browser-facing Next.js application (`frontend/`) submits exactly one source type to the FastAPI service (`backend/`). The API does not persist articles in Phase 1.

```text
Text / URL / PDF → FastAPI ingestion route → extraction & normalization
                                      └── language identification → metadata JSON → Next.js results view
```

Each input route delegates to a small service. `article_service` owns the shared metadata workflow; extraction, language detection, future analysis, discovery, comparison, and reporting remain independently replaceable modules.

URL retrieval resolves every target hostname before each request/redirect and blocks non-global addresses. It uses a bounded streamed response, a timeout, a redirect limit, supported HTML content types, and Trafilatura only for readable-text extraction. It does not circumvent paywalls or authentication.

Language detection first identifies the Indic scripts deterministically. For other text, it loads fastText if `FASTTEXT_MODEL_PATH` points to a locally supplied model, otherwise it uses `langdetect`. Confidence is reported only from a detector that supplies it.
