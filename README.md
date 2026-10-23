# Multilingual News Intelligence — Phase 6

An evidence-first multilingual news-analysis application. It accepts pasted text, a public article URL, or a text-based PDF; preserves original-language text; and returns transparent linguistic and framing cues. It does not claim to determine political bias, deception, manipulation, or author intent.

## Capabilities

- Phase 1 ingestion for text, public HTTP(S) URLs, and text PDFs, with English, Hindi, Kannada, Telugu, and Tamil language identification.
- `POST /api/analyze/article` for sentence and token counts, rule-based entity candidates, lexical sentiment, limited emotion cues, and exact original-text evidence spans.
- Explainable heuristic indicators for source attribution, attribution of responsibility/cause, and loaded or evaluative language.
- A responsive Next.js dashboard that chains ingestion into analysis and highlights only evidence spans that exactly match the original text.
- Per-task availability and limitations. POS tagging and pretrained multilingual NER are explicitly unavailable rather than simulated.

The spreadsheet artifacts in this repository are preserved and are neither read nor used as training data.

## Setup

Prerequisites: Python 3.12 (verified; 3.11+ should work) and Node.js 20+ (verified with 24).

```powershell
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
cd backend
uvicorn app.main:app --reload --port 8000
```

In another PowerShell window:

```powershell
cd frontend
Copy-Item .env.local.example .env.local
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:3000`; health is at `http://localhost:8000/api/health`.

## API example

```powershell
Invoke-RestMethod http://localhost:8000/api/analyze/article -Method Post -ContentType 'application/json' -Body '{"text":"Officials said the shocking crisis was caused by poor planning."}'
```

The response contains original text and language, linguistic counts, entities, sentiment/emotion, heuristic indicators with `quote`, `start`, and `end`, statistics, availability, warnings, and limitations. Offsets are Unicode character positions in `original_text`.

## Methods, coverage, and limitations

No model weights are downloaded at startup. The core pipeline uses deterministic punctuation/token rules and compact inspectable lexicons. Sentiment seed lexicons cover English, Hindi, Kannada, Telugu, and Tamil. Emotion is a limited English lexicon. Entity extraction is limited to dates and conservative English person candidates. These are not calibrated pretrained-model predictions and negative sentiment is never equated with bias.

Language identification reuses Phase 1's script detector for Hindi/Kannada/Telugu/Tamil and `langdetect` fallback for other text. An optional locally supplied fastText language-ID model may be used when configured. Consult the upstream licenses before redistribution (`langdetect` is Apache-2.0; fastText is MIT); this repository includes neither model weights nor a new pretrained NLP model.

## Phase 3 indicator report

`POST /api/analyze/article` remains backward compatible and now also returns a statistical report in `statistics`. It contains raw counts by indicator, sentence/token counts, sentence-level lexical sentiment distribution where supported, and `evidence_density_per_1000_tokens`. Tokens are non-empty whitespace-delimited sequences in the original normalized text; density is raw evidence spans ÷ token count × 1,000. Neither counts nor rates are bias scores.

The indicator registry provides observable cues for attributed claims, unattributed assertion candidates, responsibility/agency language, hedging, absolutist/generalising language, loaded/evaluative wording, and a narrowly defined potentially-one-sided cue (evaluative wording in an un-attributed sentence). Every cue has an exact Unicode offset and direct quote. Invalid spans fail validation in the backend; the UI additionally refuses to highlight a span unless it exactly slices the displayed original text. These rules support English, Hindi, Kannada, Telugu, and Tamil with deliberately compact lexicons, so contextual nuance, quotations, and language variation remain important limitations.

`FASTTEXT_MODEL_PATH` may point to a locally acquired fastText model for language identification. Local models/caches must remain outside version control. The core has modest CPU/RAM needs; optional pretrained models should be installed/configured deliberately and never downloaded during application startup.

For manual evaluation, run representative short articles in each target language; inspect each returned quote against the original text; record false positives/negatives and missing language-specific terms. This validates application behavior, not real-world bias-detection accuracy.

## Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
cd ..\frontend
npm.cmd run lint
npm.cmd run build
```

No OCR, translation, persistence, model training, paid APIs, cross-language discovery, or automated comparison is included.

## Phase 4 discovery and comparison

`POST /api/compare/article` accepts the same article fields as the analysis endpoint plus `settings.target_languages`, `settings.result_limit` (1–10), and `settings.include_original_language`. It uses public GDELT's document endpoint and Google News RSS without credentials, applies bounded timeouts, deduplicates URLs before retrieval, and records provider failures as partial-result metadata.

Candidates are never assumed to be the same event. A transparent rule combines shared event terms and keyword overlap to yield `verified_match`, `possible_match`, or `insufficient_evidence`; the score is a relevance signal, not a probability. Full text is fetched only through the existing safe URL service and is analyzed by the existing pipeline. Snippet-only and metadata-only results are explicitly scoped and never represented as full-article analysis. Comparison coverage reports languages, sources, full-text/snippet counts, and aggregate observable-indicator counts, not source or language bias.

Set `DISCOVERY_RESULT_LIMIT` and `DISCOVERY_TIMEOUT_SECONDS` in `.env` to bound work. Provider availability, RSS formats, paywalls, robots policies, redirects, and article extraction can limit coverage.

## Phase 5 evaluation and reliability

The repository includes a small deterministic evaluation framework in `backend/evaluation/`. Its manually reviewed fixtures cover the five supported scripts, exact evidence offsets (including repeated and Indic-script text), availability reporting, verified/possible/rejected/insufficient event outcomes, snippet-only inputs, and provider-style incomplete metadata. It is software evaluation, not a representative corpus or an estimate of real-world framing, sentiment, NER, or event-matching performance. The project does not train on, read, or alter the included spreadsheet artifacts.

Run it from `backend` after installing the backend requirements:

```powershell
python -m evaluation.run --output evaluation-report.json
```

The generated JSON records case counts, skips, exact-span failures, outcome counts, per-language coverage, feature availability, errors, and limitations. Its precision/recall/F1, when present, is explicitly limited to the small reviewed matching fixtures; the sample size appears in the report and the metric must not be generalized to live news discovery.

Evidence offsets are zero-based Unicode character offsets with an exclusive `end`, so `original_text[start:end] == quote` must always hold. Invalid, empty, truncated, malformed, or altered evidence is rejected rather than repaired. Comparison relevance is an uncalibrated rule signal, never a probability. A date conflict is reported as `rejected_match`; absence of enough information is `insufficient_evidence`.

### Reproducible checks

```powershell
# Backend
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
cd backend
python -m pytest
python -m evaluation.run --output evaluation-report.json
uvicorn app.main:app --reload --port 8000

# Frontend (separate shell)
cd frontend
npm.cmd install
npm.cmd run lint
npm.cmd run build
npm.cmd run dev
```

### Known limitations

- `backend/evaluation-report.json` is generated output; it is excluded from version control.
- `frontend/next-env.d.ts` is auto-generated by Next.js; it is excluded from version control.
- The `StarletteDeprecationWarning` from `fastapi`'s test client is an upstream issue unrelated to this project's code.
- Live discovery (GDELT, Google News RSS) cannot be unit-tested without mocks; unit tests use deterministic fixtures only.
- `FASTTEXT_MODEL_PATH` is optional; language detection falls back to `langdetect` when unset.

Normal tests and evaluation fixtures make no live provider calls. Live discovery remains bounded by configured timeouts and result limits, but it cannot guarantee comprehensive retrieval, access to paywalled pages, or correct identification of every same-event article.
