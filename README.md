# Multilingual News Intelligence — Phase 2

An evidence-first multilingual news-analysis application. It accepts pasted text, a public article URL, or a text-based PDF; preserves original-language text; and returns transparent linguistic and framing cues. It does not claim to determine political bias, deception, manipulation, or author intent.

## Capabilities

- Phase 1 ingestion for text, public HTTP(S) URLs, and text PDFs, with English, Hindi, Kannada, Telugu, and Tamil language identification.
- `POST /api/analyze/article` for sentence and token counts, rule-based entity candidates, lexical sentiment, limited emotion cues, and exact original-text evidence spans.
- Explainable heuristic indicators for source attribution, attribution of responsibility/cause, and loaded or evaluative language.
- A responsive Next.js dashboard that chains ingestion into analysis and highlights only evidence spans that exactly match the original text.
- Per-task availability and limitations. POS tagging and pretrained multilingual NER are explicitly unavailable rather than simulated.

The spreadsheet artifacts in this repository are preserved and are neither read nor used as training data.

## Setup

Prerequisites: Python 3.11+ and Node.js 20+.

```powershell
py -3.11 -m venv backend\.venv
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
