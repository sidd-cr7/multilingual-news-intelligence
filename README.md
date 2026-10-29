# Multilingual News Intelligence

A local research prototype for analysing article text, URLs, and PDFs; surfacing explainable language/framing indicators; comparing public news coverage; and exporting a PDF report.

## Important interpretation note

Bias and framing analysis uses the configured Groq chat model on the article's original text. Its structured output is validated against exact original-language evidence spans, and the API reports `not_configured` or `unavailable` rather than substituting keyword rules when Groq cannot run. Sparse keyword/phrase indicators remain separate auxiliary signals and do not determine the contextual assessment. Model output is not independent fact-checking, proof of author intent, or a validated measure of political bias. Supported prompt languages are English, Hindi, Kannada, Telugu, and Tamil; mocked contract tests cover each, but real-model inference and linguistic accuracy have not been independently evaluated for all five. Do not treat missing findings as evidence of neutrality.

XLM-RoBERTa remains part of the optional local model architecture for its catalogued sentiment and named-entity tasks when configured; it is not the contextual bias/framing engine. Groq GPT-OSS 120B is the default active contextual-analysis model. Optional local models are not loaded unless configured, and the dashboard's task availability distinguishes configured runtime tasks from architectural options.

## Hybrid NLP and overall judgment

The existing article-analysis route first runs configured local models, then uses the existing Groq framing request. Local checkpoints use `local_files_only=True`; no model is downloaded during a request. Current task-specific model scope:

| Task | Checkpoint | Documented target-language scope | Notes |
|---|---|---|---|
| Sentiment | `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` | English and Hindi | Three-way tweet sentiment. It does not document Telugu, Kannada, or Tamil. When local sentiment cannot run, the same Groq framing response may supply a separate exact-quote-validated sentiment fallback; it is not a bias score. |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` | English | Seven model-card labels: anger, disgust, fear, joy, neutral, sadness, surprise. It is not multilingual. Groq may return a controlled fallback in the existing contextual request when this local model cannot run, with an evidence quote, source distinction, and uncertainty. |
| English NER | `Davlan/xlm-roberta-base-ner-hrl` | English | Optional local person/organization/location NER. English date/person candidate rules are separate heuristics, not pretrained NER. |
| Telugu NER | `vaibhav006/indic-ner-muril-naamapadam` | Telugu | MuRIL/BERT token classification for PER/ORG/LOC. The model card reports 0.8328 Telugu F1 on its own evaluation set, not news-domain accuracy. It is routed for Telugu only; Hindi, Kannada, and Tamil remain unsupported by this checkpoint. The weights must be present in the local Hugging Face cache. |

`ai4bharat/IndicBERTv2-MLM-only` is a masked-language-model checkpoint, not a sentiment or emotion classifier. It is not used as one. A multilingual emotion candidate with Hindi/Tamil language tags uses synthetic data and a non-commercial license; it was not selected as verified task inference. A multilingual sentiment candidate lists Telugu, but uses synthetic data, lacks Kannada/Tamil support, and is not substituted silently. Model-card scope is not a guarantee of application accuracy. Existing local inference smoke checks covered CardiffNLP sentiment on English/Hindi, Hartmann emotion on English, and Davlan NER on English. Telugu NER is covered by mocked pipeline tests only; this project has not performed inference with those Telugu weights.

Local models are optional. `backend\requirements-ml.txt` provides the Transformers/PyTorch runtime. Download only models you intend to use, review their licenses, and set any required model IDs in `backend\.env` (the default Telugu NER route is `NER_MODEL_ID_TE`). The model manager retains only one active task pipeline to limit memory use, so switching tasks may reload weights. Models are loaded from the Hugging Face cache or a configured local path only. Weight memory estimates and limitations are listed in [docs/architecture.md](./docs/architecture.md).

Groq remains the backend-only contextual framing provider. If local sentiment or emotion is unavailable, its existing contextual request may include that fallback in the same response, avoiding a separate API call. NER stays local and has no Groq fallback. Fallback predictions are optional, schema-validated, and require exact original-text quotes unless the output is explicitly uncertain/not detectable. Groq is not called from the browser.

Successful Groq response JSON is cached in process memory for 30 minutes, up to 128 entries by default. `GROQ_CACHE_TTL_SECONDS=0` disables caching; `GROQ_CACHE_MAX_ENTRIES` bounds it. The SHA-256 cache key uses the exact normalized article text and includes language, model, prompt/schema version, input limit, and fallback options; it excludes credentials and title. Cache entries are not shared across workers or restarts, and caching cannot eliminate all charges. The response exposes `groq_used` and `cache_hit`.

The **Overall Bias & Framing Judgment** uses exact-span-validated Groq findings and separate auxiliary cues. Its provisional 0–100 indicator appears only with at least 80 whitespace-delimited tokens, 4 distinct validated findings, 3 authorial/unattributed directional findings, 1.5 weighted evidence units, and a successful contextual assessment. Per-finding weight is model confidence (0.6 when absent) × attribution weight (0.8 for journalist narration, 0.5 for unattributed claims); duplicated offsets count once. The indicator is absolute positive/negative weight imbalance × capped coverage (eligible findings/5) × mean confidence × 100. Attributed quotes are not scored. This is not a validated percentage or objective probability; sentiment and emotion are excluded from the formula. When the evidence threshold is not met, the service returns `insufficient_evidence` with no score.

Backend tests clear provider credentials and cached responses automatically. Tests use mocked Groq/model outputs and do not call the live API.

## Project layout

- `backend/` — FastAPI service, ingestion, analysis, comparison, PDF report generation, and tests.
- `frontend/` — Next.js UI and frontend unit tests.
- `docs/architecture.md` — architecture and method/limitation notes.

Generated caches/build outputs and local `.env.local` credentials are intentionally excluded from this archive. The frontend environment template is `frontend/.env.local.example`.

## Requirements

- Python 3.11+ recommended (use a version supported by the dependencies).
- Node.js 20+ and npm.
- Internet access is needed for live public-news comparison and URL ingestion. Those external services can fail or rate-limit; the UI should show those failures rather than inventing results.

## Run on Windows (PowerShell)

Open two terminals in the extracted project folder.

### 1. Start the backend

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
# Edit backend\.env locally and provide GROQ_API_KEY. GROQ_API_KEY_BACKUP is optional.
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check: open `http://127.0.0.1:8000/api/health`.

If PowerShell prevents environment activation, you can run the venv's Python directly instead:
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and
`.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`.

### Groq contextual analysis setup

The backend uses the official `groq` Python SDK (installed by `requirements.txt`). Configure these values in `backend\.env` or the server environment:

| Variable | Required | Default / purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Primary account key. |
| `GROQ_API_KEY_BACKUP` | No | Optional independent account key, attempted after recoverable primary failures. |
| `GROQ_MODEL` | No | Defaults to `openai/gpt-oss-120b`; this model is listed in Groq's current model documentation. |
| `GROQ_TIMEOUT_SECONDS` | No | Request timeout; defaults to 45 seconds. |
| `GROQ_MAX_RETRIES` | No | Bounded retry count for network/server errors only; defaults to 1. |
| `GROQ_MAX_INPUT_CHARS` | No | Maximum analysis input, defaults to 30,000 characters. Larger articles use representative beginning/middle/end excerpts and are explicitly marked partial. |

`GROQ_API_KEY_BACKUP` is failover, not additional guaranteed quota. Authentication, permission, and rate-limit failures switch directly to the backup; transient network/server failures receive the configured bounded retry before failover. Invalid model/request errors are returned explicitly and do not trigger account rotation. The configured model remains the same for both keys. Keys are server-side only and are never included in response bodies or logs. Never put them in `frontend\.env.local`, browser code, screenshots, or commits. If no primary key is configured, the analysis response marks contextual framing `not_configured`; if the provider cannot complete, it marks the task `unavailable`.

The model ID and currently documented context window are subject to provider changes. The implementation uses bounded inputs and output tokens rather than the full provider context window. No automatic model substitution or weight download occurs. See [Groq's model page](https://console.groq.com/docs/model/openai/gpt-oss-120b) and [docs/architecture.md](./docs/architecture.md) for limitations.

### Optional local model inference

The base installation remains small and does not install PyTorch or Transformers. To enable the catalogued local classifiers, install `backend\requirements-ml.txt`, obtain the model files yourself after reviewing their licenses and dataset terms, and configure the corresponding model IDs. Python 3.14.3 with Transformers 5.14.1 and CPU Torch 2.10.0 was exercised during the latest local validation; other environments should verify their dependency and hardware compatibility.

```powershell
py -3.14 -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
hf download cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual
$env:SENTIMENT_MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
```

Run `hf download` only for each model you intend to use, after reviewing its license and dataset terms. For emotion or NER, first obtain only the desired checkpoint (respectively `j-hartmann/emotion-english-distilroberta-base` or `Davlan/xlm-roberta-base-ner-hrl`) and set `EMOTION_MODEL_ID` or `NER_MODEL_ID` to that exact ID. The application loads only from the local Hugging Face cache or a configured local path (`local_files_only=True`) and runs on CPU. It distinguishes `not_configured`, `model_unavailable` (for example, configured files missing), `unsupported`, and `inference_failed` rather than silently falling back to a model-like result. CardiffNLP sentiment is documented for English and Hindi among the project languages; the English emotion model supports English only; the listed NER checkpoint does not cover any of the four Indic languages. The models are large (about 1.06 GB weights for each XLM-R-base task model and 313 MB for the English emotion model), and runtime memory is higher than weight size. Configure only tasks you need.

To load a reviewed checkpoint from a specific local directory instead of the default Hugging Face cache, set both its exact catalogued model ID and path, for example `$env:SENTIMENT_MODEL_PATH = "D:\models\cardiff-sentiment"`. `EMOTION_MODEL_PATH` and `NER_MODEL_PATH` work the same way. Paths are loaded locally only; the application does not download weights. Use the directory for the same checkpoint named by the corresponding model ID.

In the 2026-10-29 local smoke validation, all three catalogued checkpoints were present in cache. Offline inference succeeded for CardiffNLP sentiment on English and Hindi, Hartmann emotion on English, and Davlan NER on English; the end-to-end English analysis path also returned model-backed outputs with exact source offsets. The NER pipeline was adapted to the installed Transformers API and its recommended tokenizer regex fix. These were loading/smoke checks, not accuracy evaluation. No contextual-framing, sarcasm, or POS model is configured, and no emotion/NER support for Hindi, Kannada, Telugu, or Tamil was verified. See [docs/architecture.md](./docs/architecture.md) for model selection, language scope, licenses, and limitations.

### 2. Start the frontend

In a second terminal, from the project root:

```powershell
cd frontend
Copy-Item .env.local.example .env.local
npm ci
npm run dev
```

Open `http://localhost:3000`. The default frontend API URL is `http://localhost:8000`. If you change it, update `frontend/.env.local`.

### 3. Tests

Frontend unit tests:

```powershell
cd frontend
npm test
```

Backend tests (with the backend virtual environment active):

```powershell
cd backend
python -m pytest -q
```

Deterministic evaluation fixtures (not live news retrieval; not a generalisation/accuracy benchmark):

```powershell
cd backend
python -m evaluation.run --output evaluation-report.json
```

## Notes

- Do not commit or share real API keys, tokens, passwords, or private data. No real `.env.local` file is included.
- The comparison feature calls public GDELT and Google News RSS endpoints. It is bounded and non-exhaustive; provider availability, redirects, regional coverage, and rate limits can affect results.
- URL ingestion rejects non-public network addresses and follows redirects with validation. Use only article URLs you are authorised to access.
- PDF report generation uses installed system fonts for some scripts; inspect generated PDFs visually before using them in a submission.
