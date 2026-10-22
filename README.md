# Multilingual News Intelligence — Phase 1

An ingestion foundation for an explainable multilingual news-framing and bias research application. Phase 1 extracts article text from pasted text, public web pages, or text-based PDFs and detects its language. It deliberately does **not** claim to perform framing, bias, sentiment, NER, comparison, search, or reporting.

## Current capabilities

- English, Hindi, Kannada, Telugu, and Tamil recognition (with unsupported-language reporting).
- Text normalization, character/token/paragraph counts, timestamps, and transparent metadata.
- PDF extraction using PyMuPDF, with clear errors for encrypted, damaged, scanned, or textless files.
- Secure URL intake: HTTP(S) validation, public-address checks, DNS rechecks on redirects, timeouts, response-size and redirect limits, and HTML-only article extraction.
- Responsive Next.js dashboard with loading, validation, error, result, and reset states.

## Project structure

```text
backend/app/api/       HTTP routes
backend/app/services/  input, extraction, language, and metadata services
backend/app/schemas/   request and response contracts
backend/tests/         API and service tests
frontend/app/          Next.js interface
docs/                  architecture notes
```

The existing `NLP 100.xlsx`, `nlp 20 events.xlsx`, `nlp 30.xlsx`, and `nlp 60.xlsx` are preserved research artifacts. They are not read or used by the application.

## Prerequisites

- Python 3.11+ (the operating system must permit executing its Python launcher)
- Node.js 20+ and npm

## Install and run (Windows PowerShell)

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
cd backend
uvicorn app.main:app --reload --port 8000
```

Open a second PowerShell window:

```powershell
cd "C:\Users\l\OneDrive\Documents\PROJECTS\NLP PROJECT\frontend"
Copy-Item .env.local.example .env.local
npm.cmd install
npm.cmd run dev
```

Visit `http://localhost:3000`; API health is `http://localhost:8000/api/health`.

If PowerShell cannot run an activation script, invoke the environment directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

## Configuration

Copy `.env.example` to `.env`. It contains safe defaults for upload/text/response limits, CORS, and timeouts. `FASTTEXT_MODEL_PATH` is optional: point it to a locally acquired `lid.176.bin` model to use fastText. Install `backend\requirements-fasttext.txt` only on a compatible Python/toolchain, then set that path. Without it, the API uses its fallback language detector. No keys are required or included.

## Tests and checks

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest
cd ..\frontend
npm.cmd run lint
npm.cmd run build
```

## Limitations and roadmap

No OCR, translation, persistence, login, model training, Google News/GDELT retrieval, or cross-language matching is in this phase. Future modules will add POS/NER, framing/bias indicators, sentiment/emotion, psychological/media-framing concepts, evidence explanations, discovery/comparison, and reports/visualizations.
