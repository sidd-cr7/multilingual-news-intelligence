# Phase 2 Architecture

The browser-facing Next.js application submits one source type to the FastAPI service. Articles are not persisted.

```text
Text / URL / PDF → ingestion route → extraction & normalization → language + metadata
                                                              ↓
Dashboard ← structured response ← POST /api/analyze/article ← analysis service
                                                       ↓
segment/tokenize + lexical sentiment/emotion + entity rules + indicator registry
                                                       ↓
                   exact original-text evidence spans, availability, limitations
```

The Phase 1 ingestion routes remain responsible for safe URL/PDF/text extraction. The dashboard then passes their `extracted_text` to the analysis endpoint, preserving paste, URL, and PDF workflows.

`analysis_service` never translates input. It uses punctuation-based segmentation, whitespace tokenization, small inspectable language lexicons, conservative English date/person-candidate rules, and evidence helpers. Evidence quotes are direct slices of `original_text` using returned Unicode character offsets.

Phase 3 extends the same response with a `statistics` report: raw category counts, sentence/token counts, optional lexical-sentiment distribution, and an evidence-per-1,000-token rate. The rate is normalisation for comparison only, never a bias score. The evidence validator rejects any span whose bounds or quote fail to match `original_text` exactly.

Evidence records are deduplicated per category so repeated wording is one finding with an occurrence count, while `occurrence_spans` and `contexts` retain every counted position and sentence. Sentence-level rules anchor on a predicate token rather than the whole sentence, which keeps highlights precise and non-overlapping. `evidence_strength` names the matching method and is not a confidence probability.

Every analysis response includes per-task availability. POS tagging and pretrained multilingual NER are marked unavailable until a locally configured model integration exists; no output is fabricated. The available framing cues identify observable source-attribution, responsibility/causation, and loaded/evaluative wording. They are heuristic evidence—not findings of author intent, political bias, deception, or manipulation.

The existing URL service resolves every hostname before each request/redirect, blocks non-global addresses, bounds response sizes, and extracts readable HTML only. It does not bypass paywalls or authentication.

Phase 4 adds `POST /api/compare/article`. `comparison_service` generates entity/keyword queries, issues one bounded GDELT request covering all requested source languages plus one Google News RSS request per language, deduplicates URLs, and only then uses the existing safe URL extractor with bounded concurrency. `provider_status` separates an empty provider result from a timeout, rate limit, HTTP error, or invalid response, and a failed provider never disables the other. A deterministic same-event verifier returns rationale and an uncalibrated relevance signal. The existing analysis service runs only over available candidate text; the comparison response separates full-text, snippet-only, and metadata-only coverage and preserves provider failures as partial results.

## Phase 5: deterministic evaluation and reliability boundaries

`backend/evaluation/fixtures.json` is deliberately separate from application code and spreadsheets. `python -m evaluation.run --output evaluation-report.json` runs the analysis service and same-event verifier only against these reviewed, deterministic inputs. It does not call GDELT, Google News, or article URLs. The report identifies exact-span validation failures, language-case coverage, feature availability, comparison verification outcomes, errors, skips, and a clearly scoped fixture-only matching metric.

Evidence is fail-closed at the analysis boundary: a span must have `0 <= start < end <= len(original_text)` and an exact `original_text[start:end] == quote` match. The frontend independently declines invalid spans. The offset convention is Unicode code-point indexing as used by Python strings and JavaScript string slicing for BMP Indic text; callers should preserve the returned original text rather than normalize it again.

Comparison results use explicit verification states: `verified_match` for sufficient full-text rule evidence, `possible_match` for lower-confidence or snippet-only similarity, `rejected_match` for a conflicting date or weak full-text similarity, and `insufficient_evidence` where the system cannot decide. Scores are diagnostic relevance signals only. Coverage is bounded, partial, and must never be interpreted as exhaustive or as a statement about a source, language, or article's bias.

## Phase 7: research reporting and PDF export

`POST /api/reports/pdf` accepts the typed `ArticleAnalysisResponse` previously returned by `/api/analyze/article`, plus an optional `ComparisonResponse` previously returned by `/api/compare/article`. The route serializes those results without rerunning analysis or contacting discovery providers. It validates exact evidence spans, bounds article text to 100,000 Unicode characters, limits evidence to 5,000 spans and comparison candidates to 100, and rejects a serialized request over 5 MB. Successful responses use `application/pdf` and an attachment filename; validation failures use FastAPI's `422` response and rendering failures return a generic `500` message without internal exception details.

`report_service` composes the supplied outputs into a paginated A4 PDF with headings, tables, evidence grouped by category, generated-at metadata, source attribution, and page numbering. It reports punctuation-based approximate sentences, whitespace-delimited tokens, distinct finding counts with occurrence counts and explicit denominators, and evidence density as separate quantities. A generated summary and publication date are not available in current analysis/ingestion results and are labelled unavailable. URL credentials, query strings, and fragments are omitted from printed URLs. Empty entity results, unsupported analyses, absent comparisons, and missing language coverage are not presented as fabricated zeroes or findings.

The dashboard uses its current analysis and includes comparison data only when present. A report-generation failure is displayed separately and does not clear the analysis or comparison state. The download utility is covered by a deterministic Node built-in test. A browser end-to-end test suite is not currently configured.

The backend depends on `PyMuPDF`, `pymupdf-fonts`, `fpdf2`, and `uharfbuzz`. Noto Sans resources and HarfBuzz provide report typography and shaping; no large font assets are committed. Windows uses the installed Nirmala UI collection, while Linux deployments should install the matching Noto Sans font for the article script. The evidence appendix was visually inspected with Nirmala UI for Hindi, Kannada, Telugu, and Tamil; visual behavior still depends on the installed font and PDF viewer. A UTF-8 evidence attachment preserves exact quotations independently of viewer text extraction. If a suitable system font is absent, the report states that the shaped appendix is unavailable and retains the attachment fallback. Visually inspect PDFs before publication. Reports explain that lexical/rule indicators are observations rather than proof of intent or definitive bias.

## Phase 8: context-aware framing boundaries

The indicator record is the common internal representation for the dashboard, comparisons, and PDF: category, exact original-text quote and offsets, sentence context, explanation, method, evidence-status label, occurrence locations, and limitation. Evidence is de-duplicated by cue/category while retaining every exact occurrence location; it is not de-duplicated merely because a word repeats in a different sentence.

Attribution-marker counts are sentence-level descriptive statistics, not a count of unique sources or a measure of viewpoint balance. English has additional reviewed lexical markers for allegations/accusations, explicit disputes, and opinions/interpretations. Those markers describe how a claim is presented and never decide whether it is true. They are explicitly unavailable for Hindi, Kannada, Telugu, and Tamil until language-specific resources are reviewed.

Comparative wording observations are emitted only for `verified_match` candidates with retrieved full text. Each observation cites an evidence quote and context from both articles and states that a difference may reflect length, scope, context, or editorial choices rather than bias. Possible, rejected, insufficient-evidence, snippet-only, and metadata-only candidates do not generate comparative framing observations.
