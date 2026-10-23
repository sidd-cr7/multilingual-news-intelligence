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

Every analysis response includes per-task availability. POS tagging and pretrained multilingual NER are marked unavailable until a locally configured model integration exists; no output is fabricated. The available framing cues identify observable source-attribution, responsibility/causation, and loaded/evaluative wording. They are heuristic evidence—not findings of author intent, political bias, deception, or manipulation.

The existing URL service resolves every hostname before each request/redirect, blocks non-global addresses, bounds response sizes, and extracts readable HTML only. It does not bypass paywalls or authentication.
