"use client";

import { ChangeEvent, FormEvent, useRef, useState } from "react";
import type { ReactNode } from "react";
import { FileText, Globe2, Languages, RotateCcw, Upload } from "lucide-react";
import { browserDownloadTarget, downloadPdf, reportFilename } from "./reportDownload.mjs";
import { evidenceTotals, highlightRanges } from "./evidence.mjs";
import { framingNoFindingsMessage, framingStatusMessage } from "./framingStatus.mjs";

type Mode = "text" | "url" | "pdf";
type Language = {
  name: string;
  code: string;
  supported: boolean;
  needs_confirmation?: boolean;
  confidence?: number | null;
};
type ArticleResult = {
  title?: string | null;
  original_url?: string | null;
  language: Language;
  extracted_text: string;
  word_count: number;
  character_count: number;
  paragraph_count: number;
  warnings: string[];
};
type Evidence = {
  category: string;
  quote: string;
  start: number;
  end: number;
  context: string;
  explanation: string;
  method: string;
  uncertainty?: string | null;
  cue?: string | null;
  occurrences?: number;
  occurrence_spans?: [number, number][];
  contexts?: string[];
  evidence_strength?: string | null;
  claim_status?: string | null;
  claim_status_note?: string | null;
};
type FramingFinding = {
  id: string;
  category: string;
  label: string;
  actor?: string | null;
  speaker?: string | null;
  attribution_type: string;
  framing_direction: string;
  evidence_text: string;
  start_offset: number;
  end_offset: number;
  explanation: string;
  contextual_interpretation: string;
  potential_reader_effect: string;
  evidence_strength: string;
  alternative_interpretation: string;
  method: string;
  uncertainty: string;
  confidence?: number | null;
};
type BiasAssessment = {
  status: string;
  provider?: string | null;
  model?: string | null;
  groq_used?: boolean;
  cache_hit?: boolean;
  error_code?: string | null;
  analysis_scope: string;
  neutral_rewrite?: string | null;
  conclusion: string;
  summary: string;
  strongest_patterns: string[];
  actor_portrayals: {
    actor: string;
    direction: string;
    finding_ids: string[];
    explanation: string;
  }[];
  competing_attributed_viewpoints?: boolean | null;
  representation_comparison: string;
  selective_presentation: string;
  sufficient_evidence_for_overall_assessment: boolean;
  findings: FramingFinding[];
  limitations: string[];
};
type BiasJudgment = {
  status: string;
  summary: string;
  indicator: number | null;
  indicator_label: string;
  direction: string;
  targets: BiasAssessment["actor_portrayals"];
  reasons: {
    evidence_text: string;
    start_offset: number;
    end_offset: number;
    category: string;
    label: string;
    actor?: string | null;
    speaker?: string | null;
    attribution_type: string;
    framing_direction: string;
    explanation: string;
    confidence?: number | null;
    method: string;
  }[];
  validated_auxiliary_cues: {
    category: string;
    quote: string;
    start_offset: number;
    end_offset: number;
    explanation: string;
    claim_status?: string | null;
  }[];
  sentiment_signal: {
    status: string;
    label?: string | null;
    model_identifier?: string | null;
    limitation?: string | null;
    role: string;
  };
  emotion_signal: {
    status: string;
    label?: string | null;
    model_identifier?: string | null;
    limitation?: string | null;
    role: string;
  };
  entity_mentions: {
    text: string;
    label: string;
    start: number;
    end: number;
    method: string;
    role?: string | null;
  }[];
  evidence_sufficiency: string;
  scoring_method: string;
  limitations: string[];
};
type Analysis = {
  analysis_id: string;
  title?: string | null;
  original_url?: string | null;
  original_text: string;
  language: Language;
  linguistic: {
    sentence_count: number;
    token_count: number;
    token_count_convention: string;
  };
  entities: {
    text: string;
    label: string;
    method: string;
    start: number;
    end: number;
    confidence?: number | null;
  }[];
  sentiment: {
    status: string;
    label?: string | null;
    model_identifier?: string | null;
    class_scores?: Record<string, number>;
    score_note?: string | null;
    limitation?: string | null;
    method?: string | null;
    sentences?: {
      label: string;
      text: string;
      start: number;
      end: number;
      score?: number | null;
      class_scores?: Record<string, number>;
    }[];
  };
  emotion: {
    status: string;
    label?: string | null;
    model_identifier?: string | null;
    score_note?: string | null;
    limitation?: string | null;
    method?: string | null;
    distribution?: Record<string, number>;
    sentences?: {
      label: string;
      text: string;
      start: number;
      end: number;
      score?: number | null;
    }[];
  };
  indicators: Evidence[];
  bias_assessment: BiasAssessment;
  overall_judgment: BiasJudgment | null;
  statistics: {
    indicator_counts: Record<string, number>;
    indicator_occurrences?: Record<string, number>;
    total_evidence_findings?: number;
    total_cue_occurrences?: number;
    evidence_density_per_1000_tokens?: number | null;
    evidence_density_convention?: string;
    attribution_summary?: {
      attributed_findings: number;
      unattributed_findings: number;
      mixed_status_findings: number;
      attribution_marker_findings: number;
      denominator: string;
      note: string;
    };
  };
  methodology?: {
    approach?: string;
    categories?: Record<string, string>;
    claim_status?: Record<string, string>;
    offsets?: string;
    interpretation?: string;
  };
  task_availability: {
    task: string;
    status: string;
    method?: string | null;
    limitation?: string | null;
    supported_languages?: string[];
    model_identifier?: string | null;
    fallback?: string | null;
  }[];
  limitations: string[];
  warnings: string[];
};
type Candidate = {
  url: string;
  title: string;
  source_name?: string | null;
  published_at?: string | null;
  language?: Language | null;
  retrieval_status: string;
  analysis_scope: string;
  verification_status: string;
  relevance_score?: number | null;
  match_reasons: string[];
  uncertainty: string[];
};
type EventProfile = {
  headline: string | null;
  summary: string;
  actors: string[];
  organizations: string[];
  locations: string[];
  dates: string[];
  main_action: string | null;
  distinctive_phrases: string[];
  original_language: string;
  is_specific_event: boolean;
  uncertainty_notes: string[];
};
type FramingDifference = {
  dimension: string;
  article_a_quote: string | null;
  article_b_quote: string | null;
  explanation: string;
  significance: string | null;
  confidence: string | null;
  alternative_interpretation: string | null;
};
type Comparison = {
  event_profile?: EventProfile | null;
  discovery_queries: Record<string, string>;
  candidates: Candidate[];
  comparison: {
    coverage: {
      discovered_articles: number;
      evaluated_articles?: number;
      matched_articles: number;
      languages_represented: string[];
      sources_represented: string[];
      full_text_articles: number;
      snippet_only_articles: number;
      verification_outcomes: Record<string, number>;
      retrieval_outcomes: Record<string, number>;
    };
    indicator_counts_across_matches: Record<string, number>;
    wording_comparison_basis?: string;
    observation: string;
  };
  framing_differences?: FramingDifference[];
  provider_failures: string[];
  provider_status?: Record<string, string>;
  limitations: string[];
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const CATEGORY_LABELS: Record<string, string> = {
  loaded_or_evaluative_language: "Loaded or evaluative language",
  attribution_of_responsibility: "Attribution of responsibility",
  attributed_claim: "Attributed claim",
  hedging_or_qualification: "Hedging or qualification",
  absolutist_or_generalising_language: "Absolutist or generalising language",
};

const CLAIM_LABELS: Record<string, string> = {
  attributed: "Attributed",
  unattributed: "Unattributed",
  attribution_marker: "Attribution marker",
  mixed: "Mixed attribution",
};

const categoryLabel = (category: string) =>
  CATEGORY_LABELS[category] ?? category.replaceAll("_", " ");

const claimLabel = (status?: string | null) =>
  status ? CLAIM_LABELS[status] ?? status.replaceAll("_", " ") : "";

function HighlightedText({
  text,
  evidence,
  focusedKey,
}: {
  text: string;
  evidence: Evidence[];
  focusedKey: string;
}) {
  const items: ReactNode[] = [];
  let cursor = 0;
  for (const range of highlightRanges(evidence, text)) {
    items.push(text.slice(cursor, range.start));
    items.push(
      <mark
        id={`mark-${range.key}`}
        key={`mark-${range.key}`}
        className={range.key === focusedKey ? "mark-focus" : undefined}
      >
        {text.slice(range.start, range.end)}
      </mark>,
    );
    cursor = range.end;
  }
  items.push(text.slice(cursor));
  return <>{items}</>;
}

async function responseJson(response: Response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "The request could not be completed.");
  }
  return data;
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ArticleResult | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [error, setError] = useState("");
  const [reportMessage, setReportMessage] = useState("");
  const [reportFailed, setReportFailed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);
  const busy = useRef(false);

  const reset = () => {
    setMode("text");
    setText("");
    setUrl("");
    setFile(null);
    setResult(null);
    setAnalysis(null);
    setComparison(null);
    setError("");
    setReportMessage("");
    setReportFailed(false);
  };

  async function requestAnalysis(article: ArticleResult) {
    const analysisResponse = await fetch(`${apiBase}/api/analyze/article`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: article.extracted_text,
        title: article.title,
        original_url: article.original_url,
      }),
    });
    setAnalysis((await responseJson(analysisResponse)) as Analysis);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy.current) return;
    setError("");
    setReportMessage("");
    setComparison(null);
    setAnalysis(null);
    let endpoint: string;
    let options: RequestInit;
    if (mode === "text") {
      if (text.trim().length < 40) {
        setError("Paste at least 40 characters.");
        return;
      }
      endpoint = "/api/analyze/text";
      options = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      };
    } else if (mode === "url") {
      if (!url.trim()) {
        setError("Enter an article URL.");
        return;
      }
      endpoint = "/api/analyze/url";
      options = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      };
    } else {
      if (!file) {
        setError("Choose a PDF file.");
        return;
      }
      const form = new FormData();
      form.append("file", file);
      endpoint = "/api/analyze/pdf";
      options = { method: "POST", body: form };
    }

    busy.current = true;
    setLoading(true);
    try {
      const article = (await responseJson(
        await fetch(`${apiBase}${endpoint}`, options),
      )) as ArticleResult;
      setResult(article);
      await requestAnalysis(article);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Article processing failed.");
    } finally {
      busy.current = false;
      setLoading(false);
    }
  }

  async function retryAnalysis() {
    if (!result || loading) return;
    setLoading(true);
    setError("");
    try {
      await requestAnalysis(result);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Article analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  async function compare() {
    if (!result || busy.current) return;
    busy.current = true;
    setComparing(true);
    setError("");
    try {
      const response = await fetch(`${apiBase}/api/compare/article`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: result.extracted_text,
          title: result.title,
          original_url: result.original_url,
          settings: { result_limit: 3 },
        }),
      });
      setComparison((await responseJson(response)) as Comparison);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Comparison failed.");
    } finally {
      busy.current = false;
      setComparing(false);
    }
  }

  async function downloadReport() {
    if (!analysis || busy.current) return;
    busy.current = true;
    setGeneratingReport(true);
    setReportMessage("");
    setReportFailed(false);
    try {
      const response = await fetch(`${apiBase}/api/reports/pdf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analysis, ...(comparison ? { comparison } : {}) }),
      });
      if (!response.ok) {
        const data = await responseJson(response);
        throw new Error(data.detail || "The PDF report could not be generated.");
      }
      const blob = await response.blob();
      const signature = new Uint8Array(await blob.slice(0, 4).arrayBuffer());
      if (
        blob.type !== "application/pdf" ||
        signature[0] !== 0x25 ||
        signature[1] !== 0x50 ||
        signature[2] !== 0x44 ||
        signature[3] !== 0x46
      ) {
        throw new Error("The server returned an invalid PDF response.");
      }
      const disposition = response.headers.get("Content-Disposition") || "";
      const filename = reportFilename(disposition, analysis.title || "untitled");
      downloadPdf(blob, filename, browserDownloadTarget());
      setReportMessage("Research report downloaded.");
    } catch (cause) {
      setReportFailed(true);
      setReportMessage(
        cause instanceof Error ? cause.message : "The PDF report could not be generated.",
      );
    } finally {
      busy.current = false;
      setGeneratingReport(false);
    }
  }

  return (
    <main>
      <header>
        <div className="brand">
          <Languages size={29} />
          <div>
            <h1>Multilingual News Intelligence</h1>
            <p>Explainable News Framing &amp; Bias Analysis</p>
          </div>
        </div>
        <span className="phase">EXPLAINABLE FRAMING ANALYSIS</span>
      </header>
      <section className="hero">
        <p className="eyebrow">RESEARCH WORKSPACE</p>
        <h2>Analyse article language with traceable evidence.</h2>
        <p>
          Comparison results are limited to accessible reporting and never label
          a source or language as biased.
        </p>
        <ol className="workflow" aria-label="Workspace workflow">
          <li>Submit an article</li>
          <li>Review the evidence</li>
          <li>Compare coverage</li>
          <li>Download the report</li>
        </ol>
      </section>
      {!result ? (
        <section className="workspace">
          <form onSubmit={submit} aria-busy={loading}>
            <div className="tabs" aria-label="Article input method">
              {(
                [
                  ["text", FileText, "Paste text"],
                  ["url", Globe2, "Article URL"],
                  ["pdf", Upload, "Upload PDF"],
                ] as const
              ).map(([key, Icon, label]) => (
                <button
                  type="button"
                  aria-pressed={mode === key}
                  onClick={() => {
                    setMode(key);
                    setError("");
                  }}
                  className={mode === key ? "active" : ""}
                  key={key}
                >
                  <Icon size={17} />
                  {label}
                </button>
              ))}
            </div>
            {mode === "text" && (
              <label htmlFor="article-text">
                Article text
                <textarea
                  id="article-text"
                  name="article-text"
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  rows={13}
                  minLength={40}
                  required
                />
              </label>
            )}
            {mode === "url" && (
              <label htmlFor="article-url">
                Article URL
                <input
                  id="article-url"
                  name="article-url"
                  value={url}
                  onChange={(event) => setUrl(event.target.value)}
                  type="url"
                  placeholder="https://"
                  required
                />
              </label>
            )}
            {mode === "pdf" && (
              <label className="dropzone" htmlFor="article-pdf">
                Text-based PDF
                <input
                  id="article-pdf"
                  name="article-pdf"
                  type="file"
                  accept="application/pdf"
                  onChange={(event: ChangeEvent<HTMLInputElement>) =>
                    setFile(event.target.files?.[0] ?? null)
                  }
                />
                <span>{file?.name ?? "Choose a PDF"}</span>
              </label>
            )}
            {error && (
              <div role="alert" className="error">
                {error}
              </div>
            )}
            <button className="submit" disabled={loading}>
              {loading ? "Processing…" : "Analyse Article"}
            </button>
            <p className="meta" role="status" aria-live="polite">
              {loading
                ? "Extracting the article and analysing its wording. This can take a few seconds."
                : ""}
            </p>
          </form>
          <aside>
            <h3>How to read results</h3>
            <ul>
              <li>Inspect exact original-text spans</li>
              <li>Check availability limitations</li>
              <li>Compare coverage, not bias scores</li>
            </ul>
          </aside>
        </section>
      ) : (
        <Results
          result={result}
          analysis={analysis}
          comparison={comparison}
          comparing={comparing}
          generatingReport={generatingReport}
          reportMessage={reportMessage}
          reportFailed={reportFailed}
          error={error}
          loading={loading}
          compare={compare}
          downloadReport={downloadReport}
          retryAnalysis={retryAnalysis}
          reset={reset}
        />
      )}
    </main>
  );
}

function Results({
  result,
  analysis,
  comparison,
  comparing,
  generatingReport,
  reportMessage,
  reportFailed,
  error,
  loading,
  compare,
  downloadReport,
  retryAnalysis,
  reset,
}: {
  result: ArticleResult;
  analysis: Analysis | null;
  comparison: Comparison | null;
  comparing: boolean;
  generatingReport: boolean;
  reportMessage: string;
  reportFailed: boolean;
  error: string;
  loading: boolean;
  compare: () => void;
  downloadReport: () => void;
  retryAnalysis: () => void;
  reset: () => void;
}) {
  const [focusedKey, setFocusedKey] = useState("");
  const framingEvidence: Evidence[] = (analysis?.bias_assessment.findings ?? []).map((finding) => ({
    category: finding.category,
    quote: finding.evidence_text,
    start: finding.start_offset,
    end: finding.end_offset,
    context: finding.evidence_text,
    explanation: finding.explanation,
    method: finding.method,
  }));
  const ranges = analysis
    ? highlightRanges([...analysis.indicators, ...framingEvidence], analysis.original_text)
    : [];
  const totals = analysis ? evidenceTotals(analysis.statistics, analysis.indicators) : null;
  const attribution = analysis?.statistics.attribution_summary;
  const grouped = (analysis?.indicators ?? []).reduce<Record<string, Evidence[]>>((acc, item) => {
    (acc[item.category] ??= []).push(item);
    return acc;
  }, {});

  const showInArticle = (item: Evidence) => {
    const range =
      ranges.find((candidate) => candidate.start === item.start) ??
      ranges.find((candidate) => candidate.start >= item.start && candidate.start < item.end);
    if (!range) return;
    setFocusedKey(range.key);
    document
      .getElementById(`mark-${range.key}`)
      ?.scrollIntoView({ block: "center", behavior: "smooth" });
  };
  const showFramingFinding = (finding: FramingFinding) => {
    showInArticle({
      category: finding.category,
      quote: finding.evidence_text,
      start: finding.start_offset,
      end: finding.end_offset,
      context: finding.evidence_text,
      explanation: finding.explanation,
      method: finding.method,
    });
  };

  return (
    <section className="results">
      <div className="result-head">
        <div>
          <p className="eyebrow">
            {analysis ? "ANALYSIS COMPLETE" : "ANALYSIS INCOMPLETE"}
          </p>
          <h2>{result.title || "Untitled article"}</h2>
          {result.original_url && (
            <a href={result.original_url} target="_blank" rel="noreferrer">
              Source article
            </a>
          )}
        </div>
        <button onClick={reset} className="secondary">
          <RotateCcw size={16} />
          New article
        </button>
      </div>

      {error && (
        <div role="alert" className="error">
          {error}
        </div>
      )}

      <section aria-labelledby="overview-heading">
        <h3 id="overview-heading">Article overview</h3>
        <div className="metrics">
          <div>
            <strong>{result.language.name}</strong>
            <span>Detected language ({result.language.code})</span>
          </div>
          <div>
            <strong>{analysis?.linguistic.token_count ?? "—"}</strong>
            <span>Whitespace tokens</span>
          </div>
          <div>
            <strong>{analysis?.linguistic.sentence_count ?? "—"}</strong>
            <span>Approximate sentences</span>
          </div>
          <div>
            <strong>{totals ? totals.findings : "—"}</strong>
            <span>Distinct evidence findings</span>
          </div>
        </div>
        {analysis && (
          <p className="meta">
            Token convention: {analysis.linguistic.token_count_convention}
            {" "}Language identification: {result.language.supported ? "supported" : "unsupported"}
            {result.language.confidence != null &&
              ` · confidence ${result.language.confidence}`}
          </p>
        )}
        {result.warnings.map((warning) => (
          <p className="warning" key={warning}>{warning}</p>
        ))}
      </section>

      {analysis ? (
        <>
          <section className="framing-panel" aria-labelledby="framing-heading">
            <p className="eyebrow">ARTICLE-LEVEL INTERPRETATION</p>
            <h3 id="framing-heading">Bias and Framing Analysis</h3>
            <p className="meta">{framingStatusMessage(analysis.bias_assessment)}</p>
            <h4>{analysis.bias_assessment.conclusion}</h4>
            <p className="summary-lede">{analysis.bias_assessment.summary}</p>
            {analysis.language.code !== "en" && (
              <p className="meta">
                Explanations are in English; quoted evidence is preserved in the article&apos;s
                original language.
              </p>
            )}
            {analysis.bias_assessment.strongest_patterns.length > 0 && (
              <div className="framing-patterns">
                <strong>Strongest observable patterns</strong>
                <ul>
                  {analysis.bias_assessment.strongest_patterns.map((pattern) => (
                    <li key={pattern}>{pattern}</li>
                  ))}
                </ul>
              </div>
            )}
            {analysis.bias_assessment.actor_portrayals.length > 0 && (
              <div className="actor-portrayals">
                <h4>Actors and portrayal</h4>
                {analysis.bias_assessment.actor_portrayals.map((portrayal) => (
                  <p key={portrayal.actor}>
                    <strong>{portrayal.actor}</strong>: {portrayal.direction}.{" "}
                    {portrayal.explanation}
                  </p>
                ))}
              </div>
            )}
            {analysis.bias_assessment.findings.length > 0 ? (
              analysis.bias_assessment.findings.map((finding) => (
                <article className="framing-card" key={finding.id}>
                  <div className="framing-card-head">
                    <strong>{finding.label}</strong>
                    <span>{finding.framing_direction}</span>
                  </div>
                  <blockquote>{finding.evidence_text}</blockquote>
                  <p>{finding.explanation}</p>
                  <p><strong>Actor:</strong> {finding.actor || "Not confidently identified"}</p>
                  <p><strong>Speaker:</strong> {finding.speaker || "Not confidently identified"}</p>
                  <p><strong>Passage type:</strong> {finding.attribution_type.replaceAll("_", " ")}</p>
                  <p><strong>Context:</strong> {finding.contextual_interpretation}</p>
                  <p><strong>Why it may matter:</strong> {finding.potential_reader_effect}</p>
                  <p><strong>Alternative interpretation:</strong> {finding.alternative_interpretation}</p>
                  <p className="meta">
                    Evidence strength: {finding.evidence_strength} ·
                    Confidence: {finding.confidence == null ? "not provided" : `${Math.round(finding.confidence * 100)}%`} ·
                    Uncertainty: {finding.uncertainty}
                  </p>
                  <button
                    type="button"
                    className="link-button"
                    onClick={() => showFramingFinding(finding)}
                    disabled={
                      analysis.original_text.slice(finding.start_offset, finding.end_offset) !==
                      finding.evidence_text
                    }
                  >
                    Show passage in article
                  </button>
                </article>
              ))
            ) : (
              <p>{framingNoFindingsMessage(analysis.bias_assessment.status)}</p>
            )}
            {analysis.bias_assessment.neutral_rewrite && (
              <div className="neutral-rewrite">
                <h4>Possible neutral rewrite</h4>
                <blockquote>{analysis.bias_assessment.neutral_rewrite}</blockquote>
              </div>
            )}
            <p><strong>Representation comparison:</strong> {analysis.bias_assessment.representation_comparison}</p>
            <p><strong>Selective presentation:</strong> {analysis.bias_assessment.selective_presentation}</p>
            <details className="availability-details">
              <summary>Methodology and limitations</summary>
              {analysis.bias_assessment.limitations.map((limitation) => (
                <p className="meta" key={limitation}>{limitation}</p>
              ))}
              <p className="meta">
                Contextual findings are model-generated observations grounded in exact original
                passages. They are not independent fact-checking or proof of author intent.
              </p>
              <p className="meta">
                Indicator totals and sentiment are not bias scores; no finding is not evidence
                that an article is neutral. Selective presentation cannot be assessed from one
                article without an external comparison set.
              </p>
            </details>
          </section>

          <section className="summary-panel" aria-labelledby="summary-heading">
            <h3 id="summary-heading">Key findings</h3>
            <p className="summary-lede">
              {totals?.findings ?? 0} distinct observable findings ·{" "}
              {totals?.occurrences ?? 0} cue occurrences ·{" "}
              {analysis.linguistic.sentence_count} sentences ·{" "}
              {analysis.linguistic.token_count} whitespace tokens
            </p>
            <div className="summary-grid">
              <div>
                <strong>{totals?.findings ?? 0}</strong>
                <span>Distinct findings</span>
              </div>
              <div>
                <strong>{totals?.occurrences ?? 0}</strong>
                <span>Cue occurrences</span>
              </div>
              <div>
                <strong>{analysis.statistics.evidence_density_per_1000_tokens ?? "—"}</strong>
                <span>Per 1,000 tokens</span>
              </div>
              <div>
                <strong>{attribution?.unattributed_findings ?? "—"}</strong>
                <span>Findings without a marker</span>
              </div>
            </div>
          </section>

          <section className="analysis-grid" aria-label="Analysis sections">
            <div>
              <h3>Sentiment</h3>
              <p>
                Status: {analysis.sentiment.status}
                {analysis.sentiment.label && ` · ${analysis.sentiment.label}`}
              </p>
              {analysis.sentiment.model_identifier && (
                <p className="meta">Model: {analysis.sentiment.model_identifier}</p>
              )}
              {analysis.sentiment.class_scores &&
                Object.entries(analysis.sentiment.class_scores).map(([name, score]) => (
                  <p key={name}>{name}: {score.toFixed(3)} raw score</p>
                ))}
              {analysis.sentiment.score_note && <p className="meta">{analysis.sentiment.score_note}</p>}
              {analysis.sentiment.sentences?.map((sentence, index) => (
                <blockquote key={`${sentence.start}-${index}`}>
                  <strong>{sentence.label}</strong>: {sentence.text}
                </blockquote>
              ))}
              {analysis.sentiment.limitation && <p className="meta">{analysis.sentiment.limitation}</p>}
            </div>
            <div>
              <h3>Emotion</h3>
              <p>Status: {analysis.emotion.status}{analysis.emotion.label && ` · ${analysis.emotion.label}`}</p>
              {analysis.emotion.model_identifier && (
                <p className="meta">Model: {analysis.emotion.model_identifier}</p>
              )}
              {analysis.emotion.distribution &&
                Object.entries(analysis.emotion.distribution).map(([name, count]) => (
                  <p key={name}>{name}: {count.toFixed(3)} raw score</p>
                ))}
              {analysis.emotion.score_note && <p className="meta">{analysis.emotion.score_note}</p>}
              {analysis.emotion.sentences?.map((sentence, index) => (
                <blockquote key={`${sentence.start}-${index}`}>
                  <strong>{sentence.label}</strong>: {sentence.text}
                </blockquote>
              ))}
              {analysis.emotion.limitation && <p className="meta">{analysis.emotion.limitation}</p>}
            </div>
            <div>
              <h3>Named entities</h3>
              <p>
                {analysis.task_availability.find(
                  (task) => task.task === "named_entity_recognition",
                )?.status ?? "model_unavailable"}
                {analysis.task_availability.find(
                  (task) => task.task === "named_entity_recognition",
                )?.method && ` · ${analysis.task_availability.find(
                  (task) => task.task === "named_entity_recognition",
                )?.method}`}
              </p>
              {analysis.entities.length ? (
                <ul>
                  {analysis.entities.map((entity, index) => (
                    <li key={`${entity.text}-${index}`}>
                      {entity.text} ({entity.label}) · {entity.method}
                      {entity.confidence != null &&
                        ` · ${Math.round(entity.confidence * 100)}% model confidence`}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>
                  No entity candidates were detected. This does not establish that the
                  article contains no named entities.
                </p>
              )}
              {analysis.task_availability.find(
                (task) => task.task === "named_entity_recognition",
              )?.limitation && (
                <p className="meta">
                  {analysis.task_availability.find(
                    (task) => task.task === "named_entity_recognition",
                  )?.limitation}
                </p>
              )}
            </div>
          </section>

          <section className="report">
            <h3>Category summary</h3>
            {Object.keys(analysis.statistics.indicator_counts).length > 0 ? (
              <div className="count-list">
                {Object.entries(analysis.statistics.indicator_counts).map(([category, count]) => (
                  <span key={category}>
                    {categoryLabel(category)}: {count} finding{count === 1 ? "" : "s"}
                    {analysis.statistics.indicator_occurrences?.[category]
                      ? ` · ${analysis.statistics.indicator_occurrences[category]} occurrence${
                          analysis.statistics.indicator_occurrences[category] === 1 ? "" : "s"
                        }`
                      : ""}
                  </span>
                ))}
              </div>
            ) : (
              <p>No configured indicator cues matched. This is not a finding of absence.</p>
            )}
            <p className="meta">
              {analysis.statistics.evidence_density_convention ??
                "Total cue occurrences divided by whitespace token count, multiplied by 1,000."}
            </p>
          </section>

          <section className="evidence" aria-labelledby="evidence-heading">
            <h3 id="evidence-heading">Evidence and explanations</h3>
            <p>
              Repeated wording is reported once per category and counted. Select a finding to
              highlight its first occurrence in the article text.
            </p>
            {analysis.indicators.length === 0 && (
              <p>
                No configured indicator cues matched the analysed text. This is not evidence
                that the article is unbiased.
              </p>
            )}
            {Object.entries(grouped).map(([category, items]) => {
              const occurrences = items.reduce((sum, item) => sum + (item.occurrences ?? 1), 0);
              return (
                <section className="evidence-group" key={category}>
                  <div className="evidence-group-head">
                    <h4>{categoryLabel(category)}</h4>
                    <span>
                      {items.length} finding{items.length === 1 ? "" : "s"} · {occurrences}{" "}
                      occurrence{occurrences === 1 ? "" : "s"}
                    </span>
                  </div>
                  {analysis.methodology?.categories?.[category] && (
                    <p className="meta">{analysis.methodology.categories[category]}</p>
                  )}
                  {items.map((item) => (
                    <article
                      className="evidence-card"
                      key={`${item.category}-${item.start}-${item.end}`}
                    >
                      <div className="evidence-card-head">
                        <blockquote>{item.quote}</blockquote>
                        {item.claim_status && (
                          <span className={`badge badge-${item.claim_status}`}>
                            {claimLabel(item.claim_status)}
                          </span>
                        )}
                      </div>
                      {item.context && (
                        <p className="evidence-context">
                          <span className="meta">Sentence:</span> {item.context}
                        </p>
                      )}
                      <p>{item.explanation}</p>
                      <details className="evidence-details">
                        <summary>Technical detail</summary>
                        <p>
                          {item.method}
                          {item.evidence_strength ? ` · ${item.evidence_strength}` : ""}
                          {` · ${item.occurrences ?? 1} occurrence${
                            (item.occurrences ?? 1) === 1 ? "" : "s"
                          }`}
                          {` · ${analysis.original_text.slice(item.start, item.end) === item.quote
                            ? "exact-span validated"
                            : "invalid span"}`}
                        </p>
                        {item.claim_status_note && <p>{item.claim_status_note}</p>}
                        {item.uncertainty && <p>{item.uncertainty}</p>}
                      </details>
                      <button
                        type="button"
                        className="link-button"
                        onClick={() => showInArticle(item)}
                        disabled={!ranges.some(
                          (range) => range.start >= item.start && range.start < item.end,
                        )}
                      >
                        Show in article text
                      </button>
                    </article>
                  ))}
                </section>
              );
            })}
          </section>

          <section className="availability">
            <h3>Feature availability and limitations</h3>
            <details className="availability-details">
              <summary>Task availability</summary>
              <p>
                XLM-RoBERTa is retained for optional local sentiment and named-entity
                classifiers when configured; it does not perform contextual framing.
                Contextual bias and framing analysis uses the configured Groq model
                (default: openai/gpt-oss-120b) when available.
              </p>
              {analysis.task_availability.map((task) => (
                <p key={task.task}>
                  <strong>{task.task.replaceAll("_", " ")}:</strong> {task.status}
                  {task.method ? ` · ${task.method}` : ""}
                  {task.model_identifier ? ` · ${task.model_identifier}` : ""}
                  {task.supported_languages?.length
                    ? ` · languages: ${task.supported_languages.join(", ")}`
                    : ""}
                  {task.limitation ? ` · ${task.limitation}` : ""}
                </p>
              ))}
              {analysis.limitations.map((limitation) => (
                <p className="warning" key={limitation}>{limitation}</p>
              ))}
            </details>
          </section>
        </>
      ) : (
        <section className="report" aria-live="polite">
          <h3>Analysis details unavailable</h3>
          <p>
            Article text was extracted, but the analysis request did not complete.
            Your source input is still available; retry without starting over.
          </p>
          <button className="submit" onClick={retryAnalysis} disabled={loading}>
            {loading ? "Retrying analysis…" : "Retry Analysis"}
          </button>
        </section>
      )}

      {analysis && (
        <section className="report" aria-labelledby="overall-judgment-heading">
          <h3 id="overall-judgment-heading">Overall Bias &amp; Framing Judgment</h3>
          {analysis.overall_judgment ? (
            <>
              <p>{analysis.overall_judgment.summary}</p>
              <p>
                <strong>Estimated analytical indicator:</strong>{" "}
                {analysis.overall_judgment.indicator == null
                  ? "Insufficient evidence for a score"
                  : `${analysis.overall_judgment.indicator}/100`}
                {analysis.overall_judgment.indicator != null &&
                  ` — ${analysis.overall_judgment.indicator_label}`}
              </p>
              <p>
                <strong>Direction:</strong>{" "}
                {analysis.overall_judgment.direction.replaceAll("_", " ")}
                {" · "}
                <strong>Evidence sufficiency:</strong>{" "}
                {analysis.overall_judgment.evidence_sufficiency}
              </p>
              <p className="meta">
                This is a provisional indicator of directional framing evidence, not an
                objective truth, quality, or scientifically validated bias score.
              </p>
              <p className="meta">
                Groq contextual analysis:{" "}
                {analysis.bias_assessment.cache_hit
                  ? "cached response reused"
                  : analysis.bias_assessment.groq_used
                    ? "provider called for this analysis"
                    : analysis.bias_assessment.status === "available"
                      ? "provider result available"
                      : "no successful provider result"}
              </p>
              {analysis.overall_judgment.targets.length > 0 && (
                <div>
                  <strong>Actors or positions in validated findings</strong>
                  <ul>
                    {analysis.overall_judgment.targets.map((target) => (
                      <li key={target.actor}>
                        {target.actor}: {target.direction}. {target.explanation}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {analysis.overall_judgment.reasons.length > 0 ? (
                <div>
                  <strong>Main reasons and original-language evidence</strong>
                  {analysis.overall_judgment.reasons.map((reason, index) => {
                    const finding = analysis.bias_assessment.findings.find(
                      (item) =>
                        item.start_offset === reason.start_offset &&
                        item.end_offset === reason.end_offset &&
                        item.evidence_text === reason.evidence_text,
                    );
                    const validSpan =
                      analysis.original_text.slice(reason.start_offset, reason.end_offset) ===
                      reason.evidence_text;
                    return (
                      <article
                        className="framing-card"
                        key={`${reason.start_offset}-${reason.end_offset}-${index}`}
                      >
                        <strong>{reason.label}</strong>
                        <blockquote>{reason.evidence_text}</blockquote>
                        <p>{reason.explanation}</p>
                        <p>
                          Category: {reason.category.replaceAll("_", " ")} · Direction:{" "}
                          {reason.framing_direction} · Attribution:{" "}
                          {reason.attribution_type.replaceAll("_", " ")}
                        </p>
                        {(reason.actor || reason.speaker) && (
                          <p>
                            {reason.actor && `Target: ${reason.actor}`}
                            {reason.actor && reason.speaker && " · "}
                            {reason.speaker && `Speaker: ${reason.speaker}`}
                          </p>
                        )}
                        {validSpan && finding && (
                          <button
                            type="button"
                            className="link-button"
                            onClick={() => showFramingFinding(finding)}
                          >
                            Show passage in article
                          </button>
                        )}
                        {!validSpan && <p className="warning">Evidence span failed validation.</p>}
                      </article>
                    );
                  })}
                </div>
              ) : (
                <p>No validated contextual framing excerpts are available.</p>
              )}
              <div>
                <strong>Sentiment (supporting signal only)</strong>
                <p>
                  Status: {analysis.overall_judgment.sentiment_signal.status}
                  {analysis.overall_judgment.sentiment_signal.label
                    ? ` · ${analysis.overall_judgment.sentiment_signal.label}`
                    : ""}
                  {analysis.overall_judgment.sentiment_signal.model_identifier
                    ? ` · ${analysis.overall_judgment.sentiment_signal.model_identifier}`
                    : ""}
                </p>
                {analysis.overall_judgment.sentiment_signal.limitation && (
                  <p className="meta">{analysis.overall_judgment.sentiment_signal.limitation}</p>
                )}
              </div>
              <div>
                <strong>Emotion (supporting signal only)</strong>
                <p>
                  Status: {analysis.overall_judgment.emotion_signal.status}
                  {analysis.overall_judgment.emotion_signal.label
                    ? ` · ${analysis.overall_judgment.emotion_signal.label}`
                    : ""}
                  {analysis.overall_judgment.emotion_signal.model_identifier
                    ? ` · ${analysis.overall_judgment.emotion_signal.model_identifier}`
                    : ""}
                </p>
                {analysis.overall_judgment.emotion_signal.limitation && (
                  <p className="meta">{analysis.overall_judgment.emotion_signal.limitation}</p>
                )}
              </div>
              {analysis.overall_judgment.entity_mentions.length > 0 && (
                <div>
                  <strong>Detected entities and supported roles</strong>
                  <ul>
                    {analysis.overall_judgment.entity_mentions.map((entity) => (
                      <li key={`${entity.start}-${entity.end}-${entity.label}`}>
                        {entity.text} ({entity.label})
                        {entity.role ? ` · ${entity.role} in a contextual finding` : ""}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <details className="availability-details">
                <summary>Attribution, scoring method, and limitations</summary>
                <p>
                  Attributed speech is distinguished from narration and is not treated as
                  the article&apos;s endorsement. Sentiment and emotion do not contribute to
                  the indicator.
                </p>
                <p>{analysis.overall_judgment.scoring_method}</p>
                {analysis.overall_judgment.limitations.map((limitation) => (
                  <p className="meta" key={limitation}>{limitation}</p>
                ))}
              </details>
            </>
          ) : (
            <p>Overall judgment is unavailable in this API response.</p>
          )}
        </section>
      )}

      <section className="report">
        <h3>Cross-language coverage</h3>
        <p>
          Discovery is bounded and incomplete. Full-text, snippet-only, and
          metadata-only candidates are distinguished; a possible match is not a
          verified match.
        </p>
        <button className="submit" onClick={compare} disabled={comparing}>
          {comparing ? "Searching coverage…" : "Compare Coverage Across Languages"}
        </button>
        {comparison && <ComparisonPanel comparison={comparison} />}
      </section>

      {analysis && (
        <section className="report" aria-labelledby="download-report-heading">
          <h3 id="download-report-heading">Research report</h3>
          <p>
            The PDF uses this article analysis and includes comparison results
            only when available.
          </p>
          <button
            className="submit"
            onClick={downloadReport}
            disabled={generatingReport}
          >
            {generatingReport ? "Generating PDF…" : "Download Research Report (PDF)"}
          </button>
          {reportFailed ? (
            <p role="alert">{reportMessage}</p>
          ) : (
            <p role="status" aria-live="polite">{reportMessage}</p>
          )}
        </section>
      )}

      <article>
        <h3>Original article text</h3>
        <p>
          <HighlightedText
            text={result.extracted_text}
            evidence={[...(analysis?.indicators ?? []), ...framingEvidence]}
            focusedKey={focusedKey}
          />
        </p>
      </article>
    </section>
  );
}

function ComparisonPanel({ comparison }: { comparison: Comparison }) {
  const coverage = comparison.comparison.coverage;
  const outcomes = coverage.verification_outcomes;
  const providerEntries = Object.entries(comparison.provider_status ?? {});
  const queries = comparison.discovery_queries;
  const profile = comparison.event_profile;
  const framingDiffs = comparison.framing_differences ?? [];
  const matched = comparison.candidates.filter(
    (c) => c.verification_status === "verified_match" || c.verification_status === "possible_match",
  );
  const other = comparison.candidates.filter(
    (c) => c.verification_status !== "verified_match" && c.verification_status !== "possible_match",
  );

  const LANG_NAMES: Record<string, string> = {
    en: "English", hi: "Hindi", kn: "Kannada", te: "Telugu", ta: "Tamil",
  };

  return (
    <div className="comparison-panel">
      <h3>Comparison results</h3>

      {/* A. Input event profile */}
      {profile && (
        <section className="event-profile">
          <h4>What the system searched for</h4>
          {profile.headline && <p><strong>Headline:</strong> {profile.headline}</p>}
          <p><strong>Summary:</strong> {profile.summary}</p>
          <div className="profile-grid">
            {profile.actors.length > 0 && (
              <div><strong>Actors</strong><span>{profile.actors.join(", ")}</span></div>
            )}
            {profile.locations.length > 0 && (
              <div><strong>Locations</strong><span>{profile.locations.join(", ")}</span></div>
            )}
            {profile.dates.length > 0 && (
              <div><strong>Dates</strong><span>{profile.dates.join(", ")}</span></div>
            )}
            {profile.main_action && (
              <div><strong>Main action</strong><span>{profile.main_action}</span></div>
            )}
          </div>
          {!profile.is_specific_event && (
            <p className="warning">
              The article may describe a broad topic rather than a specific incident.
              Search results may be less precise.
            </p>
          )}
          {profile.uncertainty_notes.map((note) => (
            <p className="meta" key={note}>{note}</p>
          ))}
        </section>
      )}

      {/* B. Language coverage */}
      <section className="lang-coverage">
        <h4>Language coverage</h4>
        <div className="lang-grid">
          {Object.entries(queries).map(([lang, query]) => {
            const langCandidates = comparison.candidates.filter(
              (c) => c.language?.code === lang,
            );
            const matched_count = langCandidates.filter(
              (c) => c.verification_status === "verified_match" || c.verification_status === "possible_match",
            ).length;
            const full_text = langCandidates.filter((c) => c.analysis_scope === "full_text").length;
            return (
              <div key={lang} className="lang-card">
                <strong>{LANG_NAMES[lang] ?? lang}</strong>
                <span className="meta">Query: {query}</span>
                <span>{langCandidates.length} candidate{langCandidates.length !== 1 ? "s" : ""}</span>
                <span>{matched_count} relevant · {full_text} full-text</span>
              </div>
            );
          })}
        </div>
      </section>

      {/* Summary counts */}
      <p>
        {coverage.discovered_articles} discovered;
        {" "}{coverage.evaluated_articles ?? coverage.discovered_articles} evaluated;
        {" "}{coverage.matched_articles} possible or verified;
        {" "}languages: {coverage.languages_represented.join(", ") || "none"}.
      </p>
      <p>
        Outcomes: verified {outcomes.verified_match ?? 0}, possible {outcomes.possible_match ?? 0},
        {" "}rejected {outcomes.rejected_match ?? 0}, insufficient evidence {outcomes.insufficient_evidence ?? 0}.
      </p>
      <p>
        Retrieval: {coverage.full_text_articles} full-text; {coverage.snippet_only_articles} snippet-only.
      </p>

      {/* D. Framing differences */}
      {framingDiffs.length > 0 && (
        <section className="framing-section">
          <h4>Cross-language framing differences</h4>
          <p className="meta">
            Differences are based on exact evidence from both articles.
            They are observations, not bias verdicts.
          </p>
          {framingDiffs.map((diff, i) => (
            <div key={i} className="framing-card">
              <strong>{diff.dimension}</strong>
              {diff.article_a_quote && (
                <div className="framing-row">
                  <span className="framing-label">Article A</span>
                  <blockquote>{diff.article_a_quote}</blockquote>
                </div>
              )}
              {diff.article_b_quote && (
                <div className="framing-row">
                  <span className="framing-label">Article B</span>
                  <blockquote>{diff.article_b_quote}</blockquote>
                </div>
              )}
              <p>{diff.explanation}</p>
              {diff.alternative_interpretation && (
                <p className="meta">Alternative: {diff.alternative_interpretation}</p>
              )}
              {diff.confidence && (
                <p className="meta">Confidence: {diff.confidence}</p>
              )}
            </div>
          ))}
        </section>
      )}

      {/* C. Relevant articles */}
      {matched.length > 0 && (
        <section className="comparison-matches" aria-label="Matched candidates">
          <h4>Possible or verified matches</h4>
          {matched.map((candidate) => (
            <CandidateCard candidate={candidate} key={candidate.url} />
          ))}
        </section>
      )}
      {other.length > 0 && (
        <details className="comparison-others">
          <summary>Other evaluated candidates ({other.length})</summary>
          {other.map((candidate) => (
            <CandidateCard candidate={candidate} key={candidate.url} />
          ))}
        </details>
      )}

      {/* E. Provider status */}
      {providerEntries.length > 0 && (
        <details className="provider-details">
          <summary className="meta">Provider status</summary>
          <div className="provider-status">
            {providerEntries.map(([provider, status]) => (
              <p key={provider}>
                <strong>{provider.replaceAll("_", " ")}:</strong> {status}
              </p>
            ))}
          </div>
        </details>
      )}
      {comparison.provider_failures.map((failure) => (
        <details key={failure} className="provider-details">
          <summary className="meta warning-inline">Provider limitation (expand)</summary>
          <div className="warning">{failure}</div>
        </details>
      ))}

      {/* E. Honest failure */}
      {comparison.candidates.length === 0 && (
        <div className="empty-state">
          <p><strong>No candidates were returned.</strong></p>
          <p>This can mean:</p>
          <ul>
            <li>The providers returned no results for these queries.</li>
            <li>All results were deduplicated or had invalid URLs.</li>
            <li>A provider timed out or was rate-limited.</li>
          </ul>
          <p className="meta">This is not evidence that no other coverage exists.</p>
        </div>
      )}

      {comparison.limitations.map((lim) => (
        <p className="meta" key={lim} style={{marginTop: 6}}>{lim}</p>
      ))}
    </div>
  );
}

function CandidateCard({ candidate }: { candidate: Candidate }) {
  const statusColour: Record<string, string> = {
    verified_match: "#1a7a4a",
    possible_match: "#725300",
    rejected_match: "#932d2d",
    insufficient_evidence: "#617084",
  };
  const colour = statusColour[candidate.verification_status] ?? "#617084";
  return (
    <div className="evidence-card">
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",gap:8}}>
        <strong>{candidate.title}</strong>
        <span style={{color: colour, fontSize: 12, fontWeight: 700, whiteSpace: "nowrap"}}>
          {candidate.verification_status.replaceAll("_", " ")}
        </span>
      </div>
      <p style={{margin:"4px 0",fontSize:14}}>
        {candidate.language?.name || "Unknown language"}
        {candidate.source_name ? ` · ${candidate.source_name}` : ""}
        {candidate.published_at ? ` · ${candidate.published_at}` : ""}
      </p>
      <p style={{margin:"4px 0",fontSize:13,color:"#617084"}}>
        Scope: {candidate.analysis_scope} · Retrieval: {candidate.retrieval_status}
      </p>
      {candidate.match_reasons.length > 0 && (
        <p style={{margin:"4px 0",fontSize:13}}>
          {candidate.match_reasons.join(" ")}
        </p>
      )}
      {candidate.uncertainty.length > 0 && (
        <details style={{fontSize:13,marginTop:4}}>
          <summary style={{cursor:"pointer",color:"#1769aa"}}>Uncertainty notes</summary>
          <ul style={{margin:"4px 0",paddingLeft:16}}>
            {candidate.uncertainty.map((u) => <li key={u}>{u}</li>)}
          </ul>
        </details>
      )}
      <p style={{margin:"6px 0 0"}}>
        <a href={candidate.url} target="_blank" rel="noreferrer" style={{fontSize:13}}>
          Open article ↗
        </a>
      </p>
    </div>
  );
}
