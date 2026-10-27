"use client";

import { ChangeEvent, FormEvent, useRef, useState } from "react";
import type { ReactNode } from "react";
import { FileText, Globe2, Languages, RotateCcw, Upload } from "lucide-react";
import { browserDownloadTarget, downloadPdf, reportFilename } from "./reportDownload.mjs";
import { evidenceTotals, highlightRanges } from "./evidence.mjs";

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
  title_source?: string | null;
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
};
type Analysis = {
  analysis_id: string;
  title?: string | null;
  title_source?: string | null;
  original_url?: string | null;
  original_text: string;
  language: Language;
  linguistic: {
    sentence_count: number;
    token_count: number;
    token_count_convention: string;
  };
  entities: { text: string; label: string; method: string }[];
  sentiment: {
    status: string;
    label?: string;
    positive_cues?: number;
    negative_cues?: number;
    method?: string | null;
    sentences?: { label: string }[];
  };
  emotion: {
    status: string;
    method?: string | null;
    distribution?: Record<string, number>;
    dominant_emotion?: string | null;
    evidence?: Evidence[];
    limitation?: string | null;
  };
  framing_analysis?: {
    overall_assessment: string;
    limitation: string;
    findings: (Evidence & { label: string; evidence_text: string; start_offset: number; end_offset: number; potential_effect: string; interpretation: string; alternative_interpretation: string })[];
  };
  summary: { status: string; method?: string | null; text?: string | null; limitation?: string | null };
  indicators: Evidence[];
  statistics: {
    indicator_counts: Record<string, number>;
    indicator_occurrences?: Record<string, number>;
    total_evidence_findings?: number;
    total_cue_occurrences?: number;
    evidence_density_per_1000_tokens?: number | null;
    evidence_density_convention?: string;
    source_representation?: {
      attribution_marker_occurrences: number;
      sentences_with_attribution_markers: number;
      sentences_without_attribution_markers: number;
      interpretation: string;
    };
  };
  task_availability: {
    task: string;
    status: string;
    method?: string | null;
    limitation?: string | null;
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
type Comparison = {
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
    event_profile?: { headline?: string | null; language: string; distinctive_terms: string[]; dates: string[]; people_candidates: string[]; sufficient_for_targeted_search: boolean; limitation: string };
    evidence_comparisons?: {
      candidate_url: string;
      candidate_title: string;
      category: string;
      original_evidence: { quote: string; context: string; occurrences: number };
      candidate_evidence: { quote: string; context: string; occurrences: number };
      interpretation: string;
    }[];
    observation: string;
  };
  provider_failures: string[];
  provider_status?: Record<string, string>;
  limitations: string[];
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

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
    busy.current = true;
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
        <span className="phase">PHASE 7 · RESEARCH REPORTS</span>
      </header>
      <section className="hero">
        <p className="eyebrow">RESEARCH WORKSPACE</p>
        <h2>Analyse article language with traceable evidence.</h2>
        <p>
          Comparison results are limited to accessible reporting and never label
          a source or language as biased.
        </p>
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
  const ranges = analysis ? highlightRanges(analysis.indicators, analysis.original_text) : [];
  const totals = analysis ? evidenceTotals(analysis.statistics, analysis.indicators) : null;

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

  return (
    <section className="results">
      <div className="result-head">
        <div>
          <p className="eyebrow">
            {analysis ? "ANALYSIS COMPLETE" : "ANALYSIS INCOMPLETE"}
          </p>
          <h2>{analysis?.title || result.title || "Untitled article"}</h2>
          {(analysis?.title_source || result.title_source) && (
            <p className="meta">Title: {analysis?.title_source || result.title_source}</p>
          )}
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
          <section className="analysis-grid" aria-label="Analysis sections">
            <div>
              <h3>Article summary</h3>
              <p>{analysis.summary.text || "No extractive summary could be formed from the available text."}</p>
              <p className="meta">{analysis.summary.method || analysis.summary.limitation}</p>
            </div>
            <div>
              <h3>Sentiment</h3>
              <p>
                Status: {analysis.sentiment.status}
                {analysis.sentiment.label && ` · ${analysis.sentiment.label}`}
              </p>
              {analysis.sentiment.status === "available" && (
                <p>
                  Lexical cues: {analysis.sentiment.positive_cues ?? 0} positive,{" "}
                  {analysis.sentiment.negative_cues ?? 0} negative. This is a
                  heuristic, not a bias measure.
                </p>
              )}
            </div>
            <div>
              <h3>Emotion</h3>
              <p>Status: {analysis.emotion.status}{analysis.emotion.dominant_emotion ? ` · ${analysis.emotion.dominant_emotion}` : ""}</p>
              {analysis.emotion.distribution &&
                Object.entries(analysis.emotion.distribution).map(([name, count]) => (
                  <p key={name}>{name}: {count}</p>
                ))}
              {analysis.emotion.evidence?.map((item) => (
                <p className="meta" key={`${item.category}-${item.start}`}>“{item.quote}” · {item.explanation}</p>
              ))}
              <p className="meta">{analysis.emotion.limitation}</p>
            </div>
            <div>
              <h3>Named entities</h3>
              <p>
                {analysis.task_availability.find(
                  (task) => task.task === "named_entity_recognition",
                )?.status ?? "unavailable"}; rule-based candidates only.
              </p>
              {analysis.entities.length ? (
                <ul>
                  {analysis.entities.map((entity, index) => (
                    <li key={`${entity.text}-${index}`}>{entity.text} ({entity.label})</li>
                  ))}
                </ul>
              ) : (
                <p>No entity candidates returned; this does not establish absence.</p>
              )}
            </div>
          </section>

          <section className="report">
            <h3>Framing and statistical summary</h3>
            <p>
              <strong>{totals?.findings ?? 0}</strong> distinct findings covering{" "}
              <strong>{totals?.occurrences ?? 0}</strong> cue occurrences in{" "}
              {analysis.linguistic.sentence_count} sentences and{" "}
              {analysis.linguistic.token_count} whitespace tokens. Normalized
              evidence density:{" "}
              {analysis.statistics.evidence_density_per_1000_tokens ?? "Unavailable"}{" "}
              occurrences per 1,000 tokens. This is not an overall bias score.
            </p>
            <p className="meta">
              {analysis.statistics.evidence_density_convention ??
                "Total cue occurrences divided by whitespace token count, multiplied by 1,000."}
            </p>
            <div className="count-list">
              {Object.entries(analysis.statistics.indicator_counts).map(([category, count]) => (
                <span key={category}>
                  {category.replaceAll("_", " ")}: {count} finding{count === 1 ? "" : "s"}
                  {analysis.statistics.indicator_occurrences?.[category]
                    ? ` · ${analysis.statistics.indicator_occurrences[category]} occurrence${
                        analysis.statistics.indicator_occurrences[category] === 1 ? "" : "s"
                      }`
                    : ""}
                </span>
              ))}
            </div>
            {Object.keys(analysis.statistics.indicator_counts).length === 0 && (
              <p>No indicator cues matched. This is not a finding of absence.</p>
            )}
            {analysis.statistics.source_representation && (
              <p className="meta">
                Attribution markers: {analysis.statistics.source_representation.attribution_marker_occurrences} occurrence(s) in{" "}
                {analysis.statistics.source_representation.sentences_with_attribution_markers} of{" "}
                {analysis.linguistic.sentence_count} analysed sentence(s).{" "}
                {analysis.statistics.source_representation.interpretation}
              </p>
            )}
          </section>

          <section className="evidence" aria-labelledby="bias-framing-heading">
            <h3 id="bias-framing-heading">Bias and Framing Analysis</h3>
            <p>{analysis.framing_analysis?.overall_assessment ?? "No framing assessment was supplied."}</p>
            {analysis.framing_analysis?.findings.map((finding) => (
              <div className="evidence-card" key={`${finding.category}-${finding.start_offset}-${finding.end_offset}`}>
                <strong>{finding.label} · {finding.interpretation}</strong>
                <blockquote>{finding.evidence_text}</blockquote>
                <p>{finding.explanation}</p>
                <p><span className="meta">Potential effect:</span> {finding.potential_effect}</p>
                <p className="meta">Alternative interpretation: {finding.alternative_interpretation}</p>
                <button type="button" className="link-button" onClick={() => showInArticle({ ...finding, quote: finding.evidence_text, start: finding.start_offset, end: finding.end_offset })}>Show in article text</button>
              </div>
            ))}
            <p className="meta">{analysis.framing_analysis?.limitation}</p>
          </section>

          <section className="evidence" aria-labelledby="evidence-heading">
            <h3 id="evidence-heading">Evidence and explanations</h3>
            <p>
              Repeated wording is reported once per category and counted; select
              a finding to highlight its first occurrence in the article text.
            </p>
            {analysis.indicators.length === 0 && (
              <p>No indicator evidence was returned.</p>
            )}
            {analysis.indicators.map((item) => (
              <div className="evidence-card" key={`${item.category}-${item.start}-${item.end}`}>
                <strong>
                  {item.category.replaceAll("_", " ")}
                  {item.cue ? ` · “${item.cue}”` : ""}
                </strong>
                <blockquote>{item.quote}</blockquote>
                {item.context && (
                  <p className="context">
                    <span className="meta">Sentence:</span> {item.context}
                  </p>
                )}
                <p>{item.explanation}</p>
                <small>
                  {item.method}
                  {item.evidence_strength ? ` · ${item.evidence_strength}` : ""}
                  {` · ${item.occurrences ?? 1} occurrence${
                    (item.occurrences ?? 1) === 1 ? "" : "s"
                  }`}
                  {` · ${analysis.original_text.slice(item.start, item.end) === item.quote
                    ? "exact-span validated"
                    : "invalid span"}`}
                  {item.uncertainty ? ` · ${item.uncertainty}` : ""}
                </small>
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
              </div>
            ))}
          </section>

          <section className="availability">
            <h3>Feature availability and limitations</h3>
            {analysis.task_availability.map((task) => (
              <p key={task.task}>
                <strong>{task.task.replaceAll("_", " ")}:</strong> {task.status}
                {task.method ? ` · ${task.method}` : ""}
                {task.limitation ? ` · ${task.limitation}` : ""}
              </p>
            ))}
            {analysis.limitations.map((limitation) => (
              <p className="warning" key={limitation}>{limitation}</p>
            ))}
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
            evidence={analysis?.indicators ?? []}
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
  const profile = comparison.comparison.event_profile;
  return (
    <div className="comparison-panel">
      <h3>Comparison results</h3>
      {profile && (
        <div className="evidence-card">
          <strong>Input event profile</strong>
          <p>{profile.headline || "No source headline supplied"} · {profile.language}</p>
          <p>Distinctive search terms: {profile.distinctive_terms.join(", ") || "none"}</p>
          <p className="meta">{profile.limitation}</p>
        </div>
      )}
      <p>
        {coverage.discovered_articles} discovered;{" "}
        {coverage.evaluated_articles ?? coverage.discovered_articles} shortlisted and
        evaluated; {coverage.matched_articles} possible or verified matches; languages:{" "}
        {coverage.languages_represented.join(", ") || "none"}.
      </p>
      <p>
        Candidate outcomes (denominator{" "}
        {coverage.evaluated_articles ?? coverage.discovered_articles} evaluated): verified{" "}
        {outcomes.verified_match ?? 0}, possible {outcomes.possible_match ?? 0},
        rejected {outcomes.rejected_match ?? 0}, insufficient evidence{" "}
        {outcomes.insufficient_evidence ?? 0}.
      </p>
      <p>
        Retrieval: {coverage.full_text_articles} full-text;{" "}
        {coverage.snippet_only_articles} snippet-only. {comparison.comparison.observation}
      </p>
      {providerEntries.length > 0 && (
        <div className="provider-status">
          {providerEntries.map(([provider, status]) => (
            <p key={provider}>
              <strong>{provider.replaceAll("_", " ")}:</strong> {status}
            </p>
          ))}
        </div>
      )}
      {comparison.provider_failures.map((failure) => (
        <div className="warning" key={failure}>Provider limitation: {failure}</div>
      ))}
      {comparison.candidates.length === 0 && (
        <p className="meta">
          No candidates were returned. This can reflect provider limitations or
          genuinely limited coverage; it is not evidence that no other reporting exists.
        </p>
      )}
      {comparison.candidates.map((candidate) => (
        <div className="evidence-card" key={candidate.url}>
          <strong>{candidate.title}</strong>
          <p>
            {candidate.language?.name || "Unknown language"} ·{" "}
            {candidate.source_name || "Unknown source"} ·{" "}
            {candidate.verification_status.replaceAll("_", " ")}
            {" "} (uncalibrated relevance: {candidate.relevance_score ?? "unavailable"})
          </p>
          <small>
            Retrieval: {candidate.retrieval_status}; scope: {candidate.analysis_scope}.{" "}
            {candidate.match_reasons.join(" ")} {candidate.uncertainty.join(" ")}
          </small>
          <p>
            <a href={candidate.url} target="_blank" rel="noreferrer">Open article</a>
          </p>
        </div>
      ))}
      {(comparison.comparison.evidence_comparisons ?? []).map((finding, index) => (
        <div className="evidence-card" key={`${finding.candidate_url}-${finding.category}-${index}`}>
          <strong>Full-text comparison · {finding.category.replaceAll("_", " ")}</strong>
          <p>{finding.candidate_title}</p>
          <p><span className="meta">Original:</span> “{finding.original_evidence.quote}” ({finding.original_evidence.occurrences} occurrence(s))</p>
          <p><span className="meta">Compared article:</span> “{finding.candidate_evidence.quote}” ({finding.candidate_evidence.occurrences} occurrence(s))</p>
          <p className="meta">{finding.interpretation}</p>
        </div>
      ))}
      {comparison.limitations.map((limitation) => (
        <p className="warning" key={limitation}>{limitation}</p>
      ))}
    </div>
  );
}
