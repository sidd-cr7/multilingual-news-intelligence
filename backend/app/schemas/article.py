from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class InputMethod(str, Enum):
    text = "text"
    pdf = "pdf"
    url = "url"


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class ArticleAnalysisRequest(BaseModel):
    """Article text is always analysed in its original language; it is never translated."""
    text: str = Field(min_length=1)
    title: str | None = Field(default=None, max_length=500)
    original_url: str | None = Field(default=None, max_length=2_000)


class UrlRequest(BaseModel):
    url: HttpUrl


class LanguageResult(BaseModel):
    name: str
    code: str
    confidence: float | None = None
    supported: bool
    needs_confirmation: bool = False


class ArticleMetadata(BaseModel):
    analysis_id: str
    input_method: InputMethod
    original_url: str | None = None
    title: str | None = None
    language: LanguageResult
    extracted_text: str
    character_count: int
    word_count: int
    paragraph_count: int
    status: str = "completed"
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
    processed_at: datetime


class ErrorResponse(BaseModel):
    detail: str


class TaskAvailability(BaseModel):
    task: str
    status: str
    method: str | None = None
    supported_languages: list[str] = Field(default_factory=list)
    limitation: str | None = None
    model_identifier: str | None = None
    fallback: str | None = None


class EvidenceSpan(BaseModel):
    """One distinct observable cue, anchored on its first occurrence.

    ``quote``/``start``/``end`` always describe the anchor occurrence exactly as
    it appears in the original text. ``occurrence_spans`` lists every occurrence
    of the same cue so repeated wording is reported once with a count instead of
    being counted as an independent finding.
    """
    category: str
    quote: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    context: str
    explanation: str
    method: str
    confidence: float | None = None
    uncertainty: str | None = None
    cue: str | None = None
    occurrences: int = Field(default=1, ge=1)
    occurrence_spans: list[tuple[int, int]] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    evidence_strength: str | None = None
    # Claim status describes the anchor sentence, not the truth of the claim.
    # "attributed" means a configured reporting/attribution marker appears in
    # that sentence; "unattributed" means none was found. "attribution_marker"
    # marks the attribution finding itself. "mixed" is used when a repeated cue
    # appears in sentences with different attribution contexts.
    claim_status: str | None = None
    claim_status_note: str | None = None


class Entity(BaseModel):
    text: str
    label: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    method: str
    confidence: float | None = None


class SentenceSentiment(BaseModel):
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    label: str
    score: float | None = None
    method: str
    model_identifier: str | None = None
    class_scores: dict[str, float] = Field(default_factory=dict)
    uncertainty: str | None = None


class FramingFinding(BaseModel):
    id: str
    category: str
    label: str
    actor: str | None = None
    speaker: str | None = None
    attribution_type: str
    framing_direction: str
    evidence_text: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    explanation: str
    contextual_interpretation: str = ""
    potential_reader_effect: str
    evidence_strength: str
    alternative_interpretation: str
    method: str
    uncertainty: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class ActorPortrayal(BaseModel):
    actor: str
    direction: str
    finding_ids: list[str] = Field(default_factory=list)
    explanation: str


class BiasAssessment(BaseModel):
    status: str = "not_configured"
    provider: str | None = None
    model: str | None = None
    groq_used: bool = False
    cache_hit: bool = False
    error_code: str | None = None
    emotion_fallback: dict | None = None
    sentiment_fallback: dict | None = None
    analysis_scope: str = "full_article"
    neutral_rewrite: str | None = None
    conclusion: str
    summary: str
    strongest_patterns: list[str] = Field(default_factory=list)
    actor_portrayals: list[ActorPortrayal] = Field(default_factory=list)
    competing_attributed_viewpoints: bool | None = None
    representation_comparison: str
    selective_presentation: str
    sufficient_evidence_for_overall_assessment: bool
    findings: list[FramingFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class BiasJudgmentEvidence(BaseModel):
    evidence_text: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    category: str
    label: str
    actor: str | None = None
    speaker: str | None = None
    attribution_type: str
    framing_direction: str
    explanation: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    method: str


class BiasJudgment(BaseModel):
    status: str
    summary: str
    indicator: int | None = Field(default=None, ge=0, le=100)
    indicator_label: str
    direction: str
    targets: list[ActorPortrayal] = Field(default_factory=list)
    reasons: list[BiasJudgmentEvidence] = Field(default_factory=list)
    validated_auxiliary_cues: list[dict] = Field(default_factory=list)
    sentiment_signal: dict = Field(default_factory=dict)
    emotion_signal: dict = Field(default_factory=dict)
    entity_mentions: list[dict] = Field(default_factory=list)
    evidence_sufficiency: str
    scoring_method: str
    limitations: list[str] = Field(default_factory=list)


class ArticleAnalysisResponse(BaseModel):
    analysis_id: str
    title: str | None = None
    original_url: str | None = None
    original_text: str
    language: LanguageResult
    linguistic: dict
    entities: list[Entity]
    sentiment: dict
    emotion: dict
    indicators: list[EvidenceSpan]
    bias_assessment: BiasAssessment = Field(
        default_factory=lambda: BiasAssessment(
            conclusion="Insufficient evidence for an overall assessment.",
            summary="No article-level framing assessment was provided.",
            representation_comparison="Not assessed.",
            selective_presentation="Not assessed.",
            sufficient_evidence_for_overall_assessment=False,
        )
    )
    overall_judgment: BiasJudgment | None = None
    statistics: dict
    methodology: dict = Field(default_factory=dict)
    task_availability: list[TaskAvailability]
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ComparisonSettings(BaseModel):
    target_languages: list[str] = Field(default_factory=lambda: ["en", "hi", "kn", "te", "ta"])
    result_limit: int = Field(default=3, ge=1, le=10)
    include_original_language: bool = False


class CompareArticleRequest(ArticleAnalysisRequest):
    settings: ComparisonSettings = Field(default_factory=ComparisonSettings)


class DiscoveryCandidate(BaseModel):
    url: str
    title: str
    source_name: str | None = None
    published_at: str | None = None
    snippet: str | None = None
    provider: str
    language: LanguageResult | None = None
    retrieval_status: str
    analysis_scope: str
    verification_status: str
    relevance_score: float | None = None
    match_reasons: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    analysis: ArticleAnalysisResponse | None = None


class FramingDifference(BaseModel):
    """One observable framing difference between two articles."""
    dimension: str
    article_a_quote: str | None = None
    article_b_quote: str | None = None
    explanation: str
    significance: str | None = None
    confidence: str | None = None
    alternative_interpretation: str | None = None


class ComparisonResponse(BaseModel):
    original_analysis: ArticleAnalysisResponse
    event_profile: dict | None = None
    discovery_queries: dict[str, str]
    candidates: list[DiscoveryCandidate]
    comparison: dict
    framing_differences: list[FramingDifference] = Field(default_factory=list)
    provider_failures: list[str] = Field(default_factory=list)
    provider_status: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class ReportRequest(BaseModel):
    """Request body for PDF report generation.

    Pass the analysis result already obtained from /api/analyze/article.
    Optionally include comparison results from /api/compare/article.
    The report is generated from these results without re-running analysis.
    """
    model_config = ConfigDict(extra="forbid")
    analysis: ArticleAnalysisResponse
    comparison: ComparisonResponse | None = None

    @model_validator(mode="after")
    def enforce_report_bounds_and_evidence(self) -> "ReportRequest":
        if not self.analysis.original_text.strip():
            raise ValueError("Report article text cannot be empty.")
        if len(self.analysis.original_text) > 100_000:
            raise ValueError("Report article text exceeds the 100,000-character limit.")
        if len(self.analysis.indicators) > 5_000:
            raise ValueError("Report evidence exceeds the 5,000-item limit.")
        if self.comparison and len(self.comparison.candidates) > 100:
            raise ValueError("Report comparison exceeds the 100-candidate limit.")
        if (
            self.comparison
            and self.comparison.original_analysis.original_text != self.analysis.original_text
        ):
            raise ValueError("Comparison results must refer to the report article text.")
        if len(self.model_dump_json().encode("utf-8")) > 5 * 1024 * 1024:
            raise ValueError("Report request exceeds the 5 MB limit.")

        for evidence in self.analysis.indicators:
            if (
                not 0 <= evidence.start < evidence.end <= len(self.analysis.original_text)
                or self.analysis.original_text[evidence.start:evidence.end] != evidence.quote
            ):
                raise ValueError("Report evidence must exactly match the original article text.")
        return self
