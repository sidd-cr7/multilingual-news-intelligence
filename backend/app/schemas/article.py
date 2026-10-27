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
    title_source: str | None = None
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


class ArticleAnalysisResponse(BaseModel):
    analysis_id: str
    title: str | None = None
    title_source: str | None = None
    original_url: str | None = None
    original_text: str
    language: LanguageResult
    linguistic: dict
    entities: list[Entity]
    sentiment: dict
    emotion: dict
    framing_analysis: dict = Field(default_factory=dict)
    summary: dict = Field(default_factory=lambda: {"status": "unavailable", "method": None, "text": None, "limitation": "No summary was supplied."})
    indicators: list[EvidenceSpan]
    statistics: dict
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


class ComparisonResponse(BaseModel):
    original_analysis: ArticleAnalysisResponse
    discovery_queries: dict[str, str]
    candidates: list[DiscoveryCandidate]
    comparison: dict
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
