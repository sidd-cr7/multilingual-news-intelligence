from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


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


class EvidenceSpan(BaseModel):
    category: str
    quote: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    context: str
    explanation: str
    method: str
    confidence: float | None = None
    uncertainty: str | None = None


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
    original_url: str | None = None
    original_text: str
    language: LanguageResult
    linguistic: dict
    entities: list[Entity]
    sentiment: dict
    emotion: dict
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
    limitations: list[str] = Field(default_factory=list)
