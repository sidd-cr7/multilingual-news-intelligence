from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


class InputMethod(str, Enum):
    text = "text"
    pdf = "pdf"
    url = "url"


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


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
