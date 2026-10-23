from fastapi import APIRouter, File, UploadFile
from app.config import get_settings
from app.schemas.article import ArticleAnalysisRequest, ArticleAnalysisResponse, ArticleMetadata, TextRequest, UrlRequest
from app.services.analysis_service import analyse_article
from app.services.article_service import create_analysis
from app.services.errors import IngestionError
from app.services.language_service import LanguageDetector
from app.services.pdf_service import extract_pdf_text
from app.services.url_service import fetch_article
from app.schemas.article import InputMethod

router = APIRouter(prefix="/api")

def detector() -> LanguageDetector:
    return LanguageDetector(get_settings())

@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}

@router.post("/analyze/text", response_model=ArticleMetadata)
async def analyze_text(payload: TextRequest) -> ArticleMetadata:
    settings = get_settings()
    return create_analysis(payload.text, InputMethod.text, settings, detector())

@router.post("/analyze/article", response_model=ArticleAnalysisResponse)
async def analyze_article(payload: ArticleAnalysisRequest) -> ArticleAnalysisResponse:
    settings = get_settings()
    return analyse_article(payload.text, settings, detector(), payload.title, payload.original_url)

@router.post("/analyze/pdf", response_model=ArticleMetadata)
async def analyze_pdf(file: UploadFile = File(...)) -> ArticleMetadata:
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise IngestionError("Only PDF uploads are accepted.", 415)
    settings = get_settings()
    content = await file.read(settings.max_upload_bytes + 1)
    text, warnings = extract_pdf_text(content, settings)
    return create_analysis(text, InputMethod.pdf, settings, detector(), title=file.filename, warnings=warnings)

@router.post("/analyze/url", response_model=ArticleMetadata)
async def analyze_url(payload: UrlRequest) -> ArticleMetadata:
    settings = get_settings()
    text, title, final_url, warnings = await fetch_article(str(payload.url), settings)
    return create_analysis(text, InputMethod.url, settings, detector(), title=title, original_url=final_url, warnings=warnings)
