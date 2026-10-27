import logging
import re
from urllib.parse import quote

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response
from app.config import get_settings
from app.schemas.article import ArticleAnalysisRequest, ArticleAnalysisResponse, ArticleMetadata, CompareArticleRequest, ComparisonResponse, ErrorResponse, ReportRequest, TextRequest, UrlRequest
from app.services.analysis_service import analyse_article
from app.services.comparison_service import compare_article
from app.services.article_service import create_analysis
from app.services.errors import IngestionError
from app.services.language_service import LanguageDetector
from app.services.pdf_service import extract_pdf_text
from app.services.report_service import generate_pdf_report_with_footer
from app.services.url_service import fetch_article
from app.schemas.article import InputMethod

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

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

@router.post("/compare/article", response_model=ComparisonResponse)
async def compare_coverage(payload: CompareArticleRequest) -> ComparisonResponse:
    settings = get_settings()
    return await compare_article(payload.text, payload.title, payload.original_url, payload.settings, settings, detector())

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


@router.post(
    "/reports/pdf",
    response_class=Response,
    responses={
        200: {
            "description": "Generated research report PDF",
            "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
        },
        422: {"model": ErrorResponse, "description": "Invalid report request"},
        500: {"model": ErrorResponse, "description": "Report generation failed"},
    },
)
async def create_pdf_report(payload: ReportRequest) -> Response:
    """Generate a downloadable PDF research report from existing analysis results.

    Pass the ArticleAnalysisResponse already obtained from /api/analyze/article.
    Optionally include a ComparisonResponse from /api/compare/article.
    Analysis is not re-run; the provided results are used directly.
    """
    try:
        pdf_bytes = generate_pdf_report_with_footer(payload.analysis, payload.comparison)
    except Exception as exc:
        logger.exception("PDF report generation failed")
        raise IngestionError("The research report could not be generated.", 500) from exc
    title = payload.analysis.title or "report"
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "-", title).strip("-_ ")[:60]
    filename = f"news-report-{safe_title or 'untitled'}.pdf"
    disposition = f'attachment; filename="{filename}"'
    # Non-Latin titles have no ASCII fallback characters, so also offer the
    # RFC 5987 form. Path separators and control characters are never emitted.
    unicode_title = re.sub(r"[\\/:*?\"<>|\r\n]+", "-", title).strip("-_ ")[:60]
    if unicode_title and unicode_title != safe_title:
        disposition += f"; filename*=UTF-8''{quote(f'news-report-{unicode_title}.pdf')}"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": disposition},
    )
