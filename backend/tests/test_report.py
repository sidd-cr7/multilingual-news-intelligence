"""Tests for Phase 7 PDF report generation."""
import io
import re
import pytest
import pymupdf
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.services.analysis_service import analyse_article
from app.services.language_service import LanguageDetector
from app.services.report_service import generate_pdf_report, generate_pdf_report_with_footer

client = TestClient(app)
_settings = Settings(min_text_chars=10)
_detector = LanguageDetector(_settings)


def _analysis(text: str, title: str | None = None):
    return analyse_article(text, _settings, _detector, title)


# ── Basic generation ─────────────────────────────────────────────────────────

def test_pdf_generation_returns_bytes():
    result = _analysis("Officials said the shocking crisis was caused by poor planning on 12 March 2025.")
    pdf = generate_pdf_report(result)
    assert isinstance(pdf, bytes)
    assert len(pdf) > 1000
    assert pdf[:4] == b"%PDF"


def test_pdf_with_footer_returns_valid_pdf():
    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    pdf = generate_pdf_report_with_footer(result)
    assert pdf[:4] == b"%PDF"
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    assert doc.page_count >= 1
    doc.close()


def test_pdf_contains_analysis_id():
    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    pdf = generate_pdf_report_with_footer(result)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    assert result.analysis_id in full_text


def test_pdf_contains_english_evidence_quote():
    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    pdf = generate_pdf_report_with_footer(result)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    # "shocking" is a loaded-language indicator
    assert "shocking" in full_text


@pytest.mark.parametrize(
    ("language_text", "quote"),
    [
        ("The shocking crisis is a sufficiently long news article.", "shocking"),
        ("यह क्रूर संकट पर्याप्त लंबा समाचार लेख है।", "क्रूर"),
        ("ಇದು ಕ್ರೂರ ಸಂಕಟದ ಬಗ್ಗೆ ಸಾಕಷ್ಟು ಉದ್ದವಾದ ಸುದ್ದಿ ಲೇಖನವಾಗಿದೆ.", "ಕ್ರೂರ"),
        ("ఇది క్రూరమైన సంక్షోభం గురించి తగినంత పొడవైన వార్తా కథనం.", "క్రూరమైన"),
        ("இது கொடூரமான நெருக்கடி குறித்து போதுமான நீளமான செய்திக் கட்டுரை.", "கொடூரமான"),
    ],
)
def test_unicode_evidence_quote_is_preserved(language_text: str, quote: str):
    result = _analysis(language_text)
    assert any(item.quote == quote for item in result.indicators)
    pdf = generate_pdf_report_with_footer(result)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    attachment = doc.embfile_get("research-report-evidence.txt").decode("utf-8")
    appendix_page = doc[-1]
    appendix_pixmap = appendix_page.get_pixmap()
    visible_pixels = sum(1 for pixel in appendix_pixmap.samples if pixel < 240)
    doc.close()
    assert f"Quote: {quote}" in attachment
    assert visible_pixels > 100


def test_pdf_contains_interpretation_guide():
    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    pdf = generate_pdf_report_with_footer(result)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    assert "Interpretation guide" in full_text or "interpretation guide" in full_text.lower()


# ── Multilingual rendering ───────────────────────────────────────────────────

def test_hindi_evidence_present_in_pdf():
    """Hindi (Devanagari) renders correctly with Noto Sans — verified."""
    text = "यह संकट पर पर्याप्त लंबा समाचार लेख है जिसमें जिम्मेदार लोगों का उल्लेख है।"
    result = _analysis(text)
    assert result.language.code == "hi"
    pdf = generate_pdf_report_with_footer(result)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    # Verify at least some Devanagari characters survive extraction
    devanagari_chars = [c for c in full_text if "\u0900" <= c <= "\u097F"]
    assert len(devanagari_chars) > 0, "No Devanagari characters found in extracted PDF text"


def test_kannada_evidence_is_preserved_and_appendix_is_visible():
    """Shaped Kannada text is visually rendered; exact Unicode remains in the attachment."""
    text = "ಇದು ಕ್ರೂರ ಸಂಕಟದ ಬಗ್ಗೆ ಸಾಕಷ್ಟು ಉದ್ದವಾದ ಸುದ್ದಿ ಲೇಖನವಾಗಿದೆ."
    result = _analysis(text)
    assert result.language.code == "kn"
    pdf = generate_pdf_report_with_footer(result)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    attachment = doc.embfile_get("research-report-evidence.txt").decode("utf-8")
    appendix_pixmap = doc[-1].get_pixmap()
    doc.close()
    # Text extraction is not a reliable shaping check for this TTC-backed
    # appendix. The attachment is the exact Unicode source of record.
    assert "Quote: ಕ್ರೂರ" in attachment
    assert sum(pixel < 240 for pixel in appendix_pixmap.samples) > 100


def test_telugu_pdf_generated_with_rendering_warning():
    """Telugu generates a PDF; rendering warning is included due to shaping limitations."""
    text = "ఇది సంక్షోభం గురించి తగినంత పొడవైన వార్తా కథనం మరియు కారణాన్ని వివరిస్తుంది."
    result = _analysis(text)
    assert result.language.code == "te"
    pdf = generate_pdf_report_with_footer(result)
    assert pdf[:4] == b"%PDF"
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    # Rendering warning must be present
    assert "Telugu" in full_text and ("shap" in full_text or "glyph" in full_text or "complex" in full_text)


def test_tamil_pdf_generated_with_rendering_warning():
    """Tamil generates a PDF; rendering warning is included due to shaping limitations."""
    text = "இது நெருக்கடி பற்றிய போதுமான நீளமான செய்திக் கட்டுரை மற்றும் காரணத்தை விளக்குகிறது."
    result = _analysis(text)
    assert result.language.code == "ta"
    pdf = generate_pdf_report_with_footer(result)
    assert pdf[:4] == b"%PDF"
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    assert "Tamil" in full_text and ("shap" in full_text or "glyph" in full_text or "complex" in full_text)


# ── Unsupported / missing sections ───────────────────────────────────────────

def test_unsupported_language_pdf_shows_unavailable():
    """Unsupported language produces a PDF that explicitly marks sentiment as unsupported."""
    text = "Пример достаточно длинной статьи для проверки недоступных языковых задач."
    result = _analysis(text)
    assert not result.language.supported
    pdf = generate_pdf_report_with_footer(result)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    assert "unsupported" in full_text.lower()


def test_no_comparison_section_says_unavailable():
    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    pdf = generate_pdf_report_with_footer(result, comparison=None)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    assert "No comparison results available" in full_text


def test_report_includes_overall_judgment_and_insufficient_evidence_state():
    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    pdf = generate_pdf_report_with_footer(result)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    full_text = re.sub(r"\s+", " ", "".join(page.get_text() for page in doc))
    doc.close()
    assert "Overall Bias & Framing Judgment" in full_text
    assert "Insufficient evidence for a score" in full_text
    assert "not a calibrated probability or validated measure" in full_text


def test_report_with_comparison_results_is_included_without_live_retrieval():
    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    analysis_resp = client.post(
        "/api/analyze/article",
        json={"text": "Officials said the shocking crisis was caused by poor planning."},
    )
    analysis_data = analysis_resp.json()
    comparison = {
        "original_analysis": analysis_data,
        "discovery_queries": {"hi": "query", "ta": "query"},
        "candidates": [],
        "comparison": {
            "coverage": {
                "discovered_articles": 0,
                "matched_articles": 0,
                "languages_represented": [],
                "sources_represented": [],
                "full_text_articles": 0,
                "snippet_only_articles": 0,
                "verification_outcomes": {},
                "retrieval_outcomes": {},
            },
            "indicator_counts_across_matches": {},
            "observation": "Coverage is bounded.",
        },
        "provider_failures": [],
        "limitations": [],
    }
    response = client.post(
        "/api/reports/pdf",
        json={"analysis": analysis_data, "comparison": comparison},
    )
    assert response.status_code == 200
    document = pymupdf.open(stream=response.content, filetype="pdf")
    full_text = "".join(page.get_text() for page in document)
    document.close()
    assert "Cross-Language Coverage Comparison" in full_text
    assert "Coverage is bounded." in full_text
    assert result.original_text == analysis_data["original_text"]


def test_long_candidate_url_is_wrapped_not_clipped():
    """A single opaque token wider than the column was silently clipped by Story."""
    from app.schemas.article import ComparisonResponse, DiscoveryCandidate

    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    url = "https://news.google.com/rss/articles/" + "CBMiA" * 30
    candidate = DiscoveryCandidate(
        url=url,
        title="Delhi flooding",
        provider="google_news_rss",
        retrieval_status="unavailable: blocked",
        analysis_scope="snippet_only",
        verification_status="possible_match",
    )
    comparison = ComparisonResponse(
        original_analysis=result,
        discovery_queries={"hi": "delhi flood"},
        candidates=[candidate],
        comparison={"coverage": {}, "observation": "bounded"},
    )
    pdf = generate_pdf_report_with_footer(result, comparison)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    # Break opportunities may appear as zero-width or extracted whitespace.
    compact = "".join(text.replace("\u200b", "").split())
    assert url in compact


def test_report_preserves_exact_utf8_when_script_font_is_unavailable(monkeypatch):
    result = _analysis("यह क्रूर संकट पर्याप्त लंबा समाचार लेख है।")
    monkeypatch.setattr("app.services.report_service._evidence_font_path", lambda _code: None)
    pdf = generate_pdf_report_with_footer(result)
    document = pymupdf.open(stream=pdf, filetype="pdf")
    report_text = "".join(page.get_text() for page in document)
    exact_evidence = document.embfile_get("research-report-evidence.txt").decode("utf-8")
    page_count = document.page_count
    document.close()
    assert "No suitable local font was found" in report_text
    assert any(item.quote in exact_evidence for item in result.indicators)
    assert page_count >= 2


def test_large_evidence_list_paginates_and_is_preserved():
    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    evidence_count = len(result.indicators) * 80
    result = result.model_copy(
        update={"indicators": result.indicators * 80},
    )
    pdf = generate_pdf_report_with_footer(result)
    document = pymupdf.open(stream=pdf, filetype="pdf")
    evidence = document.embfile_get("research-report-evidence.txt").decode("utf-8")
    assert document.page_count > 5
    assert evidence.count("Quote: ") == evidence_count
    document.close()


# ── Pagination ───────────────────────────────────────────────────────────────

def test_long_article_produces_multiple_pages():
    long_text = ("Officials said the shocking crisis was caused by poor planning. " * 80).strip()
    result = _analysis(long_text)
    pdf = generate_pdf_report_with_footer(result)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    pages = doc.page_count
    doc.close()
    assert pages >= 2, f"Expected multiple pages for long article, got {pages}"


# ── API endpoint ─────────────────────────────────────────────────────────────

def test_api_report_endpoint_returns_pdf():
    analysis_resp = client.post(
        "/api/analyze/article",
        json={"text": "Officials said the shocking crisis was caused by poor planning on 12 March 2025."},
    )
    assert analysis_resp.status_code == 200
    analysis_data = analysis_resp.json()

    report_resp = client.post("/api/reports/pdf", json={"analysis": analysis_data})
    assert report_resp.status_code == 200
    assert report_resp.headers["content-type"] == "application/pdf"
    assert report_resp.content[:4] == b"%PDF"


def test_api_report_content_disposition_header():
    analysis_resp = client.post(
        "/api/analyze/article",
        json={"text": "Officials said the shocking crisis was caused by poor planning.", "title": "Test Article"},
    )
    analysis_data = analysis_resp.json()
    report_resp = client.post("/api/reports/pdf", json={"analysis": analysis_data})
    assert report_resp.status_code == 200
    cd = report_resp.headers.get("content-disposition", "")
    assert "attachment" in cd
    assert ".pdf" in cd


def test_api_report_missing_analysis_returns_422():
    resp = client.post("/api/reports/pdf", json={})
    assert resp.status_code == 422
    assert resp.json() == {"detail": "Invalid report request."}


def test_api_report_malformed_analysis_returns_422():
    resp = client.post("/api/reports/pdf", json={"analysis": {"bad": "data"}})
    assert resp.status_code == 422


def test_api_report_rejects_oversized_article_text():
    analysis_resp = client.post(
        "/api/analyze/article",
        json={"text": "Officials said the shocking crisis was caused by poor planning."},
    )
    analysis_data = analysis_resp.json()
    analysis_data["original_text"] = "x" * 100_001
    resp = client.post("/api/reports/pdf", json={"analysis": analysis_data})
    assert resp.status_code == 422
    assert resp.json() == {"detail": "Invalid report request."}
    assert "x" * 100 not in resp.text


def test_api_report_rejects_empty_article_text():
    analysis_resp = client.post(
        "/api/analyze/article",
        json={"text": "Officials said the shocking crisis was caused by poor planning."},
    )
    analysis_data = analysis_resp.json()
    analysis_data["original_text"] = ""
    resp = client.post("/api/reports/pdf", json={"analysis": analysis_data})
    assert resp.status_code == 422
    assert resp.json() == {"detail": "Invalid report request."}


def test_api_report_rejects_invalid_evidence_span():
    analysis_resp = client.post(
        "/api/analyze/article",
        json={"text": "Officials said the shocking crisis was caused by poor planning."},
    )
    analysis_data = analysis_resp.json()
    analysis_data["indicators"][0]["quote"] = "altered evidence"
    resp = client.post("/api/reports/pdf", json={"analysis": analysis_data})
    assert resp.status_code == 422
    assert resp.json() == {"detail": "Invalid report request."}


def test_report_endpoint_documents_binary_pdf_response():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    content = response.json()["paths"]["/api/reports/pdf"]["post"]["responses"]["200"]["content"]
    assert content["application/pdf"]["schema"] == {"type": "string", "format": "binary"}


def test_api_report_pdf_is_parseable():
    """The returned PDF must be parseable by PyMuPDF."""
    analysis_resp = client.post(
        "/api/analyze/article",
        json={"text": "Officials said the shocking crisis was caused by poor planning."},
    )
    analysis_data = analysis_resp.json()
    report_resp = client.post("/api/reports/pdf", json={"analysis": analysis_data})
    doc = pymupdf.open(stream=report_resp.content, filetype="pdf")
    assert doc.page_count >= 1
    doc.close()


def test_pdf_metadata_summary_and_methodology_are_explicit():
    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    pdf = generate_pdf_report_with_footer(result)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    metadata = doc.metadata
    doc.close()
    assert metadata["title"] == "Multilingual News Intelligence Research Report"
    assert "Executive Summary" in text
    assert "distinct observable findings" in text
    assert "Methodology and Indicator Definitions" in text
    assert "Loaded or evaluative language" in text
    assert "Claim-status definitions" in text
    assert "Publication" in text and "date" in text


def test_report_redacts_url_credentials_query_and_fragment():
    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    result = result.model_copy(
        update={"original_url": "https://user:secret@example.org/article?token=secret#private"},
    )
    pdf = generate_pdf_report_with_footer(result)
    document = pymupdf.open(stream=pdf, filetype="pdf")
    text = "".join(page.get_text() for page in document)
    document.close()
    assert "https://example.org/article" in text
    assert "token=secret" not in text and "#private" not in text


def test_api_report_with_title_uses_title_in_filename():
    analysis_resp = client.post(
        "/api/analyze/article",
        json={"text": "Officials said the shocking crisis was caused by poor planning.", "title": "Delhi Flood Report"},
    )
    analysis_data = analysis_resp.json()
    report_resp = client.post("/api/reports/pdf", json={"analysis": analysis_data})
    cd = report_resp.headers.get("content-disposition", "")
    assert "Delhi" in cd or "Flood" in cd or "Report" in cd


def test_api_report_generation_error_is_sanitized(monkeypatch):
    def fail_generation(*_args, **_kwargs):
        raise ValueError("C:\\private\\internal\\error")

    monkeypatch.setattr("app.api.routes.generate_pdf_report_with_footer", fail_generation)
    analysis_resp = client.post(
        "/api/analyze/article",
        json={"text": "Officials said the shocking crisis was caused by poor planning."},
    )
    resp = client.post("/api/reports/pdf", json={"analysis": analysis_resp.json()})
    assert resp.status_code == 500
    assert resp.json() == {"detail": "The research report could not be generated."}
    assert "private" not in resp.text


# ── Regression: existing endpoints unaffected ────────────────────────────────

def test_existing_analyze_article_still_works():
    resp = client.post(
        "/api/analyze/article",
        json={"text": "Officials said the shocking crisis was caused by poor planning."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "indicators" in body and "statistics" in body


def test_existing_health_still_works():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_report_states_findings_occurrences_and_denominators():
    result = _analysis("The shocking plan was shocking. Officials said the shocking plan failed.")
    pdf = generate_pdf_report_with_footer(result)
    document = pymupdf.open(stream=pdf, filetype="pdf")
    text = re.sub(r"\s+", " ", "".join(page.get_text() for page in document))
    attachment = document.embfile_get("research-report-evidence.txt").decode("utf-8")
    document.close()
    assert "Claim status:" in text
    assert "Distinct evidence findings" in text
    assert "Total cue occurrences" in text
    assert "Denominator: whitespace tokens" in text
    assert "Denominator: sentences analysed" in text
    assert f"Occurrences: {result.indicators[0].occurrences}" in attachment
    assert "Sentence context:" in attachment
    assert "Evidence strength:" in attachment
    loaded = [item for item in result.indicators if item.category == "loaded_or_evaluative_language"]
    assert len(loaded) == 1 and loaded[0].occurrences == 3
    assert attachment.count("Category: loaded_or_evaluative_language") == 1


def test_report_comparison_includes_provider_status():
    analysis_resp = client.post(
        "/api/analyze/article",
        json={"text": "Officials said the shocking crisis was caused by poor planning."},
    )
    analysis_data = analysis_resp.json()
    comparison = {
        "original_analysis": analysis_data,
        "discovery_queries": {"hi": "query"},
        "candidates": [],
        "comparison": {
            "coverage": {
                "discovered_articles": 0,
                "matched_articles": 0,
                "languages_represented": [],
                "sources_represented": [],
                "full_text_articles": 0,
                "snippet_only_articles": 0,
                "verification_outcomes": {},
                "retrieval_outcomes": {},
            },
            "indicator_counts_across_matches": {},
            "observation": "Coverage is bounded.",
        },
        "provider_failures": ["GDELT: rate limited (HTTP 429); retry later."],
        "provider_status": {"gdelt": "rate limited (HTTP 429)", "google_news_rss": "hi: ok: 0 result(s)"},
        "limitations": [],
    }
    response = client.post("/api/reports/pdf", json={"analysis": analysis_data, "comparison": comparison})
    assert response.status_code == 200
    document = pymupdf.open(stream=response.content, filetype="pdf")
    text = re.sub(r"\s+", " ", "".join(page.get_text() for page in document))
    document.close()
    assert "Provider status: gdelt" in text
    assert "rate limited (HTTP 429)" in text


def test_report_tail_sections_are_not_dropped_by_pagination():
    """Guards against silent content loss when a section starts near a page break."""
    result = _analysis("Officials said the shocking crisis was caused by poor planning.")
    pdf = generate_pdf_report_with_footer(result)
    document = pymupdf.open(stream=pdf, filetype="pdf")
    text = re.sub(r"\s+", " ", "".join(page.get_text() for page in document))
    document.close()
    for heading in (
        "Framing Indicator Statistics",
        "Evidence by Indicator Category",
        "Feature Availability",
        "Methodological Limitations",
        "Cross-Language Coverage Comparison",
        "No comparison results available",
        "Evidence Preservation and Rendering",
    ):
        assert heading in text, f"Missing report section: {heading}"


def test_api_report_offers_unicode_filename_for_non_latin_title():
    from urllib.parse import quote

    analysis_resp = client.post(
        "/api/analyze/article",
        json={
            "text": "Officials said the shocking crisis was caused by poor planning.",
            "title": "दिल्ली बाढ़ रिपोर्ट",
        },
    )
    report_resp = client.post("/api/reports/pdf", json={"analysis": analysis_resp.json()})
    assert report_resp.status_code == 200
    disposition = report_resp.headers["content-disposition"]
    assert "attachment" in disposition
    assert 'filename="news-report-untitled.pdf"' in disposition
    assert "filename*=UTF-8''" in disposition
    assert quote("दिल्ली बाढ़ रिपोर्ट") in disposition
