"""Build paginated research PDFs and preserve exact quotations as UTF-8 evidence."""
from __future__ import annotations

import html as html_module
import io
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from datetime import datetime, timezone

import pymupdf
from fpdf import FPDF
from fpdf.enums import WrapMode

from app.schemas.article import ArticleAnalysisResponse, ComparisonResponse, ReportRequest

logger = logging.getLogger(__name__)

_RENDERING_WARNING = (
    "MuPDF does not reliably render complex Indic scripts. Exact Unicode evidence "
    "is preserved in the attached UTF-8 evidence file and, when a suitable system "
    "font is available, in a separately shaped evidence appendix."
)
_SCRIPT_TAGS = {"hi": "Deva", "kn": "Knda", "te": "Telu", "ta": "Taml"}
_SCRIPT_FONTS = {
    "hi": "NotoSansDevanagari-Regular.ttf",
    "kn": "NotoSansKannada-Regular.ttf",
    "te": "NotoSansTelugu-Regular.ttf",
    "ta": "NotoSansTamil-Regular.ttf",
}


def _evidence_font_path(language_code: str) -> Path | None:
    if language_code not in _SCRIPT_TAGS:
        return None
    windows = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates = [windows / "Fonts" / "Nirmala.ttc"]
    font_name = _SCRIPT_FONTS.get(language_code)
    if font_name:
        candidates.extend(
            Path(directory) / font_name
            for directory in (
                "/usr/share/fonts/truetype/noto",
                "/usr/share/fonts/opentype/noto",
                "/usr/local/share/fonts",
            )
        )
    return next((path for path in candidates if path.is_file()), None)


def _font_covers_evidence(font_path: Path, analysis: ArticleAnalysisResponse) -> bool:
    try:
        font = pymupdf.Font(fontfile=str(font_path))
    except (pymupdf.FileDataError, RuntimeError, ValueError):
        logger.warning("The installed evidence font could not be loaded.")
        return False
    sampled = [
        content
        for item in analysis.indicators
        for content in (item.quote, item.context)
        if content
    ]
    return all(font.has_glyph(ord(character)) for content in sampled for character in content)


def _appendix_cell(pdf: FPDF, text: str, height: float) -> None:
    pdf.multi_cell(
        pdf.epw,
        height,
        text,
        new_x="LMARGIN",
        new_y="NEXT",
        wrapmode=WrapMode.WORD,
    )


def _evidence_attachment(analysis: ArticleAnalysisResponse) -> bytes:
    lines = [
        "Exact Unicode evidence quotations",
        f"Analysis ID: {analysis.analysis_id}",
        f"Language: {analysis.language.name} ({analysis.language.code})",
        "Offsets are zero-based Unicode character positions with an exclusive end.",
        "",
    ]
    for item in analysis.indicators:
        occurrences = ", ".join(f"{start}:{end}" for start, end in item.occurrence_spans) or f"{item.start}:{item.end}"
        lines.extend(
            [
                f"Category: {item.category}",
                f"Cue: {item.cue or 'not applicable'}",
                f"Offsets: {item.start}:{item.end}",
                f"Quote: {item.quote}",
                f"Occurrences: {item.occurrences}",
                f"Occurrence offsets: {occurrences}",
                f"Evidence strength: {item.evidence_strength or 'Not stated'}",
                f"Sentence context: {item.context or 'Not reported'}",
                f"Explanation: {item.explanation}",
                f"Method: {item.method}",
                f"Uncertainty: {item.uncertainty or 'Not reported'}",
                "",
            ]
        )
    return "\n".join(lines).encode("utf-8")


def _evidence_appendix(analysis: ArticleAnalysisResponse) -> bytes | None:
    if not analysis.indicators:
        return None
    # The shaped appendix exists to work around complex-script rendering. For
    # Latin-script reports the main body already shows the quotations, so an
    # appendix would only duplicate pages.
    if analysis.language.code not in _SCRIPT_TAGS:
        return None
    font_path = _evidence_font_path(analysis.language.code)
    if font_path is None or not _font_covers_evidence(font_path, analysis):
        return None

    pdf = FPDF(format="A4")
    pdf.set_margins(20, 18, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    _appendix_cell(pdf, "Original Unicode Evidence", 8)
    pdf.set_font("Helvetica", size=9)
    _appendix_cell(
        pdf,
        "Quotations below are direct slices of the supplied original text. "
        "The attached UTF-8 evidence file preserves exact source characters.",
        5,
    )

    if font_path is not None:
        pdf.add_font("Evidence", fname=str(font_path), collection_font_number=0)
        pdf.set_font("Evidence", size=10)
        pdf.set_text_shaping(
            script=_SCRIPT_TAGS.get(analysis.language.code),
            language=analysis.language.code,
        )
    else:
        pdf.set_font("Helvetica", size=10)

    from collections import defaultdict

    by_category: dict[str, list] = defaultdict(list)
    for item in analysis.indicators:
        by_category[item.category].append(item)
    for category, items in sorted(by_category.items()):
        pdf.set_font("Helvetica", "B", 10)
        occurrences = sum(item.occurrences for item in items)
        finding_label = "finding" if len(items) == 1 else "findings"
        occurrence_label = "occurrence" if occurrences == 1 else "occurrences"
        _appendix_cell(pdf, f"{category.replace('_', ' ').title()} ({len(items)} {finding_label}, {occurrences} {occurrence_label})", 6)
        for item in items:
            if font_path is not None:
                pdf.set_font("Evidence", size=10)
                pdf.set_text_shaping(
                    script=_SCRIPT_TAGS.get(analysis.language.code),
                    language=analysis.language.code,
                )
            else:
                pdf.set_font("Helvetica", size=10)
            _appendix_cell(pdf, f"Quote: {item.quote}", 6)
            pdf.set_font("Helvetica", size=9)
            _appendix_cell(pdf, f"Offsets: {item.start}-{item.end}; occurrences: {item.occurrences}", 5)
            _appendix_cell(pdf, f"Evidence strength: {item.evidence_strength or 'Not stated'}", 5)
            if item.context and font_path is not None:
                pdf.set_font("Evidence", size=10)
                pdf.set_text_shaping(
                    script=_SCRIPT_TAGS.get(analysis.language.code),
                    language=analysis.language.code,
                )
                _appendix_cell(pdf, f"Sentence context: {item.context}", 5)
                pdf.set_font("Helvetica", size=9)
            else:
                _appendix_cell(pdf, "Sentence context: see the attached UTF-8 evidence file.", 5)
            _appendix_cell(pdf, f"Explanation: {item.explanation}", 5)
            _appendix_cell(pdf, f"Method: {item.method}; uncertainty: {item.uncertainty or 'Not reported'}", 5)
            pdf.ln(3)
    return bytes(pdf.output())


def _e(text: str) -> str:
    """HTML-escape a string for safe embedding."""
    return html_module.escape(str(text), quote=True)


def _safe_source_url(value: str | None) -> str:
    if not value:
        return "Not provided"
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        return "Unavailable"
    if parts.scheme.lower() not in {"http", "https"} or not hostname:
        return "Unavailable"
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parts.scheme.lower(), netloc, parts.path, "", ""))


def _break_long_token(value: str, size: int = 48) -> str:
    """Insert invisible break opportunities into an overlong unbroken token.

    PyMuPDF's Story renderer clips a token wider than the text column instead of
    breaking it, which silently dropped the tail of long candidate URLs. Zero-width
    spaces keep every visible character while allowing the token to wrap. Short
    URLs that already fit are left untouched so their text stays verbatim.
    """
    if len(value) <= 72:
        return value
    return "\u200b".join(value[index:index + size] for index in range(0, len(value), size))


def _noto_css() -> tuple[str, pymupdf.Archive]:
    """Return CSS and Archive for Noto Sans (covers Latin, Devanagari, Kannada)."""
    archive = pymupdf.Archive()
    css = pymupdf.css_for_pymupdf_font("notos", archive=archive)
    css += "\nbody { font-family: notos, sans-serif; font-size: 10pt; line-height: 1.5; color: #1a1a2e; margin: 0; }\n"
    css += "h1 { font-size: 16pt; color: #102a43; margin-bottom: 4pt; }\n"
    css += "h2 { font-size: 12pt; color: #102a43; border-bottom: 1px solid #dce4ec; padding-bottom: 2pt; margin-top: 14pt; margin-bottom: 6pt; }\n"
    css += "h3 { font-size: 10pt; color: #1769aa; margin-top: 10pt; margin-bottom: 4pt; }\n"
    css += "table { width: 100%; border-collapse: collapse; margin: 6pt 0; font-size: 9pt; table-layout: fixed; }\n"
    # Percentage cell widths caused PyMuPDF Story to silently drop the tail of the
    # document, so column sizing relies on the fixed layout and the header cell.
    css += "th { background: #f4f8fb; text-align: left; padding: 4pt 6pt; border: 1px solid #dce4ec; font-weight: bold; }\n"
    css += "th, td { word-wrap: break-word; vertical-align: top; }\n"
    css += "td { padding: 3pt 6pt; border: 1px solid #dce4ec; }\n"
    css += ".warn { background: #fff4d6; padding: 4pt 8pt; border-left: 3px solid #e6a817; margin: 6pt 0; font-size: 9pt; }\n"
    css += ".quote { background: #f4f8fb; padding: 4pt 8pt; border-left: 3px solid #1769aa; margin: 4pt 0; font-size: 9pt; }\n"
    css += ".meta { color: #617084; font-size: 9pt; }\n"
    css += ".avail-ok { color: #1a7a4a; }\n"
    css += ".avail-no { color: #932d2d; }\n"
    css += ".avail-partial { color: #725300; }\n"
    css += "p { margin: 3pt 0; }\n"
    css += "ul { margin: 3pt 0; padding-left: 16pt; }\n"
    css += "li { margin: 2pt 0; }\n"
    return css, archive


def _section(title: str, content: str) -> str:
    return f"<h2>{_e(title)}</h2>{content}"


def _kv_table(rows: list[tuple[str, str]]) -> str:
    """Render label/value rows as paragraphs.

    A two-column table in the PyMuPDF Story renderer produced overlapping cells
    and an unreadable key column, so these rows are laid out as text instead.
    """
    return "".join(f"<p><b>{_e(k)}:</b> {v}</p>" for k, v in rows)


def _avail_span(status: str) -> str:
    cls = {
        "available": "avail-ok",
        "available_with_limitations": "avail-partial",
        "partial": "avail-partial",
    }.get(status, "avail-no")
    return f"<span class='{cls}'>{_e(status)}</span>"


_CATEGORY_LABELS = {
    "loaded_or_evaluative_language": "Loaded or evaluative language",
    "attribution_of_responsibility": "Attribution of responsibility",
    "attributed_claim": "Attributed claim",
    "hedging_or_qualification": "Hedging or qualification",
    "absolutist_or_generalising_language": "Absolutist or generalising language",
    "attributed": "Attributed",
    "unattributed": "Unattributed",
    "attribution_marker": "Attribution marker",
    "mixed": "Mixed attribution",
}


def _label(value: str | None) -> str:
    if not value:
        return "Not reported"
    return _CATEGORY_LABELS.get(value, value.replace("_", " ").capitalize())


def _executive_summary(analysis: ArticleAnalysisResponse, comparison: ComparisonResponse | None) -> str:
    ling = analysis.linguistic or {}
    stat = analysis.statistics or {}
    sentences = ling.get("sentence_count", "N/A")
    tokens = ling.get("token_count", stat.get("word_count", "N/A"))
    findings = len(analysis.indicators)
    occurrences = stat.get("total_cue_occurrences", sum(item.occurrences for item in analysis.indicators))
    density = stat.get("evidence_density_per_1000_tokens")
    counts = stat.get("indicator_counts") or {}
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    parts = [
        f"This report examines a {_e(analysis.language.name)} article containing approximately "
        f"{_e(str(sentences))} sentences and {_e(str(tokens))} whitespace tokens."
    ]
    if findings:
        density_text = f" ({_e(str(density))} cue occurrences per 1,000 tokens)" if density is not None else ""
        parts.append(
            f"The rule-based engine identified {findings} distinct observable findings covering "
            f"{occurrences} cue occurrences{density_text}. The most frequent categories were "
            + ", ".join(f"{_e(_label(category))} ({count})" for category, count in top)
            + "."
        )
    else:
        parts.append(
            "No configured indicator cues matched the analysed text. This is not evidence that the article is unbiased or balanced."
        )
    claim = stat.get("attribution_summary") or {}
    if claim:
        markers = claim.get('attribution_marker_findings', 0)
        parts.append(
            "Attribution context was recorded for the anchor sentence of each finding: "
            f"{_e(str(claim.get('attributed_findings', 0)))} attributed, "
            f"{_e(str(claim.get('unattributed_findings', 0)))} unattributed, and "
            f"{_e(str(markers))} attribution-marker finding{'' if markers == 1 else 's'}. "
            "These are descriptive counts; they are not counts of unique sources and do not measure fairness."
        )
    if comparison and comparison.candidates:
        cov = (comparison.comparison or {}).get("coverage", {})
        matched = cov.get('matched_articles', 'N/A')
        parts.append(
            f"Coverage comparison evaluated {_e(str(cov.get('evaluated_articles', 'N/A')))} of "
            f"{_e(str(cov.get('discovered_articles', 'N/A')))} discovered candidates; "
            f"{_e(str(matched))} {'was' if matched == 1 else 'were'} possible or verified "
            f"match{'' if matched == 1 else 'es'}. "
            "Wording comparisons are restricted to verified full-text matches."
        )
    parts.append(
        "Every finding is a heuristic observation tied to an exact quotation. The report does not "
        "establish author intent, deception, or a validated article-level bias verdict. "
        "A provisional directional-framing indicator may be reported separately when evidence "
        "meets its stated threshold."
    )
    return " ".join(parts)


def _methodology_html(analysis: ArticleAnalysisResponse) -> str:
    methodology = analysis.methodology or {}
    categories = methodology.get("categories") or {}
    claim_defs = methodology.get("claim_status") or {}

    def block(title: str, items: dict) -> str:
        if not items:
            return ""
        rows = "".join(f"<li><b>{_e(_label(key))}</b> — {_e(value)}</li>" for key, value in items.items())
        return f"<h3>{_e(title)}</h3><ul>{rows}</ul>"

    approach = methodology.get("approach")
    offsets = methodology.get("offsets")
    interpretation = methodology.get("interpretation")
    html = f"<p>{_e(approach)}</p>" if approach else ""
    html += block("Indicator definitions", categories)
    html += block("Claim-status definitions", claim_defs)
    if offsets:
        html += f"<p><b>Offsets:</b> {_e(offsets)}</p>"
    if interpretation:
        html += f"<p><b>Interpretation:</b> {_e(interpretation)}</p>"
    return html or "<p>No methodology metadata was supplied.</p>"


def _build_html(
    analysis: ArticleAnalysisResponse,
    comparison: ComparisonResponse | None,
    generated_at: datetime,
) -> str:
    parts: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    ts = generated_at.strftime("%Y-%m-%d %H:%M UTC")
    parts.append(
        f"<h1>Multilingual News Intelligence — Research Report</h1>"
        f"<p class='meta'>Generated: {_e(ts)} &nbsp;|&nbsp; "
        f"Analysis ID: {_e(analysis.analysis_id)}</p>"
        f"<p class='warn'><b>Interpretation guide:</b> Indicators identify "
        f"observable linguistic patterns in the original text. They are not "
        f"proof of author intent, political bias, deception, or manipulation. "
        f"When sufficient validated evidence is available, a provisional directional-framing "
        f"indicator may be shown; it is not a calibrated probability or validated bias score.</p>"
    )
    parts.append(_section("Executive Summary", f"<p>{_executive_summary(analysis, comparison)}</p>"))

    # ── Article metadata ─────────────────────────────────────────────────────
    meta_rows: list[tuple[str, str]] = [
        ("Title", _e(analysis.title or "Untitled")),
        ("Source URL", _e(_break_long_token(_safe_source_url(analysis.original_url)))),
        ("Publication date", "Unavailable — no publication date was provided."),
        ("Language", f"{_e(analysis.language.name)} ({_e(analysis.language.code)})"),
        ("Language confidence", _e(f"{analysis.language.confidence:.3f}" if analysis.language.confidence is not None else "N/A (script-based detection)")),
        ("Language supported", _e("Yes" if analysis.language.supported else "No")),
    ]
    parts.append(_section("Article Metadata", _kv_table(meta_rows)))

    # ── Linguistic counts ────────────────────────────────────────────────────
    ling = analysis.linguistic
    stat = analysis.statistics
    ling_rows: list[tuple[str, str]] = [
        ("Sentence count", _e(str(ling.get("sentence_count", "N/A")))),
        ("Token count", _e(str(ling.get("token_count", stat.get("word_count", "N/A"))))),
        ("Token convention", _e(ling.get("token_count_convention", "Whitespace-delimited non-empty tokens"))),
        ("Sentence convention", "Punctuation-based segmentation; approximate for all languages."),
        ("Character count", _e(str(stat.get("character_count", "N/A")))),
        ("Paragraph count", _e(str(stat.get("paragraph_count", "N/A")))),
        ("POS distribution", _e("Unavailable — no pretrained POS model configured")),
    ]
    parts.append(_section("Linguistic Counts", _kv_table(ling_rows)))
    parts.append(_section("Methodology and Indicator Definitions", _methodology_html(analysis)))

    # ── Named entities ───────────────────────────────────────────────────────
    entities = analysis.entities
    if entities:
        rows_html = "".join(
            f"<tr><td>{_e(e.text)}</td><td>{_e(e.label)}</td>"
            f"<td>{_e(str(e.start))}–{_e(str(e.end))}</td>"
            f"<td>{_e(e.method)}</td></tr>"
            for e in entities
        )
        ent_table = (
            "<table><tr><th>Text</th><th>Label</th><th>Offsets</th><th>Method</th></tr>"
            + rows_html + "</table>"
            "<p class='meta'>Entity methods are shown per candidate. An empty result is not evidence "
            "that no entities are present; consult task availability for model scope.</p>"
        )
    else:
        entity_status = next(
            (task.status for task in analysis.task_availability if task.task == "named_entity_recognition"),
            "unavailable",
        )
        ent_table = (
            f"<p>No entity candidates were returned. Feature status: "
            f"{_e(entity_status)}. An empty result is not evidence that no entities are present.</p>"
        )
    parts.append(_section("Named Entities", ent_table))

    # ── Sentiment ────────────────────────────────────────────────────────────
    sent = analysis.sentiment
    if sent.get("status") == "available_with_limitations" and sent.get("class_scores"):
        score_text = ", ".join(
            f"{_e(label)}: {_e(format(score, '.3f'))}"
            for label, score in sent["class_scores"].items()
            if isinstance(score, (int, float))
        )
        sent_rows: list[tuple[str, str]] = [
            ("Model label", _e(sent.get("label") or "Not resolved")),
            ("Raw mean class scores", score_text or "Not reported"),
            ("Model", _e(sent.get("model_identifier", "Not reported"))),
            ("Method", _e(sent.get("method", "N/A"))),
        ]
        sent_html = (
            _kv_table(sent_rows)
            + f"<p class='meta'>{_e(sent.get('score_note', 'Scores are uncalibrated model outputs.'))} "
            "Quoted reactions, sarcasm, and speaker attribution are not resolved. Negative sentiment "
            "is never equated with bias.</p>"
        )
    else:
        status = sent.get("status", "unavailable")
        sent_html = (
            f"<p>Sentiment analysis: {_avail_span(status)} for language "
            f"'{_e(analysis.language.code)}'. {_e(sent.get('limitation', 'No model prediction is available.'))}</p>"
        )
    parts.append(_section("Sentiment Analysis", sent_html))

    # ── Emotion ──────────────────────────────────────────────────────────────
    emo = analysis.emotion
    if emo.get("status") == "available_with_limitations" and emo.get("distribution"):
        emo_rows = [
            (k, _e(f"{v:.3f} raw score"))
            for k, v in emo["distribution"].items()
            if isinstance(v, (int, float))
        ]
        emo_html = (
            _kv_table(emo_rows)
            + f"<p class='meta'>Model: {_e(emo.get('model_identifier', 'Not reported'))}. "
            f"{_e(emo.get('score_note', 'Scores are uncalibrated model outputs.'))} "
            "This is emotion classification, not sentiment or sarcasm detection.</p>"
        )
    else:
        emo_html = (
            f"<p>Emotion analysis: {_avail_span(emo.get('status', 'unavailable'))}. "
            f"{_e(emo.get('limitation', 'No model prediction is available.'))}</p>"
        )
    parts.append(_section("Emotion Analysis", emo_html))

    # ── Indicator statistics ─────────────────────────────────────────────────
    ind_counts = stat.get("indicator_counts")
    density = stat.get("evidence_density_per_1000_tokens")
    occurrence_counts = stat.get("indicator_occurrences")
    total_occurrences = stat.get("total_cue_occurrences")
    sentence_count = ling.get("sentence_count")
    token_count = ling.get("token_count", stat.get("word_count"))
    stat_rows: list[tuple[str, str]] = [
        ("Distinct evidence findings", _e(str(len(analysis.indicators)))),
        (
            "Total cue occurrences",
            _e(
                str(total_occurrences)
                if total_occurrences is not None
                else str(sum(item.occurrences for item in analysis.indicators))
            ),
        ),
        ("Denominator: sentences analysed", _e(str(sentence_count) if sentence_count is not None else "N/A")),
        ("Denominator: whitespace tokens", _e(str(token_count) if token_count is not None else "N/A")),
        ("Evidence density (occurrences per 1,000 tokens)", _e(str(density) if density is not None else "N/A")),
        ("Density convention", _e(stat.get("evidence_density_convention", "Total cue occurrences ÷ token count × 1,000"))),
    ]
    if isinstance(ind_counts, dict) and ind_counts:
        per_category: list[str] = []
        for cat, count in sorted(ind_counts.items()):
            occurrences = occurrence_counts.get(cat) if isinstance(occurrence_counts, dict) else None
            finding_label = "finding" if count == 1 else "findings"
            suffix = (
                f" ({occurrences} {'occurrence' if occurrences == 1 else 'occurrences'})"
                if occurrences is not None
                else ""
            )
            per_category.append(f"{cat.replace('_', ' ').title()}: {count} {finding_label}{suffix}")
        stat_rows.append(("Per-category counts", "<br/>".join(_e(line) for line in per_category)))
    else:
        stat_rows.append(("Per-category indicator counts", "Unavailable — no indicator category matched, or counts were not supplied."))
    parts.append(_section(
        "Framing Indicator Statistics",
        _kv_table(stat_rows)
        + "<p class='meta'>A finding is one distinct observable cue; its occurrences count "
        "every time that cue appears in the analysed text. Repeated wording is not counted "
        "as independent findings, and counts are never bias scores.</p>"
    ))

    # ── Evidence by category ─────────────────────────────────────────────────
    lang_code = analysis.language.code
    needs_rendering_warning = lang_code in _SCRIPT_TAGS
    evidence_html = (
        "<p>Evidence validation: all supplied quotation offsets were checked against "
        "the exact original text before report generation.</p>"
    )
    if needs_rendering_warning:
        evidence_html += f"<p class='warn'>{_e(_RENDERING_WARNING)}</p>"
        font_path = _evidence_font_path(lang_code)
        if font_path is None or not _font_covers_evidence(font_path, analysis):
            evidence_html += (
                "<p class='warn'>No suitable local font was found for this script; "
                "the shaped appendix is unavailable. Use the attached UTF-8 evidence file.</p>"
            )
        else:
            evidence_html += (
                "<p class='meta'>A shaped evidence appendix is included when rendering succeeds; "
                "the attached UTF-8 text remains authoritative.</p>"
            )

    from collections import defaultdict
    by_cat: dict[str, list] = defaultdict(list)
    for item in analysis.indicators:
        by_cat[item.category].append(item)

    if by_cat:
        for cat, items in sorted(by_cat.items()):
            occurrences = sum(item.occurrences for item in items)
            evidence_html += (
                f"<h3>{_e(_label(cat))} "
                f"({len(items)} {'finding' if len(items) == 1 else 'findings'}, "
                f"{occurrences} {'occurrence' if occurrences == 1 else 'occurrences'})</h3>"
            )
            for item in items:
                quote_html = (
                    "<b>Exact quotation:</b> See the Original Unicode Evidence appendix; "
                    "the attached UTF-8 file preserves the exact characters.<br/>"
                    if lang_code in _SCRIPT_TAGS
                    else f"<b>Quote:</b> &#8220;{_e(item.quote)}&#8221;<br/>"
                )
                context_html = f"<b>Sentence context:</b> {_e(item.context)}<br/>" if item.context else ""
                claim_html = (
                    f"<b>Claim status:</b> {_e(_label(item.claim_status))}"
                    + (f" — {_e(item.claim_status_note)}" if item.claim_status_note else "")
                    + "<br/>"
                ) if item.claim_status else ""
                evidence_html += (
                    f"<div class='quote'>{quote_html}"
                    f"<b>Cue:</b> {_e(item.cue or 'not applicable')} &nbsp; "
                    f"<b>Occurrences:</b> {_e(str(item.occurrences))} &nbsp; "
                    f"<b>Evidence strength:</b> {_e(item.evidence_strength or 'Not stated')}<br/>"
                    f"<b>Offsets:</b> {_e(str(item.start))}–{_e(str(item.end))} &nbsp; "
                    f"<b>Method:</b> {_e(item.method)}<br/>"
                    f"{claim_html}"
                    f"{context_html}"
                    f"<b>Explanation:</b> {_e(item.explanation)}<br/>"
                    f"<b>Uncertainty:</b> {_e(item.uncertainty or 'None stated')}"
                    f"</div>"
                )
    else:
        evidence_html += (
            "<p>No configured indicator cues matched the analysed text. "
            "This is not evidence that the article is unbiased or balanced.</p>"
        )
    parts.append(_section("Evidence by Indicator Category", evidence_html))

    # ── Feature availability ─────────────────────────────────────────────────
    avail_rows: list[tuple[str, str]] = []
    for task in analysis.task_availability:
        status_html = _avail_span(task.status)
        method = f" — {_e(task.method)}" if task.method else ""
        limit = f"<br/><span class='meta'>{_e(task.limitation)}</span>" if task.limitation else ""
        avail_rows.append((task.task.replace("_", " ").title(), f"{status_html}{method}{limit}"))
    avail_html = (
        _kv_table(avail_rows)
        if avail_rows
        else "<p>Per-task feature availability was not supplied.</p>"
    )
    parts.append(_section("Feature Availability", avail_html))

    # ── Limitations ──────────────────────────────────────────────────────────
    lim_items = "".join(f"<li>{_e(lim)}</li>" for lim in analysis.limitations)
    if analysis.warnings:
        lim_items += "".join(f"<li><b>Warning:</b> {_e(w)}</li>" for w in analysis.warnings)
    if not lim_items:
        lim_items = "<li>No methodological limitations were supplied with this result.</li>"
    parts.append(_section("Methodological Limitations", f"<ul>{lim_items}</ul>"))

    # ── Overall bias and framing judgment ────────────────────────────────────
    judgment = analysis.overall_judgment
    if judgment:
        indicator = (
            f"{judgment.indicator}/100 ({_e(judgment.indicator_label)})"
            if judgment.indicator is not None
            else "Insufficient evidence for a score"
        )
        judgment_html = (
            f"<p><b>Status:</b> {_e(judgment.status)}</p>"
            f"<p><b>Provisional indicator:</b> {indicator}</p>"
            f"<p><b>Direction:</b> {_e(judgment.direction.replace('_', ' '))}</p>"
            f"<p><b>Evidence sufficiency:</b> {_e(judgment.evidence_sufficiency)}</p>"
            f"<p>{_e(judgment.summary)}</p>"
        )
        for reason in judgment.reasons[:5]:
            judgment_html += (
                "<div class='quote'>"
                f"<b>Evidence ({_e(reason.attribution_type.replace('_', ' '))}):</b> "
                f"{_e(reason.evidence_text)}<br/>"
                f"<b>Direction:</b> {_e(reason.framing_direction.replace('_', ' '))}<br/>"
                f"<b>Explanation:</b> {_e(reason.explanation)}"
                "</div>"
            )
        if judgment.limitations:
            judgment_html += "<ul>" + "".join(
                f"<li>{_e(limitation)}</li>" for limitation in judgment.limitations
            ) + "</ul>"
        judgment_html += (
            f"<p class='meta'>{_e(judgment.scoring_method)} "
            "This is a provisional indicator, not a calibrated probability or validated "
            "measure of political bias. Attributed quotations, sentiment, and emotion are "
            "not treated as authorial directional evidence.</p>"
        )
        parts.append(_section("Overall Bias & Framing Judgment", judgment_html))

    # ── Cross-language comparison ────────────────────────────────────────────
    if comparison:
        raw_coverage = comparison.comparison.get("coverage")
        cov = raw_coverage if isinstance(raw_coverage, dict) else {}
        requested_languages = set(comparison.discovery_queries)
        raw_represented = cov.get("languages_represented")
        represented_list = (
            [language for language in raw_represented if isinstance(language, str)]
            if isinstance(raw_represented, list)
            else []
        )
        represented_languages = set(represented_list)
        missing_languages = sorted(requested_languages - represented_languages)
        comp_rows: list[tuple[str, str]] = [
            ("Discovered articles", _e(str(cov.get("discovered_articles", "Not reported")))),
            ("Candidates evaluated", _e(str(cov.get("evaluated_articles", "Not reported")))),
            ("Matched articles", _e(str(cov.get("matched_articles", "Not reported")))),
            ("Languages represented", _e(", ".join(represented_list) or "None reported")),
            ("Requested languages with no represented match", _e(", ".join(missing_languages) or "None reported")),
            ("Full-text articles", _e(str(cov.get("full_text_articles", "Not reported")))),
            ("Snippet-only articles", _e(str(cov.get("snippet_only_articles", "Not reported")))),
            ("Retrieval outcomes", _e(str(cov.get("retrieval_outcomes", "Not reported")))),
        ]
        wording_basis = comparison.comparison.get("wording_comparison_basis")
        if wording_basis:
            comp_rows.append(("Wording comparison basis", _e(str(wording_basis))))
        outcomes = cov.get("verification_outcomes", {})
        for outcome, count in sorted(outcomes.items()):
            comp_rows.append((f"Outcome: {outcome.replace('_', ' ')}", _e(str(count))))
        for provider, status in (comparison.provider_status or {}).items():
            comp_rows.append((f"Provider status: {provider.replace('_', ' ')}", _e(status)))
        if comparison.provider_failures:
            comp_rows.append(("Provider failures", _e("; ".join(comparison.provider_failures))))
        comp_html = _kv_table(comp_rows)

        if comparison.candidates:
            comp_html += "<h3>Candidate Articles</h3>"
            for candidate in comparison.candidates:
                comp_html += (
                    f"<div class='quote'><b>{_e(candidate.title)}</b><br/>"
                    f"<b>Verification:</b> {_e(candidate.verification_status.replace('_', ' '))} &nbsp; "
                    f"<b>Relevance (uncalibrated):</b> "
                    f"{_e(str(candidate.relevance_score) if candidate.relevance_score is not None else 'N/A')}<br/>"
                    f"<b>Source:</b> {_e(candidate.source_name or 'Not provided')} &nbsp; "
                    f"<b>Published:</b> {_e(candidate.published_at or 'Not provided')}<br/>"
                    f"<b>Scope:</b> {_e(candidate.analysis_scope)} &nbsp; "
                    f"<b>Retrieval:</b> {_e(candidate.retrieval_status)}<br/>"
                    f"<b>URL:</b> {_e(_break_long_token(_safe_source_url(candidate.url)))}<br/>"
                    f"<b>Reasons:</b> {_e(' '.join(candidate.match_reasons) or 'None recorded')}<br/>"
                    f"<b>Uncertainty:</b> {_e(' '.join(candidate.uncertainty) or 'None recorded')}"
                    f"</div>"
                )
            comp_html += (
                "<p class='meta'>Snippet-only and metadata-only candidates do not represent "
                "full-text analysis. Only candidates explicitly marked verified_match are "
                "verified by the configured rules; possible matches remain uncertain.</p>"
            )

        obs = comparison.comparison.get("observation", "")
        if obs:
            comp_html += f"<p class='meta'>{_e(obs)}</p>"
        for lim in comparison.limitations:
            comp_html += f"<p class='warn'>{_e(lim)}</p>"
        parts.append(
            f"<h2 style='page-break-before: always;'>"
            f"{_e('Cross-Language Coverage Comparison')}</h2>{comp_html}"
        )
    else:
        parts.append(_section(
            "Cross-Language Coverage Comparison",
            "<p>No comparison results available. Use the comparison endpoint to discover related coverage.</p>"
        ))

    # ── Evidence preservation and rendering ─────────────────────────────────
    render_lims = [
        "Every evidence quotation is validated against the supplied original text before the report is generated.",
        "For Indic scripts, the main body identifies the evidence category, cue, offsets, and claim status; exact quotations are rendered in the Original Unicode Evidence appendix.",
        "An attached UTF-8 evidence file preserves every exact quotation, occurrence offset, and explanation independently of PDF text extraction.",
        "If a suitable local script font is unavailable, the appendix is omitted and that limitation is stated; the exact UTF-8 attachment remains available.",
        "Visual rendering depends on the fonts installed on the viewing system. Inspect the appendix before publication.",
    ]
    render_items = "".join(f"<li>{_e(lim)}</li>" for lim in render_lims)
    parts.append(_section("Evidence Preservation and Rendering", f"<ul>{render_items}</ul>"))

    body = "\n".join(parts)
    return f"<html><body>{body}</body></html>"


def generate_pdf_report(
    analysis: ArticleAnalysisResponse,
    comparison: ComparisonResponse | None = None,
) -> bytes:
    """Generate a PDF research report from analysis results.

    Returns raw PDF bytes. Raises ValueError on generation failure.
    """
    ReportRequest(analysis=analysis, comparison=comparison)
    generated_at = datetime.now(timezone.utc)
    css, archive = _noto_css()
    html_content = _build_html(analysis, comparison, generated_at)

    try:
        story = pymupdf.Story(html=html_content, user_css=css, archive=archive)
        buf = io.BytesIO()
        writer = pymupdf.DocumentWriter(buf)
        mediabox = pymupdf.paper_rect("a4")
        margin = 54  # ~19mm
        where = pymupdf.Rect(margin, margin, mediabox.width - margin, mediabox.height - margin)

        page_num = 0
        more = True
        while more:
            page_num += 1
            device = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(device)
            # Page number footer
            writer.end_page()

        writer.close()
        buf.seek(0)
        document = pymupdf.open(stream=buf.read(), filetype="pdf")
        appendix_bytes = _evidence_appendix(analysis)
        if appendix_bytes:
            appendix = pymupdf.open(stream=appendix_bytes, filetype="pdf")
            document.insert_pdf(appendix)
            appendix.close()
        document.embfile_add(
            "research-report-evidence.txt",
            _evidence_attachment(analysis),
            desc="Exact original Unicode evidence quotations and explanations",
        )
        combined = io.BytesIO()
        document.save(combined, garbage=4, deflate=True)
        document.close()
        return combined.getvalue()
    except Exception as exc:
        logger.exception("PDF report generation failed")
        raise ValueError("PDF report generation failed.") from exc


def generate_pdf_report_with_footer(
    analysis: ArticleAnalysisResponse,
    comparison: ComparisonResponse | None = None,
) -> bytes:
    """Generate PDF and add page-number footers via a second pass."""
    raw = generate_pdf_report(analysis, comparison)
    doc = pymupdf.open(stream=raw, filetype="pdf")
    try:
        doc.set_metadata(
            {
                "title": "Multilingual News Intelligence Research Report",
                "subject": "Research-oriented linguistic indicator report",
                "creator": "Multilingual News Intelligence",
                "creationDate": datetime.now(timezone.utc).strftime("D:%Y%m%d%H%M%S+00'00'"),
            }
        )
        total = doc.page_count
        for i, page in enumerate(doc, 1):
            footer = f"Page {i} of {total}  |  Multilingual News Intelligence"
            page.insert_text(
                pymupdf.Point(54, page.rect.height - 24),
                footer,
                fontsize=7,
                color=(0.38, 0.44, 0.52),
            )
        buf = io.BytesIO()
        doc.save(buf, garbage=4, deflate=True)
        return buf.getvalue()
    except Exception as exc:
        logger.exception("PDF report post-processing failed")
        raise ValueError("PDF report generation failed.") from exc
    finally:
        doc.close()
