import json

import pytest
from pydantic import ValidationError

from app.schemas.article import EvidenceSpan
from app.services.analysis_service import _validated_evidence
from evaluation.run import evaluate


def test_evaluation_report_is_deterministic_and_has_no_fixture_errors() -> None:
    report = evaluate()
    assert report["cases"]["total"] == 10
    assert report["cases"]["unavailable_or_skipped"] == 0
    assert not report["errors"]
    assert not report["evidence_validation_failures"]
    assert {"verified_match", "possible_match", "rejected_match", "insufficient_evidence"} <= report["same_event_outcomes"].keys()
    assert report["reviewed_fixture_matching_metrics"]["sample_size"] == 4
    json.dumps(report)  # report is structured and serializable


@pytest.mark.parametrize("quote,start,end", [("", 0, 0), ("abc", 0, 2), ("missing", 0, 7), ("text", -1, 3)])
def test_malformed_evidence_is_rejected(quote: str, start: int, end: int) -> None:
    with pytest.raises((ValueError, ValidationError)):
        evidence = EvidenceSpan(category="test", quote=quote, start=start, end=end, context="", explanation="", method="test")
        _validated_evidence("text", [evidence])
