"""Run deterministic Phase 5 checks without calling live discovery providers.

Usage from backend: python -m evaluation.run --output evaluation-report.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.config import Settings
from app.services.analysis_service import analyse_article
from app.services.comparison_service import verify_event
from app.services.language_service import LanguageDetector


FIXTURES = Path(__file__).with_name("fixtures.json")


def _metric(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"sample_size": tp + fn, "precision": round(precision, 3), "recall": round(recall, 3), "f1": round((2 * precision * recall / (precision + recall)) if precision + recall else 0.0, 3)}


def evaluate() -> dict:
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    settings = Settings(min_text_chars=10)
    detector = LanguageDetector(settings)
    per_language: dict[str, Counter] = defaultdict(Counter)
    feature_availability: Counter = Counter()
    errors: list[dict[str, str]] = []
    evidence_failures: list[dict[str, str]] = []
    analysis_processed = 0
    comparison_processed = 0

    for case in fixtures["analysis_cases"]:
        try:
            result = analyse_article(case["text"], settings, detector, case.get("title"))
            analysis_processed += 1
            language = result.language.code
            per_language[language]["cases"] += 1
            expected_language = case.get("language")
            if expected_language is None or language == expected_language:
                per_language[language]["identified"] += 1
            else:
                errors.append({"case": case["id"], "error": f"expected language {expected_language}, got {language}"})
            if expected := case.get("expect_indicator"):
                if not any(item.category == expected for item in result.indicators):
                    errors.append({"case": case["id"], "error": f"missing expected indicator {expected}"})
            if expected := case.get("expect_sentiment"):
                if result.sentiment["status"] != expected:
                    errors.append({"case": case["id"], "error": f"expected sentiment {expected}, got {result.sentiment['status']}"})
            for item in result.indicators:
                if not (0 <= item.start < item.end <= len(result.original_text)) or result.original_text[item.start:item.end] != item.quote:
                    evidence_failures.append({"case": case["id"], "error": "invalid exact Unicode evidence slice"})
            for task in result.task_availability:
                feature_availability[f"{task.task}:{task.status}"] += 1
        except Exception as exc:  # reported, never hidden
            errors.append({"case": case["id"], "error": f"{type(exc).__name__}: {exc}"})

    outcomes: Counter = Counter()
    matching = Counter()
    for case in fixtures["comparison_cases"]:
        try:
            original = analyse_article(case["text"], settings, detector, case["title"])
            actual, _, _, _ = verify_event(original, case["candidate_title"], case["candidate_text"])
            comparison_processed += 1
            outcomes[actual] += 1
            if actual != case["expected"]:
                errors.append({"case": case["id"], "error": f"expected {case['expected']}, got {actual}"})
            # Ground truth is only the reviewed fixture label; this is not a production accuracy claim.
            expected_positive = case["expected"] in {"verified_match", "possible_match"}
            actual_positive = actual in {"verified_match", "possible_match"}
            matching["tp" if expected_positive and actual_positive else "fp" if actual_positive else "fn" if expected_positive else "tn"] += 1
        except Exception as exc:
            errors.append({"case": case["id"], "error": f"{type(exc).__name__}: {exc}"})

    return {
        "methodology": fixtures["notes"],
        "scope": "Deterministic software evaluation; no live providers, training data, or claim of real-world model performance.",
        "cases": {"analysis": len(fixtures["analysis_cases"]), "comparison": len(fixtures["comparison_cases"]), "total": len(fixtures["analysis_cases"]) + len(fixtures["comparison_cases"]), "processed_successfully": analysis_processed + comparison_processed, "unavailable_or_skipped": len(fixtures["analysis_cases"]) + len(fixtures["comparison_cases"]) - analysis_processed - comparison_processed},
        "evidence_validation_failures": evidence_failures,
        "same_event_outcomes": dict(outcomes),
        "per_language_coverage": {language: dict(counts) for language, counts in sorted(per_language.items())},
        "feature_availability": dict(sorted(feature_availability.items())),
        "reviewed_fixture_matching_metrics": _metric(matching["tp"], matching["fp"], matching["fn"]),
        "errors": errors,
        "known_limitations": ["The fixtures are small and manually reviewed, not a representative benchmark.", "The matching metric is only over reviewed deterministic fixtures and must not be generalized to live news coverage.", "Provider failures, inaccessible pages, and language variation are represented by unit fixtures rather than live network calls."],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evaluation-report.json"))
    args = parser.parse_args()
    report = evaluate()
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0 if not report["errors"] and not report["evidence_validation_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
