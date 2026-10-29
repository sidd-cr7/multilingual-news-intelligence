// Evidence highlighting helpers shared by the dashboard and its unit tests.
// Every span is re-validated against the displayed article text so an invalid
// span can never be highlighted, even if a response were tampered with.

export function isExactEvidence(item, text) {
  return (
    Number.isInteger(item.start) &&
    Number.isInteger(item.end) &&
    item.start >= 0 &&
    item.end > item.start &&
    item.end <= text.length &&
    text.slice(item.start, item.end) === item.quote
  );
}

function occurrenceSpans(item) {
  if (Array.isArray(item.occurrence_spans) && item.occurrence_spans.length) {
    return item.occurrence_spans;
  }
  return [[item.start, item.end]];
}

// One range per distinct occurrence of a cue. Overlapping ranges are skipped so
// a span is highlighted once, while every category that pointed at it remains
// visible in the evidence list.
export function highlightRanges(evidence, text) {
  const accepted = [];
  for (const item of evidence) {
    if (!isExactEvidence(item, text)) continue;
    for (const [start, end] of occurrenceSpans(item)) {
      if (
        !Number.isInteger(start) ||
        !Number.isInteger(end) ||
        start < 0 ||
        end <= start ||
        end > text.length
      ) {
        continue;
      }
      if (text.slice(start, end).toLowerCase() !== item.quote.toLowerCase()) continue;
      accepted.push({ start, end, key: `${start}-${end}` });
    }
  }
  accepted.sort((a, b) => a.start - b.start || b.end - a.end);
  const ranges = [];
  let cursor = 0;
  for (const range of accepted) {
    if (range.start < cursor) continue;
    ranges.push(range);
    cursor = range.end;
  }
  return ranges;
}

export function evidenceKey(item) {
  return `${item.start}-${item.end}`;
}

// Article-level totals are reported as distinct findings and as total cue
// occurrences so a repeated word is never counted as separate evidence.
export function evidenceTotals(statistics, evidence) {
  const findings = Number.isInteger(statistics?.total_evidence_findings)
    ? statistics.total_evidence_findings
    : evidence.length;
  const occurrences = Number.isInteger(statistics?.total_cue_occurrences)
    ? statistics.total_cue_occurrences
    : evidence.reduce((total, item) => total + (item.occurrences ?? 1), 0);
  return { findings, occurrences };
}
