export interface EvidenceLike {
  quote: string;
  start: number;
  end: number;
  occurrences?: number;
  occurrence_spans?: [number, number][];
}

export interface HighlightRange {
  start: number;
  end: number;
  key: string;
}

export function isExactEvidence(item: EvidenceLike, text: string): boolean;
export function highlightRanges(evidence: EvidenceLike[], text: string): HighlightRange[];
export function evidenceKey(item: EvidenceLike): string;
export function evidenceTotals(
  statistics: Record<string, unknown> | null | undefined,
  evidence: EvidenceLike[],
): { findings: number; occurrences: number };
