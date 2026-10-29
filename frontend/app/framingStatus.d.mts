export interface FramingStatusInput {
  status: string;
  model?: string | null;
  error_code?: string | null;
  analysis_scope?: string;
}

export function framingStatusMessage(assessment: FramingStatusInput): string;
export function framingNoFindingsMessage(status: string): string;
