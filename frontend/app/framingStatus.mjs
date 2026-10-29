export function framingStatusMessage(assessment) {
  if (assessment.status === "available") {
    const model = assessment.model ? ` · ${assessment.model}` : "";
    const coverage = assessment.analysis_scope === "sampled_excerpts"
      ? " · sampled excerpts only"
      : " · full article";
    return `Contextual analysis available${model}${coverage}`;
  }
  if (assessment.status === "not_configured") {
    return "Contextual analysis is not configured. Set GROQ_API_KEY on the backend.";
  }
  const reason = assessment.error_code ? ` (${assessment.error_code})` : "";
  return `Contextual analysis unavailable${reason}. No keyword fallback was used.`;
}

export function framingNoFindingsMessage(status) {
  return status === "available"
    ? "The contextual analysis produced no supported findings; this is not proof that the article is unbiased."
    : "No contextual findings are available because the Groq analysis did not complete.";
}
