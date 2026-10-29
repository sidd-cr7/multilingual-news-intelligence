import test from "node:test";
import assert from "node:assert/strict";

import {
  framingNoFindingsMessage,
  framingStatusMessage,
} from "../app/framingStatus.mjs";

test("successful contextual analysis displays the provider model and coverage", () => {
  assert.equal(
    framingStatusMessage({
      status: "available",
      provider: "groq",
      model: "openai/gpt-oss-120b",
      analysis_scope: "full_article",
    }),
    "Contextual analysis available · openai/gpt-oss-120b · full article",
  );
  assert.match(
    framingStatusMessage({ status: "available", analysis_scope: "sampled_excerpts" }),
    /sampled excerpts only/,
  );
  assert.match(framingNoFindingsMessage("available"), /not proof/);
});

test("missing and failed provider states are visibly distinct without fallback claims", () => {
  assert.match(
    framingStatusMessage({ status: "not_configured" }),
    /not configured.*GROQ_API_KEY/,
  );
  assert.match(
    framingStatusMessage({ status: "unavailable", error_code: "provider_unavailable" }),
    /unavailable \(provider_unavailable\).*No keyword fallback/,
  );
  assert.match(framingNoFindingsMessage("unavailable"), /did not complete/);
});
