import assert from "node:assert/strict";
import { test } from "node:test";
import {
  evidenceKey,
  evidenceTotals,
  highlightRanges,
  isExactEvidence,
} from "../app/evidence.mjs";

const text = "The shocking plan was shocking. Officials said the shocking plan failed.";
const firstCue = text.indexOf("shocking");
const secondCue = text.indexOf("shocking", firstCue + 1);
const thirdCue = text.indexOf("shocking", secondCue + 1);
const cueSpan = (start) => [start, start + "shocking".length];
const planStart = text.indexOf("plan");
const officialsStart = text.indexOf("Officials");

test("accepts only spans that exactly slice the original text", () => {
  assert.equal(
    isExactEvidence({ quote: "shocking", start: firstCue, end: firstCue + 8 }, text),
    true,
  );
  assert.equal(
    isExactEvidence({ quote: "Shocking", start: firstCue, end: firstCue + 8 }, text),
    false,
  );
  assert.equal(
    isExactEvidence({ quote: "shocking", start: firstCue, end: text.length + 1 }, text),
    false,
  );
  assert.equal(isExactEvidence({ quote: "", start: firstCue, end: firstCue }, text), false);
});

test("highlights every occurrence of a repeated cue once", () => {
  const ranges = highlightRanges(
    [
      {
        quote: "shocking",
        start: firstCue,
        end: firstCue + 8,
        occurrences: 3,
        occurrence_spans: [cueSpan(firstCue), cueSpan(secondCue), cueSpan(thirdCue)],
      },
    ],
    text,
  );
  assert.deepEqual(
    ranges.map((range) => range.start),
    [firstCue, secondCue, thirdCue],
  );
  assert.equal(ranges.filter((range) => range.start === firstCue).length, 1);
});

test("drops invalid occurrence spans but keeps validated ones", () => {
  const ranges = highlightRanges(
    [
      {
        quote: "plan",
        start: planStart,
        end: planStart + 4,
        occurrence_spans: [
          [planStart, planStart + 4],
          [firstCue, planStart + 4],
          [-1, 4],
          [planStart + 1, planStart + 5],
        ],
      },
    ],
    text,
  );
  assert.deepEqual(
    ranges.map((range) => range.key),
    [`${planStart}-${planStart + 4}`],
  );
});

test("overlapping cues are highlighted once without losing later spans", () => {
  const sentenceEnd = text.indexOf(".") + 1;
  const ranges = highlightRanges(
    [
      { quote: text.slice(0, sentenceEnd), start: 0, end: sentenceEnd },
      { quote: "shocking", start: firstCue, end: firstCue + 8 },
      { quote: "plan", start: planStart, end: planStart + 4 },
      { quote: "Officials", start: officialsStart, end: officialsStart + 9 },
    ],
    text,
  );
  assert.deepEqual(
    ranges.map((range) => range.key),
    [`0-${sentenceEnd}`, `${officialsStart}-${officialsStart + 9}`],
  );
});

test("falls back to the anchor when occurrence spans are absent", () => {
  const ranges = highlightRanges([{ quote: "plan", start: planStart, end: planStart + 4 }], text);
  assert.deepEqual(ranges.map((range) => range.key), [`${planStart}-${planStart + 4}`]);
});

test("reports distinct findings and total occurrences separately", () => {
  const evidence = [{ quote: "plan", start: planStart, end: planStart + 4, occurrences: 2 }];
  assert.deepEqual(evidenceTotals({}, evidence), { findings: 1, occurrences: 2 });
  assert.deepEqual(
    evidenceTotals(
      { total_evidence_findings: 4, total_cue_occurrences: 9 },
      evidence,
    ),
    { findings: 4, occurrences: 9 },
  );
});

test("evidence keys identify a finding by its anchor span", () => {
  assert.equal(
    evidenceKey({ quote: "plan", start: planStart, end: planStart + 4 }),
    `${planStart}-${planStart + 4}`,
  );
});
