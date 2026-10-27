import assert from "node:assert/strict";
import { test } from "node:test";
import { downloadPdf, reportFilename } from "../app/reportDownload.mjs";

test("uses the API filename when supplied", () => {
  assert.equal(
    reportFilename('attachment; filename="news-report-Delhi-Flood.pdf"', "ignored"),
    "news-report-Delhi-Flood.pdf",
  );
});

test("creates a safe fallback filename", () => {
  assert.equal(
    reportFilename("", "తెలుగు / Floods"),
    "news-report-Floods.pdf",
  );
});

test("triggers and cleans up a browser download", () => {
  const events = [];
  const link = {
    href: "",
    download: "",
    click: () => events.push("click"),
    remove: () => events.push("remove"),
  };
  const blob = new Blob(["%PDF"]);

  downloadPdf(blob, "research.pdf", {
    createObjectURL: (received) => {
      assert.equal(received, blob);
      events.push("create");
      return "blob:report";
    },
    revokeObjectURL: (url) => events.push(`revoke:${url}`),
    createLink: () => link,
    append: (received) => {
      assert.equal(received, link);
      events.push("append");
    },
    scheduleRevoke: (callback) => callback(),
  });

  assert.equal(link.href, "blob:report");
  assert.equal(link.download, "research.pdf");
  assert.deepEqual(events, ["create", "append", "click", "remove", "revoke:blob:report"]);
});

test("prefers the RFC 5987 unicode filename when provided", () => {
  const filename = reportFilename(
    "attachment; filename=\"news-report-untitled.pdf\"; filename*=UTF-8''news-report-%E0%A4%A6%E0%A4%BF%E0%A4%B2%E0%A5%8D%E0%A4%B2%E0%A5%80.pdf",
    "ignored",
  );
  assert.equal(filename, "news-report-दिल्ली.pdf");
});

test("rejects a traversal filename and falls back to the title", () => {
  assert.equal(
    reportFilename('attachment; filename="../../etc/passwd.pdf"', "Delhi Flood"),
    "passwd.pdf",
  );
});
