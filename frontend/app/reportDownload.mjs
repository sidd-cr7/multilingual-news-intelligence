export function reportFilename(contentDisposition, title) {
  const encodedName = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  const quotedName = contentDisposition.match(/filename="([^"]+)"/i)?.[1];
  const suppliedName = encodedName
    ? decodeURIComponent(encodedName.trim())
    : quotedName;
  const filename = suppliedName?.split(/[\\/]/).pop()?.trim();
  if (filename && filename.toLowerCase().endsWith(".pdf")) return filename;

  const safeTitle = title
    .replace(/[^A-Za-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
  return `news-report-${safeTitle || "untitled"}.pdf`;
}

export function downloadPdf(blob, filename, target) {
  const objectUrl = target.createObjectURL(blob);
  let link;
  try {
    link = target.createLink();
    link.href = objectUrl;
    link.download = filename;
    target.append(link);
    link.click();
  } finally {
    link?.remove();
    target.scheduleRevoke(() => target.revokeObjectURL(objectUrl));
  }
}

export function browserDownloadTarget() {
  return {
    createObjectURL: (blob) => URL.createObjectURL(blob),
    revokeObjectURL: (url) => URL.revokeObjectURL(url),
    createLink: () => document.createElement("a"),
    append: (link) => document.body.appendChild(link),
    scheduleRevoke: (callback) => window.setTimeout(callback, 1000),
  };
}
