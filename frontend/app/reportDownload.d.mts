export interface DownloadLink {
  href: string;
  download: string;
  click(): void;
  remove(): void;
}

export interface DownloadTarget {
  createObjectURL(blob: Blob): string;
  revokeObjectURL(url: string): void;
  createLink(): DownloadLink;
  append(link: DownloadLink): void;
  scheduleRevoke(callback: () => void): void;
}

export function reportFilename(contentDisposition: string, title: string): string;
export function downloadPdf(blob: Blob, filename: string, target: DownloadTarget): void;
export function browserDownloadTarget(): DownloadTarget;
