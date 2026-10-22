import ipaddress
import socket
from urllib.parse import urlparse
import httpx
import trafilatura
from app.config import Settings
from app.services.errors import IngestionError


def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise IngestionError("Provide a valid HTTP or HTTPS URL.")
    if parsed.username or parsed.password:
        raise IngestionError("URLs with embedded credentials are not accepted.")
    try:
        addresses = {entry[4][0] for entry in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
    except socket.gaierror as error:
        raise IngestionError("The website hostname could not be resolved.") from error
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise IngestionError("Private, local, and reserved network addresses are not allowed.", 403)


async def fetch_article(url: str, settings: Settings) -> tuple[str, str | None, str, list[str]]:
    validate_public_url(url)
    headers = {"User-Agent": "MultilingualNewsIntelligence/0.1 (research article extraction)"}
    timeout = httpx.Timeout(settings.http_timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, headers=headers) as client:
            current_url = url
            for _ in range(settings.max_redirects + 1):
                validate_public_url(current_url)  # Re-resolve every redirect to mitigate DNS rebinding.
                async with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise IngestionError("The website returned an invalid redirect.")
                        current_url = str(response.url.join(location))
                        continue
                    if response.status_code >= 400:
                        raise IngestionError(f"The website could not be accessed (HTTP {response.status_code}).", 422)
                    content_type = response.headers.get("content-type", "").lower()
                    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                        raise IngestionError("The URL did not return an HTML page.")
                    chunks, total = [], 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > settings.max_url_response_bytes:
                            raise IngestionError("The webpage exceeds the configured response-size limit.", 413)
                        chunks.append(chunk)
                    html = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
                    text = trafilatura.extract(html, include_comments=False, include_tables=False)
                    metadata = trafilatura.extract_metadata(html)
                    title = metadata.title if metadata else None
                    if not text or len(text.strip()) < settings.min_text_chars:
                        raise IngestionError("Readable article text could not be extracted. The page may be blocked, paywalled, or not an article.")
                    return text, title, current_url, []
            raise IngestionError("The URL exceeded the redirect limit.")
    except httpx.TimeoutException as error:
        raise IngestionError("The website took too long to respond.", 504) from error
    except httpx.HTTPError as error:
        raise IngestionError("The website could not be retrieved.", 422) from error
