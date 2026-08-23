"""HTTP helpers for polite FIA website access."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

import requests

DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _scraper(cfg: Any) -> dict[str, Any]:
    return getattr(cfg, "scraper", {}) or {}


def _base_url(cfg: Any) -> str:
    return str(_scraper(cfg).get("fia_base_url", "https://www.fia.com")).rstrip("/")


def _browser_headers(cfg: Any, referer: str | None = None) -> dict[str, str]:
    scraper = _scraper(cfg)
    headers = {
        "User-Agent": scraper.get("user_agent", DEFAULT_BROWSER_USER_AGENT),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def _championship_url(cfg: Any) -> str:
    return f"{_base_url(cfg)}/documents/championships/fia-formula-one-world-championship-14"


def _warmup_steps(cfg: Any, target_url: str) -> list[tuple[str, str | None]]:
    base = _base_url(cfg)
    championship = _championship_url(cfg)
    steps: list[tuple[str, str | None]] = [
        (f"{base}/", None),
        (championship, f"{base}/"),
    ]
    if target_url.rstrip("/") != championship.rstrip("/"):
        steps.append((target_url, championship))
    return steps


def create_fia_session(cfg: Any) -> requests.Session:
    session = requests.Session()
    session.headers.update(_browser_headers(cfg))
    return session


def warmup_fia_session(session: requests.Session, cfg: Any, target_url: str) -> None:
    """Visit homepage and championship pages before document URLs, like browser navigation."""
    if not _scraper(cfg).get("warmup_enabled", True):
        return
    for url, referer in _warmup_steps(cfg, target_url):
        if urlparse(url).path.rstrip("/") == urlparse(target_url).path.rstrip("/"):
            continue
        _request_with_retries(session, cfg, url, referer=referer, expect_html=True)


def _request_with_retries(
    session: requests.Session,
    cfg: Any,
    url: str,
    *,
    referer: str | None = None,
    expect_html: bool = True,
) -> requests.Response:
    scraper = _scraper(cfg)
    retries = int(scraper.get("max_retries", 3))
    timeout = int(scraper.get("timeout_seconds", 60))
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = session.get(
                url,
                headers=_browser_headers(cfg, referer=referer),
                timeout=timeout,
            )
            if response.status_code == 403 and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            if expect_html and "maintenance" in response.text.lower() and "/documents/" in url:
                raise RuntimeError(
                    "FIA returned the maintenance/WAF page instead of document content. "
                    "Wait and retry, use a browser session export, or enable Playwright fetching."
                )
            return response
        except Exception as exc:  # noqa: BLE001 - retry loop
            last_error = exc
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def fetch_fia_html(
    session: requests.Session,
    cfg: Any,
    url: str,
    *,
    referer: str | None = None,
    warmed: bool = False,
) -> str:
    if not warmed and _scraper(cfg).get("warmup_enabled", True):
        warmup_fia_session(session, cfg, url)
        warmed = True
    response = _request_with_retries(session, cfg, url, referer=referer, expect_html=True)
    return response.text


def download_fia_file(
    session: requests.Session,
    cfg: Any,
    url: str,
    *,
    referer: str | None = None,
) -> bytes:
    response = _request_with_retries(session, cfg, url, referer=referer, expect_html=False)
    time.sleep(float(_scraper(cfg).get("rate_limit_seconds", 1.0)))
    return response.content
