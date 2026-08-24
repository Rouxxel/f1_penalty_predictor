"""FIA website access via requests or Playwright."""

from __future__ import annotations

import time
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class FiaClient(Protocol):
    def fetch_html(self, url: str, *, referer: str | None = None) -> str: ...

    def download_bytes(self, url: str, *, referer: str | None = None) -> bytes: ...

    def close(self) -> None: ...


def _scraper(cfg: Any) -> dict[str, Any]:
    return getattr(cfg, "scraper", {}) or {}


def _base_url(cfg: Any) -> str:
    return str(_scraper(cfg).get("fia_base_url", "https://www.fia.com")).rstrip("/")


def championship_url(cfg: Any) -> str:
    return f"{_base_url(cfg)}/documents/championships/fia-formula-one-world-championship-14"


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


def _warmup_steps(cfg: Any, target_url: str) -> list[str]:
    base = f"{_base_url(cfg)}/"
    championship = championship_url(cfg)
    steps = [base, championship]
    if target_url.rstrip("/") not in {base.rstrip("/"), championship.rstrip("/")}:
        steps.append(target_url)
    return steps


def _rate_limit(cfg: Any) -> None:
    time.sleep(float(_scraper(cfg).get("rate_limit_seconds", 3.0)))


def _is_blocked_html(html: str, url: str) -> bool:
    lowered = html.lower()
    if "/documents/" not in url:
        return False
    if "facebook.com/fia" in lowered:
        return True
    if "maintenance" in lowered and 'value="/documents/championships/fia-formula-one-world-championship-14/season/' not in html:
        return True
    return False


class RequestsFiaClient:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(_browser_headers(cfg))
        self._visited: set[str] = set()

    def _path_key(self, url: str) -> str:
        return urlparse(url).path.rstrip("/")

    def _warmup(self, target_url: str) -> None:
        if not _scraper(self.cfg).get("warmup_enabled", True):
            return
        target_key = self._path_key(target_url)
        if target_key in self._visited:
            return
        for step_url in _warmup_steps(self.cfg, target_url):
            step_key = self._path_key(step_url)
            if step_key == target_key or step_key in self._visited:
                continue
            self._fetch_html_direct(step_url)
            self._visited.add(step_key)
        self._visited.add(target_key)

    def _fetch_html_direct(self, url: str, *, referer: str | None = None) -> str:
        scraper = _scraper(self.cfg)
        retries = int(scraper.get("max_retries", 3))
        timeout = int(scraper.get("timeout_seconds", 60))
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    headers=_browser_headers(self.cfg, referer=referer),
                    timeout=timeout,
                )
                if response.status_code == 403 and attempt < retries - 1:
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                if _is_blocked_html(response.text, url):
                    raise RuntimeError(
                        "FIA returned the maintenance/WAF page instead of document content."
                    )
                _rate_limit(self.cfg)
                return response.text
            except Exception as exc:  # noqa: BLE001 - retry loop
                last_error = exc
                time.sleep(2**attempt)

        raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error

    def fetch_html(self, url: str, *, referer: str | None = None) -> str:
        self._warmup(url)
        return self._fetch_html_direct(url, referer=referer)

    def download_bytes(self, url: str, *, referer: str | None = None) -> bytes:
        response = self.session.get(
            url,
            headers=_browser_headers(self.cfg, referer=referer),
            timeout=int(_scraper(self.cfg).get("timeout_seconds", 60)),
        )
        response.raise_for_status()
        _rate_limit(self.cfg)
        return response.content

    def close(self) -> None:
        self.session.close()


class PlaywrightFiaClient:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._visited: set[str] = set()

    def _path_key(self, url: str) -> str:
        return urlparse(url).path.rstrip("/")

    def _ensure_started(self) -> None:
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for scraper.fetch_backend=playwright. "
                "Install with: pip install playwright && playwright install chromium"
            ) from exc

        scraper = _scraper(self.cfg)
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=bool(scraper.get("playwright_headless", True)),
        )
        self._context = self._browser.new_context(
            user_agent=scraper.get("user_agent", DEFAULT_BROWSER_USER_AGENT),
            locale="en-US",
        )
        self._page = self._context.new_page()
        timeout_ms = int(scraper.get("timeout_seconds", 60)) * 1000
        self._page.set_default_timeout(timeout_ms)

    def _warmup(self, target_url: str) -> None:
        if not _scraper(self.cfg).get("warmup_enabled", True):
            return
        target_key = self._path_key(target_url)
        if target_key in self._visited:
            return
        for step_url in _warmup_steps(self.cfg, target_url):
            step_key = self._path_key(step_url)
            if step_key == target_key or step_key in self._visited:
                continue
            self._fetch_html_direct(step_url)
            self._visited.add(step_key)
        self._visited.add(target_key)

    def _fetch_html_direct(self, url: str, *, referer: str | None = None) -> str:
        self._ensure_started()
        assert self._page is not None

        scraper = _scraper(self.cfg)
        retries = int(scraper.get("max_retries", 3))
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                self._page.goto(url, wait_until="domcontentloaded")
                html = self._page.content()
                if _is_blocked_html(html, url):
                    raise RuntimeError(
                        "FIA returned the maintenance/WAF page instead of document content."
                    )
                _rate_limit(self.cfg)
                return html
            except Exception as exc:  # noqa: BLE001 - retry loop
                last_error = exc
                time.sleep(2**attempt)

        raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error

    def fetch_html(self, url: str, *, referer: str | None = None) -> str:
        self._warmup(url)
        return self._fetch_html_direct(url, referer=referer)

    def download_bytes(self, url: str, *, referer: str | None = None) -> bytes:
        self._ensure_started()
        assert self._context is not None
        response = self._context.request.get(
            url,
            headers=_browser_headers(self.cfg, referer=referer),
        )
        if not response.ok:
            raise RuntimeError(f"Failed to download {url}: HTTP {response.status}")
        _rate_limit(self.cfg)
        return response.body()

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()


def create_fia_client(cfg: Any) -> FiaClient:
    backend = str(_scraper(cfg).get("fetch_backend", "playwright")).lower()
    if backend == "playwright":
        return PlaywrightFiaClient(cfg)
    if backend == "requests":
        return RequestsFiaClient(cfg)
    raise ValueError(f"Unknown scraper.fetch_backend: {backend!r}")


# Backwards-compatible helpers used by older imports.
def _championship_url(cfg: Any) -> str:
    return championship_url(cfg)


def create_fia_session(cfg: Any) -> requests.Session:
    return RequestsFiaClient(cfg).session


def warmup_fia_session(session: requests.Session, cfg: Any, target_url: str) -> None:
    client = RequestsFiaClient(cfg)
    client.session = session
    client._warmup(target_url)


def fetch_fia_html(
    session: requests.Session,
    cfg: Any,
    url: str,
    *,
    referer: str | None = None,
    warmed: bool = False,
) -> str:
    client = RequestsFiaClient(cfg)
    client.session = session
    if warmed:
        client._visited.add(client._path_key(url))
    return client.fetch_html(url, referer=referer)


def download_fia_file(
    session: requests.Session,
    cfg: Any,
    url: str,
    *,
    referer: str | None = None,
) -> bytes:
    client = RequestsFiaClient(cfg)
    client.session = session
    return client.download_bytes(url, referer=referer)
