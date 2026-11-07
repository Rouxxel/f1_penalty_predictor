"""FIA season URL â†’ event pages â†’ filtered PDF download."""

from __future__ import annotations

import hashlib
import html as html_lib
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from fia_ml.data.config import PipelineConfig
from fia_ml.data.fia_http import FiaClient, championship_url, create_fia_client
from fia_ml.paths import PROJECT_ROOT, ensure_dir
from fia_ml.utils import secure_file_io as sio


@dataclass
class DocumentEntry:
    document_id: str
    url: str
    local_path: str
    event: str
    event_slug: str
    title: str
    sha256: str
    season: int


def slugify_event(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def discover_event_urls(
    season_url: str,
    cfg: PipelineConfig,
    client: FiaClient,
) -> list[tuple[str, str]]:
    html = client.fetch_html(season_url, referer=championship_url(cfg))
    pattern = re.compile(
        r'value="(/documents/championships/fia-formula-one-world-championship-14/season/[^"]+/event/[^"]+)"'
    )
    events: list[tuple[str, str]] = []
    base = cfg.scraper.get("fia_base_url", "https://www.fia.com")
    seen: set[str] = set()
    for match in pattern.finditer(html):
        rel = match.group(1)
        if rel in seen:
            continue
        seen.add(rel)
        name = urllib.parse.unquote(rel.rsplit("/", 1)[-1])
        events.append((name, base + rel))
    return events


def _normalize_title_for_matching(title: str) -> str:
    """Normalize PDF filenames so keyword matching works across FIA naming styles."""
    stem = Path(title).stem if title.lower().endswith(".pdf") else title
    normalized = re.sub(r"[_\-.]+", " ", stem.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(keyword.lower())}\b", re.IGNORECASE)


def _title_has_keyword(title: str, keyword: str) -> bool:
    normalized = _normalize_title_for_matching(title)
    return _keyword_pattern(keyword).search(normalized) is not None


def _title_matches_include_patterns(title: str, patterns: list[str]) -> bool:
    return any(_title_has_keyword(title, pattern) for pattern in patterns)


def _should_include_pdf(title: str, cfg: PipelineConfig) -> bool:
    if not _title_matches_include_patterns(title, cfg.document_include_patterns):
        return False

    lowered = title.lower()
    for pattern in cfg.document_exclude_patterns:
        if pattern.lower() not in lowered:
            continue
        if pattern.lower() == "provisional":
            continue
        if "correction" in lowered and _title_matches_include_patterns(
            title, cfg.document_include_patterns
        ):
            continue
        return False
    return True


def clean_title(title: str) -> str:
    """Decode HTML entities left in a PDF title (e.g. ``&amp;`` -> ``&``)."""
    return html_lib.unescape(title)


def normalize_pdf_path(path: str) -> str:
    """Return a valid URL path for an href scraped from FIA HTML.

    Fixes two known failure causes: HTML entities (``&amp;`` -> ``&``, which gave
    HTTP 404) and bare ``%`` characters such as ``107%.pdf`` (encoded to ``%25``,
    which gave HTTP 400). Already-encoded paths round-trip unchanged.
    """
    decoded = urllib.parse.unquote(html_lib.unescape(path))
    return urllib.parse.quote(decoded, safe="/&()',!$@=+;:~-_.")


def normalize_failure(item: dict[str, str]) -> dict[str, str]:
    """Repair URL/title of a failure record written by an older crawler version."""
    parts = urllib.parse.urlsplit(item["url"])
    fixed = dict(item)
    fixed["url"] = urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, normalize_pdf_path(parts.path), parts.query, "")
    )
    fixed["title"] = clean_title(item["title"])
    if fixed["url"] != item["url"]:
        fixed["error"] = item.get("error", "").replace(item["url"], fixed["url"])
    return fixed


def discover_pdfs_for_event(
    event_url: str,
    event_name: str,
    cfg: PipelineConfig,
    client: FiaClient,
    *,
    season_url: str,
) -> list[tuple[str, str]]:
    html = client.fetch_html(event_url, referer=season_url)
    base = cfg.scraper.get("fia_base_url", "https://www.fia.com")
    pdfs: list[tuple[str, str]] = []
    for match in re.finditer(
        r'href="((?:/sites/default/files|/system/files)/decision-document/([^"]+\.pdf))"',
        html,
        re.IGNORECASE,
    ):
        rel_path, filename = match.group(1), match.group(2)
        title = clean_title(urllib.parse.unquote(html_lib.unescape(filename)))
        if _should_include_pdf(title, cfg):
            pdfs.append((title, base + normalize_pdf_path(rel_path)))
    return pdfs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(sio.read_bytes(path))
    return digest.hexdigest()


def _download_file(
    url: str,
    dest: Path,
    client: FiaClient,
    *,
    referer: str,
) -> None:
    """Download ``url``; on HTTP 400/404 retry alternate spellings of the same URL."""
    last_exc: RuntimeError | None = None
    for candidate in url_variants(url):
        try:
            data = client.download_bytes(candidate, referer=referer)
        except RuntimeError as exc:
            last_exc = exc
            if not _is_url_error(exc):
                raise
            continue
        sio.write_bytes(dest, data)
        return
    assert last_exc is not None
    raise last_exc


def _is_url_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return "HTTP 400" in message or "HTTP 404" in message


def url_variants(url: str) -> list[str]:
    """Alternate spellings of a FIA PDF URL, most likely to work first.

    1. Normalised (entities decoded, ``%`` encoded as ``%25``).
    2. As given.
    3. Fully decoded path with a bare ``%`` and ``&`` left literal.
    """
    parts = urllib.parse.urlsplit(url)

    def rebuild(path: str) -> str:
        return urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, path, parts.query, "")
        )

    normalized = rebuild(normalize_pdf_path(parts.path))
    literal = rebuild(
        urllib.parse.unquote(html_lib.unescape(parts.path)).replace(" ", "%20")
    )
    return list(dict.fromkeys([normalized, url, literal]))


def make_document_id(season: int, event_slug: str, title: str) -> str:
    digest = hashlib.sha1(f"{season}|{event_slug}|{title}".encode()).hexdigest()[:12]
    return f"{season}_{event_slug}_{digest}"


def merge_failures(
    previous: list[dict[str, str]],
    current: list[dict[str, str]],
    downloaded_urls: set[str],
) -> list[dict[str, str]]:
    """Drop previously failed URLs that are now downloaded; current failures win."""
    current_urls = {item["url"] for item in current}
    kept = [
        item
        for item in previous
        if item["url"] not in downloaded_urls and item["url"] not in current_urls
    ]
    return kept + current



def _title_key(name: str) -> str:
    """Loose key so a manually saved file still matches its failure record."""
    stem = name[:-4] if name.lower().endswith(".pdf") else name
    return re.sub(r"[^a-z0-9]+", "", html_lib.unescape(stem).lower())


def _build_entry(
    season: int,
    event_slug: str,
    event_name: str,
    title: str,
    url: str,
    local_path: Path,
) -> DocumentEntry:
    return DocumentEntry(
        document_id=make_document_id(season, event_slug, title),
        url=url,
        local_path=str(local_path.relative_to(PROJECT_ROOT)),
        event=event_name,
        event_slug=event_slug,
        title=title,
        sha256=_sha256_file(local_path),
        season=season,
    )


def _event_name_for(slug: str, entries: dict[str, DocumentEntry]) -> str:
    for entry in entries.values():
        if entry.event_slug == slug:
            return entry.event
    return slug.replace("_", " ").title()


def adopt_manual_files(
    season: int,
    raw_root: Path,
    entries: dict[str, DocumentEntry],
    failures: list[dict[str, str]],
) -> list[DocumentEntry]:
    """Register PDFs found under ``raw_root/<event>/`` that are not in the manifest.

    This covers files downloaded by hand after an automatic download failed.
    The URL is taken from a matching failure record when there is one.
    """
    known_paths = {
        (PROJECT_ROOT / e.local_path).resolve() for e in entries.values()
    }
    failure_by_key = {_title_key(f["title"]): f for f in failures}
    adopted: list[DocumentEntry] = []
    for event_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for pdf in sorted(event_dir.glob("*.pdf")):
            if pdf.resolve() in known_paths:
                continue
            match = failure_by_key.get(_title_key(pdf.name))
            entry = _build_entry(
                season,
                event_dir.name,
                _event_name_for(event_dir.name, entries),
                pdf.name,
                match["url"] if match else "",
                pdf,
            )
            entries[entry.document_id] = entry
            adopted.append(entry)
    return adopted


def _infer_event_slug(failure: dict[str, str], raw_root: Path) -> str | None:
    """Event folder for a failure: stored slug, else longest folder name inside the title."""
    if failure.get("event_slug"):
        return failure["event_slug"]
    title_slug = slugify_event(failure["title"])
    candidates = [
        p.name for p in raw_root.iterdir() if p.is_dir() and p.name in title_slug
    ]
    return max(candidates, key=len) if candidates else None


def retry_failures(
    season: int,
    raw_root: Path,
    failures: list[dict[str, str]],
    entries: dict[str, DocumentEntry],
    client: FiaClient,
    cfg: PipelineConfig,
) -> list[dict[str, str]]:
    """Retry each recorded failure directly from its URL; return those still failing."""
    remaining: list[dict[str, str]] = []
    referer = cfg.season_url
    for item in failures:
        slug = _infer_event_slug(item, raw_root)
        if slug is None:
            remaining.append(item)  # left for the season-page scan to resolve
            continue
        local_path = ensure_dir(raw_root / slug) / item["title"].replace("/", "-")
        try:
            if not local_path.exists():
                _download_file(item["url"], local_path, client, referer=referer)
        except RuntimeError as exc:
            remaining.append({**item, "event_slug": slug, "error": str(exc)})
            continue
        entry = _build_entry(
            season, slug, _event_name_for(slug, entries), item["title"], item["url"], local_path
        )
        entries[entry.document_id] = entry
    return remaining


def download_season(cfg: PipelineConfig) -> list[DocumentEntry]:
    """Download a season's stewarding PDFs.

    Order of work:
      1. Load the manifest and any ``download_failures.json``.
      2. Adopt PDFs added by hand into the event folders (added to the manifest).
      3. Retry the remaining failures directly from their recorded URLs.
      4. Only if failures remain (or there were none recorded), scan the FIA
         season pages for everything missing.
      5. Rewrite the manifest and ``download_failures.json``.
    """
    season = cfg.season
    raw_root = ensure_dir(cfg.path("raw_fia") / str(season))
    manifest_path = raw_root / "manifest.json"
    failures_path = raw_root / "download_failures.json"

    entries: dict[str, DocumentEntry] = {}
    if manifest_path.exists():
        for item in sio.read_json(manifest_path):
            entry = DocumentEntry(**item)
            if (PROJECT_ROOT / entry.local_path).exists():
                entries[entry.document_id] = entry

    previous_failures = (
        [normalize_failure(item) for item in sio.read_json(failures_path)]
        if failures_path.exists()
        else []
    )

    adopt_manual_files(season, raw_root, entries, previous_failures)
    adopted_keys = {_title_key(e.title) for e in entries.values()}
    pending = [f for f in previous_failures if _title_key(f["title"]) not in adopted_keys]

    failures: list[dict[str, str]] = []
    scan_needed = not previous_failures  # no failure history: normal full scan
    if pending or scan_needed:
        client = create_fia_client(cfg)
        try:
            if pending:
                pending = retry_failures(season, raw_root, pending, entries, client, cfg)
                scan_needed = bool(pending)
            if scan_needed:
                failures = _scan_season(cfg, raw_root, entries, client)
        finally:
            client.close()

    downloaded_keys = {_title_key(e.title) for e in entries.values()}
    final_failures = [
        f
        for f in merge_failures(pending, failures, {e.url for e in entries.values()})
        if _title_key(f["title"]) not in downloaded_keys
    ]

    result = list(entries.values())
    sio.write_json(manifest_path, [entry.__dict__ for entry in result])
    if final_failures or failures_path.exists():
        sio.write_json(failures_path, final_failures)
    return result


def _scan_season(
    cfg: PipelineConfig,
    raw_root: Path,
    entries: dict[str, DocumentEntry],
    client: FiaClient,
) -> list[dict[str, str]]:
    """Crawl the season page and event pages; download anything not yet on disk."""
    season = cfg.season
    failures: list[dict[str, str]] = []
    event_urls = discover_event_urls(cfg.season_url, cfg, client)
    if not event_urls:
        raise RuntimeError(f"No event URLs found at {cfg.season_url}")

    for event_name, event_url in event_urls:
        event_slug = slugify_event(event_name)
        event_dir = ensure_dir(raw_root / event_slug)
        pdfs = discover_pdfs_for_event(
            event_url, event_name, cfg, client, season_url=cfg.season_url
        )
        for title, url in pdfs:
            local_path = event_dir / title.replace("/", "-")
            document_id = make_document_id(season, event_slug, title)

            prior = entries.get(document_id)
            if prior and local_path.exists() and prior.sha256 == _sha256_file(local_path):
                continue

            if not local_path.exists():
                try:
                    _download_file(url, local_path, client, referer=event_url)
                except RuntimeError as exc:
                    failures.append(
                        {
                            "title": title,
                            "url": url,
                            "event": event_name,
                            "event_slug": event_slug,
                            "error": str(exc),
                        }
                    )
                    continue

            entries[document_id] = _build_entry(
                season, event_slug, event_name, title, url, local_path
            )
    return failures
