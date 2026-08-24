"""FIA season URL → event pages → filtered PDF download."""

from __future__ import annotations

import hashlib
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
        title = urllib.parse.unquote(filename)
        if _should_include_pdf(title, cfg):
            pdfs.append((title, base + rel_path.replace(" ", "%20")))
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
    data = client.download_bytes(url, referer=referer)
    sio.write_bytes(dest, data)


def make_document_id(season: int, event_slug: str, title: str) -> str:
    digest = hashlib.sha1(f"{season}|{event_slug}|{title}".encode()).hexdigest()[:12]
    return f"{season}_{event_slug}_{digest}"


def download_season(cfg: PipelineConfig) -> list[DocumentEntry]:
    season = cfg.season
    raw_root = ensure_dir(cfg.path("raw_fia") / str(season))
    manifest_path = raw_root / "manifest.json"

    existing: dict[str, DocumentEntry] = {}
    if manifest_path.exists():
        for item in sio.read_json(manifest_path):
            existing[item["document_id"]] = DocumentEntry(**item)

    entries: list[DocumentEntry] = []
    failures: list[dict[str, str]] = []
    client = create_fia_client(cfg)
    try:
        event_urls = discover_event_urls(cfg.season_url, cfg, client)
        if not event_urls:
            raise RuntimeError(f"No event URLs found at {cfg.season_url}")

        for event_name, event_url in event_urls:
            event_slug = slugify_event(event_name)
            event_dir = ensure_dir(raw_root / event_slug)
            pdfs = discover_pdfs_for_event(
                event_url,
                event_name,
                cfg,
                client,
                season_url=cfg.season_url,
            )

            for title, url in pdfs:
                safe_name = title.replace("/", "-")
                local_path = event_dir / safe_name
                document_id = make_document_id(season, event_slug, title)

                if local_path.exists() and document_id in existing:
                    prior = existing[document_id]
                    current_hash = _sha256_file(local_path)
                    if prior.sha256 == current_hash:
                        entries.append(prior)
                        continue

                if not local_path.exists():
                    try:
                        _download_file(url, local_path, client, referer=event_url)
                    except RuntimeError as exc:
                        failures.append({"title": title, "url": url, "error": str(exc)})
                        continue

                sha256 = _sha256_file(local_path)
                entry = DocumentEntry(
                    document_id=document_id,
                    url=url,
                    local_path=str(local_path.relative_to(PROJECT_ROOT)),
                    event=event_name,
                    event_slug=event_slug,
                    title=title,
                    sha256=sha256,
                    season=season,
                )
                entries.append(entry)
    finally:
        client.close()

    sio.write_json(manifest_path, [entry.__dict__ for entry in entries])
    if failures:
        sio.write_json(raw_root / "download_failures.json", failures)
    return entries
