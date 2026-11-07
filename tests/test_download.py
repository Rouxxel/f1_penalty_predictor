import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.download import _should_include_pdf, merge_failures, slugify_event
from fia_ml.data.config import PipelineConfig


@pytest.fixture
def cfg():
    return PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml")


def test_slugify_event():
    assert slugify_event("Monaco Grand Prix") == "monaco_grand_prix"


def test_should_include_decision_pdf(cfg):
    title = "2019 Monaco Grand Prix - Decision - Car 88 (incident with car 99 in turn 12).pdf"
    assert _should_include_pdf(title, cfg) is True


def test_should_include_offence_pdf(cfg):
    title = "Offence - Car 5 - Pit lane speeding.pdf"
    assert _should_include_pdf(title, cfg) is True


def test_should_include_infringement_with_doc_prefix(cfg):
    title = "Doc 52 - Infringement - Car 18 - More than one change of direction.pdf"
    assert _should_include_pdf(title, cfg) is True


def test_should_include_slugified_infringement_filename(cfg):
    title = "2025_abu_dhabi_grand_prix_-_infringement_-_car_18_-_more_than_one_change_of_direction.pdf"
    assert _should_include_pdf(title, cfg) is True


def test_should_include_summons_anywhere_in_title(cfg):
    title = "2019 Monaco Grand Prix - Summons - Car 44 - Alleged impeding.pdf"
    assert _should_include_pdf(title, cfg) is True


def test_should_exclude_classification_pdf(cfg):
    title = "2019 Monaco Grand Prix - Final Race Classification.pdf"
    assert _should_include_pdf(title, cfg) is False


def test_should_not_match_partial_keyword(cfg):
    title = "Race Directors Note - Pre-Race Procedure.pdf"
    assert _should_include_pdf(title, cfg) is False


def test_merge_failures_removes_recovered_and_keeps_others():
    previous = [
        {"title": "a", "url": "u_a", "error": "HTTP 429"},
        {"title": "b", "url": "u_b", "error": "HTTP 404"},
        {"title": "c", "url": "u_c", "error": "HTTP 429"},
    ]
    current = [{"title": "c", "url": "u_c", "error": "HTTP 400"}]
    merged = merge_failures(previous, current, downloaded_urls={"u_a"})
    assert merged == [
        {"title": "b", "url": "u_b", "error": "HTTP 404"},
        {"title": "c", "url": "u_c", "error": "HTTP 400"},
    ]


def test_merge_failures_empty_when_all_recovered():
    previous = [{"title": "a", "url": "u_a", "error": "HTTP 429"}]
    assert merge_failures(previous, [], downloaded_urls={"u_a"}) == []


def test_config_has_all_seasons(cfg):
    years = [season.year for season in cfg.seasons]
    assert years
    assert len(years) == len(set(years))
    assert all(f"season-{season.year}-" in season.url for season in cfg.seasons)


class _FakeClient:
    def __init__(self, ok_urls):
        self.ok_urls = ok_urls

    def download_bytes(self, url, *, referer=None):
        if url not in self.ok_urls:
            raise RuntimeError(f"Failed to download {url}: HTTP 404")
        return b"%PDF-fake"


def test_normalize_pdf_path_fixes_amp_and_percent():
    from fia_ml.data.download import normalize_pdf_path

    assert normalize_pdf_path("/a/turns%208%20&amp;%209.pdf") == "/a/turns%208%20&%209.pdf"
    assert normalize_pdf_path("/a/within%20107%.pdf") == "/a/within%20107%25.pdf"
    assert normalize_pdf_path("/a/within%20107%25.pdf") == "/a/within%20107%25.pdf"


def test_adopt_manual_file_added_to_manifest_and_failure_resolved(tmp_path, monkeypatch):
    from fia_ml.data import download

    monkeypatch.setattr(download, "PROJECT_ROOT", tmp_path)
    raw_root = tmp_path / "2022"
    event_dir = raw_root / "saudi_arabian_grand_prix"
    event_dir.mkdir(parents=True)
    (event_dir / "2022 Saudi Arabian Grand Prix - Offence - Car 23 - x & y.pdf").write_bytes(b"x")
    failures = [
        {
            "title": "2022 Saudi Arabian Grand Prix - Offence - Car 23 - x &amp; y.pdf",
            "url": "https://fia/x",
            "error": "HTTP 404",
        }
    ]
    entries = {}
    adopted = download.adopt_manual_files(2022, raw_root, entries, failures)
    assert len(adopted) == 1
    assert adopted[0].url == "https://fia/x"
    assert adopted[0].event_slug == "saudi_arabian_grand_prix"
    assert adopted[0].local_path == str(
        Path("2022/saudi_arabian_grand_prix/2022 Saudi Arabian Grand Prix - Offence - Car 23 - x & y.pdf")
    )
    # idempotent: a second pass adopts nothing new
    assert download.adopt_manual_files(2022, raw_root, entries, failures) == []


def test_retry_failures_downloads_and_keeps_still_failing(tmp_path, monkeypatch, cfg):
    from fia_ml.data import download

    monkeypatch.setattr(download, "PROJECT_ROOT", tmp_path)
    raw_root = tmp_path / "2022"
    (raw_root / "saudi_arabian_grand_prix").mkdir(parents=True)
    failures = [
        {"title": "2022 Saudi Arabian Grand Prix - Offence - A.pdf", "url": "https://fia/ok", "error": "e"},
        {"title": "2022 Saudi Arabian Grand Prix - Offence - B.pdf", "url": "https://fia/bad", "error": "e"},
    ]
    entries = {}
    remaining = download.retry_failures(
        2022, raw_root, failures, entries, _FakeClient({"https://fia/ok"}), cfg
    )
    assert [r["url"] for r in remaining] == ["https://fia/bad"]
    assert len(entries) == 1
    assert (raw_root / "saudi_arabian_grand_prix" / "2022 Saudi Arabian Grand Prix - Offence - A.pdf").exists()


def test_url_variants_cover_percent_and_entity_forms():
    from fia_ml.data.download import url_variants

    base = "https://www.fia.com/sites/default/files/decision-document/"
    variants = url_variants(base + "a%20107%.pdf")
    assert variants[0] == base + "a%20107%25.pdf"
    assert base + "a%20107%.pdf" in variants
    assert url_variants(base + "x%20&amp;%209.pdf")[0] == base + "x%20&%209.pdf"


def test_download_file_falls_back_to_next_variant(tmp_path):
    from fia_ml.data.download import _download_file

    good = "https://www.fia.com/sites/default/files/decision-document/a%20107%25.pdf"

    class Client:
        def __init__(self):
            self.calls = []

        def download_bytes(self, url, *, referer=None):
            self.calls.append(url)
            if url != good:
                raise RuntimeError(f"Failed to download {url}: HTTP 400")
            return b"%PDF"

    client = Client()
    dest = tmp_path / "a.pdf"
    # Bare-% input: the normalised variant is tried first and succeeds.
    _download_file(good.replace("%25", "%"), dest, client, referer="r")
    assert dest.read_bytes() == b"%PDF"
    assert client.calls == [good]

    # A non-URL error (e.g. 403) is not retried with other variants.
    class Blocked:
        def download_bytes(self, url, *, referer=None):
            raise RuntimeError(f"Failed to download {url}: HTTP 403")

    with pytest.raises(RuntimeError, match="403"):
        _download_file(good, tmp_path / "b.pdf", Blocked(), referer="r")
