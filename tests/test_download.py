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
