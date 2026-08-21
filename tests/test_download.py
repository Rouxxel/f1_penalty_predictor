import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.download import _should_include_pdf, slugify_event
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


def test_should_include_summons_anywhere_in_title(cfg):
    title = "2019 Monaco Grand Prix - Summons - Car 44 - Alleged impeding.pdf"
    assert _should_include_pdf(title, cfg) is True


def test_should_exclude_classification_pdf(cfg):
    title = "2019 Monaco Grand Prix - Final Race Classification.pdf"
    assert _should_include_pdf(title, cfg) is False


def test_should_not_match_partial_keyword(cfg):
    title = "Race Directors Note - Pre-Race Procedure.pdf"
    assert _should_include_pdf(title, cfg) is False


def test_config_has_all_seasons(cfg):
    assert len(cfg.seasons) == 7
    assert {season.year for season in cfg.seasons} == {2019, 2020, 2021, 2022, 2023, 2024, 2025}
