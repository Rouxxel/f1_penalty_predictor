"""Tests for clean.py / fia_ml.cleanup. All run against a temporary fake project root."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml import cleanup  # noqa: E402

FILES = [
    # dataset
    "data/raw/fia/2020/british_grand_prix/doc.pdf",
    "data/raw/fia/2020/manifest.json",
    "data/raw/fia/2021/manifest.json",
    "data/raw/fia/.gitkeep",
    "data/interim/extracted_documents/2020/a.json",
    "data/interim/extracted_documents/2021/b.json",
    "data/interim/extracted_documents/.gitkeep",
    "data/interim/enrichment_meta/2020.json",
    "dataset/csv/raw_incidents_2020.csv",
    "dataset/csv/raw_incidents_2020.meta.json",
    "dataset/csv/processed_2020.csv",
    "dataset/csv/processed_2021.csv",
    "dataset/csv/review_queue_2020.csv",
    "reports/tables/data_quality_2020.json",
    "reports/tables/enrichment_report_2020.json",
    "reports/model_reports/enrichment_improvement_2026-11-08.md",
    "data/raw/race_data/ergast/2020/2020.json",
    "data/raw/race_data/fastf1_cache/fastf1_http_cache.sqlite",
    "data/raw/race_data/openf1_cache/2023/sessions/x.json",
    # v1
    "data/processed/incidents.parquet",
    "data/processed/train.parquet",
    "ml_models/preprocessor.joblib",
    "ml_models/xgboost/model.json",
    "ml_models/baseline/model.pkl",
    "ml_models/baseline/__init__.py",
    "reports/figures/confusion_matrix_val.png",
    "reports/model_reports/v1_training_report_2026-11-08.md",
    # v2
    "data/processed/train_v2.parquet",
    "ml_models/xgboost_v2/model.json",
    "reports/ablation_results.json",
    # nlp
    "data/processed/nlp_train.jsonl",
    "ml_models/nlp/checkpoints/checkpoint-10/weights.bin",
    "ml_models/nlp/metrics.json",
    "ml_models/nlp/__init__.py",
    "reports/model_reports/nlp_training_report_2026-11-08.md",
    # normative
    "data/processed/incidents_with_normative.parquet",
    "ml_models/normative/predictions.json",
    "reports/normative/deviation_summary_2026-08-25.md",
    "reports/normative/figures/x.png",
    # never deleted
    "configs/data.yaml",
    "src/fia_ml/__init__.py",
    "data/reference/seasons.json",
    "documentation/spec.md",
]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    for rel in FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    # minimal configs so OutputPaths reads real values
    (tmp_path / "configs" / "data.yaml").write_text(
        "paths:\n  ergast_cache: data/raw/race_data/ergast\n  fastf1_cache: data/raw/race_data/fastf1_cache\n",
        encoding="utf-8",
    )
    return tmp_path


def _exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def _run(root: Path, monkeypatch: pytest.MonkeyPatch, *argv: str, answer: str | None = None) -> int:
    monkeypatch.setattr(cleanup, "PROJECT_ROOT", root)
    return cleanup.main(list(argv), confirm_fn=(lambda _msg: answer) if answer is not None else None)


def test_bare_invocation_deletes_nothing(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(project, monkeypatch) == 2
    assert all(_exists(project, rel) for rel in FILES)


def test_dry_run_deletes_nothing(project: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    assert _run(project, monkeypatch, "--dry-run", "--purge") == 0
    assert all(_exists(project, rel) for rel in FILES)
    assert "Dry run: nothing deleted." in capsys.readouterr().out


def test_declined_confirmation_deletes_nothing(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(project, monkeypatch, "--run", "--only", "v1", answer="n") == 1
    assert _exists(project, "ml_models/xgboost/model.json")


def test_only_v1_deletes_v1_and_keeps_placeholders(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(project, monkeypatch, "--run", "--only", "v1", "--yes") == 0
    for rel in (
        "data/processed/incidents.parquet",
        "ml_models/preprocessor.joblib",
        "ml_models/xgboost/model.json",
        "ml_models/baseline/model.pkl",
        "reports/figures/confusion_matrix_val.png",
    ):
        assert not _exists(project, rel), rel
    assert _exists(project, "ml_models/baseline/__init__.py")
    assert _exists(project, "ml_models/xgboost")  # structural folder kept
    assert _exists(project, "ml_models/xgboost_v2/model.json")  # v2 untouched
    assert _exists(project, "dataset/csv/processed_2020.csv")  # dataset untouched


def test_from_v1_cleans_downstream_not_dataset(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(project, monkeypatch, "--run", "--from", "v1", "--yes") == 0
    for rel in (
        "ml_models/xgboost/model.json",
        "ml_models/xgboost_v2/model.json",
        "ml_models/nlp/metrics.json",
        "ml_models/nlp/checkpoints/checkpoint-10/weights.bin",
        "data/processed/incidents_with_normative.parquet",
        "reports/normative/deviation_summary_2026-08-25.md",
    ):
        assert not _exists(project, rel), rel
    assert not _exists(project, "ml_models/nlp/checkpoints")  # emptied non-structural dir removed
    assert _exists(project, "ml_models/nlp/__init__.py")
    assert _exists(project, "dataset/csv/processed_2020.csv")


def test_dataset_season_filter_and_pdfs_kept_by_default(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(project, monkeypatch, "--run", "--only", "dataset", "--season", "2020", "--yes") == 0
    for rel in (
        "data/interim/extracted_documents/2020/a.json",
        "dataset/csv/raw_incidents_2020.csv",
        "dataset/csv/processed_2020.csv",
        "reports/tables/data_quality_2020.json",
        "data/interim/enrichment_meta/2020.json",
    ):
        assert not _exists(project, rel), rel
    # other season, PDFs, caches, season-spanning report and model outputs survive
    assert _exists(project, "dataset/csv/processed_2021.csv")
    assert _exists(project, "data/interim/extracted_documents/2021/b.json")
    assert _exists(project, "data/raw/fia/2020/british_grand_prix/doc.pdf")
    assert _exists(project, "data/raw/race_data/ergast/2020/2020.json")
    assert _exists(project, "reports/model_reports/enrichment_improvement_2026-11-08.md")
    assert _exists(project, "ml_models/xgboost/model.json")


def test_dataset_stage_only_removes_that_stage(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(project, monkeypatch, "--run", "--dataset-stage", "validate", "--yes") == 0
    assert not _exists(project, "dataset/csv/processed_2020.csv")
    assert not _exists(project, "dataset/csv/review_queue_2020.csv")
    assert _exists(project, "dataset/csv/raw_incidents_2020.csv")
    assert _exists(project, "data/interim/extracted_documents/2020/a.json")


def test_include_pdfs_requires_typing_yes(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(project, monkeypatch, "--run", "--only", "dataset", "--include-pdfs", answer="y") == 1
    assert _exists(project, "data/raw/fia/2020/british_grand_prix/doc.pdf")
    assert _run(project, monkeypatch, "--run", "--only", "dataset", "--include-pdfs", answer="yes") == 0
    assert not _exists(project, "data/raw/fia/2020/british_grand_prix/doc.pdf")
    assert not _exists(project, "data/raw/fia/2020")  # season folder pruned
    assert _exists(project, "data/raw/fia/.gitkeep")
    assert _exists(project, "data/raw/race_data/ergast/2020/2020.json")  # caches still opt-in


def test_purge_removes_everything_generated_but_never_inputs(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(project, monkeypatch, "--run", "--purge", "--yes") == 0
    keep = {
        "configs/data.yaml",
        "src/fia_ml/__init__.py",
        "data/reference/seasons.json",
        "documentation/spec.md",
        "data/raw/fia/.gitkeep",
        "data/interim/extracted_documents/.gitkeep",
        "ml_models/baseline/__init__.py",
        "ml_models/nlp/__init__.py",
    }
    for rel in FILES:
        assert _exists(project, rel) == (rel in keep), rel


def test_misconfigured_path_outside_outputs_is_refused(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (project / "configs" / "normative.yaml").write_text(
        "paths:\n  output: configs/data.yaml\n  models_dir: src\n  reports_dir: data/reference\n",
        encoding="utf-8",
    )
    assert _run(project, monkeypatch, "--run", "--only", "normative", "--yes") == 0
    assert _exists(project, "configs/data.yaml")
    assert _exists(project, "src/fia_ml/__init__.py")
    assert _exists(project, "data/reference/seasons.json")


def test_season_with_model_phase_is_rejected(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _run(project, monkeypatch, "--dry-run", "--from", "dataset", "--season", "2020")
    with pytest.raises(SystemExit):
        _run(project, monkeypatch, "--dry-run", "--only", "v1", "--include-pdfs")


def test_non_interactive_without_yes_aborts(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cleanup, "PROJECT_ROOT", project)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert cleanup.main(["--run", "--only", "v1"]) == 1
    assert _exists(project, "ml_models/xgboost/model.json")
