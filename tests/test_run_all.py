import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.orchestration.run_all import (
    PipelinePhase,
    PrerequisiteError,
    _check_phase_prerequisites,
    _phases_to_run,
    _validate_prerequisites,
    main,
    run_full_pipeline,
)


def test_dry_run_default_plan():
    result = run_full_pipeline(dry_run=True)
    assert result["status"] == "dry_run"
    assert result["plan"]["phases"] == [
        "dataset",
        "v1",
        "v2",
        "nlp",
        "normative",
    ]
    assert result["plan"]["seasons"] == [2019, 2025]


def test_dry_run_dataset_only():
    result = run_full_pipeline(dry_run=True, dataset_only=True)
    assert result["plan"]["phases"] == ["dataset"]


def test_main_without_run_exits_without_executing():
    assert main([]) == 2


def test_main_dry_run_does_not_require_run_flag():
    assert main(["--dry-run"]) == 0


def test_phases_to_run_from_v1_skips_dataset():
    phases = _phases_to_run(
        only=None,
        start_from=PipelinePhase.V1,
        skip_v2=True,
        skip_nlp=False,
        skip_normative=False,
        dataset_only=False,
    )
    assert phases[0] == PipelinePhase.V1
    assert PipelinePhase.DATASET not in phases
    assert PipelinePhase.V2 not in phases


def test_phases_to_run_only_v1():
    phases = _phases_to_run(
        only=PipelinePhase.V1,
        start_from=None,
        skip_v2=False,
        skip_nlp=False,
        skip_normative=False,
        dataset_only=False,
    )
    assert phases == [PipelinePhase.V1]


def test_normative_prerequisites_do_not_require_nlp(monkeypatch):
    fake_root = ROOT / "tests" / "fixtures" / "orchestration_empty" / "no_artifacts"
    fake_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("fia_ml.orchestration.run_all.PROJECT_ROOT", fake_root)
    missing = _check_phase_prerequisites(
        PipelinePhase.NORMATIVE,
        seasons=[2019, 2025],
        nlp_fusion=True,
        normative_config=ROOT / "configs" / "normative.yaml",
    )
    assert missing == ["data/processed/incidents.parquet"]
    assert not any("nlp" in item.lower() for item in missing)


def test_validate_prerequisites_raises(monkeypatch):
    fake_root = ROOT / "tests" / "fixtures" / "orchestration_empty" / "no_artifacts"
    fake_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("fia_ml.orchestration.run_all.PROJECT_ROOT", fake_root)
    try:
        _validate_prerequisites(
            [PipelinePhase.V2],
            seasons=[2019, 2025],
            nlp_fusion=False,
            normative_config=ROOT / "configs" / "normative.yaml",
        )
    except PrerequisiteError as exc:
        assert "incidents.parquet" in str(exc)
    else:
        raise AssertionError("expected PrerequisiteError")
