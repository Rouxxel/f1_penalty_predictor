import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.training.fusion_nlp import (
    fuse_concat_logits,
    merge_modalities,
    run_fusion_nlp,
)
from fia_ml.training.nlp_config import NlpTrainingConfig


@pytest.fixture
def fixture_cfg():
    return NlpTrainingConfig.from_yaml(ROOT / "tests" / "fixtures" / "nlp" / "nlp_dataset_test.yaml")


def _sample_records():
    return [
        {
            "row_id": "r1",
            "penalty_severity": 0,
            "proba_no_penalty": 0.7,
            "proba_minor": 0.2,
            "proba_major": 0.1,
        },
        {
            "row_id": "r2",
            "penalty_severity": 1,
            "proba_no_penalty": 0.1,
            "proba_minor": 0.8,
            "proba_major": 0.1,
        },
        {
            "row_id": "r3",
            "penalty_severity": 2,
            "proba_no_penalty": 0.1,
            "proba_minor": 0.2,
            "proba_major": 0.7,
        },
    ]


def test_fuse_concat_logits_prefers_agreement():
    nlp = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    tab = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    preds = fuse_concat_logits(nlp, tab)
    assert preds.tolist() == [0, 1]


def test_merge_modalities_inner_join():
    import pandas as pd

    proba_cols = ["proba_no_penalty", "proba_minor", "proba_major"]
    nlp_df = pd.DataFrame(_sample_records())
    tab_df = pd.DataFrame(_sample_records())
    tab_df.loc[0, "proba_minor"] = 0.3
    merged = merge_modalities(nlp_df, tab_df, proba_cols=proba_cols)
    assert len(merged) == 3
    assert "nlp_proba_minor" in merged.columns
    assert "tab_proba_minor" in merged.columns


def test_run_fusion_concat_logits_with_fixtures(fixture_cfg):
    out_dir = ROOT / "tests" / "fixtures" / "nlp" / "_out" / "fusion_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    nlp_records = _sample_records()
    tab_records = _sample_records()
    tab_records[1]["proba_minor"] = 0.55
    tab_records[1]["proba_no_penalty"] = 0.35

    nlp_pred_path = out_dir / "predictions_val.json"
    tab_pred_path = out_dir / "tabular_predictions_val.json"
    nlp_pred_path.write_text(json.dumps(nlp_records), encoding="utf-8")
    tab_pred_path.write_text(json.dumps(tab_records), encoding="utf-8")

    cfg = NlpTrainingConfig(
        paths={**fixture_cfg.paths, "models": "tests/fixtures/nlp/_out/fusion_smoke"},
        inputs=fixture_cfg.inputs,
        splits=fixture_cfg.splits,
        text=fixture_cfg.text,
        model=fixture_cfg.model,
        training=fixture_cfg.training,
        evaluation=fixture_cfg.evaluation,
        fusion={
            **fixture_cfg.fusion,
            "enabled": True,
            "method": "concat_logits",
            "tabular_predictions": "tests/fixtures/nlp/_out/fusion_smoke/tabular_predictions_val.json",
        },
    )
    result = run_fusion_nlp(cfg)
    assert result["method"] == "concat_logits"
    assert Path(result["outputs"]["fusion_metrics"]).exists()
    assert Path(result["outputs"]["report"]).exists()
    report = Path(result["outputs"]["report"]).read_text(encoding="utf-8")
    assert "NLP vs Tabular Fusion" in report
