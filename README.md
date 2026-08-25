# f1_penalty_predictor

ML pipeline to predict FIA penalty decisions from stewarding incident data, with a separate rule-based normative engine for comparison against actual FIA outcomes.

## Models (`ml_models/`)

### V1 XGBoost — FIA behavior model

Trained on 2019, validated on 2025. Validation macro-F1: **0.402**.

| Artifact | Path |
|----------|------|
| Model weights | `ml_models/xgboost/model.json` |
| Label encoder | `ml_models/xgboost/label_encoder.json` |
| Preprocessor | `ml_models/preprocessor.joblib` |
| Preprocessor metadata | `ml_models/preprocessor.meta.json` |
| Validation metrics | `ml_models/xgboost/metrics.json` |
| Validation predictions | `ml_models/xgboost/predictions_val.json` |
| Feature importance | `ml_models/xgboost/feature_importance.json` |
| Error analysis (val) | `ml_models/xgboost/error_analysis_val.json` |

Config: `configs/xgboost.yaml`

### V2 XGBoost — feature-engineering model

Same train/val split as V1, with history/precedent/championship features and selection pruning. Validation macro-F1: **0.381**.

| Artifact | Path |
|----------|------|
| Model weights | `ml_models/xgboost_v2/model.json` |
| Label encoder | `ml_models/xgboost_v2/label_encoder.json` |
| Preprocessor | `ml_models/preprocessor_xgboost_v2.joblib` |
| Preprocessor metadata | `ml_models/preprocessor_xgboost_v2.meta.json` |
| Validation metrics | `ml_models/xgboost_v2/metrics.json` |
| Validation predictions | `ml_models/xgboost_v2/predictions_val.json` |
| Feature importance | `ml_models/xgboost_v2/feature_importance.json` |
| Error analysis (val) | `ml_models/xgboost_v2/error_analysis_val.json` |

Config: `configs/xgboost_v2.yaml`

### Baselines

| Artifact | Path |
|----------|------|
| Fitted baseline models | `ml_models/baseline/model.pkl` |
| Validation metrics | `ml_models/baseline/metrics.json` |
| Validation predictions | `ml_models/baseline/predictions_val.json` |

### Normative rule engine (not trained)

Deterministic rule engine — no learned weights. Rules live in YAML; outputs are versioned JSON artifacts.

| Artifact | Path |
|----------|------|
| Rule definitions | `configs/normative_rules.yaml` |
| Runtime config | `configs/normative.yaml` |
| Rules version + content hash | `ml_models/normative/rules_version.json` |
| Per-incident normative predictions | `ml_models/normative/predictions.json` |
| FIA vs normative evaluation metrics | `ml_models/normative/evaluation_metrics.json` |

## Results & reports (`reports/`)

### Model training

| Report | Path |
|--------|------|
| V1 training report | `reports/model_reports/v1_training_report_2026-08-24.md` |
| V2 feature-engineering report | `reports/model_reports/v2_feature_engineering_report_2026-08-25.md` |

### Figures

| Figure | Path |
|--------|------|
| V1 confusion matrix (validation) | `reports/figures/confusion_matrix_val.png` |
| V2 confusion matrix (validation) | `reports/figures/confusion_matrix_v2_val.png` |
| V1 vs V2 macro-F1 comparison | `reports/figures/v1_vs_v2_macro_f1.png` |
| V1 feature importance (top 20) | `reports/figures/feature_importance_top20.png` |
| V2 feature importance (top 25) | `reports/figures/feature_importance_v2_top25.png` |

### Feature engineering

| Artifact | Path |
|----------|------|
| Ablation results | `reports/ablation_results.json` |
| V2 feature selection report | `reports/selection_report_v2.json` |

### Normative deviation analysis

| Report | Path |
|--------|------|
| Deviation summary (markdown) | `reports/normative/deviation_summary_2026-08-25.md` |
| Breakdown by incident type | `reports/normative/deviation_by_incident_type.csv` |
| Breakdown by session | `reports/normative/deviation_by_session.csv` |
| Breakdown by circuit | `reports/normative/deviation_by_circuit.csv` |
| Breakdown by season | `reports/normative/deviation_by_season.csv` |
| FIA vs normative confusion matrix | `reports/normative/figures/fia_vs_normative_confusion.png` |
| Disagreement rate by incident type | `reports/normative/figures/deviation_rate_by_incident_type.png` |

### Data quality

| Report | Path |
|--------|------|
| 2019 data quality | `reports/tables/data_quality_2019.json` |
| 2025 data quality | `reports/tables/data_quality_2025.json` |

## Processed datasets (`data/processed/`)

| Dataset | Path |
|---------|------|
| Flattened incidents (labels) | `data/processed/incidents.parquet` |
| Incidents + normative outcomes | `data/processed/incidents_with_normative.parquet` |
| V1 encoded features | `data/processed/features.parquet` |
| V2 encoded features | `data/processed/features_v2.parquet` |
| V1 train / validation splits | `data/processed/train.parquet`, `data/processed/validation.parquet` |
| V2 train / validation splits | `data/processed/train_v2.parquet`, `data/processed/validation_v2.parquet` |

## Quick reference

```text
ml_models/
├── xgboost/              # V1 model (macro-F1 0.402)
├── xgboost_v2/           # V2 model (macro-F1 0.381)
├── baseline/             # Majority + session-stratified baselines
├── normative/            # Rule-engine outputs (not ML weights)
├── preprocessor.joblib   # V1 preprocessor
└── preprocessor_xgboost_v2.joblib

reports/
├── model_reports/        # V1 and V2 training write-ups
├── figures/              # Confusion matrices, importance, V1 vs V2
├── normative/            # Deviation analysis vs FIA labels
├── ablation_results.json
└── selection_report_v2.json
```
