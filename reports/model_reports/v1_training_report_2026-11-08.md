# V1 Training Report — 2026-11-08

## Configuration

- Train seasons: [2019, 2020, 2021, 2022, 2023]
- Validation season: 2024
- Test season: [2025, 2026]
- Random seed: 42

## Data counts

- Train rows: 917
- Validation rows: 248

## Model comparison (validation macro-F1)

| Model | Macro-F1 |
|-------|----------|
| Majority baseline | 0.265 |
| Session-stratified baseline | 0.406 |
| XGBoost | 0.456 |

## XGBoost validation metrics

- Accuracy: 0.569
- Macro-F1: 0.456
- Weighted F1: 0.579
- Log loss: 0.8447301392040182
- Best iteration: 167

### Per-class

- Class 0: precision=0.333, recall=0.373, f1=0.352, support=59
- Class 1: precision=0.733, recall=0.671, f1=0.701, support=164
- Class 2: precision=0.281, recall=0.360, f1=0.316, support=25

## Quality checks

- Leakage audit: PASS
- Domain sanity (incident_type/severity in top-10): PASS
- Top features: cat__session, num__lap_remaining, cat__incident_type, num__num_drivers, num__completion_percentage, num__full_laps, num__lap, num__season, cat__circuit, cat__sector
- Misclassified validation rows: 107

## Figures

- confusion_matrix_val: `reports\figures\confusion_matrix_val.png`
- feature_importance: `reports\figures\feature_importance_top20.png`
- confusion_matrix_test: `reports/figures/confusion_matrix_test.png`

## Success criteria

- Macro-F1 > 0.4: YES
- Beats session baseline: YES