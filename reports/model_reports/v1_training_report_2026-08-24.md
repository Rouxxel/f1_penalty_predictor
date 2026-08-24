# V1 Training Report — 2026-08-24

## Configuration

- Train seasons: [2019]
- Validation season: 2025
- Test season: None
- Random seed: 42

## Data counts

- Train rows: 90
- Validation rows: 144

## Model comparison (validation macro-F1)

| Model | Macro-F1 |
|-------|----------|
| Majority baseline | 0.267 |
| Session-stratified baseline | 0.359 |
| XGBoost | 0.402 |

## XGBoost validation metrics

- Accuracy: 0.528
- Macro-F1: 0.402
- Weighted F1: 0.552
- Log loss: 0.9705789747222645
- Best iteration: 30

### Per-class

- Class 0: precision=0.326, recall=0.412, f1=0.364, support=34
- Class 1: precision=0.747, recall=0.615, f1=0.674, support=96
- Class 2: precision=0.136, recall=0.214, f1=0.167, support=14

## Quality checks

- Leakage audit: PASS
- Domain sanity (incident_type/severity in top-10): PASS
- Top features: num__full_laps, cat__incident_type, cat__safety_car, cat__sector, cat__session, cat__driver, cat__circuit, num__round, num__first_season, cat__country
- Misclassified validation rows: 68

## Figures

- confusion_matrix_val: `reports\figures\confusion_matrix_val.png`
- feature_importance_top20: `reports\figures\feature_importance_top20.png`

## Success criteria

- Macro-F1 > 0.4: YES
- Beats session baseline: YES