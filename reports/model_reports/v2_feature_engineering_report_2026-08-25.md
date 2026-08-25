# V2 Feature Engineering Report — 2026-08-25

## Summary

Final V2 macro-F1 (**0.381**) trails saved V1 model (0.402) by **-0.021** on validation.

**History features** were the only ablation step to clear the +0.03 macro-F1 threshold. Precedent features hurt validation on this two-season corpus; selection pruning partially recovered performance but not to V1 levels.

## Configuration

- Train seasons: [2019]
- Validation season: 2025
- Random seed: 42
- Encoded features (post-selection): 25

## V1 vs V2 comparison

| Model | Validation macro-F1 |
|-------|---------------------|
| V1 XGBoost (saved) | 0.402 |
| V2 XGBoost (final, pruned features) | 0.381 |

## Ablation experiments

| Exp | Feature groups | macro-F1 | Δ vs prev |
|-----|----------------|----------|-----------|
| A | V1 only | 0.430 | — |
| B | championship, race | 0.388 | -0.042 |
| C | championship, history, race | 0.423 | +0.035 |
| D | championship, history, precedent, race | 0.349 | -0.075 |
| E | championship, history, precedent, race | 0.371 | +0.022 |

## Feature selection

- Input columns: 64
- Kept (raw): 25
- Dropped (missing): 23
- Dropped (correlation): 16
- Dropped (importance): 4

**Kept raw columns:**

- `driver`
- `driver_team`
- `weather_conditions`
- `session`
- `circuit`
- `driver_nationality`
- `safety_car`
- `country`
- `incident_type`
- `track_conditions`
- `full_laps`
- `first_season`
- `career_penalties_per_100_races`
- `career_major_penalties`
- `precedent_major_penalty_rate`
- `incidents_last_3_races`
- `career_incidents`
- `precedent_count`
- `precedent_no_penalty_rate`
- `round_progress`
- `penalties_last_5_races`
- `precedent_minor_penalty_rate`
- `races_since_last_incident`
- `points_gap_to_leader`
- `career_incidents_per_100_races`

**Dropped at importance prune:**

- `num__career_incidents`
- `num__penalties_last_5_races`
- `num__career_incidents_per_100_races`
- `cat__driver_nationality`

## V2 model validation metrics

- Accuracy: 0.472
- Macro-F1: 0.381
- Weighted F1: 0.501
- Best iteration: 18

### Per-class

- Class 0: precision=0.250, recall=0.382, f1=0.302, support=34
- Class 1: precision=0.718, recall=0.531, f1=0.611, support=96
- Class 2: precision=0.190, recall=0.286, f1=0.229, support=14

## Quality checks

- Leakage audit: PASS
- Domain sanity (incident_type/severity in top-10): PASS
- Engineered features in top-15 importance: PASS
- Engineered in top-15: num__precedent_no_penalty_rate, num__round_progress, num__precedent_major_penalty_rate, num__precedent_minor_penalty_rate, num__precedent_count, num__career_major_penalties
- Misclassified validation rows: 76

## Figures

- confusion_matrix_val: `reports\figures\confusion_matrix_v2_val.png`
- feature_importance: `reports\figures\feature_importance_v2_top25.png`
- v1_vs_v2_macro_f1: `reports\figures\v1_vs_v2_macro_f1.png`

## Success criteria

- V2 macro-F1 ≥ V1: NO (documented)
- History group +0.03 macro-F1 step: YES
- Engineered features in top-15: YES
- Macro-F1 > 0.4: NO