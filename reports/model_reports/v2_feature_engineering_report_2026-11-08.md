# V2 Feature Engineering Report — 2026-11-08

## Summary

Final V2 macro-F1 (**0.456**) trails saved V1 model (0.456) by **-0.001** on validation.

**History features** were the only ablation step to clear the +0.03 macro-F1 threshold. Precedent features hurt validation on this two-season corpus; selection pruning partially recovered performance but not to V1 levels.

## Configuration

- Train seasons: [2019, 2020, 2021, 2022, 2023]
- Validation season: 2024
- Random seed: 42
- Encoded features (post-selection): 18

## V1 vs V2 comparison

| Model | Validation macro-F1 |
|-------|---------------------|
| V1 XGBoost (saved) | 0.456 |
| V2 XGBoost (final, pruned features) | 0.456 |

## Ablation experiments

| Exp | Feature groups | macro-F1 | Δ vs prev |
|-----|----------------|----------|-----------|
| A | V1 only | 0.464 | — |
| B | championship, race | 0.483 | +0.019 |
| C | championship, history, race | 0.506 | +0.023 |
| D | championship, history, precedent, race | 0.499 | -0.007 |
| E | championship, history, precedent, race | 0.462 | -0.037 |

## Feature selection

- Input columns: 64
- Kept (raw): 18
- Dropped (missing): 23
- Dropped (correlation): 23
- Dropped (importance): 3

**Kept raw columns:**

- `session`
- `country`
- `driver`
- `circuit`
- `driver_nationality`
- `incident_type`
- `driver_team`
- `is_first_round`
- `num_drivers`
- `years_in_sport`
- `first_season`
- `precedent_no_penalty_rate`
- `round_progress`
- `precedent_count`
- `precedent_minor_penalty_rate`
- `precedent_major_penalty_rate`
- `incidents_last_3_races`
- `career_penalties_per_100_races`

**Dropped at importance prune:**

- `cat__driver_nationality`
- `num__years_in_sport`
- `num__career_penalties_per_100_races`

## V2 model validation metrics

- Accuracy: 0.548
- Macro-F1: 0.456
- Weighted F1: 0.553
- Best iteration: 113

### Per-class

- Class 0: precision=0.254, recall=0.254, f1=0.254, support=59
- Class 1: precision=0.701, recall=0.659, f1=0.679, support=164
- Class 2: precision=0.371, recall=0.520, f1=0.433, support=25

## Quality checks

- Leakage audit: PASS
- Domain sanity (incident_type/severity in top-10): PASS
- Engineered features in top-15 importance: PASS
- Engineered in top-15: num__precedent_major_penalty_rate, num__precedent_minor_penalty_rate, num__precedent_no_penalty_rate, num__precedent_count, num__round_progress, num__is_first_round, num__incidents_last_3_races
- Misclassified validation rows: 112

## Figures

- confusion_matrix_val: `reports\figures\confusion_matrix_v2_val.png`
- feature_importance: `reports\figures\feature_importance_v2_top25.png`
- v1_vs_v2_macro_f1: `reports\figures\v1_vs_v2_macro_f1.png`
- confusion_matrix_test: `reports/figures/confusion_matrix_test.png`

## Success criteria

- V2 macro-F1 ≥ V1: NO (documented)
- History group +0.03 macro-F1 step: NO
- Engineered features in top-15: YES
- Macro-F1 > 0.4: YES