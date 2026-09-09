# NLP Training Report — 2026-09-09

## Summary

- Backbone: `distilbert-base-uncased`
- Text profile: `fact_offence`
- Validation macro-F1: **0.500**
- V1 tabular macro-F1: **0.402** (+0.098 vs NLP)

## Configuration

- Train seasons: [2019]
- Validation season: 2025
- Max sequence length: 512
- Random seed: 42

## Data counts

- Train rows: 2
- Validation rows: 1

## Model comparison (validation macro-F1)

| Model | Macro-F1 |
|-------|----------|
| V1 XGBoost (tabular) | 0.402 |
| NLP (distilbert-base-uncased) | 0.500 |

## NLP validation metrics

- Accuracy: 0.500
- Macro-F1: 0.500
- Weighted F1: 0.500

### Per-class

- Class 0: precision=0.500, recall=0.500, f1=0.500, support=1

## Dataset audit

- Labeled incident rows: n/a
- Rows with text: 3
- Missing interim docs: n/a
- Text build errors: n/a

## Quality checks

- Misclassified validation rows: 1
- Text profile leakage-safe: YES

## Figures

- confusion_matrix_val: `reports/figures/nlp_confusion_matrix_val.png`

## Success criteria

- Macro-F1 ≥ 0.35: YES
- NLP macro-F1 ≥ V1 tabular: YES