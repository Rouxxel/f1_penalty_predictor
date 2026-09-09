# NLP Training Report — 2026-09-09

## Summary

- Backbone: `distilbert-base-uncased`
- Text profile: `fact_offence`
- Validation macro-F1: **0.642**
- V1 tabular macro-F1: **0.402** (+0.241 vs NLP)

## Configuration

- Train seasons: [2019]
- Validation season: 2025
- Max sequence length: 512
- Random seed: 42

## Data counts

- Train rows: 90
- Validation rows: 154

## Model comparison (validation macro-F1)

| Model | Macro-F1 |
|-------|----------|
| V1 XGBoost (tabular) | 0.402 |
| NLP (distilbert-base-uncased) | 0.642 |

## NLP validation metrics

- Accuracy: 0.831
- Macro-F1: 0.642
- Weighted F1: 0.805
- Log loss: 0.564

### Per-class

- Class 0: precision=0.714, recall=0.972, f1=0.824, support=36
- Class 1: precision=0.892, recall=0.892, f1=0.892, support=102
- Class 2: precision=0.667, recall=0.125, f1=0.211, support=16

## Dataset audit

- Labeled incident rows: 374
- Rows with text: 244
- Missing interim docs: 0
- Text build errors: 0

## Quality checks

- Misclassified validation rows: 26
- Text profile leakage-safe: YES

## Figures

- confusion_matrix_val: `reports\figures\nlp_confusion_matrix_val.png`

## Success criteria

- Macro-F1 ≥ 0.35: YES
- NLP macro-F1 ≥ V1 tabular: YES