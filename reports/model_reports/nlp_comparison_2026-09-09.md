# NLP vs Tabular Fusion — 2026-09-09

## Summary

- Fusion method: `concat_logits`
- Merged validation rows: 144
- NLP validation rows: 154
- Tabular validation rows: 144
- Best single modality: **nlp_only**
- Fusion beats best single modality: **NO**

## Validation macro-F1 (merged rows)

| Model | Macro-F1 | Accuracy |
|-------|----------|----------|
| NLP text-only | 0.648 | 0.833 |
| V1 tabular-only | 0.402 | 0.528 |
| Fused (concat_logits) | 0.632 | 0.806 |

## Notes

- Metrics are computed on the inner join of NLP and tabular validation `row_id`s.
- A negative fusion result is acceptable on the current two-season corpus.
