# NLP vs Tabular Fusion — 2026-09-09

## Summary

- Fusion method: `concat_logits`
- Merged validation rows: 3
- NLP validation rows: 3
- Tabular validation rows: 3
- Best single modality: **nlp_only**
- Fusion beats best single modality: **YES**

## Validation macro-F1 (merged rows)

| Model | Macro-F1 | Accuracy |
|-------|----------|----------|
| NLP text-only | 1.000 | 1.000 |
| V1 tabular-only | 1.000 | 1.000 |
| Fused (concat_logits) | 1.000 | 1.000 |

## Notes

- Metrics are computed on the inner join of NLP and tabular validation `row_id`s.
- A negative fusion result is acceptable on the current two-season corpus.
