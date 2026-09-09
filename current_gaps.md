# Current Gaps Registry

> **Last updated:** 2026-09-09  
> **Purpose:** Open gaps on **existing** pipelines — missing data, unmet targets, modeling limitations.  
> **Future capabilities** (CNN, NLP, telemetry, multimodal): [`documentation/FUTURE_FEATURES.md`](documentation/FUTURE_FEATURES.md)  
> **Schema:** [`documentation/f1_dataset_example.csv`](documentation/f1_dataset_example.csv)

### What this file covers

| Section | Topic |
|---------|--------|
| §1–§4 | **Dataset** — seasons, columns, enrichment, row quality |
| §5–§6 | **Current models** — limitations on today’s V1/V2 (not “plans to implement”) |
| §7 | **Normative rules** — coverage and iteration (engine already runs) |

Pipeline docs: [`README.md`](README.md) · dataset runbook: [`documentation/dataset_generation_runbook.md`](documentation/dataset_generation_runbook.md) · fill rates: [`reports/tables/data_quality_{season}.json`](reports/tables/).

---

## 1. Seasons & rows

### Missing seasons

| Season | Status |
|--------|--------|
| **2020** | No `processed_2020.csv` — FIA download blocked (403 / WAF) |
| **2021** | No `processed_2021.csv` — FIA download blocked |
| **2022** | No `processed_2022.csv` — FIA download blocked |
| **2023** | No `processed_2023.csv` — FIA download blocked |
| **2024** | No `processed_2024.csv` — FIA download blocked |

**Backfill:** `python dataset/scripts/run_pipeline.py --stage all --season 2020` or manual PDFs under `data/raw/fia/{season}/`. See [`dataset/scripts/README.md`](dataset/scripts/README.md).

### Row coverage limits

| Issue | Detail |
|-------|--------|
| Train / val only | 90 train (2019) / 144 val (2025) — no held-out **test** season |
| Flatten loss | 546 raw incidents → 234 driver-rows; summons-only, unmapped penalties, ~40 misaligned multi-driver incidents skipped |
| Distribution shift | 2019 train → 2025 val spans 6 years |

---

## 2. Columns below target

Fill rates from `data_quality_2019.json` and `data_quality_2025.json`. Only columns that are missing, sparse, or below plan targets.

### Race progression

| Column | 2019 | 2025 | Planned fix |
|--------|------|------|-------------|
| `lap` | 0% | 0% | PDF `time` → FastF1 session timeline |
| `lap_remaining` | 0% | 0% | `validation.py` once `lap` works |
| `completion_percentage` | 0% | 0% | Same — blocks V2 `race_stage` |
| `full_laps` | 88% | 69% | `circuits.json` + FastF1 |

**Target not met:** FastF1 lap/time for > 50% of race incidents (actual: 0%).

### Environmental / race control

| Column | 2019 | 2025 | Planned fix |
|--------|------|------|-------------|
| `flag` | 0% | 0% | FastF1 race control messages |
| `safety_car` | 82% | 62% | FastF1 race control |
| `track_conditions` | 82% | 62% | FastF1 session weather |
| `weather_conditions` | 82% | 62% | FastF1 session weather |
| `sector` | 48% | 3% | Turn→sector in `circuits.json` |

**Target not met:** weather + SC/VSC for > 70% of race-session incidents on 2025 (62%).

### Incident & driver fields

| Column | 2019 | 2025 | Notes |
|--------|------|------|-------|
| `severity` | 0% | 0% | Manual via `review_queue_{season}.csv` |
| `positions_of_involved parties` | 0% | 0% | FastF1 positions at incident time |
| `drivers` | 99% | 82% | Ergast fallback when PDF lacks driver |
| `nationalities` | 99% | 75% | Multi-driver misalignment |
| `driver_standings` | 99% | 75% | Season totals, not round N−1 |
| `driver_points` | 99% | 75% | Same |
| `construct_standings` | 55% | 68% | Incomplete |
| `construct_points` | 55% | 68% | Incomplete |
| `years_in_sport` | 99% | 75% | Multi-driver misalignment |
| `superlicense_points_before_incident` | 0% | 0% | Not implemented in enricher |
| `driver_at_fault` | 5% | 4% | Weak PDF heuristics |
| `penalty` | 91% | 92% | Target > 95% |
| `first_season` | 100% | 90% | Degraded on 2025 |

---

## 3. Dataset pipeline

| Gap | Detail |
|-----|--------|
| **Standings timing** | `reference_enrich.py` uses season-end totals; Ergast round N−1 only fills empty cells |
| **`superlicense_points_before_incident`** | Specified in plan — not implemented |
| **`test_enrichment_ergast.py`** | Not created |
| **`test_enrichment_fastf1.py`** | Not created |
| **Point-in-time standings test** | Not created |
| **PDF `raw_text` NLP** | Interim JSON exists; no NLP pipeline |
| **Review queue** | 20 (2019) + 37 (2025) rows — low confidence, missing `severity`/`lap` |
| **FIA WAF** | Blocks automated download for 2020–2024 |
| **Multi-car dedup** | Ambiguous cases → review queue |

### Suggested fix order

1. Multi-driver column alignment
2. `lap` / time → unlocks `lap_remaining`, `completion_percentage`, `race_stage`
3. Point-in-time standings (Ergast round N−1)
4. `positions_of_involved parties` + `flag`
5. `superlicense_points_before_incident`
6. `severity` (manual labeling)

---

## 4. Data quality (non-column)

| Issue | Impact |
|-------|--------|
| Multi-driver misalignment | ~19 (2019) + ~21 (2025) incidents skipped on flatten |
| Two-season corpus | Precedent/history cannot use 2020–2024; 6-year val shift |
| Class 2 (major) sparse | ~14 val support; recall ~21% (V1 report) |

---

## 5. Model training

| Gap | Detail |
|-----|--------|
| **No test season** | `test_season: null` in `configs/xgboost.yaml` |
| **LightGBM** | `tabular_classifier.py` not built |
| **Leave-one-season-out CV** | Not implemented |
| **Macro-F1 vs session baseline** | +0.043 improvement — below +0.10 plan target |
| **Ordinal encoding** | Nominal fields (`session`, `circuit`) use `OrdinalEncoder` |
| **Small train set** | 90 rows |
| **Class 2 (major)** | Weak precision/recall on validation |
| **Columns dropped at encode** | `flag`, `severity`, `lap`, `lap_remaining`, `completion_percentage`, `opponent_*`, `standing_difference`, `points_difference`, `superlicense_points_before_incident` — no train observations |

---

## 6. Feature engineering

| Gap | Detail |
|-----|--------|
| **V2 macro-F1 < V1** | 0.381 vs 0.402 on validation — V2 did not beat V1 |
| **Precedent hurt ablation** | Exp D −0.075 macro-F1 vs C — sparse groups, two-season corpus |
| **`race_stage` unusable** | 0% `completion_percentage` fill |
| **Championship features** | `points_gap_to_leader`, `title_contender` use season totals not round N−1 |
| **`severity` unlabeled** | Cannot activate `(incident_type, severity, session)` precedent key |
| **Precedent fallback** | 32% of rows use global prior (`precedent_count < 3`) |
| **Opponent history (Group F)** | Deferred |
| **Extended leakage audit** | No dedicated helper for precedent/history columns |
| **Hyperparameter tuning** | Not done |
| **`races_since_last_*`** | Same-season only; `NaN` when prior event was previous season |

---

## 7. Normative rules

| Gap | Impact | Planned fix |
|-----|--------|-------------|
| **`manual_review` rate 59.8%** | Target ≤20% unmet | Expand rules; reduce `other` incidents |
| **`incident_type: other`** (128/234) | → `other_unclassified` → `manual_review` | Better classifier or sub-rules |
| **Fact text empty** | `fact_contains_any` rules don't fire | Populate `data/interim/extracted_documents/{season}/` |
| **Collision rules unused** | 12 collisions → `collision_unclassified` | Depends on Fact join |
| **Top deviations not reviewed** | Rule iteration not started | Manual review + YAML updates |
| **`normative_history` ablation** | Not run | Second-pass with normative escalation outcomes |
| **Fixture file** | `tests/fixtures/normative_incidents.json` missing | 20–30 synthetic labeled rows |
| **Session corruption (2025)** | Garbage `session` strings on some rows | Dataset cleanup |

**Deviation snapshot (234 rows):** FIA vs normative agreement 52.6%; FIA harsher 41.5%; highest disagreement on `other` (70.3%). Report: `reports/normative/deviation_summary_2026-08-25.md`.

---

## 8. Open checklist

### Data
- [ ] Seasons **2020–2024**
- [ ] Third season for **test** split
- [ ] Fix **multi-driver misalignment** (~40 incidents)
- [ ] Clear **review queue** (57 rows)
- [ ] `lap`, `flag`, `severity`, `positions_of_involved parties`, `superlicense_points_before_incident`, `sector`
- [ ] Point-in-time **standings** (round N−1)
- [ ] `test_enrichment_ergast.py` + point-in-time standings test

### Modeling
- [ ] Held-out **test** season
- [ ] LightGBM, leave-one-season-out CV (optional)
- [ ] Opponent history (Group F)
- [ ] Improve V2 or accept V1 as primary model until more data

### Normative
- [ ] Reduce `manual_review` rate
- [ ] Populate Fact text from parsed PDFs
- [ ] Manual review of top deviations + rule iteration

---

## 9. Key artifact paths

| Path | Use when fixing gaps |
|------|----------------------|
| `dataset/csv/review_queue_{season}.csv` | Manual review backlog |
| `reports/tables/data_quality_{season}.json` | Column fill rates |
| `data/interim/extracted_documents/{season}/` | Fact text for normative rules |
| `configs/normative_rules.yaml` | Rule iteration |
| `configs/features.yaml` | Precedent key / feature toggles |
| `ml_models/preprocessor_xgboost_v2.meta.json` | Encode-time column drops |
