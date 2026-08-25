# Current Gaps Registry

> **Last updated:** 2026-08-25  
> **Purpose:** Single place to track missing data, incomplete columns, pipeline backlog, and deferred work.  
> **Authoritative schema:** [`documentation/f1_dataset_example.csv`](documentation/f1_dataset_example.csv)

Sources consolidated from:

- [`dataset/scripts/README.md`](dataset/scripts/README.md)
- [`src/fia_ml/data/README.md`](src/fia_ml/data/README.md)
- [`src/fia_ml/data/enrichment/README.md`](src/fia_ml/data/enrichment/README.md)
- [`reports/tables/data_quality_2019.json`](reports/tables/data_quality_2019.json)
- [`reports/tables/data_quality_2025.json`](reports/tables/data_quality_2025.json)
- [`ml_models/preprocessor_xgboost_v2.meta.json`](ml_models/preprocessor_xgboost_v2.meta.json)

---

## 1. Season coverage (dataset)

| Season | `processed_{season}.csv` | Rows | Status |
|--------|--------------------------|------|--------|
| **2019** | Yes | 203 | Ready for training |
| **2020** | No | — | **Missing** — FIA download blocked (403 / WAF) |
| **2021** | No | — | **Missing** — FIA download blocked |
| **2022** | No | — | **Missing** — FIA download blocked |
| **2023** | No | — | **Missing** — FIA download blocked |
| **2024** | No | — | **Missing** — FIA download blocked |
| **2025** | Yes | 343 | Ready for training |

**Available now:** 546 incident rows (2019 + 2025).  
**Training split today:** train = 2019 (90 driver-rows after flatten), validation = 2025 (144 rows).  
**Test season:** not configured (`test_season: null` in `configs/xgboost.yaml`).

### How to backfill missing seasons

```powershell
python dataset/scripts/run_pipeline.py --stage all --season 2020
```

If download fails, drop PDFs under `data/raw/fia/{season}/` and run `parse` → `build` → `enrich` → `validate`.

FIA access troubleshooting: `configs/data.yaml` → `scraper.fetch_backend: playwright`, `playwright install chromium`, `python scripts/probe_fia.py`.

---

## 2. Raw schema columns — fill status

Fill rates from latest `data_quality_{season}.json` runs. Legend:

| Symbol | Meaning |
|--------|---------|
| ✅ | ≥ 90% fill on both seasons |
| ⚠️ | Partial / degraded on at least one season |
| ❌ | 0% or not implemented |
| 🏷️ | Label / leakage — not used as model features |
| ✋ | Manual review required |

### Race progression

| Column | 2019 | 2025 | Status | Planned fix |
|--------|------|------|--------|-------------|
| `lap` | 0% | 0% | ❌ | PDF `time` → FastF1 session timeline (`fastf1_enrich.py`) |
| `lap_remaining` | 0% | 0% | ❌ | Derived in `validation.py` once `lap` works |
| `completion_percentage` | 0% | 0% | ❌ | Derived in `validation.py` once `lap` works |
| `full_laps` | 88% | 69% | ⚠️ | `circuits.json` + FastF1 fallback; gaps on non-race sessions |

**Downstream impact:** V2 `race_stage` is empty for most rows until `completion_percentage` is filled.

### Environmental / race control

| Column | 2019 | 2025 | Status | Planned fix |
|--------|------|------|--------|-------------|
| `flag` | 0% | 0% | ❌ | FastF1 race control messages (yellow/red/etc.) |
| `safety_car` | 82% | 62% | ⚠️ | FastF1 race control — partial, session-dependent |
| `track_conditions` | 82% | 62% | ⚠️ | FastF1 session weather |
| `weather_conditions` | 82% | 62% | ⚠️ | FastF1 session weather |
| `sector` | 48% | 3% | ❌ | Turn→sector map in `circuits.json`; sparse in 2025 |

### Incident classification

| Column | 2019 | 2025 | Status | Planned fix |
|--------|------|------|--------|-------------|
| `incident_type` | 100% | 100% | ✅ | From PDF parsing |
| `severity` | 0% | 0% | ✋ ❌ | **Manual only** — fill via `review_queue_{season}.csv` |
| `positions_of_involved parties` | 0% | 0% | ❌ | FastF1 running positions at incident lap/time |
| `incident_classification` | 100% | 100% | 🏷️ | Parsed from PDF; **excluded from model X** (leakage risk) |

### Drivers & teams (multi-value `**` columns)

| Column | 2019 | 2025 | Status | Planned fix |
|--------|------|------|--------|-------------|
| `drivers` | 99% | 82% | ⚠️ | Ergast car-number fallback; gaps when PDF lacks driver |
| `nationalities` | 99% | 75% | ⚠️ | `drivers.json` + Ergast |
| `respective_teams` | 100% | 98% | ✅ | Reference + Ergast |
| `driver_standings` | 99% | 75% | ⚠️ | **Season totals**, not round N−1; multi-driver misalignment |
| `driver_points` | 99% | 75% | ⚠️ | Same as standings |
| `construct_standings` | 55% | 68% | ⚠️ | Ergast constructor standings per round |
| `construct_points` | 55% | 68% | ⚠️ | Same |
| `years_in_sport` | 99% | 75% | ⚠️ | `drivers.json` debut year |
| `superlicense_points_before_incident` | 0% | 0% | ❌ | Rolling sum of `superlicense_points_added` from prior incidents |
| `current_top_4_drivers` | 100% | 100% | ✅ | `seasons.json` — blocked from model X (raw string) |

### Labels (training targets — never features)

| Column | 2019 | 2025 | Status | Notes |
|--------|------|------|--------|-------|
| `penalty` | 91% | 92% | 🏷️ | Primary target source → `penalty_severity` |
| `penalty_severity` | derived | derived | 🏷️ | 3-class model target |
| `driver_at_fault` | 5% | 4% | 🏷️ | Weak PDF heuristics; **excluded from X** |
| `superlicense_points_added` | 15% | 11% | 🏷️ | Outcome field; **excluded from X** |
| `mentioned_article` | 90% | 92% | 🏷️ | **Excluded from X** |
| `investigation` | 100% | 100% | 🏷️ | **Excluded from X** |

### Context columns (generally well filled)

| Column | 2019 | 2025 | Status |
|--------|------|------|--------|
| `incident_id` | 100% | 100% | ✅ (identifier — not in X) |
| `circuit` | 100% | 100% | ✅ |
| `country` | 100% | 100% | ✅ |
| `first_season` | 100% | 90% | ⚠️ |
| `round` | 100% | 100% | ✅ |
| `season` | 100% | 100% | ✅ |
| `rounds` | 100% | 100% | ✅ |
| `num_teams` | 100% | 100% | ✅ |
| `session` | 90% | 91% | ✅ |
| `num_drivers` | 100% | 100% | ✅ |

---

## 3. Data quality issues (non-column)

| Issue | 2019 | 2025 | Impact |
|-------|------|------|--------|
| **Multi-driver column misalignment** | ~19 incidents flagged | ~21 incidents flagged | Two-driver rows: `driver_standings`, `nationalities`, `years_in_sport`, etc. have 1 value but `drivers` has 2 — flatten skips misaligned incidents |
| **Review queue rows** | 20 | 37 | Rows needing manual check (`severity`, missing `lap`, low parse confidence) |
| **Standings timing** | All seasons | All seasons | `seasons.json` uses **end-of-season totals**, not standings before round N |
| **FIA download / WAF** | 2020–2024 | — | Blocks entire seasons at `download` stage |
| **Summons-only rows** | some | some | No mapped `penalty` → excluded from training |

### Suggested enrichment fix order

(from [`src/fia_ml/data/enrichment/README.md`](src/fia_ml/data/enrichment/README.md))

1. Multi-driver column alignment (per-driver Ergast lookup)
2. `lap` / time alignment → unlocks `lap_remaining`, `completion_percentage`, `race_stage`
3. Point-in-time standings (Ergast round N−1)
4. `positions_of_involved parties` + `flag`
5. `superlicense_points_before_incident` (dataset-internal rolling)
6. `severity` (manual review workflow)

---

## 4. Model training gaps

| Gap | Status | Notes |
|-----|--------|-------|
| **Held-out test season** | Not configured | Need a third season (e.g. 2024) for final test split |
| **V2 XGBoost not trained yet** | Pending | `features_v2.parquet` built; `ml_models/xgboost_v2/` empty until `--stage train` |
| **Ablation experiments A–E** | Not implemented | Phase E of `FEATURE_ENGINEERING_PLAN.md` |
| **Precedent features (Group E)** | Not implemented | Phase D — next FE step |
| **Feature selection / prune** | Not implemented | Phase E — correlation + importance prune |
| **V1 vs V2 comparison report** | Not generated | Requires V2 train + evaluate |

### Columns in V2 feature list but dropped at encode time (2019 train split)

Many columns exist in `feature_columns` but have **no observed values in the 2019 train split**, so sklearn imputers skip them and they never reach the encoded matrix. See `ml_models/preprocessor_xgboost_v2.meta.json` (`output_columns` vs `feature_columns`).

| Column group | Examples | Why empty in train |
|--------------|----------|-------------------|
| Race progression | `lap`, `lap_remaining`, `completion_percentage`, `race_stage` | 0% lap fill in source CSV |
| Environmental | `flag`, `severity` | Not implemented / manual |
| Opponent (single-driver incidents) | `opponent_*`, `standing_difference`, `points_difference` | Many 2019 rows have no opponent |
| History (V2) | `career_*`, `incidents_last_*`, `penalties_last_*`, `races_since_*` | May be present but often dropped when all-NaN in train slice — verify after more seasons |
| Championship (V2) | `points_gap_to_leader`, `title_contender`, `round_progress` | Partial; some dropped if train slice lacks variance |

**Improving 2019/2025 column fill rates will directly increase usable model features.**

---

## 5. Feature engineering roadmap (remaining)

| Phase | Status | Deliverable |
|-------|--------|-------------|
| A — Scaffolding | ✅ Done | `configs/features.yaml`, `xgboost_v2.yaml`, feature module stubs |
| B — Race + championship | ✅ Done | `race.py`, `driver.py` |
| C — History | ✅ Done | `history.py` (career + rolling windows) |
| D — Precedent | ⏳ **Next** | `precedent.py` — groupby penalty rates by `(incident_type, severity, session)` |
| E — Selection + ablation | Pending | `selection.py`, experiments A–E, `ablation_results.json` |
| F — Re-train + report | Pending | `xgboost_v2`, v1 vs v2 figures and markdown report |

### V2 engineered columns (code exists; usability depends on source data)

| Column | Group | Blocked by |
|--------|-------|------------|
| `race_stage` | A | `completion_percentage` empty |
| `round_progress`, `is_first_round`, `is_last_round` | A | Usually OK (`round`/`rounds` filled) |
| `points_gap_to_leader`, `title_contender`, etc. | B | Standings are season totals, not point-in-time |
| `career_*`, `incidents_last_*`, `penalties_last_*`, `races_since_*` | C | Works from existing labels; stronger with more seasons |
| `precedent_*` | D | Not built yet |

---

## 6. Deferred / out-of-scope (other plans)

| Item | Plan | Notes |
|------|------|-------|
| NLP / BERT text model | `project_spec.md` V2 | Needs `data/interim/extracted_documents/` text pipeline |
| Normative rules engine | `NORMATIVE_RULES_PLAN.md` | Not started |
| Embedding / vector precedent search | Feature spec §6 | Deferred until ~2000+ incidents |
| Opponent history features (Group F) | `FEATURE_ENGINEERING_PLAN.md` V2.1 | Optional ablation only |
| Nationality bias ablation | Feature spec §28 | Optional fairness experiment |
| Track characteristic features | Feature spec | Not in current schema |

---

## 7. Quick reference — columns to fill after core work

Use this checklist when returning to data enrichment:

- [ ] `lap` → unlocks `lap_remaining`, `completion_percentage`, `race_stage`
- [ ] `flag`
- [ ] `positions_of_involved parties`
- [ ] `severity` (manual)
- [ ] `superlicense_points_before_incident`
- [ ] `sector` (especially 2025 — 3% fill)
- [ ] `driver_standings`, `driver_points`, `nationalities`, `years_in_sport` on **multi-driver** rows
- [ ] `construct_standings`, `construct_points` (raise from ~55–68%)
- [ ] Point-in-time standings (replace season totals)
- [ ] Seasons **2020–2024** (full rows, not just columns)

---

## 8. Related files

| File | Purpose |
|------|---------|
| `dataset/csv/review_queue_{season}.csv` | Manual review backlog |
| `reports/tables/data_quality_{season}.json` | Per-column fill rates |
| `configs/data.yaml` | FIA scraper + enrichment toggles |
| `configs/features.yaml` | V2 feature engineering thresholds |
| `FEATURE_ENGINEERING_PLAN.md` | FE implementation roadmap |
| `MODEL_TRAINING_PLAN.md` | V1 training (complete) |
| `NORMATIVE_RULES_PLAN.md` | Next major plan after FE |
