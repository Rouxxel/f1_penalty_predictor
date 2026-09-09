# Current Gaps Registry

> **Last updated:** 2026-09-09  
> **Purpose:** Open gaps on **existing** pipelines — missing data, unmet targets, modeling limitations.  
> **Future capabilities** (CNN, telemetry, embedding precedent, multimodal): [`documentation/FUTURE_FEATURES.md`](documentation/FUTURE_FEATURES.md)  
> **Schema:** [`documentation/f1_dataset_example.csv`](documentation/f1_dataset_example.csv)

### What this file covers

| Section | Topic |
|---------|--------|
| §1–§4 | **Dataset** — seasons, columns, enrichment blockers, row quality |
| §5–§6 | **Current models** — limitations on today’s V1/V2/NLP (not “plans to implement”) |
| §7 | **Normative rules** — coverage and iteration (engine already runs) |

Pipeline docs: [`README.md`](README.md) · enrichment detail: [`src/fia_ml/data/enrichment/README.md`](src/fia_ml/data/enrichment/README.md) · fill rates: [`reports/tables/data_quality_{season}.json`](reports/tables/data_quality_{season}.json) · quality gates: [`reports/tables/enrichment_report_{season}.json`](reports/tables/enrichment_report_{season}.json).

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
| **Parquet not refreshed** | `data/processed/*.parquet` still built from pre-enrichment CSVs — re-run `prepare` → `features_v2` after enrichment stabilizes |

---

## 2. Columns below target

Fill rates from `reports/tables/data_quality_{season}.json` (post-enrichment run, 2026-09-09). Only columns that are missing, sparse, or below plan targets.

### Race progression (critical path)

| Column | 2019 | 2025 | Blocker |
|--------|------|------|---------|
| `lap` | 1.5% | 0.3% | PDF `time` parses as **document clock**, not session time — only ~1.5% valid `session_offset_seconds` |
| `lap_remaining` | 1.5% | 0.3% | Derived in `validation.py` once `lap` works |
| `completion_percentage` | 1.5% | 0.3% | Same — blocks V2 `race_stage` |
| `flag` | 0% | 0% | Needs valid session timestamp for OpenF1/FastF1 race-control join |
| `positions_of_involved parties` | 0% | 0% | Same |

**Target not met:** > 50% race-session `lap` / `flag` fill (actual: ~0–4%).

### Environmental / race control

| Column | 2019 | 2025 | Notes |
|--------|------|------|-------|
| `sector` | 48% | 3% | Turn→sector map + RC sector need timestamp; 2019 below 70% target |
| `full_laps` | 90% | 89% | FastF1 + circuits — acceptable |
| `safety_car` / weather | 90% | 89% | FastF1 session-level — acceptable |

### Incident & driver fields

| Column | 2019 | 2025 | Notes |
|--------|------|------|-------|
| `severity` | 0% | 0% | Suggestions in review queue only; CSV column not auto-filled |
| `driver_standings` / `driver_points` | 96% | 78% | Ergast round N−1 wired; **2025 below 90% gate** (unresolved driver slugs) |
| `construct_standings` / `construct_points` | 96% | 78% | Same |
| `drivers` | 99% | 81% | Ergast car-number fallback |
| `nationalities` / `years_in_sport` | 96–99% | 77% | **Multi-driver misalignment** on `car_*` placeholder rows |
| `driver_at_fault` | 25% | 25% | Rule-based assist; low-confidence rows → review queue |
| `penalty` | 91% | 92% | Target > 95% |
| `first_season` | 100% | 90% | Degraded on 2025 |

---

## 3. Dataset pipeline

| Gap | Detail |
|-----|--------|
| **PDF session timestamps** | `timestamp.py` parses all rows via `document_clock`; ~1.5% map to session offset — blocks lap/flag/positions despite OpenF1 + FastF1 v2 being wired |
| **Multi-driver alignment** | ~20 incidents/season: `drivers` has 2 values but per-driver `**` columns have length 1 (often `car_*` placeholder) |
| **Review queue** | 20 (2019) + 37 (2025) rows — low confidence, missing `severity`/`lap`, text-field suggestions |
| **FIA WAF** | Blocks automated download for 2020–2024 |
| **Season backfill** | Enrichment modules are season-agnostic; need PDFs for 2020–2024 |
| **Downstream refresh** | Re-run flatten + V2 feature build not done since enrichment — `race_stage`, championship features still use old parquet |

### Suggested fix order

1. **Session timestamp extraction** — move off document header clock; unlock lap/flag/positions
2. **Multi-driver column alignment** — per-driver Ergast fields for `car_*` rows
3. **2025 standings coverage** — driver slug ↔ car number gaps
4. **`sector`** — expand turn maps + timestamp-dependent RC sector
5. **`severity`** — manual promotion from review queue
6. **Seasons 2020–2024** — PDF backfill when WAF allows
7. **Re-run `prepare` → `features_v2`** — consume enriched CSVs

---

## 4. Data quality (non-column)

| Issue | Impact |
|-------|--------|
| Multi-driver misalignment | ~19 (2019) + ~20 (2025) incidents flagged in validation; skipped on flatten |
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
| **Columns dropped at encode** | `flag`, `severity`, `lap`, `lap_remaining`, `completion_percentage`, `opponent_*`, `standing_difference`, `points_difference`, `superlicense_points_before_incident` — sparse or no train observations in current parquet |

### NLP text model (spec V2)

Trained on 2019+2025 interim JSON (2026-09-09): **val macro-F1 0.642** (154 rows) vs V1 **0.402** (144 merged rows). Artifacts in `ml_models/nlp/`.

| Gap | Detail |
|-----|--------|
| **Class 2 (major) recall** | 12.5% on validation (2/16 correct) — major penalties still hard |
| **Val row count mismatch** | NLP 154 vs V1 144 validation rows — inner join used for fusion |
| **Fusion below NLP-only** | `concat_logits` macro-F1 0.632 < NLP-only 0.648 on merged rows — use text-only for now |
| **Interim doc join** | 100% join on labeled rows after parse; 112 misaligned multi-driver rows skipped (same as tabular) |
| **Summons-only rows** | Excluded (no offence text) |
| **HF runtime** | Set `USE_TF=0` and `PYTHONPATH=src` on Windows if TensorFlow conflicts; `numpy>=1.26,<2` recommended |

Reports: `reports/model_reports/nlp_training_report_2026-09-09.md`, `nlp_comparison_2026-09-09.md`.

---

## 6. Feature engineering

| Gap | Detail |
|-----|--------|
| **V2 macro-F1 < V1** | 0.381 vs 0.402 on validation — V2 did not beat V1 |
| **Precedent hurt ablation** | Exp D −0.075 macro-F1 vs C — sparse groups, two-season corpus |
| **`race_stage` unusable** | ~0% `completion_percentage` fill in current parquet |
| **Championship features** | `points_gap_to_leader`, `title_contender` use stale parquet (pre round N−1 enrichment) |
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
- [ ] Fix **PDF session timestamps** (lap / flag / positions blocker)
- [ ] Fix **multi-driver misalignment** (~40 incidents)
- [ ] Clear **review queue** (57 rows)
- [ ] `lap`, `flag`, `severity`, `positions_of_involved parties`, `sector` fill targets
- [ ] **2025** round N−1 standings coverage (≥ 90%)
- [ ] Re-run **flatten + V2 features** on enriched CSVs

### Modeling
- [ ] Held-out **test** season
- [ ] LightGBM, leave-one-season-out CV (optional)
- [ ] Opponent history (Group F)
- [ ] Improve V2 or accept V1 as primary model until more data
- [x] **NLP** — train/evaluate on interim JSON; beats V1 on macro-F1; fusion report (concat_logits does not beat NLP-only)

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
| `reports/tables/enrichment_report_{season}.json` | Quality-gate pass/fail vs targets |
| `data/interim/enrichment_meta/{season}.json` | Per-field enrichment provenance |
| `configs/enrichment.yaml` | Source priority, thresholds, quality targets |
| `data/interim/extracted_documents/{season}/` | Fact text for normative rules and NLP dataset join |
| `configs/bert.yaml` | NLP text profile, training, fusion settings |
| `ml_models/nlp/nlp_dataset_audit.json` | Incident → interim document join coverage |
| `configs/normative_rules.yaml` | Rule iteration |
| `configs/features.yaml` | Precedent key / feature toggles |
| `ml_models/preprocessor_xgboost_v2.meta.json` | Encode-time column drops |
