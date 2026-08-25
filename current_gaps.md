# Current Gaps Registry

> **Last updated:** 2026-08-25 (Normative Phase B complete)  
> **Purpose:** Single registry of missing data, incomplete columns, unmet plan success criteria, and deferred work across the project.  
> **Authoritative schema:** [`documentation/f1_dataset_example.csv`](documentation/f1_dataset_example.csv)

### Source documents

| Document | Scope |
|----------|--------|
| [`DATASET_GENERATION_PLAN.md`](DATASET_GENERATION_PLAN.md) | FIA PDF → `processed_{season}.csv` |
| [`MODEL_TRAINING_PLAN.md`](MODEL_TRAINING_PLAN.md) | Flatten, encode, V1 XGBoost |
| [`FEATURE_ENGINEERING_PLAN.md`](FEATURE_ENGINEERING_PLAN.md) | V2 features, ablation, `xgboost_v2` |
| [`NORMATIVE_RULES_PLAN.md`](NORMATIVE_RULES_PLAN.md) | Rule engine (not started) |
| [`dataset/scripts/README.md`](dataset/scripts/README.md) | CLI + live season status |
| [`src/fia_ml/data/README.md`](src/fia_ml/data/README.md) | Pipeline modules |
| [`src/fia_ml/data/enrichment/README.md`](src/fia_ml/data/enrichment/README.md) | Enrichment detail |
| [`reports/tables/data_quality_{season}.json`](reports/tables/) | Column fill rates |
| [`ml_models/preprocessor_xgboost_v2.meta.json`](ml_models/preprocessor_xgboost_v2.meta.json) | Encode-time column drops |

---

## 1. Season & row coverage

### Missing seasons (dataset generation)

| Season | `processed_{season}.csv` | Rows | Status |
|--------|--------------------------|------|--------|
| **2019** | Yes | 203 | Ready |
| **2020** | No | — | **Missing** — FIA download blocked (403 / WAF) |
| **2021** | No | — | **Missing** — FIA download blocked |
| **2022** | No | — | **Missing** — FIA download blocked |
| **2023** | No | — | **Missing** — FIA download blocked |
| **2024** | No | — | **Missing** — FIA download blocked |
| **2025** | Yes | 343 | Ready |

**Available:** 546 raw incident rows (2019 + 2025).

### Row loss between raw CSV and model training

| Stage | Count | Gap |
|-------|-------|-----|
| Raw incidents (2019 + 2025) | 546 | — |
| After flatten + mapped penalty | 234 driver-rows | Summons-only / unmapped penalties excluded; ~19–21 misaligned multi-driver incidents **skipped** per season |
| Train split (2019) | 90 | Small single-season train set |
| Validation split (2025) | 144 | No held-out test season |

**Backfill:** `python dataset/scripts/run_pipeline.py --stage all --season 2020` or manual PDFs under `data/raw/fia/{season}/`. See [`dataset/scripts/README.md`](dataset/scripts/README.md) for Playwright/WAF troubleshooting.

---

## 2. Raw schema columns — fill status

Fill rates from `data_quality_2019.json` and `data_quality_2025.json`.

| Symbol | Meaning |
|--------|---------|
| ✅ | ≥ 90% fill on both seasons |
| ⚠️ | Partial / degraded on at least one season |
| ❌ | 0% or not implemented |
| 🏷️ | Label / leakage — excluded from model **X** |
| ✋ | Manual review required |

### Race progression

| Column | 2019 | 2025 | Status | Planned fix (`DATASET_GENERATION_PLAN` §4b) |
|--------|------|------|--------|---------------------------------------------|
| `lap` | 0% | 0% | ❌ | PDF `time` → FastF1 session timeline |
| `lap_remaining` | 0% | 0% | ❌ | `validation.py` once `lap` works |
| `completion_percentage` | 0% | 0% | ❌ | Same |
| `full_laps` | 88% | 69% | ⚠️ | `circuits.json` + FastF1; gaps on non-race sessions |

**Plan target:** FastF1 lap/time for **> 50%** of race incidents — **not met (0%)**.  
**Downstream:** V2 `race_stage` empty until `completion_percentage` is filled.

### Environmental / race control

| Column | 2019 | 2025 | Status | Planned fix |
|--------|------|------|--------|-------------|
| `flag` | 0% | 0% | ❌ | FastF1 race control messages |
| `safety_car` | 82% | 62% | ⚠️ | FastF1 race control |
| `track_conditions` | 82% | 62% | ⚠️ | FastF1 session weather |
| `weather_conditions` | 82% | 62% | ⚠️ | FastF1 session weather |
| `sector` | 48% | 3% | ❌ | Turn→sector in `circuits.json`; very sparse in 2025 |

**Plan target:** weather + SC/VSC for **> 70%** of race-session incidents — **2019 met, 2025 below (62%)**.

### Incident classification

| Column | 2019 | 2025 | Status | Notes |
|--------|------|------|--------|-------|
| `incident_type` | 100% | 100% | ✅ | PDF keyword classifier |
| `severity` | 0% | 0% | ✋ ❌ | Manual via `review_queue_{season}.csv` |
| `positions_of_involved parties` | 0% | 0% | ❌ | FastF1 positions at incident time |
| `incident_classification` | 100% | 100% | 🏷️ | Leakage risk — blocked from **X** |

### Drivers & teams (multi-value `**` columns)

| Column | 2019 | 2025 | Status | Notes |
|--------|------|------|--------|-------|
| `drivers` | 99% | 82% | ⚠️ | Ergast car-number fallback when PDF lacks driver |
| `nationalities` | 99% | 75% | ⚠️ | `drivers.json` + Ergast |
| `respective_teams` | 100% | 98% | ✅ | |
| `driver_standings` | 99% | 75% | ⚠️ | See **standings timing** below; multi-driver misalignment |
| `driver_points` | 99% | 75% | ⚠️ | Same |
| `construct_standings` | 55% | 68% | ⚠️ | Ergast constructor standings; incomplete |
| `construct_points` | 55% | 68% | ⚠️ | Same |
| `years_in_sport` | 99% | 75% | ⚠️ | `drivers.json` debut year |
| `superlicense_points_before_incident` | 0% | 0% | ❌ | Planned: rolling sum in Ergast/dataset (`DATASET_GENERATION_PLAN` §4a) |
| `current_top_4_drivers` | 100% | 100% | ✅ | Blocked from **X** (raw string); was `top_4_driver` booleans — **removed** as redundant |

### Labels (never model features)

| Column | 2019 | 2025 | Status | Notes |
|--------|------|------|--------|-------|
| `penalty` | 91% | 92% | 🏷️ | Plan target **> 95%** for Decision/Offence — **slightly below** |
| `penalty_severity` | derived | derived | 🏷️ | 3-class target |
| `driver_at_fault` | 5% | 4% | 🏷️ | Weak PDF heuristics |
| `superlicense_points_added` | 15% | 11% | 🏷️ | Outcome field |
| `mentioned_article` | 90% | 92% | 🏷️ | |
| `investigation` | 100% | 100% | 🏷️ | |

### Context columns (generally well filled)

| Column | 2019 | 2025 | Status |
|--------|------|------|--------|
| `circuit`, `country`, `round`, `season`, `rounds`, `num_teams` | 100% | 100% | ✅ |
| `first_season` | 100% | 90% | ⚠️ |
| `session` | 90% | 91% | ✅ |
| `num_drivers` | 100% | 100% | ✅ |

---

## 3. Dataset pipeline gaps (`DATASET_GENERATION_PLAN.md`)

### Enrichment architecture gap (reference vs Ergast)

| Issue | Detail |
|-------|--------|
| **Standings timing** | `reference_enrich.py` fills from `seasons.json` **season-end totals** first. Ergast fallback *can* use round N−1 (`ergast.py` `standings_round = round - 1`) but only for **empty** cells (`fill_gaps_only=True`). Most rows keep season totals. |
| **Point-in-time test** | Plan requires automated test: “British GP round N must not include British GP results” — **not implemented** (`test_enrichment_ergast.py` missing). |
| **Ergast `superlicense_points_before_incident`** | Specified in plan §4a — **not implemented** in any enricher. |
| **Enrichment order** | Reference → Ergast → FastF1 is implemented; reference data dominates standings. |

### Parsing & incident-building gaps

| Issue | Status | Notes |
|-------|--------|-------|
| `parse_confidence` in review queue | ⚠️ Partial | Flagged when `< 0.7` in `validation.py`; 20 + 37 review rows |
| PDF `raw_text` NLP sidecar | ❌ | Stored in interim JSON; no NLP pipeline (`project_spec` §5.4) |
| Season-specific PDF templates | ⚠️ Unknown | Plan mentions parser templates by era — single parser today |
| Multi-car dedup ambiguity | ⚠️ | Conservative linking; ambiguous cases → review queue |
| `driver_at_fault` extraction | ❌ | 4–5% fill; Reason/Fact heuristics weak |
| Correction doc superseding | ✅ | Implemented in `incident_builder.py` |
| FIA HTML / WAF changes | ⚠️ Ongoing | 403 blocks 2020–2024; selectors may break |

### Missing tests (planned in `DATASET_GENERATION_PLAN.md`)

| Planned test file | Status |
|-------------------|--------|
| `test_enrichment_ergast.py` | ❌ Not created |
| `test_enrichment_fastf1.py` | ❌ Not created |
| Point-in-time standings test | ❌ Not created |
| `test_download.py` | ✅ Exists |
| `test_parsing.py` | ✅ Exists |
| `test_incident_builder.py` | ✅ Exists |
| `test_validation.py` | ✅ Exists |

### Dataset plan success criteria vs actual

| Criterion | Target | Actual | Met? |
|-----------|--------|--------|------|
| Incident rows per season | 150+ | 203 / 343 | ✅ |
| Label columns filled (Decision/Offence) | > 95% | ~91–92% `penalty` | ⚠️ |
| Ergast point-in-time standings | Verified by test | Season totals dominate | ❌ |
| FastF1 weather/SC (race) | > 70% | 82% / 62% | ⚠️ |
| FastF1 lap/time | > 50% | 0% | ❌ |
| Seasons 2020–2024 | Pipeline ready | No CSVs | ❌ |
| Idempotent re-runs | Identical output | Implemented | ✅ |

---

## 4. Data quality issues (non-column)

| Issue | 2019 | 2025 | Impact |
|-------|------|------|--------|
| Multi-driver column misalignment | ~19 incidents | ~21 incidents | Flatten **skips** row; loses training data |
| Review queue rows | 20 | 37 | Manual backlog (`severity`, `lap`, low confidence) |
| Summons-only / unmapped penalty | some | some | Excluded from training pool |
| Two-season temporal gap | — | — | Train 2019 → val 2025 is a 6-year distribution shift |
| Class imbalance (major penalties) | — | — | Class 2: ~14 val support; recall ~21% (V1 report) |

### Suggested enrichment fix order

1. Multi-driver column alignment (per-driver Ergast lookup)
2. `lap` / time alignment → unlocks `lap_remaining`, `completion_percentage`, `race_stage`
3. Point-in-time standings (prefer Ergast round N−1 over `seasons.json` totals)
4. `positions_of_involved parties` + `flag`
5. `superlicense_points_before_incident` (dataset-internal rolling)
6. `severity` (manual review)

---

## 5. Model training gaps (`MODEL_TRAINING_PLAN.md`)

### Pipeline status

| Component | Status |
|-----------|--------|
| Prepare / flatten / encode / split | ✅ |
| Baselines (majority + session-stratified) | ✅ |
| XGBoost V1 + early stopping | ✅ |
| Evaluation + training report | ✅ |
| Held-out **test** season | ❌ `test_season: null` |
| LightGBM drop-in (`tabular_classifier.py`) | ❌ Not built |
| `test_training_smoke.py` | ❌ Not created (covered partially by `test_training_scaffold.py`, `test_xgboost.py`) |
| Leave-one-season-out CV (small-data mode) | ❌ Not implemented |

### V1 success criteria vs actual

| Criterion | Status | Notes |
|-----------|--------|-------|
| End-to-end `processed_*.csv` → `ml_models/xgboost/` | ✅ | |
| Flattened parquet, no multi-value cells | ✅ | 234 rows in `incidents.parquet` |
| Leakage audit passes | ✅ | Automated in evaluate + tests |
| Temporal split verified | ✅ | 2019 train / 2025 val |
| Macro-F1 > 0.40 | ✅ | **0.402** (after redundant-feature cleanup) |
| Macro-F1 > baseline + **0.10** | ⚠️ | +0.135 vs majority ✅; +0.043 vs session baseline ❌ |
| Training report + figures | ✅ | `reports/model_reports/v1_training_report_*.md` |
| Reproducible metrics (seed=42) | ✅ | |

### Modeling & encoding gaps

| Gap | Notes |
|-----|-------|
| **Ordinal encoding** for categoricals | `session`, `circuit`, etc. use `OrdinalEncoder` — not ideal for nominal categories; no native XGBoost categorical |
| **High missingness drop** | Columns with > 30% missing excluded (`positions_of_involved parties`, etc.) |
| **Small train set** | 90 rows — overfitting risk; shallow trees mitigate but limits performance |
| **Removed redundant V1 features** | `is_race_session`, `is_qualifying`, `top_4_driver`, `top_4_opponent` removed intentionally (schema-aligned); plan text still lists them |
| **Class 2 (major) weak** | Precision ~0.17, recall ~0.21 on validation (V1 report) |

### V2 preprocessor encode drops (2019 train slice)

64 columns in `feature_columns` → **49** in `output_columns` (`preprocessor_xgboost_v2.meta.json`) after fixing `encoding.py` to include V2 feature sets (was 26 before fix).

| Status | Columns |
|--------|---------|
| ✅ Now encoded (V2) | `precedent_*`, `career_*`, `incidents_last_*`, `penalties_last_*`, `races_since_*`, `round_progress`, `points_gap_to_leader`, `points_available_remaining`, `title_contender`, `construct_title_contender`, `is_first_round`, `is_last_round`, `race_stage` |
| ❌ Still dropped (no train observations) | `flag`, `severity`, `lap`, `lap_remaining`, `completion_percentage`, `opponent_*`, `standing_difference`, `points_difference`, `points_gap_to_opponent`, `superlicense_points_before_incident` |

**Fixed in Phase D:** `encoding.py` previously only mapped V1 categorical/numeric/boolean sets, silently dropping all V2 engineered columns at fit time.

---

## 6. Feature engineering gaps (`FEATURE_ENGINEERING_PLAN.md`)

### Implementation phases

| Phase | Status | Deliverable |
|-------|--------|-------------|
| A — Scaffolding | ✅ | `configs/features.yaml`, `xgboost_v2.yaml`, CLI stages |
| B — Race + championship | ✅ | `race.py`, `driver.py` |
| C — History | ✅ | `history.py` + `test_history_rolling.py` |
| D — Precedent | ✅ | `precedent.py` + `test_precedent_temporal.py` |
| E — Selection + ablation | ✅ | `selection.py`, `ablation.py`, `reports/ablation_results.json` |
| F — Re-train + report | ✅ | `ml_models/xgboost_v2/`, `v2_feature_engineering_report_*.md` |

### V2 feature groups — implementation vs usability

| Group | Code | Usability blocker |
|-------|------|-------------------|
| A — `race_stage`, `round_progress`, … | ✅ | `race_stage` needs `completion_percentage` (0% fill) |
| B — `points_gap_to_leader`, `title_contender`, … | ✅ | Leader/gap uses standings that are season totals, not round N−1 |
| C — career + rolling history | ✅ | Works from labels; limited by 2 seasons + small train |
| D — `precedent_*` | ✅ | Active key `(incident_type, session)` — see §6.1 |
| F — opponent history (optional) | ❌ | Deferred to V2.1 ablation |

### 6.1 Phase D — precedent implementation notes

**Implemented:** `src/fia_ml/features/precedent.py` with temporal strict-prior filtering (same rule as history: same-season round `<` only).

| Feature | Definition |
|---------|------------|
| `precedent_count` | Prior incidents matching active similarity key |
| `precedent_no_penalty_rate` | Share with `penalty_severity == 0` among priors |
| `precedent_minor_penalty_rate` | Share with `penalty_severity == 1` |
| `precedent_major_penalty_rate` | Share with `penalty_severity == 2` |

**Active similarity key:** `(incident_type, session)` via `active_similarity_key` in `configs/features.yaml`.  
**Reason:** `severity` is 0% filled — the planned `(incident_type, severity, session)` key is configured but **dormant** until manual labeling.

**Fallback:** When `precedent_count < min_precedent_count` (3), rates impute to the **global** temporally-prior distribution (all incident types/sessions).

**Coverage on 234 driver-rows (post-flatten):**

| Metric | Value |
|--------|-------|
| `precedent_count == 0` | 43 rows (18%) |
| `precedent_count < 3` (uses global prior for rates) | 75 rows (32%) |
| `precedent_count >= 3` (group-specific rates) | 159 rows (68%) |
| NaN rate columns | 3 rows (first incidents with no global prior) |

**Remaining precedent gaps:**

| Gap | Impact |
|-----|--------|
| `severity` unlabeled | Cannot activate `(incident_type, severity, session)` key — coarser groups |
| Two-season corpus (2019 + 2025 only) | 2025 rows cannot see 2020–2024 precedents; 6-year distribution shift |
| Sparse `incident_type` × `session` cells | 32% of rows fall back to global prior |
| `circuit` in precedent key | Ablation-only; not active (risk of overfitting) |
| Cross-season precedent | 2019 incidents **do** count as priors for 2025 (strict temporal order) |

### Critical blocker for severity-based precedent (deferred)

Precedent similarity key `(incident_type, severity, session)` per original plan requires **`severity` manual labeling** (0% fill). Switch `active_similarity_key` in `configs/features.yaml` once `severity` is populated.

| Precedent column | Status |
|------------------|--------|
| `precedent_count` | ✅ |
| `precedent_no_penalty_rate` | ✅ |
| `precedent_minor_penalty_rate` | ✅ |
| `precedent_major_penalty_rate` | ✅ |

### 6.2 Phase E — selection & ablation results

**Implemented:** `src/fia_ml/features/selection.py` (missing-rate + correlation + importance prune), `src/fia_ml/training/ablation.py` (experiments A–E), integrated into `build_features_v2.py`.

**Selection steps (on 2019 train):**
1. Drop columns with >40% missing
2. Correlation prune (|r| > 0.95) — drop lower mutual-information column
3. Preliminary XGBoost → drop bottom 20% encoded features by gain

**Ablation macro-F1 (validation 2025):**

| Exp | Features | macro-F1 | Δ vs prev |
|-----|----------|----------|-----------|
| A | V1 baseline | **0.430** | — |
| B | + race/championship | 0.388 | −0.042 |
| C | + history | 0.423 | **+0.035** |
| D | + precedent | 0.349 | −0.075 |
| E | + selection prune | 0.371 | +0.022 |

**Key findings:**
- **History group (C)** is the only step meeting the +0.03 macro-F1 ablation target (+0.035 over B).
- **Precedent (D)** hurts validation (−0.075) — likely sparse groups + two-season corpus + 6-year shift.
- **Selection (E)** recovers some loss vs D (+0.022) but **full V2 still below V1** (0.371 vs 0.430).
- V1 ablation baseline (0.430) exceeds prior V1 trained report (0.402) — ablation retrains in-memory with current encode pipeline; small differences expected.

**Artifacts:**
- `reports/ablation_results.json`
- `reports/selection_report_v2.json`
- `train_v2.parquet` now has **21** post-selection encoded features (down from 49 pre-prune)

**Remaining Phase E gaps:**

| Gap | Notes |
|-----|-------|
| V2 does not beat V1 | Valid negative result — document in Phase F report |
| Correlation prune on small train (90 rows) | Aggressive drops (e.g. `round`, `season`, `driver_standing`) — may over-prune |
| `race_stage` dropped at missing step | 0% `completion_percentage` fill |

### 6.3 Phase F — final V2 model & report

**Trained:** `ml_models/xgboost_v2/model.json` on pruned `train_v2.parquet` (25 encoded features).

| Model | Validation macro-F1 | Best iteration |
|-------|---------------------|----------------|
| V1 XGBoost (saved) | **0.402** | 30 |
| V2 XGBoost (final) | 0.381 | 18 |

**V2 does not beat V1** (−0.021). Documented as valid negative result — small train set (90 rows), two-season gap (2019→2025), precedent group hurt ablation, aggressive selection on sparse columns.

**Artifacts:**
- `reports/model_reports/v2_feature_engineering_report_2026-08-25.md`
- `reports/figures/v1_vs_v2_macro_f1.png`
- `reports/figures/feature_importance_v2_top25.png`
- `reports/figures/confusion_matrix_v2_val.png`
- `ml_models/xgboost_v2/feature_importance.json`

**FE plan success criteria (final):**

| Criterion | Result |
|-----------|--------|
| `features_v2.parquet` with Groups A–E | ✅ |
| Temporal leakage tests | ✅ |
| Ablation A–E | ✅ |
| V2 macro-F1 ≥ V1 | ❌ 0.381 vs 0.402 |
| Engineered group +0.03 step | ✅ History (C) +0.035 |
| Engineered in top-15 importance | ✅ (precedent/history features present) |
| `v2_feature_engineering_report_{date}.md` | ✅ |

| Planned | Status |
|---------|--------|
| `test_history_rolling.py` | ✅ |
| `test_precedent_temporal.py` | ✅ (6 tests) |
| `test_selection.py` | ✅ (3 tests) |
| Extended leakage audit (precedent/history) in `leakage_filter.py` | ⚠️ Partial — precedent columns registered in `V2_NUMERIC_FEATURES`; no dedicated audit helper |
| Ablation experiments A–E | ✅ |
| `ablation_results.json` | ✅ `reports/ablation_results.json` |
| `reports/figures/v1_vs_v2_macro_f1.png` | ✅ |
| `reports/figures/feature_importance_v2_top25.png` | ✅ |
| `v2_feature_engineering_report_{date}.md` | ✅ |

### FE success criteria vs actual

| Criterion | Status |
|-----------|--------|
| `features_v2.parquet` with Groups A–E columns | ✅ A–D engineered; E selection applied at build |
| Temporal leakage tests (history + precedent) | ✅ |
| Ablation A–E in `ablation_results.json` | ✅ |
| V2 macro-F1 ≥ V1 | ❌ 0.381 vs 0.402 (documented) |
| Engineered group +0.03 macro-F1 step | ✅ History (C) +0.035 |
| Engineered features in top-15 importance | ✅ |

### Known V2 design limitations

| Limitation | Notes |
|------------|-------|
| `races_since_last_penalty` / `races_since_last_incident` | Same-season only; `NaN` when last event was prior season |
| `is_final_laps` | Intentionally **not** implemented (redundant with `race_stage.final_laps`) |
| `selection.py` correlation + importance prune | ✅ Phase E |
| `circuit` in precedent key | Configured for ablation only; risk of sparse groups |

---

## 7. Normative rules (`NORMATIVE_RULES_PLAN.md`)

### Implementation phases

| Phase | Status | Deliverable |
|-------|--------|-------------|
| A — Rule schema & loader | ✅ | `configs/normative_rules.yaml`, `configs/normative.yaml`, `rules_loader.py`, `--validate-rules` CLI |
| B — Condition engine | ✅ | `conditions.py` + `test_conditions.py` (16 tests) |
| C — Escalation pre-pass | ❌ **Next** | `escalation.py` stub |
| D — Rule engine + batch predict | ❌ | `rule_engine.py`, `predict.py`; need ≥15 rules (currently **10**) |
| E — Comparison & reporting | ❌ | `compare.py`, `report.py` |
| F — Review & iteration | ❌ | Manual review of top deviations |

### Phase A deliverables (done)

| Artifact | Status |
|----------|--------|
| `configs/normative_rules.yaml` | ✅ v1.0.0, 7 assumptions, **10 starter rules** + default catch-all |
| `configs/normative.yaml` | ✅ escalation/comparison/unmatched settings |
| `src/fia_ml/normative/schema.py` | ✅ dataclass schema + validation |
| `src/fia_ml/normative/rules_loader.py` | ✅ load + SHA-256 content hash |
| `src/fia_ml/normative/run_normative.py` | ✅ `--validate-rules` works |
| Module stubs | ✅ `escalation`, `rule_engine`, `predict`, `compare`, `report` (conditions ✅) |
| `tests/normative/test_rules_loader.py` | ✅ 6 tests passing |

**Validate:** `python -m fia_ml.normative.run_normative --validate-rules`

### Phase B deliverables (done)

| Capability | Status |
|------------|--------|
| `conditions.py` — `evaluate_conditions()` | ✅ |
| Implicit `eq`, `session_in`, `fact_contains_any`, `gte`/`lt`/`eq`/`in`/`contains` | ✅ |
| Nested `and` / `or`, `default: true`, AND semantics on leaf keys | ✅ |
| `normalize_session()` for race/qualifying/sprint/practice | ✅ |
| `tests/normative/test_conditions.py` | ✅ 16 tests |

### Normative gaps & blockers

| Gap | Impact | Planned fix |
|-----|--------|-------------|
| **Only 10 rules** (plan target ≥15) | Many rows will hit `default_unmatched` | Phase D — author rules for `other` (128 rows), `technical`, `pit_lane` without fact text |
| **`fact` text not on `incidents.parquet`** | `fact_contains_any` always false until joined | Join from `data/interim/extracted_documents/{season}/` during predict (Phase D) |
| **Escalation counters not computed** | `driver_track_limits_last_5_races` etc. undefined on rows | Phase C `escalation.py` |
| **Condition evaluator missing** | ~~No normative predictions yet~~ | ✅ Phase B — still needs rule engine (Phase D) |
| **`incident_type: other` dominates** (128/234 rows) | Low rule coverage until `other` is split or keyword rules added | Inspect PDF Facts; add subtype rules or improve classifier |
| **`session` normalization edge cases** | Long corrupted session strings in 2025 data map to non-race tokens | Data cleanup or stricter session parser in dataset pipeline |
| **No `rules_version.json` written yet** | Hash computed at load only | Write to `ml_models/normative/` in Phase D predict pass |
| **No deviation report / parquet output** | Success criteria unmet | Phases D–E |
| **FIA history vs normative history modes** | Config flag exists; neither implemented | Phase C |
| **No fixture file** `tests/fixtures/normative_incidents.json` | Planned 20–30 synthetic rows | Phase D/F |

### Success criteria vs actual (plan §393–404)

| Criterion | Status |
|-----------|--------|
| ≥15 documented rules | ❌ 10 starter rules |
| Deterministic run on full `incidents.parquet` | ❌ Phase D |
| ≥80% rows matched (≤20% manual_review) | ❌ not measured |
| Deviation report + breakdowns | ❌ Phase E |
| Unit tests all categories | ⚠️ loader + conditions ✅; escalation/engine/compare pending |
| Non-trivial deviation shown | ❌ |
| Top deviation cases reviewed | ❌ Phase F |
| `rules_version.json` tracks hash | ❌ load-time hash only |

Depends on point-in-time history (V2 Groups C/D logic reusable in Phase C escalation).

---

## 8. Deferred / out-of-scope

| Item | Plan | Notes |
|------|------|-------|
| NLP / BERT on `raw_text` | `project_spec` V2 | Interim JSON exists; no training pipeline |
| Embedding / FAISS precedent search | Feature spec §6 | Deferred until ~2000+ incidents |
| Normative rules engine | `NORMATIVE_RULES_PLAN.md` | Phase A done — see §7 |
| Opponent history (Group F) | `FEATURE_ENGINEERING_PLAN.md` V2.1 | Ablation-gated |
| Nationality bias ablation | Feature spec §28 | Optional |
| Track characteristic features | Feature spec | Not in current CSV schema |
| Telemetry / multimodal | `project_spec` | Out of scope |
| Hyperparameter tuning (V2) | `FEATURE_ENGINEERING_PLAN.md` | After feature set locked |

---

## 9. Master checklist — fill / fix after core ML work

### Seasons & rows
- [ ] Seasons **2020–2024** (full pipeline or manual PDFs)
- [ ] Third season for **test** split (e.g. hold out 2024)
- [ ] Fix **multi-driver misalignment** (~40 incidents total)
- [ ] Clear **review queue** (57 rows combined)

### High-priority columns
- [ ] `lap` → unlocks `lap_remaining`, `completion_percentage`, `race_stage`
- [ ] `severity` (manual) → enables `(incident_type, severity, session)` precedent key
- [ ] `flag`
- [ ] `positions_of_involved parties`
- [ ] `superlicense_points_before_incident`
- [ ] `sector` (especially 2025 — 3% fill)
- [ ] Point-in-time **standings** (round N−1, not season totals)
- [ ] `construct_standings`, `construct_points` (raise from ~55–68%)
- [ ] Multi-driver `driver_standings`, `nationalities`, `years_in_sport`

### Feature engineering & training
- [x] Phase D — `precedent.py` (fallback key without `severity` active)
- [x] Phase E — `selection.py`, ablation A–E
- [x] Phase F — train `xgboost_v2`, v1 vs v2 report
- [x] Fix V2 columns dropped at encode (`encoding.py` V2 feature sets)
- [ ] `test_enrichment_ergast.py` + point-in-time standings test

### Normative rules
- [x] Phase A — schema, loader, `--validate-rules` CLI
- [x] Phase B — `conditions.py` + tests
- [ ] Phase C — `escalation.py` + temporal tests
- [ ] Phase D — rule engine, ≥15 rules, `incidents_with_normative.parquet`
- [ ] Phase E — deviation report + figures
- [ ] Phase F — manual review top deviations

---

## 10. Related artifacts

| Path | Purpose |
|------|---------|
| `dataset/csv/review_queue_{season}.csv` | Manual review backlog |
| `reports/tables/data_quality_{season}.json` | Per-column fill rates + validation errors |
| `data/interim/extracted_documents/{season}/` | Parsed PDF JSON + `raw_text` (NLP-ready) |
| `configs/data.yaml` | Scraper + enrichment toggles |
| `configs/features.yaml` | V2 feature thresholds |
| `reports/ablation_results.json` | Ablation A–E macro-F1 |
| `reports/model_reports/v2_feature_engineering_report_*.md` | V1 vs V2 comparison + selection rationale |
| `ml_models/xgboost_v2/` | Final V2 model artifacts |
| `configs/xgboost.yaml` / `xgboost_v2.yaml` | Training splits + paths |
| `current_gaps.md` | This file |
| `configs/normative.yaml` / `normative_rules.yaml` | Normative engine config + rule set |
| `src/fia_ml/normative/` | Rule engine package (Phase A scaffold) |
