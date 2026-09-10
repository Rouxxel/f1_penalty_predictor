# Future Features & Roadmap

> **Purpose:** Planned capabilities that are **not built yet** — distinct from [`current_gaps.md`](../current_gaps.md), which tracks incomplete work on **existing** pipelines (dataset quality, normative rule coverage, modeling limitations on current data).

**Authoritative vision docs:** [`f1_project.md`](f1_project.md) · [`project_spec.md`](project_spec.md) · [`FIA_stewarding_dataset_feature_specification.md`](FIA_stewarding_dataset_feature_specification.md)

**Note:** Dataset generation, V1/V2 tabular training, and the normative engine are built. This file reflects **current** future intent.

---

## What exists today

| Capability | Artifacts |
|------------|-----------|
| Dataset pipeline (2019 + 2025) | `dataset/scripts/run_pipeline.py`, `processed_{season}.csv` |
| Data enrichment (hybrid stack) | `configs/enrichment.yaml`, `src/fia_ml/data/enrichment/`, `data/interim/enrichment_meta/` |
| V1 tabular model | `ml_models/xgboost/` |
| V2 tabular model + ablation | `ml_models/xgboost_v2/` |
| NLP text model (spec V2) | `ml_models/nlp/` — val macro-F1 **0.642** (2026-09-09); beats V1 tabular on 2025 |
| Normative rule engine + deviation report | `configs/normative_rules.yaml`, `ml_models/normative/`, `reports/normative/` |

See [`README.md`](../README.md) for paths and metrics.

---

## Roadmap overview (from feature specification)

The feature spec defines a **version ladder**. Versions below that are not yet implemented in this repo are future work.

| Version | Focus | Status in repo |
|---------|--------|----------------|
| **V1** | Tabular ML (structured features + XGBoost) | Built |
| **V2 (spec)** | NLP on FIA report text (BERT / DistilBERT) | **Built** (pipeline) — weights pending; see §2 |
| **V3** | Embedding / similarity precedent retrieval | **Deferred** — groupby precedent only today (`precedent_*` in V2); see §5 |
| **V4** | Telemetry (speed, braking, gaps, positions from FastF1) | **Deferred** — Track A (timestamps) open in §1; Track B (car telemetry) §4 |
| **V5** | Visual / CNN (onboard frames, replay stills) + multimodal fusion | **Not built** — `ml_models/cnn/` is empty |

Normative rules sit **alongside** the ML track (comparison layer), not as a version number — engine is built; rule **coverage** gaps are in [`current_gaps.md`](../current_gaps.md) §7.

---

## 1. Dataset & enrichment

**Built (2026-09):** Hybrid enrichment pipeline — reference → Ergast (round N−1) → `timestamp.py` → OpenF1 (2023+) → FastF1 → `superlicense.py` → `text_fields.py` → provenance sidecar → quality gates. Config: [`configs/enrichment.yaml`](../configs/enrichment.yaml). Tests: `tests/test_enrichment_*.py` (37 tests). Reports: `reports/tables/enrichment_report_{season}.json`.

Runbook: [`dataset_generation_runbook.md`](dataset_generation_runbook.md) · open gaps: [`current_gaps.md`](../current_gaps.md) §1–§4.

| Feature | Status | Notes |
|---------|--------|-------|
| **Point-in-time standings** | **Built** | Ergast round N−1 overwrite; 2019 96%, 2025 78% provenance |
| **`superlicense_points_before_incident`** | **Built** | Rolling per driver; 99% (2019) / 81% (2025) fill |
| **Provenance sidecar** | **Built** | `data/interim/enrichment_meta/{season}.json` |
| **Quality gates** | **Built** | Merged into `data_quality_{season}.json` |
| **`driver_at_fault` assist** | **Built** (partial) | Rule-based; ~25% fill; low-confidence → review queue |
| **OpenF1 + FastF1 v2** | **Built** (blocked on time) | `flag`, `positions`, lap join wired but need session timestamp |
| **Lap / session timestamp (V4 Track A)** | **Open — active** | PDF `time` is document clock — ~1.5% valid offsets; unlocks lap/flag/positions **and** is prerequisite for V4 telemetry; see [`v4_research.md`](../v4_research.md) |
| **`flag`**, **positions**, **`sector` targets** | **Open** | Depend on session timestamp + turn maps |
| **`severity` in CSV** | **Open** | Suggestions in review queue only; manual promotion |
| **Seasons 2020–2024** | **Open** | FIA WAF / manual PDF backfill |
| **Re-run flatten + V2** | **Open** | Parquet still pre-enrichment |
| **Season-specific PDF templates** | Future | Parser maintenance per era |
| **NLP sidecar on `raw_text`** | **Built** | DistilBERT classifier — §2 |

---

## 2. NLP / text models (spec “Version 2”) — built

From [`project_spec.md`](project_spec.md) §5.2, §8 Phase 5, and feature spec §5.4, §33 Version 2.

**Goal:** Classify penalties from FIA steward **report text** (Fact + Offence by default; Decision/Reason excluded for leakage safety).

| Item | Status |
|------|--------|
| **Input** | `data/interim/extracted_documents/{season}/*.json` joined via `raw_incidents_{season}.meta.json` |
| **Default text profile** | `fact_offence` — Fact + Offence + session metadata |
| **Model** | DistilBERT (`distilbert-base-uncased`) → 3-class `penalty_severity` |
| **Config** | [`configs/bert.yaml`](../configs/bert.yaml) |
| **Code** | `src/fia_ml/nlp/`, `src/fia_ml/models/nlp_model.py`, `train_nlp.py`, `evaluate_nlp.py`, `fusion_nlp.py` |
| **CLI** | `python -m fia_ml.training.run_nlp_training --config configs/bert.yaml --stage all` |
| **Artifacts** | `ml_models/nlp/model/`, `tokenizer/`, `metrics.json`, `predictions_val.json`, `nlp_dataset_audit.json` |
| **Fusion (optional)** | `concat_logits` or `stack_xgb` vs V1 tabular — `--fusion` or `fusion.enabled: true` |
| **Tests** | `tests/test_nlp_*.py` (config, leakage, dataset join, evaluate, fusion; CPU smoke) |

**Trained (2026-09-09):** 90 train / 154 val rows; macro-F1 **0.642** vs V1 **0.402**; fusion `concat_logits` **0.632** (does not beat NLP-only). Class 2 recall remains weak (12.5%).

**Still optional:**

- Exploratory notebook `06_nlp_experiments.ipynb` — not in repo
- Re-train when 2020–2024 seasons are backfilled

**Also helps:** normative `fact_contains_any` rules, `incident_type` refinement for `other` rows — shared interim JSON path.

Open limitations: [`current_gaps.md`](../current_gaps.md) §5 (NLP).

---

## 3. CNN / visual models (spec “Version 5”)

From [`project_spec.md`](project_spec.md) §5.1 (`ml_models/cnn/`), §8 Phase 6+, and feature spec §33 Version 5.

**Goal:** Use **visual evidence** (onboard camera, race control replay frames) as an additional input modality for sanction prediction or incident understanding.

| Item | Planned detail |
|------|----------------|
| **Input** | `data/raw/video_metadata/` — frame paths, timestamps linked to `incident_id` |
| **Architecture** | CNN backbone (e.g. ResNet, EfficientNet, or transfer learning) → visual embedding |
| **Fusion** | Combine with tabular + (optional) BERT text + telemetry embeddings |
| **Artifacts** | `ml_models/cnn/`, `configs/cnn.yaml` (spec — not created) |
| **Code** | `src/fia_ml/models/cnn_model.py`, `multimodal.py` (spec — not created) |
| **F1-specific use cases** | Incident clip classification, track-limit camera evidence, collision context |

**Repo today:** `ml_models/cnn/__init__.py` only — no F1-trained weights.

**Personal reference (not F1):** Imported examples for learning CNN patterns — [`DL_MODEL_TRANSFER_LEARNING.md`](../DL_MODEL_TRANSFER_LEARNING.md) (emotion detection + DenseNet169), [`robust_cnn_model.py`](../robust_cnn_model.py) (audio spectrogram CNN). These are **not** connected to the F1 pipeline.

---

## 4. Telemetry (spec “Version 4”) — deferred (Track B)

From feature spec §33 Version 4 and `project_spec` Phase 6+.

**Decision (2026-09):** Car-level telemetry ML (`telemetry.py`, `configs/telemetry.yaml`, orchestration) is **not on the critical path**. Defer until timeline gates pass **and** enough seasons exist for a meaningful ablation. Research spec: [`v4_research.md`](../v4_research.md).

V4 splits into two tracks — do not conflate them:

| Track | Scope | Status |
|-------|--------|--------|
| **Track A — timeline alignment** | `session_offset_seconds`, lap/flag/positions joins | **Active** — enrichment gap in §1; runnable on 2019 + 2025 without 2020–2024 |
| **Track B — car telemetry features** | speed, brake, throttle, gap aggregates → XGBoost | **Deferred** — needs Track A + revisit gates below |

### What exists today

FastF1 enrichment (`fastf1_enrich.py`) loads `telemetry=False` — weather, laps, and race-control only. OpenF1 (2023+) is wired for flag/positions but blocked by the same ~1.5% timestamp rate. DistilBERT NLP at **0.642** macro-F1 is the current best predictor; telemetry must show incremental signal on joined rows to justify build-out.

### Why defer Track B (especially without 2020–2024)

| Factor | Detail |
|--------|--------|
| **Timeline gate** | ~1.5% valid `session_offset_seconds` — G0 not passed |
| **Two seasons only** | ~234 flattened rows; collisions/track-limits are a small subset — ablation would be inconclusive |
| **Missing 2020–2024** | Cannot validate join strategy across eras (2019 FastF1-only vs 2023+ OpenF1) or build collision sample size |
| **NLP overlap** | Text may already encode “heavy braking collision” — same redundancy risk as V3 |
| **Track A still wins** | Fixing timestamps improves `lap`, `flag`, `positions` for V1/V2 even if telemetry ML never ships |

### Revisit gates (Track B)

| Stage | Gate | Action |
|-------|------|--------|
| **Track A complete** | ≥50% race incidents with high-confidence temporal anchors (`v4_research.md` Gate A) | Continue enrichment; optional 5–10 manual car_data sanity joins |
| **V4 population** | ≥50% of collision/track-limit incidents with usable driver/session/telemetry join (Gate B) | Run join audit → `reports/v4/telemetry_join_audit.md` |
| **Predictive signal** | Telemetry aggregates beat V1/NLP on joined rows (Gate G5) | `ADOPT` or `COLLISION-SPECIFIC` per `v4_research.md` §46 |
| **Infrastructure** | Positive ablation **and** cache/scale needs (Gate G6) | `telemetry.py`, `processed_telemetry/`, FastF1 `telemetry=True` at scale |

Until gates pass: no `src/fia_ml/features/telemetry.py`, no V4 orchestration phase. Negative results (timeline not ready, no ML signal) are valid outcomes.

### Future architecture (when justified)

```text
FIA incident → timeline reconstruction → confidence → session + driver join
                                                      → telemetry window (±2s / ±5s)
                                                      → small interpretable feature set → ablation
```

| Signal | Source |
|--------|--------|
| Speed, braking, throttle, steering | FastF1 / OpenF1 car data |
| Gap, relative speed, track position | FastF1 timing + telemetry |
| Aggregated session features | `data/interim/processed_telemetry/` (spec layout) |

**Start with:** `timestamp_alignment_report.md` and manual join validation — not `telemetry.py`.

---

## 5. Precedent & similarity (spec “Version 3”) — deferred

From feature spec §12 (embedding precedent) and `project_spec` §5.3.

**Decision (2026-09):** Embedding-based precedent retrieval is **not on the critical path**. Defer production implementation until the corpus and ablation evidence justify it. Research spec: [`v3.md`](../v3.md).

### What exists today

Groupby precedent rates (`precedent_*` in `src/fia_ml/features/precedent.py`, V2 features) — implemented but **hurt validation** on the two-season corpus (ablation −0.075 macro-F1 vs history-only). DistilBERT on the same `fact_offence` text already reaches **0.642** macro-F1, so semantic retrieval may largely duplicate NLP (see `v3.md` H3).

### Corpus reality

| Estimate | Detail |
|----------|--------|
| **Today** | ~234 flattened driver-rows (2019 + 2025); ~90 train |
| **After 2019–2025 backfill** | ~700–900 flattened rows (2020–2021 had fewer races — COVID) |
| **~2000 incidents** | Unlikely before **~2028** at current F1 incident rates |

The old “~2000 incidents” guideline from `project_spec` §6 remains a **hypothesis**, not a hard gate — but even optimistic backfill does not reach it soon.

### Revisit gates

| Stage | Gate | Action |
|-------|------|--------|
| **Research script** | ≥500 labeled flattened rows **and** thesis needs an H3 redundancy check | Optional standalone brute-force prototype (`v3.md` Steps 1–8); no FAISS/Chroma/orchestration |
| **Feature integration** | Brute-force beats groupby precedent **and** adds signal on top of NLP on a temporal split | Add V3 feature group to tabular pipeline; keep groupby for ablation |
| **Infrastructure** | ≥1000 rows **and** brute-force too slow or live “similar cases” API needed | Consider FAISS / ChromaDB |

Until a gate is met, treat V3 as a **future explainability layer** (“what did FIA do in similar cases?”) alongside the normative engine — not a predictor priority.

### Future architecture (when justified)

```text
Incident text (Fact + Offence) → embedding → strict historical filter → top-K
                                                      → precedent probability features
                                                      → optional “similar cases” audit UI
```

| Item | Notes |
|------|-------|
| **Start with** | NumPy brute-force cosine — not FAISS/Chroma (`v3.md` Phase 1) |
| **Later** | FAISS / ChromaDB only if corpus size or API needs require it |
| **Negative result** | Valid outcome — document redundancy with NLP and skip infrastructure |

---

## 6. Tabular ML enhancements

From feature spec and current V1/V2 training pipeline.

| Feature | Description |
|---------|-------------|
| **Held-out test season** | Third season (e.g. 2024) when data exists |
| **LightGBM** | Alternative tree model (`tabular_classifier.py`) |
| **Leave-one-season-out CV** | Small-data evaluation mode |
| **Better categorical encoding** | Native categorical / target encoding vs ordinal |
| **Hyperparameter tuning** | After feature set stabilizes |
| **Opponent history (Group F)** | Deferred in V2 feature engineering |
| **Severity-based precedent key** | `(incident_type, severity, session)` after manual `severity` labels |
| **SHAP / deeper explainability** | Beyond gain-based importance |
| **Multimodal tabular + text** | Late fusion (`fusion_nlp.py`) implemented; early/embedding fusion still future |

**Modeling limitations on current data** (small train set, V2 &lt; V1, class 2 weakness) are tracked in [`current_gaps.md`](../current_gaps.md) §5–§6 — fixable with more data and the items above.

---

## 7. Normative rules (future iteration)

Engine is built. Future work is **rule quality and coverage**, not greenfield implementation.

| Item | Description |
|------|-------------|
| Expand rules for `incident_type: other` | Reduce 59.8% `manual_review` rate |
| Fact text join from parsed PDFs | Enable collision fact-triggered rules |
| Manual review of top deviations | Documented rule changes + YAML version bumps |
| `normative_history` ablation | Escalation on FIA vs normative prior outcomes |
| `tests/fixtures/normative_incidents.json` | Synthetic regression set |
| Automated rule extraction from ISC | Explicitly out of scope for V1 — long-term research |

---

## 8. Analysis, fairness & explainability

From feature spec §27–§28, §34.

| Item | Description |
|------|-------------|
| **Nationality bias ablation** | Train with/without nationality features; compare metrics |
| **Fairness reporting** | By driver nationality, team, circuit — research interpretability |
| **Deviation storytelling** | Season/circuit/session clusters in normative vs FIA reports |
| **Precedent explainability** | “Similar historical cases” for stewards research UI — deferred with V3 (§5); normative deviation reports cover part of this today |

---

## 9. Infrastructure & tooling (spec)

From `project_spec` directory layout — planned but not present.

| Item | Spec path |
|------|-----------|
| `configs/bert.yaml` | NLP training config (**created**) |
| `configs/cnn.yaml` | CNN training config (future) |
| `notebooks/01`–`08` | Exploration notebooks |
| `experiments/experiment_003_nlp/` | Experiment tracking |
| `data/raw/telemetry/` | Telemetry raw store |
| `data/raw/video_metadata/` | Frame metadata for CNN |
| `src/fia_ml/features/telemetry.py` | Telemetry feature module (**deferred** — §4 Track B) |

---

## 10. Suggested priority order

When choosing what to build next:

1. **V4 Track A — session timestamps** — audit/fix alignment on 2019 + 2025; unlocks lap/flag/positions (does not need 2020–2024)  
2. **Data volume** — seasons 2020–2024 backfill, multi-driver rows, re-run flatten + V2  
3. **Normative iteration** — Fact text + rules for `other` / collisions (low engineering risk)  
4. **NLP** — re-train when backfill lands; improve class 2 recall; fusion only if it beats NLP-only  
5. **Re-train tabular models** — with richer data; retry V2 history features (not groupby precedent until corpus grows)  
6. **CNN / multimodal (V5)** — after video metadata pipeline exists  
7. **V4 Track B — telemetry ML** — **deferred** until §4 gates (timeline + seasons + positive ablation)  
8. **Embedding precedent (V3)** — **deferred** until revisit gates in §5 (~2027–2028 realistic; optional research script earlier)  

---

## Related files

| File | Role |
|------|------|
| [`current_gaps.md`](../current_gaps.md) | Open gaps on **existing** systems |
| [`v3.md`](../v3.md) | V3 research spec and revisit gates (deferred) |
| [`v4_research.md`](../v4_research.md) | V4 research spec — Track A active, Track B deferred |
| [`README.md`](../README.md) | What is built and where artifacts live |
| [`DL_MODEL_TRANSFER_LEARNING.md`](../DL_MODEL_TRANSFER_LEARNING.md) | CNN / transfer-learning study guide (external projects) |
