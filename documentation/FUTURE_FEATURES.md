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
| Normative rule engine + deviation report | `configs/normative_rules.yaml`, `ml_models/normative/`, `reports/normative/` |

See [`README.md`](../README.md) for paths and metrics.

---

## Roadmap overview (from feature specification)

The feature spec defines a **version ladder**. Versions below that are not yet implemented in this repo are future work.

| Version | Focus | Status in repo |
|---------|--------|----------------|
| **V1** | Tabular ML (structured features + XGBoost) | Built |
| **V2 (spec)** | NLP on FIA report text (BERT / DistilBERT) | **Not built** — `ml_models/nlp/` is empty |
| **V3** | Embedding / similarity precedent retrieval (FAISS, sentence transformers) | **Not built** — simplified groupby precedent only (V2 features) |
| **V4** | Telemetry (speed, braking, gaps, positions from FastF1) | **Not built** |
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
| **Lap / session timestamp** | **Open** | PDF `time` is document clock — ~1.5% valid offsets; critical path |
| **`flag`**, **positions**, **`sector` targets** | **Open** | Depend on session timestamp + turn maps |
| **`severity` in CSV** | **Open** | Suggestions in review queue only; manual promotion |
| **Seasons 2020–2024** | **Open** | FIA WAF / manual PDF backfill |
| **Re-run flatten + V2** | **Open** | Parquet still pre-enrichment |
| **Season-specific PDF templates** | Future | Parser maintenance per era |
| **NLP sidecar on `raw_text`** | Future → §2 | Beyond rule-based `text_fields.py` |

---

## 2. NLP / text models (spec “Version 2”)

From [`project_spec.md`](project_spec.md) §5.2, §8 Phase 5, and feature spec §5.4, §33 Version 2.

**Goal:** Classify penalties from FIA steward **report text** (Fact, Reason, Decision), not only tabular fields.

| Item | Planned detail |
|------|----------------|
| **Input** | `data/interim/extracted_documents/{season}/*.json` → `raw_text`, `parsed_fields` |
| **Models** | DistilBERT, BERT, or RoBERTa fine-tuned for 3-class `penalty_severity` |
| **Artifacts** | `ml_models/nlp/`, `configs/bert.yaml` (spec — not created) |
| **Code** | `src/fia_ml/models/nlp_model.py` (spec — not created) |
| **Experiments** | Text-only vs tabular-only vs fused features |
| **Notebook** | `06_nlp_experiments.ipynb` (spec — not in repo) |

**Also helps:** normative `fact_contains_any` rules, `incident_type` refinement for `other` rows, `driver_at_fault` extraction.

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

## 4. Telemetry (spec “Version 4”)

From feature spec §33 Version 4 and `project_spec` Phase 6+.

| Signal | Source |
|--------|--------|
| Speed, braking, throttle, steering | FastF1 car data |
| Gap, relative speed, track position | FastF1 timing + telemetry |
| Aggregated session features | `data/interim/processed_telemetry/` (spec layout) |

**Goal:** Represent physical circumstances of the incident; feed tabular NN or multimodal fusion.

**Dependency:** Incident **session** time alignment — enrichment modules exist but only ~1.5% of rows have valid `session_offset_seconds` (PDF document clock vs on-track time).

---

## 5. Precedent & similarity (spec “Version 3”)

From feature spec §12 (embedding precedent) and `project_spec` §5.3.

**Current approach:** Groupby precedent rates (`precedent_*` in V2 features) — implemented, limited by small corpus.

**Future approach:**

```text
Incident text / features → Sentence Transformer → vector DB (FAISS / ChromaDB)
                                                      → top-K similar historical cases
                                                      → precedent probability features
```

| Item | Notes |
|------|-------|
| **Threshold** | Spec suggests revisiting when **~2000+** incidents |
| **Technologies** | Sentence Transformers, FAISS, ChromaDB |
| **Deliverables** | Similarity search API, enriched precedent features, explainability UI |

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
| **Multimodal tabular + text** | Early fusion experiments before full V5 |

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
| **Precedent explainability** | “Similar historical cases” for stewards research UI |

---

## 9. Infrastructure & tooling (spec)

From `project_spec` directory layout — planned but not present.

| Item | Spec path |
|------|-----------|
| `configs/bert.yaml`, `configs/cnn.yaml` | NLP / CNN training configs |
| `notebooks/01`–`08` | Exploration notebooks |
| `experiments/experiment_003_nlp/` | Experiment tracking |
| `data/raw/telemetry/` | Telemetry raw store |
| `data/raw/video_metadata/` | Frame metadata for CNN |
| `src/fia_ml/features/telemetry.py` | Telemetry feature module |

---

## 10. Suggested priority order

When choosing what to build next:

1. **Data volume** — seasons 2020–2024, fix PDF session timestamps (lap/flag), multi-driver rows, re-run flatten + V2  
2. **Normative iteration** — Fact text + rules for `other` / collisions (low engineering risk)  
3. **NLP on steward text** — uses existing interim JSON; complements tabular V1  
4. **Re-train tabular models** — with richer data; retry V2 features + opponent history  
5. **Embedding precedent (V3)** — when incident count &gt; ~2000  
6. **Telemetry (V4)** — after incident timestamp alignment works  
7. **CNN / multimodal (V5)** — after video metadata pipeline exists  

---

## Related files

| File | Role |
|------|------|
| [`current_gaps.md`](../current_gaps.md) | Open gaps on **existing** systems |
| [`README.md`](../README.md) | What is built and where artifacts live |
| [`DL_MODEL_TRANSFER_LEARNING.md`](../DL_MODEL_TRANSFER_LEARNING.md) | CNN / transfer-learning study guide (external projects) |
