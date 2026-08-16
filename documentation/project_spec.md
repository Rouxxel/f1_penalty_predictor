# FIA Penalty Predictor — Project Specification

## 1. Project Overview

A machine learning system that predicts Formula 1 stewarding decisions (sanctions) given an on-track incident. The system operates in two modes:

1. **FIA Behavior Model** — Supervised classifier that learns historical steward decision patterns
2. **Normative Rules Model** — Rule-based system encoding a consistent interpretation of FIA regulations

The comparison between both outputs quantifies decision variability across seasons, incident types, and contexts.

---

## 2. Project Structure

```text
f1_penalty_predictor/
│
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── base.yaml
│   ├── data.yaml
│   ├── xgboost.yaml
│   ├── bert.yaml
│   └── cnn.yaml
│
├── data/
│   ├── raw/
│   │   ├── fia/                        # Original FIA steward decision PDFs
│   │   ├── race_data/                  # Race results, standings, calendar
│   │   ├── telemetry/                  # FastF1 telemetry (Version 4+)
│   │   └── video_metadata/             # Video frame metadata (Version 5+)
│   │
│   ├── interim/
│   │   ├── extracted_documents/        # Parsed text from FIA PDFs
│   │   ├── cleaned_reports/            # Cleaned/normalized report text
│   │   └── processed_telemetry/        # Aggregated telemetry features
│   │
│   └── processed/
│       ├── incidents.parquet           # Full incident dataset (wide)
│       ├── features.parquet            # Engineered feature matrix
│       ├── train.parquet               # Temporal split: training set
│       ├── validation.parquet          # Temporal split: validation set
│       └── test.parquet                # Temporal split: test set
│
├── dataset/
│   ├── csv/                            # Generated CSV datasets
│   └── scripts/
│       └── generate_from_pdf.py        # PDF → structured CSV pipeline
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_fia_documents.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_xgboost.ipynb
│   ├── 06_nlp_experiments.ipynb
│   ├── 07_similarity_search.ipynb
│   └── 08_telemetry_analysis.ipynb
│
├── src/
│   └── fia_ml/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── download.py             # Fetch FIA PDFs from sources
│       │   ├── ingestion.py            # Load raw data into pipeline
│       │   ├── parsing.py              # PDF text extraction + structuring
│       │   └── validation.py           # Schema validation, type checks
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── cleaning.py             # Text normalization, encoding fixes
│       │   ├── encoding.py             # Categorical encoding strategies
│       │   ├── feature_engineering.py  # Derived feature computation
│       │   └── splitting.py            # Temporal train/val/test splits
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── incident.py             # Incident-level features
│       │   ├── driver.py               # Driver identity + experience
│       │   ├── history.py              # Historical behavior features
│       │   ├── race.py                 # Race context + championship
│       │   ├── telemetry.py            # Telemetry features (Version 4+)
│       │   └── precedent.py            # Groupby-based precedent stats
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── baseline.py             # Simple frequency/majority baseline
│       │   ├── xgboost_model.py        # XGBoost/LightGBM classifier
│       │   ├── nlp_model.py            # BERT-based text classifier
│       │   ├── cnn_model.py            # CNN for visual data (Version 5+)
│       │   └── multimodal.py           # Fusion model (Version 5+)
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train.py                # Training loop orchestration
│       │   ├── evaluate.py             # Metrics, confusion matrix, reports
│       │   └── cross_validation.py     # Temporal cross-validation
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   └── predict.py              # Single-incident prediction
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── reproducibility.py      # Seed management, determinism
│
├── ml_models/
│   ├── baseline/                       # Saved baseline model artifacts
│   ├── xgboost/                        # Saved XGBoost model artifacts
│   ├── nlp/                            # Saved NLP model artifacts
│   └── cnn/                            # Saved CNN model artifacts
│
├── experiments/
│   ├── experiment_001_baseline/
│   ├── experiment_002_xgboost/
│   ├── experiment_003_nlp/
│   └── experiment_004_multimodal/
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── model_reports/
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── docs/
│   ├── dataset.md
│   ├── feature_schema.md
│   ├── methodology.md
│   ├── experiments.md
│   └── architecture.md
│
└── documentation/                      # Project planning docs (existing)
    ├── f1_dataset_arch.xlsx
    ├── f1_dataset_example.csv
    ├── f1_project.md
    ├── FIA_stewarding_dataset_feature_specification.md
    └── project_spec.md                 # This file
```

---

## 3. Data Architecture

### 3.1 Unit of Data

One row = **one driver under investigation for one incident**.

If an incident involves two drivers and both are investigated, that produces two rows sharing the same `incident_id` but with swapped driver/opponent roles. If only one driver is investigated (the common case), one row.

This eliminates the multi-value column problem (`**` concern) at the schema level.

### 3.2 Schema Design — Resolving the Multi-Value Problem

**Problem:** Columns like `drivers`, `nationalities`, `driver_standings` pack multiple values into a single cell (e.g., `"max_verstappen,lewis_hamilton"`).

**Solution:** Restructure from incident-centric to driver-investigation-centric:

```text
BEFORE (incident-centric, multi-value):
─────────────────────────────────────────────────────────
incident_id | drivers                          | standings
a001        | max_verstappen,lewis_hamilton     | 1,2

AFTER (driver-investigation-centric, flat):
─────────────────────────────────────────────────────────
incident_id | driver         | opponent        | driver_standing | opponent_standing
a001        | max_verstappen | lewis_hamilton   | 1               | 2
```

Each column now holds exactly one value. No parsing needed at training time.

### 3.3 Schema Design — Resolving the Single-String Problem

**Problem:** Columns like `circuit`, `country`, `track_conditions`, `weather_conditions`, `session`, `incident_type` contain categorical strings.

**Solution:** Encoding strategy depends on the model type:

| Encoding Method | When to Use | Example |
|---|---|---|
| Label encoding (ordinal int) | XGBoost, LightGBM (native categorical support) | `monza → 12` |
| One-hot encoding | Logistic regression, neural nets, low-cardinality features | `monza → [0,0,...,1,...,0]` |
| Target encoding | High-cardinality categoricals (30+ unique values) | `monza → 0.34` (mean penalty rate) |
| Frequency encoding | When category frequency itself is informative | `monza → 47` (number of incidents) |

**Implementation:** Encoding happens in `src/fia_ml/preprocessing/encoding.py`, not in the raw dataset. The raw CSV stores human-readable strings. The processed parquet files store encoded values.

### 3.4 Revised Column Schema (Flat, Single-Value)

```text
# Identity
incident_id             string      unique per incident
row_id                  string      unique per driver-investigation (incident_id + driver)

# Circuit & Race Context
circuit                 string *    → label/target encode
country                 string *    → label/target encode
first_season            int         year circuit joined F1 calendar
round                   int         championship round number
season                  int         year
total_rounds            int         total rounds in that season
num_teams               int         teams on the grid

# Race Progression
lap                     int         lap of incident
total_laps              int         race distance
laps_remaining          int         derived: total_laps - lap
completion_percentage   float       derived: lap / total_laps * 100
sector                  int         track sector (1, 2, or 3)

# Conditions
flag                    string *    → one-hot (yellow, red, none, etc.)
safety_car              string *    → one-hot (none, safety_car, vsc)
track_conditions        string *    → one-hot (dry, wet, damp)
weather_conditions      string *    → one-hot (sunny, cloudy, rain, etc.)

# Session
session                 string *    → one-hot (race, qualifying, sprint, practice)

# Incident
incident_type           string *    → one-hot or label encode
severity                string *    → ordinal encode (low=0, medium=1, high=2)

# Driver Under Investigation
driver                  string      → label encode or entity embedding
driver_nationality      string *    → label encode
driver_team             string *    → label encode
driver_position         int         position at time of incident
driver_standing         int         championship position before incident
driver_points           float       championship points before incident
driver_construct_standing   int     team WCC position
driver_construct_points     float   team WCC points
driver_years_in_sport   int         years of F1 experience
driver_superlicense_pts float       superlicense points before incident

# Opponent
opponent                string      → label encode or entity embedding
opponent_nationality    string *    → label encode
opponent_team           string *    → label encode
opponent_position       int         position at time of incident
opponent_standing       int         championship position before incident
opponent_points         float       championship points before incident
opponent_construct_standing int     team WCC position
opponent_construct_points   float   team WCC points
opponent_years_in_sport int         years of F1 experience
opponent_superlicense_pts   float   superlicense points before incident

# Relational
same_team               bool        → 0/1
standing_difference     int         derived: driver_standing - opponent_standing
points_difference       float       derived: driver_points - opponent_points

# Championship Context (derived)
top_4_driver            bool        → 0/1 (is driver in top 4 of standings?)
top_4_opponent          bool        → 0/1

# Investigation Context
investigation           bool        → 0/1 (was it formally investigated?)
incident_classification string *    → label encode (what stewards classified it as)

# TARGET VARIABLES (labels — never used as input features)
driver_at_fault         string      who stewards found at fault
penalty                 string      → ordinal or multi-class encode
superlicense_pts_added  int         penalty points assigned
mentioned_article       string      FIA regulation cited
```

### 3.5 Target Variable Encoding

Primary target for classification:

```text
penalty_class:
  0 = no_further_action
  1 = warning / reprimand
  2 = time_penalty (5s, 10s, drive-through)
  3 = grid_penalty
  4 = penalty_points_only
  5 = disqualification
```

Simplified target for first model (addresses class imbalance):

```text
penalty_severity:
  0 = no_penalty
  1 = minor (warning, reprimand, small time penalty)
  2 = major (grid drop, large time penalty, DSQ, points)
```

---

## 4. PDF-to-CSV Pipeline

### 4.1 Location

```text
dataset/
├── csv/                    # Output CSVs
│   ├── raw_incidents.csv   # Direct extraction from PDFs
│   └── processed.csv       # After flattening + validation
└── scripts/
    └── generate_from_pdf.py
```

### 4.2 Pipeline Steps

```text
data/raw/fia/*.pdf
       │
       ▼
  PDF text extraction (pdfplumber / PyMuPDF)
       │
       ▼
  Structured field parsing (regex + heuristics)
       │
       ▼
  Validation against schema
       │
       ▼
  dataset/csv/raw_incidents.csv
       │
       ▼
  Flatten multi-value columns → one-row-per-driver
       │
       ▼
  dataset/csv/processed.csv
```

### 4.3 Key Implementation Notes

- FIA decision documents follow a semi-structured format (incident number, drivers involved, fact, decision, reason). This is parseable with regex patterns.
- Supplementary data (standings, points, lap counts) must be joined from external sources (Ergast API, official F1 results).
- The script should be idempotent: re-running on the same PDFs produces the same output.

---

## 5. ML Model Pipelines

### 5.1 Location

```text
ml_models/
├── baseline/           # Majority-class + simple heuristic models
│   ├── model.pkl
│   └── metrics.json
│
├── xgboost/            # Gradient boosted trees (primary V1 model)
│   ├── model.json
│   ├── metrics.json
│   └── feature_importance.json
│
├── nlp/                # BERT-based text model (V2)
│   ├── model/
│   ├── tokenizer/
│   └── metrics.json
│
└── cnn/                # Visual model (V5, future)
    └── ...
```

### 5.2 Training Pipelines (in `src/fia_ml/models/`)

**Version 1 — Tabular Baseline:**
```text
processed.csv → encoding → temporal split → XGBoost/LightGBM → evaluation
```

**Version 2 — NLP Integration:**
```text
FIA report text → tokenization → BERT/DistilBERT → classification head → evaluation
```

**Version 3 — Precedent Features (simplified):**
```text
For each incident:
  group historical incidents by (incident_type, severity)
  compute penalty distribution for that group
  inject as additional features → XGBoost
```

This replaces the full embedding/vector-search similarity system with a simple groupby approach that works with limited data.

### 5.3 Temporal Split Strategy

```text
Train:       All incidents from seasons before the validation year
Validation:  One full season
Test:        One full season (held out entirely until final evaluation)

Example:
  Train:      2014–2022
  Validation: 2023
  Test:       2024
```

Never split randomly. Always by time.

---

## 6. Precedent System — Simplified Approach

### Why Not Embeddings/Vector DB

With an expected dataset of 200–1000 incidents, a vector similarity system offers no real advantage over a grouped statistical approach. The signal is the same: "what happened historically when a similar incident type occurred?"

### Simplified Precedent Features

For each incident, compute using only prior data:

```text
precedent_count                  # how many similar incidents existed before this one
precedent_no_penalty_rate        # proportion that got no penalty
precedent_minor_penalty_rate     # proportion that got a minor penalty
precedent_major_penalty_rate     # proportion that got a major penalty
```

"Similar" is defined as matching on `incident_type` (and optionally `severity`). This is a groupby, not a neural retrieval system.

If the dataset grows to 2000+ incidents in the future, the vector approach can be revisited.

---

## 7. Leakage Prevention Rules

Features that must NEVER be used as model inputs:

| Feature | Reason |
|---|---|
| `driver_at_fault` | Determined by stewards as part of the decision |
| `penalty` | This IS the target variable |
| `superlicense_pts_added` | Part of the penalty outcome |
| `mentioned_article` | Cited in the decision document |
| `incident_classification` | Potentially leaked if it reflects the steward's conclusion rather than the observed event |

Features that require temporal filtering:

| Feature | Rule |
|---|---|
| `driver_standing` | Must be standings BEFORE the incident race |
| `driver_points` | Must be points BEFORE the incident race |
| `driver_superlicense_pts` | Must exclude points from this incident |
| Any `*_last_N_races` | Must exclude current race and future races |

---

## 8. Development Roadmap

### Phase 1 — Data Collection & Schema (Current)
- Finalize flat schema (one-row-per-driver-investigation)
- Build PDF parsing pipeline
- Manually collect + validate 1–2 seasons of data
- Deliverable: `dataset/csv/processed.csv` with 100–300 validated rows

### Phase 2 — Baseline Model
- Implement encoding pipeline
- Implement temporal split
- Train XGBoost on tabular features
- Evaluate with accuracy, macro-F1, confusion matrix
- Deliverable: Working V1 model in `ml_models/xgboost/`

### Phase 3 — Feature Engineering
- Add derived features (standings differences, completion %, same_team)
- Add simplified precedent features (groupby-based)
- Re-train and compare against baseline
- Deliverable: Improved V1 with feature importance analysis

### Phase 4 — Normative Model
- Encode FIA Sporting Code as rule-based decision tree
- Map (incident_type, severity, context) → expected penalty
- Compare normative output vs FIA actual decisions
- Deliverable: Rule engine + deviation analysis report

### Phase 5 — NLP Integration (V2)
- Extract raw FIA report text as separate dataset
- Fine-tune DistilBERT on report text → penalty classification
- Compare text-only vs tabular-only vs combined
- Deliverable: NLP model in `ml_models/nlp/`

### Phase 6+ — Future (Telemetry, Visual)
- FastF1 integration for speed/braking data
- CNN for video frame analysis
- Multimodal fusion

---

## 9. Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Row granularity | One row per driver-investigation | Eliminates multi-value columns cleanly |
| Categorical encoding | Label encode for trees, one-hot for NNs | Standard practice, applied in preprocessing |
| Split strategy | Temporal (by season) | Prevents data leakage from future races |
| Similarity system | Groupby-based precedent stats | Sufficient for dataset size of 200–1000 |
| Primary model | XGBoost / LightGBM | Best for tabular data with limited samples |
| Target variable | 3-class simplified initially | Handles class imbalance realistically |
| PDF parsing | pdfplumber + regex | FIA docs are semi-structured, parseable |
| Data format | CSV for human review, Parquet for ML | CSV is editable, Parquet is fast |

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Small dataset (< 300 incidents) | Model underfits or overfits | Use simpler models, fewer features, cross-validation |
| Class imbalance (rare penalties) | Model ignores minority classes | Class weighting, simplified target, stratified splits |
| FIA document format changes | Parser breaks on older/newer docs | Modular parser with per-era templates |
| Temporal leakage | Inflated metrics, useless model | Strict temporal split, automated leakage tests |
| Inconsistent manual labeling | Noise in training data | Validation schema, inter-rater checks if multiple annotators |
| Normative model subjectivity | "Correct" penalty is debatable | Document assumptions explicitly, treat as one interpretation |

---

## 11. Success Criteria

**Version 1 is successful if:**
- Dataset contains 150+ validated incidents with correct temporal ordering
- XGBoost achieves > random baseline on 3-class prediction (macro-F1 > 0.40)
- No temporal leakage detected in feature pipeline
- Feature importance aligns with domain knowledge (incident_type and severity should rank high)

**The overall project is successful if:**
- The FIA behavior model produces interpretable predictions
- The normative model provides a defensible alternative ruling
- The deviation analysis identifies non-trivial patterns in FIA decision-making
