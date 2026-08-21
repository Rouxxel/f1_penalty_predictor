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

### 3.1 Two-Layer Schema Model

The project uses **two related schemas**, not one:

| Layer | File | Row unit | Purpose |
|---|---|---|---|
| **Raw** | `documentation/f1_dataset_example.csv` | One row per **incident** | Human review, data collection, wide multi-value columns |
| **Model** | `data/processed/incidents.parquet` | One row per **driver under investigation** | ML training after flattening and encoding |

The raw CSV is the authoritative collection format. The model dataset is derived from it.

**Raw row** = an incident instance involving one or more drivers in a race context (matches `f1_dataset_example.csv`).

**Model row** = one driver under investigation for one incident. If two drivers are both investigated, that produces two model rows sharing the same `incident_id` with swapped driver/opponent roles.

### 3.2 Authoritative Raw Schema (`f1_dataset_example.csv`)

```text
incident_id
circuit, country, first_season, round, season
current_top_4_drivers, rounds, num_teams
lap, lap_remaining, full_laps, completion_percentage, sector
flag, safety_car, track_conditions, weather_conditions, session
incident_type, severity, positions_of_involved parties, num_drivers
drivers, nationalities, respective_teams                          # ** multi-value
driver_standings, driver_points                                     # ** multi-value
construct_standings, construct_points                               # ** multi-value
years_in_sport, superlicense_points_before_incident                 # ** multi-value
investigation, incident_classification
driver_at_fault, penalty, superlicense_points_added, mentioned_article  # labels
```

Legend (row 3 of example file): `*` = categorical string, `**` = comma-separated multi-value in one cell.

Naming note: `rounds` in the raw CSV = total rounds in the season (equivalent to `total_rounds` in the flat model schema). `full_laps` = total race distance (equivalent to `total_laps`).

### 3.3 Schema Design — Resolving the Multi-Value Problem (Model Layer)

**Problem:** Raw columns like `drivers`, `nationalities`, `driver_standings` pack multiple values into a single cell (e.g., `"max_verstappen,lewis_hamilton"`).

**Solution:** At preprocessing time, restructure from incident-centric to driver-investigation-centric:

```text
BEFORE (raw CSV — incident-centric, multi-value):
─────────────────────────────────────────────────────────
incident_id | drivers                          | driver_standings
a001        | max_verstappen,lewis_hamilton     | 1,2

AFTER (model dataset — flat):
─────────────────────────────────────────────────────────
incident_id | driver         | opponent        | driver_standing | opponent_standing
a001        | max_verstappen | lewis_hamilton   | 1               | 2
```

Each model column holds exactly one value. Multi-value parsing happens once in `preprocessing/`, not at training time.

### 3.4 Schema Design — Resolving the Single-String Problem

**Problem:** Columns like `circuit`, `country`, `track_conditions`, `weather_conditions`, `session`, `incident_type` contain categorical strings.

**Solution:** Encoding strategy depends on the model type:

| Encoding Method | When to Use | Example |
|---|---|---|
| Label encoding (ordinal int) | XGBoost, LightGBM (native categorical support) | `monza → 12` |
| One-hot encoding | Logistic regression, neural nets, low-cardinality features | `monza → [0,0,...,1,...,0]` |
| Target encoding | High-cardinality categoricals (30+ unique values) | `monza → 0.34` (mean penalty rate) |
| Frequency encoding | When category frequency itself is informative | `monza → 47` (number of incidents) |

**Implementation:** Encoding happens in `src/fia_ml/preprocessing/encoding.py`, not in the raw dataset. The raw CSV stores human-readable strings. The processed parquet files store encoded values.

### 3.5 Flat Model Column Schema (Derived from Raw CSV)

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

### 3.6 Target Variable Encoding

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

### 3.7 Raw Column → Data Source Mapping

Not every column in `f1_dataset_example.csv` can be extracted from FIA PDFs. The enrichment stage fills gaps.

| Column(s) | Primary source | Notes |
|---|---|---|
| `incident_id` | Generated | Stable ID; link multiple docs to same incident |
| `season`, `round`, `circuit`, `country`, `rounds`, `num_teams` | Ergast API + PDF filename | PDF title includes GP name and year |
| `first_season` | Static circuit reference | One-time lookup table |
| `session` | FIA PDF | Race, Qualifying, Practice |
| `drivers`, `respective_teams` | FIA PDF + Ergast | PDF has car number; map via entry list / results |
| `nationalities` | Ergast driver metadata | Not in PDF |
| `fact`-derived: turn, incident text | FIA PDF | Parse Fact/Offence fields |
| `incident_type`, `incident_classification` | FIA PDF + heuristics | Classify from Fact/Offence keywords |
| `penalty`, `superlicense_points_added`, `mentioned_article` | FIA PDF | **Target labels** |
| `driver_at_fault` | FIA PDF Reason field | **Target / leakage risk** |
| `investigation` | Document type | `Summons` → true; `Decision`/`Offence` → varies |
| `lap`, `lap_remaining`, `full_laps`, `completion_percentage` | FastF1 / manual | Rarely in PDF; derive when possible |
| `sector` | Corner→sector map or manual | Infer from “turn N” in fact text |
| `positions_of_involved parties` | FastF1 / Ergast lap data | Not in PDF |
| `flag`, `safety_car`, `track_conditions`, `weather_conditions` | FastF1 | Not in PDF |
| `driver_standings`, `driver_points`, `construct_standings`, `construct_points` | Ergast | **Before** incident race only |
| `current_top_4_drivers` | Derived from Ergast standings | Snapshot before race |
| `years_in_sport`, `superlicense_points_before_incident` | Ergast + historical penalties | Rolling calculation |
| `severity` | Manual or heuristic | Subjective; not in PDF |

---

## 4. FIA Document → CSV Pipeline

### 4.1 Location

```text
data/
├── raw/fia/{season}/{event_slug}/          # Downloaded PDFs
├── interim/extracted_documents/{season}/   # Parsed JSON per PDF
└── processed/                              # Parquet for ML

dataset/
├── csv/
│   ├── raw_incidents_{season}.csv          # Matches f1_dataset_example.csv schema
│   └── processed_{season}.csv              # After enrichment + validation
└── scripts/
    └── run_pipeline.py                     # CLI entry point

src/fia_ml/data/
├── download.py                             # Season URL → event pages → PDFs
├── parsing.py                              # PDF text → structured fields
├── enrichment.py                           # Ergast / FastF1 joins
├── validation.py                           # Schema checks
└── pipeline.py                             # Orchestrator
```

### 4.2 Input: FIA Season URL

Accept a season index URL such as:
```text
https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2019-971
```

**Scraper behavior:**
1. Fetch season page HTML.
2. Parse `<option value=".../event/...">` elements to discover all Grand Prix event URLs (typically ~21 per season).
3. For each event page, collect PDF links matching `/sites/default/files/decision-document/`.
4. Filter to relevant types: titles containing `Decision`, `Offence`, or `Summons`.
5. Download to `data/raw/fia/{season}/{event_slug}/`, skipping existing files (idempotent).

**Do not** scrape only the season root page — it exposes PDFs for one expanded event only.

### 4.3 Pipeline Stages

```text
                    FIA season URL
                          │
                          ▼
              ┌───────────────────────┐
              │  1. download.py       │
              │  Event pages → PDFs   │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  2. parsing.py        │
              │  PDF → structured JSON│
              │  (PyMuPDF/pdfplumber) │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  3. build raw rows    │
              │  JSON → CSV columns   │
              │  (many fields null)   │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  4. enrichment.py     │
              │  Ergast (+ FastF1)    │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  5. validation.py     │
              │  Schema + dedup       │
              └───────────┬───────────┘
                          ▼
         dataset/csv/processed_{season}.csv
                          │
                          ▼
              ┌───────────────────────┐
              │  6. flatten (ML prep) │
              │  → incidents.parquet  │
              └───────────────────────┘
```

### 4.4 FIA PDF Structure (Parse Targets)

Decision documents follow a semi-structured layout:

```text
Document / No
Date / Time
Grand Prix + event dates
No / Driver          → car number + name
Competitor           → team
Session
Fact                 → incident description, often mentions other cars / turns
Offence              → alleged regulation breach
Decision             → sanction outcome (TARGET)
Reason               → steward reasoning
```

Example extracted outcome: `No further action`, `5 second time penalty`, `3 place grid drop`.

Use regex + section headers. Normalize non-breaking spaces and encoding artifacts from PDF text extraction.

### 4.5 Deduplication Rules

Multiple documents can describe the same underlying incident:
- Summons → Decision for same car/session/fact
- Correction documents replacing earlier versions
- Separate Decision docs for each car in a two-car incident

Maintain:
```text
incident_id     — shared across related rows/docs
document_id     — unique per PDF
decision_id     — unique per steward ruling
```

Prefer the latest non-superseded Decision/Offence for labels.

### 4.6 CSV Output Granularity

| Output | Granularity | When to use |
|---|---|---|
| `raw_incidents_{season}.csv` | **One file per season** | Canonical human-editable dataset |
| `processed_{season}.csv` | One file per season | After automated enrichment |
| `data/raw/fia/{season}/{event}/` | Per race weekend | PDF cache only |
| Combined `incidents.parquet` | All seasons | ML pipeline after flattening |

Filter by `round` or `circuit` within a season file — do not split into per-weekend CSVs for modeling.

### 4.7 Key Implementation Notes

- FIA PDFs provide **labels and narrative**; Ergast/FastF1 provide **context**. Plan for ~40% of raw schema columns to be empty after PDF-only extraction.
- Enrichment must respect point-in-time rules (standings before the incident race).
- Pipeline must be idempotent: re-running on the same inputs produces the same output.
- Parser should be modular by document era if FIA format shifts between seasons.

Future detail: **Dataset Generation Implementation Plan** (separate document).

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
- Schema locked in `f1_dataset_example.csv` (raw) + flat model schema (derived)
- Build FIA scraper: season URL → event pages → filtered PDFs
- Build PDF parser + Ergast enrichment pipeline
- Validate and manually review 1 season (~150–300 incident rows)
- Deliverable: `dataset/csv/processed_{season}.csv`
- **Next doc:** Dataset Generation Implementation Plan

### Phase 2 — Baseline Model
- Flatten raw CSV → one-row-per-driver-investigation
- Implement encoding pipeline
- Implement temporal split
- Train XGBoost on tabular features
- Evaluate with accuracy, macro-F1, confusion matrix
- Deliverable: Working V1 model in `ml_models/xgboost/`
- **Next doc:** Model Training Implementation Plan

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
| Raw CSV schema | `f1_dataset_example.csv` (incident-centric) | Human review, matches collection workflow |
| Model row granularity | One row per driver-investigation | Eliminates multi-value columns for ML |
| Canonical CSV scope | One file per season | Easy to review; filter by `round`/`circuit` |
| PDF storage | Per-event folders under `data/raw/fia/` | Matches FIA site structure; cache-friendly |
| FIA scraping | Season URL → event sub-pages | Season root page alone misses most PDFs |
| Document filter | Decision, Offence, Summons only | Excludes classifications, entry lists, etc. |
| Enrichment | Ergast (V1), FastF1 (V2+) | PDFs lack standings, weather, lap, positions |
| Categorical encoding | Label encode for trees, one-hot for NNs | Applied in preprocessing, not raw CSV |
| Split strategy | Temporal (by season) | Prevents data leakage from future races |
| Similarity system | Groupby-based precedent stats | Sufficient for dataset size of 200–1000 |
| Primary model | XGBoost / LightGBM | Best for tabular data with limited samples |
| Target variable | 3-class simplified initially | Handles class imbalance realistically |
| PDF parsing | PyMuPDF/pdfplumber + regex | FIA docs are semi-structured |
| Data format | CSV for human review, Parquet for ML | CSV editable; Parquet fast for training |

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| FIA site structure changes | Scraper breaks | Event URL discovery from dropdown; versioned scraper |
| Incomplete PDF fields | Many empty columns after parse | Enrichment stage + manual review queue |
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

---

## 12. Future Implementation Plans

This specification defines architecture and schema. Step-by-step build instructions will live in separate plans:

| Document | Status | Covers |
|---|---|---|
| Dataset Generation Implementation Plan | Not started | `download.py`, `parsing.py`, `enrichment.py`, validation, CSV output |
| Model Training Implementation Plan | Not started | Flattening, encoding, temporal split, XGBoost baseline |
| Feature Engineering Implementation Plan | Not started | Derived features, precedent stats, leakage tests |
| Normative Rules Implementation Plan | Not started | Rule engine, deviation analysis |

When requesting implementation, reference the relevant plan name and phase from §8.
