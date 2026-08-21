# FIA Decision Modeling Project

## Core Idea of the Project

This project aims to build a dual-model system that analyzes Formula 1 stewarding decisions from two complementary perspectives: real-world FIA behavior and rule-based ideal behavior.

### 1. FIA Behavior Model (Descriptive / Empirical)

The first model is a supervised machine learning classifier that predicts how the FIA actually behaves when making sanction decisions during races.

**Objective:**
Given an on-track incident, predict the probability and type of FIA sanction.

**What it learns:**
- Historical steward decisions
- Race context (track, weather, session type)
- Driver and team history
- Incident characteristics (collision type, advantage gained, severity, etc.)

**Output classes:**
- No penalty
- Warning
- Time penalty
- Grid penalty
- Penalty points

This model is not concerned with fairness or correctness. It models real institutional decision-making, including inconsistencies or potential biases present in historical data.

---

### 2. Normative Rules Model (Prescriptive / Idealized)

The second model represents how decisions should be made under a strict and consistent interpretation of FIA rules.

**Objective:**
Given the same incident, determine the correct penalty outcome based on a standardized rule system.

**How it is built:**
- Manual interpretation of FIA Sporting Code
- Structured rule mapping (if-then logic or scoring system)
- Optional expert-labeled “ideal decisions”

**Output:**
- Ideal penalty decision under consistent rules

This model does not learn from FIA decisions. It encodes a consistent version of racing rules.

---

### 3. Comparison Layer (Analysis Output)

Once both models exist, their outputs can be compared:

**Deviation = FIA decision − Normative decision**

This enables:
- Detection of inconsistent stewarding decisions
- Measurement of rule interpretation variability
- Identification of patterns where decisions diverge from rules

The system does not assume bias; it quantifies divergence between behavior and rules.

---

### Key Insight

This is a dual-lens system:

- One model learns what actually happens (FIA behavior)
- One model encodes what should happen (rules-based system)
- The difference between them becomes the most informative signal

---

# How to Start the Project

## Step 1 — Define the Unit of Data

Define a single **raw dataset row** as:

An **incident instance** involving one or more drivers in a specific race context.

Each row should include:
- Drivers involved (possibly multiple, comma-separated)
- Incident type
- Session type (race, qualifying, etc.)
- Lap number and track location (when available)
- FIA decision (label)

The authoritative column layout is defined in `documentation/f1_dataset_example.csv`. Row 1 is the header, row 2 is an example incident, row 3 is a legend (`*` = categorical, `**` = multi-value).

For machine learning, this incident-centric raw table is later **flattened** to one row per driver under investigation (see `project_spec.md` §3). Do not change the raw CSV format for that reason — flattening is a preprocessing step.

---

## Step 2 — Start Small (One Season Only)

Do not attempt full F1 history.

Start with:
- One season (e.g., 2019 or 2023)
- All sessions that produce steward documents (race, qualifying, sprint where applicable)
- Only incidents with official FIA **Decision**, **Offence**, or **Summons** documents

Target dataset size:
- ~200 to 500 incident rows after deduplication

---

## Step 3 — Data Sources

### Primary: FIA Decision Documents Portal

Official steward PDFs are published on the FIA website, organized by season and event.

**Season index URL pattern:**
```text
https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-{YEAR}-{ID}
```

Example: [2019 season documents](https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2019-971)

**Important scraping facts:**
- The season page HTML only lists PDFs for one expanded event by default (usually the last race of the season).
- Each Grand Prix has its own event sub-page, discoverable from the season page dropdown:
  ```text
  .../season/season-{YEAR}-{ID}/event/{Grand%20Prix%20Name}
  ```
- PDF files live at:
  ```text
  https://www.fia.com/sites/default/files/decision-document/{filename}.pdf
  ```
- Typical event page contains 40–60 PDFs; only a subset is relevant for incidents.

**Relevant document types (download these):**

| Type | Use |
|---|---|
| Decision | Primary source for penalty outcome, fact, reason, cited article |
| Offence | Post-race penalties; often contains full sanction detail |
| Summons | Marks formal investigation; may precede a Decision |

**Exclude** (administrative, not incident data): Entry List, Classification, Starting Grid, Scrutineering, Championship Points, PU/gearbox notices, Circuit Map, Race Director notes.

**Fields extractable directly from FIA PDFs:**
- Car number, driver name, team (competitor)
- Session, document date/time
- Fact, Offence, Decision, Reason
- Regulation article cited
- Turn/corner references in fact text (e.g., “turn 12”)
- Penalty type and superlicence points (when stated)

**Fields NOT in FIA PDFs** (must come from elsewhere):
- Lap number (rarely stated)
- Live race positions at incident time
- Weather, track conditions, safety car state
- Championship standings and points (point-in-time)
- Sector mapping from corner number
- Severity (subjective — heuristic or manual)
- `current_top_4_drivers`, `years_in_sport`, historical superlicence totals

### Secondary: Enrichment APIs

| Source | Provides |
|---|---|
| [Ergast F1 API](http://ergast.com/api/f1/) | Calendar, round, circuit, country, driver/constructor standings before each race, car-number-to-driver mapping via results |
| FastF1 (later) | Lap timing, positions, weather, SC/VSC periods |
| Manual review | Severity, ambiguous incident classification, missing lap numbers |

### Optional later
- Race footage annotations
- Motorsport journalism incident breakdowns

---

## Step 4 — Dataset Schema

The project schema is **not** the minimal 8-column starter set anymore. Use the full schema in:

```text
documentation/f1_dataset_example.csv
```

Column groups:

| Group | Columns |
|---|---|
| Identity | `incident_id`, `season`, `round`, `circuit`, `country`, `first_season` |
| Race context | `rounds`, `num_teams`, `current_top_4_drivers`, `lap`, `lap_remaining`, `full_laps`, `completion_percentage`, `sector` |
| Conditions | `flag`, `safety_car`, `track_conditions`, `weather_conditions`, `session` |
| Incident | `incident_type`, `severity`, `positions_of_involved parties`, `num_drivers`, `drivers`, `nationalities`, `respective_teams` |
| Standings (multi-value) | `driver_standings`, `driver_points`, `construct_standings`, `construct_points`, `years_in_sport`, `superlicense_points_before_incident` |
| Labels | `investigation`, `incident_classification`, `driver_at_fault`, `penalty`, `superlicense_points_added`, `mentioned_article` |

See `FIA_stewarding_dataset_feature_specification.md` for feature semantics, leakage rules, and data provenance per column.

---

## Step 5 — Automated Data Collection Pipeline

Manual one-by-one extraction does not scale. The target pipeline:

```text
FIA season URL
      │
      ▼
Discover event sub-pages → download relevant PDFs
      │
      ▼
Extract text → parse structured FIA fields
      │
      ▼
Build raw incident rows (many columns empty)
      │
      ▼
Enrichment join (Ergast, later FastF1)
      │
      ▼
Validate against schema → output CSV
```

### Output file strategy

| Artifact | Granularity | Purpose |
|---|---|---|
| `data/raw/fia/{season}/{event}/` | Per race weekend | Cached PDFs (re-runnable) |
| `data/interim/extracted_documents/{season}/` | Per document | Parsed JSON/text |
| `dataset/csv/raw_incidents_{season}.csv` | **One file per season** | Human-reviewable raw table matching `f1_dataset_example.csv` |
| `dataset/csv/processed_{season}.csv` | One file per season | After enrichment + validation |
| `data/processed/incidents.parquet` | Combined seasons | ML pipeline input after flattening |

**Do not** use one CSV per race weekend as the canonical dataset — it fragments review and training. Use `season` + `round` + `circuit` columns to filter within a single season file. Per-event folders are fine for raw PDF storage only.

Detailed pipeline design: `project_spec.md` §4.

---

## Step 6 — Build First FIA Prediction Model

Once enough data is collected:

- Flatten raw CSV to one-row-per-driver-investigation
- Train/test split by season (never random)
- Start with XGBoost or LightGBM
- Evaluate with accuracy, macro-F1, confusion matrix

Future implementation plan: **Model Training Plan** (to be written separately).

---

## Step 7 — Build Rule-Based Model

Create a simple rule system:

Examples:
- Collision + avoidable + gained advantage → time penalty
- Repeated track limits → warning then penalty escalation

This can be:
- If/else logic system
- Or a scoring system mapping severity → penalty class

This represents the normative “by-the-book” system.

Future implementation plan: **Normative Rules Engine Plan** (to be written separately).

---

## Step 8 — Compare Both Models

After both models exist:

Analyze differences:
- Where FIA deviates from rules
- Which incident types show highest disagreement
- Whether inconsistencies cluster by context

This becomes the main analytical output of the project.

---

## Planned Implementation Documents

The following documents will be generated as separate implementation plans (not yet written):

| Plan | Scope |
|---|---|
| **Dataset Generation Plan** | Scraper, PDF parser, enrichment, validation, CSV output |
| **Model Training Plan** | Encoding, temporal split, XGBoost baseline, evaluation |
| **Feature Engineering Plan** | Derived features, precedent stats, leakage tests |
| **Normative Model Plan** | Rule encoding, deviation analysis |

This file (`f1_project.md`) and the two companion specs define *what* and *why*. Implementation plans will define *how* step-by-step.

---

## Key Advice

- Start small and structured — one season first
- Data quality matters more than model complexity
- Incident definition is the foundation of everything
- FIA PDFs give labels and narrative; Ergast/FastF1 give context — neither is sufficient alone
- Keep the raw CSV wide and human-readable; flatten and encode only for modeling
- Avoid adding complex features (social media, nationality bias analysis, etc.) until the baseline pipeline works
