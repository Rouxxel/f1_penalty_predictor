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

Define a single row as:

An “incident instance” involving one or more drivers in a specific race context.

Each row should include:
- Drivers involved
- Incident type
- Session type (race, qualifying, etc.)
- Lap number and track location
- FIA decision (label)

This definition is critical before any data collection begins.

---

## Step 2 — Start Small (One Season Only)

Do not attempt full F1 history.

Start with:
- One season (e.g., 2023 or 2024)
- Only race sessions initially
- Only incidents with official FIA decisions

Target dataset size:
- ~200 to 500 incidents

---

## Step 3 — Data Sources

Primary sources:
- FIA steward documents (official post-race reports)
- Official race summaries
- Wikipedia race incident summaries (for structure support)

Optional later:
- Race footage annotations
- Motorsport journalism incident breakdowns

---

## Step 4 — Define a Simple Dataset Schema

Start with a minimal structure:

- race_id
- driver_a
- driver_b (optional)
- lap_number
- corner/sector (if available)
- incident_type
- session_type
- weather
- FIA_decision (label)

Keep it simple at the beginning.

---

## Step 5 — Manual Data Collection Pipeline

At this stage, most data will be manually constructed:

- Read steward reports
- Extract incidents one by one
- Convert them into structured rows
- Store in CSV or database format

This is the most time-intensive but most important step.

---

## Step 6 — Build First FIA Prediction Model

Once enough data is collected:

- Train/test split (preferably by race, not random split)
- Start with XGBoost or LightGBM
- Evaluate with accuracy, F1 score, confusion matrix

This model learns FIA behavior patterns.

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

---

## Step 8 — Compare Both Models

After both models exist:

Analyze differences:
- Where FIA deviates from rules
- Which incident types show highest disagreement
- Whether inconsistencies cluster by context

This becomes the main analytical output of the project.

---

## Key Advice

- Start small and structured
- Data quality matters more than model complexity
- Incident definition is the foundation of everything
- Avoid adding complex features (social media, nationality, etc.) early unless clearly justified