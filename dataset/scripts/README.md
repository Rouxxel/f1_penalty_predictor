# Pipeline CLI (`dataset/scripts`)

Command-line entry point for the FIA dataset generation pipeline.

| Script | Purpose |
|--------|---------|
| [`run_pipeline.py`](run_pipeline.py) | Run download → parse → build → enrich → validate |

Library code lives in [`src/fia_ml/data/`](../../src/fia_ml/data/README.md). Run all commands from the **project root**.

---

## Quick start

```powershell
pip install -r requirements.txt

# Full pipeline for seasons that already have PDFs or when FIA access works
python dataset/scripts/run_pipeline.py --stage all --season 2019

# Re-enrich + validate existing raw CSVs (no download needed)
python dataset/scripts/run_pipeline.py --stage enrich --season 2019 --season 2025
python dataset/scripts/run_pipeline.py --stage validate --season 2019 --season 2025
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--stage` | `all`, `download`, `parse`, `build`, `enrich`, `validate` (default: `all`) |
| `--season` | Year to process (repeatable). Default: all seasons in `configs/data.yaml` |
| `--config` | Path to config (default: `configs/data.yaml`) |

---

## Current dataset coverage

Status as of the last successful pipeline runs. See [`src/fia_ml/data/README.md`](../../src/fia_ml/data/README.md) for full pipeline docs.

| Season | `processed_{season}.csv` | Rows | Notes |
|--------|------------------------|------|-------|
| **2019** | Yes | 203 | Complete — ready for model training |
| 2020 | No | — | **Missing** — FIA download blocked (403/WAF) |
| 2021 | No | — | **Missing** — FIA download blocked |
| 2022 | No | — | **Missing** — FIA download blocked |
| 2023 | No | — | **Missing** — FIA download blocked |
| 2024 | No | — | **Missing** — FIA download blocked |
| **2025** | Yes | 343 | Complete — ready for model training |

**Available now:** 546 incident rows across 2019 + 2025 — enough to proceed to [`MODEL_TRAINING_PLAN.md`](../../MODEL_TRAINING_PLAN.md).

```mermaid
flowchart LR
    subgraph done["Processed seasons"]
        S2019["2019\n203 rows"]
        S2025["2025\n343 rows"]
    end

    subgraph missing["Missing seasons"]
        S2020["2020"]
        S2021["2021"]
        S2022["2022"]
        S2023["2023"]
        S2024["2024"]
    end

    done --> TRAIN["MODEL_TRAINING_PLAN"]
    missing -.->|"backfill later"| done
```

---

## Missing seasons — how to backfill later

When FIA access works again (or PDFs are obtained manually):

```powershell
# One season at a time — avoid bulk runs that trigger FIA blocks
python dataset/scripts/run_pipeline.py --stage all --season 2020
```

If download still fails but you have PDFs on disk, place them under `data/raw/fia/{season}/{event_slug}/` and skip download:

```powershell
python dataset/scripts/run_pipeline.py --stage parse --season 2020
python dataset/scripts/run_pipeline.py --stage build --season 2020
python dataset/scripts/run_pipeline.py --stage enrich --season 2020
python dataset/scripts/run_pipeline.py --stage validate --season 2020
```

FIA blocking troubleshooting:

- Set `scraper.fetch_backend: playwright` in `configs/data.yaml`
- Install: `pip install playwright` then `playwright install chromium`
- Try `playwright_headless: false` if headless mode is blocked
- Probe access: `python scripts/probe_fia.py`

---

## Enrichment gaps (future improvements)

Several schema columns are **not fully enriched** yet. Existing `processed_*.csv` files are usable for model training but these columns need future work. Details: [`src/fia_ml/data/enrichment/README.md`](../../src/fia_ml/data/enrichment/README.md).

| Column | Status | Planned source |
|--------|--------|----------------|
| `lap`, `lap_remaining`, `completion_percentage` | **0% fill** — time→lap alignment not working | FastF1 session timeline |
| `positions_of_involved parties` | **Not implemented** | FastF1 running positions at incident lap |
| `flag` | **Not implemented** | FastF1 race control messages |
| `superlicense_points_before_incident` | **Not implemented** | Rolling sum from prior penalties in dataset |
| `severity` | **Manual only** (0% fill) | Human review via `review_queue_{season}.csv` |
| `driver_standings`, `driver_points` (multi-driver rows) | **Partial** — misaligned when `drivers` has 2+ values | Per-driver Ergast lookup |
| `construct_standings`, `construct_points` | **~55–68% fill** | Ergast point-in-time constructor standings |
| Standings (all drivers) | **Season totals**, not round N−1 | Ergast point-in-time standings per round |

Observed fill rates from `reports/tables/data_quality_{season}.json`:

| Column | 2019 | 2025 |
|--------|------|------|
| `lap` | 0% | 0% |
| `full_laps` | 88% | 69% |
| `safety_car` / weather | 82% | 62% |
| `severity` | 0% | 0% |
| `drivers` | 99% | 82% |

---

## Stage reference

```mermaid
flowchart TD
    CLI["run_pipeline.py"] --> DL["download"]
    DL --> PA["parse"]
    PA --> BU["build"]
    BU --> EN["enrich"]
    EN --> VA["validate"]

    DL --> PDF["data/raw/fia/{season}/"]
    PA --> JSON["data/interim/extracted_documents/{season}/"]
    BU --> RAW["dataset/csv/raw_incidents_{season}.csv"]
    EN --> RAW2["raw_incidents updated in place"]
    VA --> PROC["dataset/csv/processed_{season}.csv"]
    VA --> REV["dataset/csv/review_queue_{season}.csv"]
    VA --> QUA["reports/tables/data_quality_{season}.json"]
```

| Stage | When to use without download |
|-------|------------------------------|
| `enrich` | `raw_incidents_{season}.csv` exists |
| `validate` | After enrich (or to refresh review queue) |
| `parse` + `build` | PDFs already in `data/raw/fia/{season}/` |

---

## Outputs per season

```text
dataset/csv/raw_incidents_{season}.csv
dataset/csv/raw_incidents_{season}.meta.json
dataset/csv/processed_{season}.csv          # primary training input
dataset/csv/review_queue_{season}.csv       # regenerated on validate
reports/tables/data_quality_{season}.json   # column fill rates
```

---

## Related docs

- [`../README.md`](../README.md) — dataset folder overview
- [`../../src/fia_ml/data/README.md`](../../src/fia_ml/data/README.md) — pipeline modules and architecture
- [`../../src/fia_ml/data/enrichment/README.md`](../../src/fia_ml/data/enrichment/README.md) — enrichment modules and gaps
- [`../../DATASET_GENERATION_PLAN.md`](../../DATASET_GENERATION_PLAN.md) — full implementation plan
- [`../../MODEL_TRAINING_PLAN.md`](../../MODEL_TRAINING_PLAN.md) — next step with 2019 + 2025 data
