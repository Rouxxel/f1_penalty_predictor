# Dataset outputs and CLI

This folder holds **generated CSV datasets** and the **pipeline CLI script**.

| Path | Purpose |
|------|---------|
| `csv/` | `raw_incidents_{season}.csv`, `processed_{season}.csv`, review queues |
| `scripts/run_pipeline.py` | Command-line entry point for the full pipeline |

Pipeline **library code** lives in [`src/fia_ml/data/`](../src/fia_ml/data/README.md).

---

## End-to-end flow

```mermaid
flowchart TD
    CLI["python dataset/scripts/run_pipeline.py\n--stage all --season 2020"]

    CLI --> DL["download"]
    DL --> PDF["data/raw/fia/2020/"]

    CLI --> PA["parse"]
    PDF --> PA
    PA --> JSON["data/interim/extracted_documents/2020/"]

    CLI --> BU["build"]
    JSON --> BU
    BU --> RAW["dataset/csv/raw_incidents_2020.csv"]

    CLI --> EN["enrich"]
    RAW --> EN
    EN --> RAW2["raw_incidents_2020.csv\n(updated)"]

    CLI --> VA["validate"]
    RAW2 --> VA
    VA --> PROC["dataset/csv/processed_2020.csv"]
    VA --> REV["dataset/csv/review_queue_2020.csv"]
```

---

## Quick commands

Run from the **project root**:

```bash
# Full pipeline for one season
python dataset/scripts/run_pipeline.py --stage all --season 2020

# Raw CSV only (stages 1–3)
python dataset/scripts/run_pipeline.py --stage download --season 2020
python dataset/scripts/run_pipeline.py --stage parse   --season 2020
python dataset/scripts/run_pipeline.py --stage build   --season 2020

# Processed CSV only (stages 4–5, needs raw first)
python dataset/scripts/run_pipeline.py --stage enrich    --season 2020
python dataset/scripts/run_pipeline.py --stage validate  --season 2020
```

---

## Further reading

- [`src/fia_ml/data/README.md`](../src/fia_ml/data/README.md) — full pipeline docs, stage diagrams, module reference
- [`src/fia_ml/data/enrichment/README.md`](../src/fia_ml/data/enrichment/README.md) — enrichment cascade and column sources
