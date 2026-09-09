# Dataset Generation Runbook

## Setup

```bash
pip install -r requirements.txt
```

## Run full pipeline (all configured seasons: 2019–2025)

```bash
python dataset/scripts/run_pipeline.py --config configs/data.yaml
```

## Run one season

```bash
python dataset/scripts/run_pipeline.py --config configs/data.yaml --season 2021
python dataset/scripts/run_pipeline.py --config configs/data.yaml --season 2019 --season 2020
```

## Run individual stages

```bash
python dataset/scripts/run_pipeline.py --config configs/data.yaml --stage download
python dataset/scripts/run_pipeline.py --config configs/data.yaml --stage parse
python dataset/scripts/run_pipeline.py --config configs/data.yaml --stage build
python dataset/scripts/run_pipeline.py --config configs/data.yaml --stage enrich
python dataset/scripts/run_pipeline.py --config configs/data.yaml --stage validate

# Re-enrich existing raw CSVs (uses configs/enrichment.yaml by default)
python dataset/scripts/run_pipeline.py --stage enrich --season 2019 --season 2025
python dataset/scripts/run_pipeline.py --stage validate --season 2019 --season 2025
```

## Outputs

| File | Description |
|---|---|
| `data/raw/fia/{season}/` | Downloaded PDFs + `manifest.json` |
| `data/interim/extracted_documents/{season}/` | Parsed JSON per document |
| `dataset/csv/raw_incidents_{season}.csv` | PDF-only incident rows |
| `dataset/csv/processed_{season}.csv` | Enriched + validated dataset |
| `dataset/csv/review_queue_{season}.csv` | Rows needing manual review |
| `reports/tables/data_quality_{season}.json` | Column fill rates + enrichment quality gates |
| `reports/tables/enrichment_report_{season}.json` | Per-gate pass/fail vs `configs/enrichment.yaml` targets |
| `data/interim/enrichment_meta/{season}.json` | Per-field enrichment provenance |

## Notes

- Season URLs live in `configs/data.yaml` under `seasons`. Use `--season` to limit runs.
- The downloader keeps PDFs whose filenames contain **Infringement**, **Decision**, **Offence**, or **Summons** (whole-word match, anywhere in the title). Naming varies by year (e.g. `Doc 52 - Infringement - ...` in 2025 vs `Offence - ...` in 2019).
- **Enrichment order:** reference → Ergast (round N−1) → timestamp → OpenF1 (2023+) → FastF1 → superlicense → text fields → provenance → validate. See [`src/fia_ml/data/enrichment/README.md`](../src/fia_ml/data/enrichment/README.md).
- Enrichment settings: `configs/enrichment.yaml` (loaded automatically; override with `--enrichment-config`).
- FastF1 enrichment is slow on first run; cache lives in `data/raw/race_data/fastf1_cache/`.
- OpenF1 responses cached under `data/raw/race_data/openf1_cache/`.
- Ergast responses are cached under `data/raw/race_data/ergast/{season}/`.
- `severity` is not written to CSV automatically — suggestions go to `review_queue_{season}.csv` for manual promotion.
- **Known blocker:** PDF `time` fields are usually document-header clocks, not session time — lap/flag/positions fill remains low until timestamp parsing improves.
