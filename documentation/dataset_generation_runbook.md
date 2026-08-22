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
```

## Outputs

| File | Description |
|---|---|
| `data/raw/fia/{season}/` | Downloaded PDFs + `manifest.json` |
| `data/interim/extracted_documents/{season}/` | Parsed JSON per document |
| `dataset/csv/raw_incidents_{season}.csv` | PDF-only incident rows |
| `dataset/csv/processed_{season}.csv` | Enriched + validated dataset |
| `dataset/csv/review_queue_{season}.csv` | Rows needing manual review |
| `reports/tables/data_quality_{season}.json` | Column fill rates |

## Notes

- Season URLs live in `configs/data.yaml` under `seasons`. Use `--season` to limit runs.
- The downloader keeps PDFs whose filenames contain **Infringement**, **Decision**, **Offence**, or **Summons** (whole-word match, anywhere in the title). Naming varies by year (e.g. `Doc 52 - Infringement - ...` in 2025 vs `Offence - ...` in 2019).
- **Enrichment order:** local reference JSON (`data/reference/`) first, then Ergast API fallback, then FastF1 for lap/weather/SC. See `data/reference/README.md`.
- FastF1 enrichment is slow on first run; cache lives in `data/raw/race_data/fastf1_cache/`.
- Ergast responses are cached under `data/raw/race_data/ergast/{season}/`.
- Fill `severity` manually via review queue — not inferred automatically.
