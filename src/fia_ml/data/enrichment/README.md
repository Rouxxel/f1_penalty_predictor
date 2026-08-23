# Dataset enrichment (`fia_ml.data.enrichment`)

Fills empty columns in `raw_incidents_{season}.csv` after the **build** stage. Invoked by `--stage enrich` in the CLI (or as part of `--stage all`).

---

## Enrichment flow

Enrichment runs in a **fixed order**. Reference data is applied first; Ergast and FastF1 only fill cells that are still empty (`fill_gaps_only=True`).

```mermaid
flowchart TD
    IN["raw_incidents_{season}.csv\n+ raw_incidents_{season}.meta.json"]

    IN --> REF["1 · reference_enrich.py\ndata/reference/*.json"]
    REF --> GAP1{"Empty cells\nremaining?"}

    GAP1 -->|yes| ERG["2 · ergast.py\nErgast API + cache"]
    GAP1 -->|no| FF
    ERG --> GAP2{"Empty cells\nremaining?"}

    GAP2 -->|yes| FF["3 · fastf1_enrich.py\nFastF1 sessions + cache"]
    GAP2 -->|no| OUT
    FF --> OUT["raw_incidents_{season}.csv\n(enriched in place)"]

    OUT --> VAL["validation.py\n--stage validate"]
    VAL --> PROC["processed_{season}.csv"]
    VAL --> REV["review_queue_{season}.csv"]
```

---

## Where each column comes from

```mermaid
flowchart LR
    subgraph ref["Reference JSON"]
        R1["seasons.json\nround · standings · points"]
        R2["circuits.json\ncircuit · country · full_laps"]
        R3["drivers.json\nnationality · years_in_sport"]
        R4["teams.json\nlegacy team IDs"]
    end

    subgraph ergast["Ergast API"]
        E1["calendar fallback"]
        E2["driver from car number"]
        E3["standings fallback"]
    end

    subgraph fastf1["FastF1"]
        F1["lap at incident time"]
        F2["weather · track conditions"]
        F3["safety_car"]
        F4["full_laps fallback"]
    end

    subgraph manual["Manual / not yet"]
        M1["severity"]
        M2["superlicense_points_before_incident"]
        M3["positions_of_involved parties"]
        M4["flag"]
    end

    ref --> CSV["incident rows"]
    ergast --> CSV
    fastf1 --> CSV
    manual -.-> CSV
```

---

## Event → round mapping

Reference enrichment maps FIA event titles to calendar rounds:

```mermaid
flowchart LR
    PDF["PDF / parsed doc\nevent title"] --> META["meta.json\nevent_name"]
    META --> MAP["map_event_to_round()\ncircuits.json event_name"]
    MAP --> SEA["seasons.json\ncalendar[round]"]
    SEA --> COLS["round · circuit · country\nfull_laps"]
```

---

## How to run

From the project root, with `raw_incidents_{season}.csv` already built:

```bash
python dataset/scripts/run_pipeline.py --stage enrich --season 2020
```

Then export the processed dataset:

```bash
python dataset/scripts/run_pipeline.py --stage validate --season 2020
```

Or combine with the full pipeline:

```bash
python dataset/scripts/run_pipeline.py --stage all --season 2020
```

**Note:** `--stage enrich` writes back to `dataset/csv/raw_incidents_{season}.csv` (enriched in place). `--stage validate` reads that file and writes `processed_{season}.csv`.

```mermaid
sequenceDiagram
    participant CLI as run_pipeline.py
    participant REF as reference_enrich
    participant ERG as ergast
    participant FF as fastf1_enrich
    participant VAL as validation
    participant CSV as raw_incidents_*.csv
    participant OUT as processed_*.csv

    CLI->>REF: enrich_with_reference()
    REF->>CSV: fill from data/reference/
    CLI->>ERG: enrich_with_ergast() [if enabled]
    ERG->>CSV: fill gaps only
    CLI->>FF: enrich_with_fastf1() [if enabled]
    FF->>CSV: fill gaps only
    CLI->>VAL: validate()
    VAL->>OUT: write processed + review_queue
```

---

## Modules

### `reference_enrich.py` — primary source

Loads verified local files from `data/reference/`:

| File | Columns filled |
|------|----------------|
| `seasons.json` | `round`, `rounds`, `num_teams`, `current_top_4_drivers`, driver/constructor standings & points |
| `circuits.json` | `circuit`, `country`, `first_season`, `full_laps` (race sessions via `total_laps`) |
| `drivers.json` | `nationalities`, `years_in_sport` (from `debut`) |
| `teams.json` | Resolves `legacy_ids` when matching constructor slugs |

Uses `event_name` on each circuit to map FIA event titles (e.g. `"Abu Dhabi Grand Prix"`) to calendar rounds in `seasons.json`.

**Limitation:** standings in `seasons.json` are **season totals**, not point-in-time per round.

---

### `ergast.py` — API fallback

Runs only when `enrichment.ergast_fallback_enabled: true` in `configs/data.yaml` (default).

```mermaid
flowchart TD
    ROW["Incident row\nempty column"] --> CHECK{"ergast_fallback_enabled?"}
    CHECK -->|no| SKIP["Leave blank"]
    CHECK -->|yes| CACHE{"Cache hit in\ndata/raw/race_data/ergast/?"}
    CACHE -->|yes| FILL["Fill cell"]
    CACHE -->|no| API["Call Ergast API\njolpi.ca → ergast.com fallback"]
    API --> STORE["Write cache"]
    STORE --> FILL
```

Fills **gaps** left by reference enrichment:

- `round`, `circuit`, `country` from Ergast calendar
- Car number → driver slug via race results
- Standings, nationalities, points when reference data is missing

---

### `fastf1_enrich.py` — session context

Uses [FastF1](https://github.com/theOehrly/FastF1) with cache at `data/raw/race_data/fastf1_cache/`.

```mermaid
flowchart TD
    ROW["Incident row\nseason + round + session + time"] --> SESS["Load FastF1 session\n(cached after first run)"]
    SESS --> LAP["lap — time → lap number"]
    SESS --> WX["track_conditions · weather_conditions"]
    SESS --> SC["safety_car from race control"]
    SESS --> FL["full_laps if still empty"]
```

Fills (gap-only unless reference did not set the field):

| Column | Source |
|--------|--------|
| `full_laps` | Session lap count (if not set from `circuits.json`) |
| `lap` | Incident time → lap number (race sessions only) |
| `track_conditions`, `weather_conditions` | Session weather |
| `safety_car` | Race control messages (SC / VSC / none) |

**Not implemented yet:** `positions_of_involved parties`, `flag`.

First run per season is **slow** (downloads session data per round). Later runs use the cache.

`lap` often stays empty when PDF `time` fields are malformed — those rows appear in `review_queue_{season}.csv`.

---

### `common.py` — shared helpers

- `load_meta()` — reads `raw_incidents_{season}.meta.json` (event, car number, parse confidence)
- `map_event_to_round()` — event title → round via reference calendar
- `is_blank()` — gap detection for fallback enrichers
- `resolve_team_id()` — legacy team id → canonical slug

---

## Columns still manual / unimplemented

| Column | Why |
|--------|-----|
| `severity` | Subjective — fill via review queue |
| `superlicense_points_before_incident` | Not in reference data yet |
| `positions_of_involved parties` | Planned FastF1 feature |
| `flag` | Not implemented |

---

## Config flags (`configs/data.yaml`)

```yaml
enrichment:
  ergast_base_url: "https://api.jolpi.ca/ergast/f1"
  ergast_fallback_url: "https://ergast.com/api/f1"
  ergast_fallback_enabled: true   # set false to use reference JSON only
  fastf1_cache_enabled: true
```

To skip external APIs entirely (reference JSON only):

```yaml
enrichment:
  ergast_fallback_enabled: false
  fastf1_cache_enabled: false
```

You will lose lap-at-incident-time, weather, and safety-car fields unless added elsewhere.

```mermaid
flowchart LR
    subgraph modes["Enrichment modes"]
        FULL["Default\nreference + Ergast + FastF1"]
        REFONLY["Reference only\nergast_fallback_enabled: false\nfastf1_cache_enabled: false"]
    end

    FULL --> RICH["Highest column fill rate\nrequires network + cache"]
    REFONLY --> LIGHT["Fast, offline-friendly\nmany session columns empty"]
```

---

## Public API

```python
from fia_ml.data.enrichment import (
    enrich_with_reference,
    enrich_with_ergast,
    enrich_with_fastf1,
)
```

Normally you do not call these directly; use `dataset/scripts/run_pipeline.py --stage enrich`.

Parent pipeline docs: [`../README.md`](../README.md)
