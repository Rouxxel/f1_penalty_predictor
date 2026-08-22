# Reference data (manual only)

Files in this directory are **manually curated** and must not be modified by pipeline scripts.

Pipeline code loads them via `fia_ml.data.reference_data` (read-only). Writes to this
directory are blocked by `secure_file_io` and raise `ReadOnlyPathError`.

To update reference data, edit the JSON files directly in your editor.

## Files

### `circuits.json`

Per-circuit metadata keyed by circuit slug (e.g. `monza`, `yas_marina`).

| Field | Description |
|-------|-------------|
| `country`, `region`, `state`, `city` | Location metadata |
| `first_season` | First F1 championship season at this circuit |
| `total_laps` | Scheduled race lap count (used for `full_laps` before FastF1) |
| `length.km` | Circuit length in kilometres |
| `event_name` | FIA event title used to map PDFs to this circuit (e.g. `"Italian Grand Prix"`) |
| `corners` | Turn number → sector (`1`–`3`) for inferring `sector` from steward fact text |

### `drivers.json`

Driver profiles keyed by slug id (e.g. `lewis_hamilton`).

| Field | Description |
|-------|-------------|
| `id`, `name` | Canonical driver slug and display name |
| `nationality`, `nationality_code`, `nationality_flag` | Used for `nationalities` column |
| `debut` | First F1 season — used to compute `years_in_sport` |
| `birth_date`, `birth_place`, `world_championships`, `relevant_relatives` | Biographical context (not all exported to CSV yet) |

### `teams.json`

Constructor profiles keyed by slug id (e.g. `red_bull`, `mercedes`).

| Field | Description |
|-------|-------------|
| `id`, `name` | Canonical team slug and display name |
| `legacy_ids` | Historical team ids (e.g. `toro_rosso`, `alphatauri`) for matching season tables |
| `nationality`, `nationality_code`, `hq_location` | Team metadata |
| `team_first_entry`, `team_last_entry` | Team history bounds |

### `seasons.json`

Per-season championship data keyed by year string (e.g. `"2019"`).

| Field | Description |
|-------|-------------|
| `year`, `rounds` | Season year and total round count (`rounds` column) |
| `calendar_in_order` | Circuit slugs in championship order — used for `round` lookup |
| `drivers.standings_order` | End-of-season driver table: `id`, `name`, `team`, `total_points` |
| `teams.standings_order` | End-of-season constructor table: `id`, `name`, `total_points` |
| `champion`, `constructors_champion`, `constructors_subchampion` | Season champions |
| `url` | Source link (e.g. Wikipedia season page) |

**Note:** Standings in `seasons.json` are season totals (not point-in-time per round). Ergast fallback can fill gaps when reference data is missing.

### `incident_type_keywords.json`

Keyword lists per incident type slug. Used during **build** (not enrich) to classify `incident_type` from PDF fact/offence text.

## Enrichment order

When you run `--stage enrich`, the pipeline applies sources in this order:

1. **`reference_enrich.py`** — local JSON files (primary)
2. **`ergast.py`** — API fallback for cells still empty (`ergast_fallback_enabled` in config)
3. **`fastf1_enrich.py`** — lap at incident time, weather, safety car (`lap`, `track_conditions`, etc.)

Columns still expected to be manual or unimplemented: `severity`, `flag`, `positions_of_involved parties`, `superlicense_points_before_incident`.
