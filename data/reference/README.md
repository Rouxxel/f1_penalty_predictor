# Reference data (manual only)

Files in this directory are **manually curated** and must not be modified by pipeline scripts.

| File | Purpose |
|------|---------|
| `circuits.json` | Circuit slugs, countries, first F1 season, turn→sector maps, event name lookup |
| `incident_type_keywords.json` | Keyword rules for classifying incident types from PDF text |

Pipeline code loads these via `fia_ml.data.reference_data` (read-only). Writes to this
directory are blocked by `secure_file_io` and raise `ReadOnlyPathError`.

To update reference data, edit the JSON files directly in your editor.
