# Normative Deviation Report — 2026-08-25

## Interpretation

Normative outcomes reflect one documented, rule-based interpretation of racing
regulations. They are **not** ground truth and are intended for research comparison
against FIA stewarding decisions.

- Rules version: `1.0.0`

## Rule assumptions

- Avoidable contact with lasting advantage → time penalty (minor)
- Racing incident with no driver predominantly at fault → no further action
- Repeated same-class offence within 5 races → escalate one severity level
- Qualifying blocking / impeding → grid penalty for next session (major)
- Output is one documented interpretation — not certified regulatory ground truth
- Escalation counters use FIA actual prior outcomes by default (see configs/normative.yaml)
- Rows without a matching rule receive manual_review until rules are expanded

## Aggregate metrics (FIA vs normative)

- Rows compared: **234**
- Agreement rate: **52.6%**
- Cohen's kappa: **0.223**
- Mean deviation direction (normative − FIA): **-0.449**
- FIA harsher rate: **41.5%**
- Normative harsher rate: **6.0%**
- `manual_review` rate: **59.8%**

### Excluding `manual_review` rows

- Matched rows: **94**
- Agreement rate: **79.8%**
- Cohen's kappa: **0.251**

## Optional ML comparison (validation overlap only)

- Overlap rows: **144**
- FIA vs ML agreement: **47.2%**
- Normative vs ML agreement: **60.4%**
- FIA vs normative agreement (overlap): **49.3%**

## Highest disagreement by incident type

| Incident type | n | Disagreement | FIA harsher | Normative harsher |
|---|---:|---:|---:|---:|
| other | 128 | 70.3% | 70.3% | 0.0% |
| yellow_flag | 11 | 63.6% | 9.1% | 54.5% |
| unsafe_release | 12 | 25.0% | 0.0% | 25.0% |
| technical | 5 | 20.0% | 0.0% | 20.0% |
| collision | 12 | 16.7% | 16.7% | 0.0% |
| pit_lane | 45 | 13.3% | 8.9% | 4.4% |
| track_limits | 15 | 13.3% | 0.0% | 13.3% |
| overtaking | 6 | 0.0% | 0.0% | 0.0% |

## Top deviation cases

1. `2019_qualifying_3045785150_alexander_albon` — FIA severity 2 vs normative 0 (`other_unclassified`)
2. `2019_practice_671a21b2ec_pierre_gasly` — FIA severity 2 vs normative 0 (`other_unclassified`)
3. `2019_practice_8bd5d27bb0_antonio_giovinazzi` — FIA severity 2 vs normative 0 (`other_unclassified`)
4. `2019_qualifying_9da7e3994b_kimi_raikkonen` — FIA severity 2 vs normative 0 (`other_unclassified`)
5. `2019_qualifying_a5dc574293_pierre_gasly` — FIA severity 2 vs normative 0 (`other_unclassified`)
6. `2019_qualifying_212e376927_george_russell` — FIA severity 2 vs normative 0 (`other_unclassified`)
7. `2019_qualifying_1e3fe1eb62_kevin_magnussen` — FIA severity 2 vs normative 0 (`other_unclassified`)
8. `2019_qualifying_7fe543b12c_kevin_magnussen` — FIA severity 2 vs normative 0 (`other_unclassified`)
9. `2019_qualifying_27d5ca004a_sebastian_vettel` — FIA severity 2 vs normative 0 (`other_unclassified`)
10. `2019_practice_72ecc3f346_daniil_kvyat` — FIA severity 2 vs normative 0 (`other_unclassified`)
