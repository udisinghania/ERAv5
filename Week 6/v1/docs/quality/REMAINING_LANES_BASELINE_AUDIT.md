# Remaining lanes: baseline cleaning audit

This is a diagnostic pass. It measures each lane with the existing source transform and cleaner; it does not copy Wikipedia admission thresholds or modify any source snapshot.

## Inventory

| Lane | Source | Raw candidates | Accepted before quota | Snapshot | Truncated | PII-redacted |
|---|---|---:|---:|---:|---:|---:|
| long_context | wikipedia_long_en | 7800 | 1004 | 1000 | 236 | 51 |
| science_math | openwebmath_science_math | 2300 | 2075 | 2000 | 332 | 443 |
| code | codeparrot_permissive_python | 5100 | 3016 | 3000 | 494 | 746 |
| reasoning | gsm8k_reasoning_train | 3000 | 3000 | 3000 | 0 | 49 |
| indic | sangraha_verified_hindi | 2100 | 2098 | 2000 | 63 | 25 |
| indic | sangraha_unverified_hindi | 1000 | 1000 | 1000 | 38 | 16 |
| indic | sangraha_synthetic_hindi | 1200 | 1041 | 1000 | 85 | 22 |
| indic | assignment4_samanantar_translated | n/a | n/a | 1000 | n/a | n/a |
| agentic | hermes_function_calling | 1000 | 1000 | 1000 | 35 | 220 |
| agentic | hermes_json_agentic | 500 | 500 | 500 | 0 | 60 |
| agentic | hermes_glaive_function_calling | 500 | 500 | 500 | 0 | 4 |

## How to interpret this

- Raw candidates are cached rows examined, not the full upstream dataset.
- Accepted-before-quota shows what passed the current deterministic transform and basic gate.
- Snapshot count is the pinned quota actually preserved for downstream curation.
- Truncation and generic text signals are diagnostic. Their meaning differs by lane: punctuation and low alpha can be healthy in code or mathematics, while role markers are required in agentic data.

## Measured priority order

1. **Source-aware PII policy:** the generic phone pattern masks valid numeric constants, mathematical answers, years, identifiers, and JSON values. This is label corruption, not merely a sampling preference.
2. **Boundary-aware retention instead of slicing:** science/math, code, and long-context sources lose millions of characters at hard maximum-length slices. Each requires boundaries appropriate to its structure.
3. **Lane validators:** code needs syntax/file-boundary checks; reasoning needs question/derivation/final-answer integrity; agentic data needs role, tool-call, and JSON consistency; Indic data needs script/language and provenance-tier checks.
4. **Only then freeze the tokenizer input:** tokenization must consume repaired, versioned snapshots rather than the current diagnostic baseline.

Read the per-source start/end samples in the JSON artifact before defining any thresholds. The order above is chosen from measured corruption risk, truncation, and structural extremes rather than corpus size alone.
