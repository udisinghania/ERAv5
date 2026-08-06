# Remaining lanes: cleaning experiment v1

This versioned experiment rebuilds the same selected source parents with numeric-safe PII policies and boundary-aware chunks. Baseline snapshots remain unchanged.

| Lane | Source | Parents | Records | Split parents | Recovered `[PHONE]` | Character gain |
|---|---|---:|---:|---:|---:|---:|
| long_context | wikipedia_long_en | 1,000 | 1,341 | 236 | 72 | 3,019,317 |
| science_math | openwebmath_science_math | 2,000 | 2,898 | 318 | 1,884 | 8,191,349 |
| code | codeparrot_permissive_python | 3,000 | 4,099 | 490 | 2,593 | 9,588,603 |
| reasoning | gsm8k_reasoning_train | 3,000 | 3,000 | 0 | 264 | -115 |
| indic | sangraha_verified_hindi | 2,000 | 2,091 | 60 | 23 | 377,379 |
| indic | sangraha_unverified_hindi | 1,000 | 1,073 | 38 | 20 | 357,692 |
| indic | sangraha_synthetic_hindi | 1,000 | 1,178 | 83 | 48 | 948,360 |
| indic | assignment4_samanantar_translated | 1,000 | 1,000 | 0 | 0 | 0 |
| agentic | hermes_function_calling | 1,000 | 1,035 | 35 | 942 | 13,150 |
| agentic | hermes_json_agentic | 500 | 500 | 0 | 182 | 852 |
| agentic | hermes_glaive_function_calling | 500 | 500 | 0 | 8 | 35 |

Recovered `[PHONE]` markers are candidate false-positive repairs; review packets must confirm them before this experiment replaces any baseline supply. Character gain comes from retaining boundary chunks instead of slicing oversized parents.
