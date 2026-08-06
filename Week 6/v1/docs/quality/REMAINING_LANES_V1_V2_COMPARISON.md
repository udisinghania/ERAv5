# Remaining lanes: cleaning experiment v2

This candidate applies the completed v1 human review without modifying either baseline snapshots or the reviewed v1 experiment.

| Lane | Source | Parents | Records | Excluded | Phone masks | Bank-account masks |
|---|---|---:|---:|---:|---:|---:|
| long_context | wikipedia_long_en | 1,000 | 1,341 | 0 | 0 | 0 |
| science_math | openwebmath_science_math | 2,000 | 2,898 | 0 | 0 | 0 |
| code | codeparrot_permissive_python | 3,000 | 4,099 | 0 | 0 | 0 |
| reasoning | gsm8k_reasoning_train | 3,000 | 3,000 | 0 | 0 | 0 |
| indic | sangraha_verified_hindi | 1,999 | 2,089 | 1 | 2 | 2 |
| indic | sangraha_unverified_hindi | 1,000 | 1,073 | 0 | 11 | 0 |
| indic | sangraha_synthetic_hindi | 999 | 1,177 | 1 | 0 | 0 |
| indic | assignment4_samanantar_translated | 1,000 | 1,000 | 0 | 0 | 0 |
| agentic | hermes_function_calling | 1,000 | 1,035 | 0 | 0 | 0 |
| agentic | hermes_json_agentic | 500 | 500 | 0 | 0 | 0 |
| agentic | hermes_glaive_function_calling | 500 | 500 | 0 | 0 | 0 |

Human-review actions: the WhatsApp number remains masked, both exposed bank-account numbers are masked, the garbled synthetic-Hindi parent is excluded, the anomalous Sanskrit/OCR parent is held pending source verification, and the translated false marker is restored to 250,000 from its paired English state.
