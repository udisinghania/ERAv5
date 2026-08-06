# Curriculum and executable mixture schedule v1

Difficulty is a scheduling proxy, not a quality judgment. Records are ranked by token length within each source, while the independent human-reviewed quality weight remains available for record selection.

- Curriculum records: 24,842
- Demonstration loss-token budget: 262,144
- Pre-anneal loss tokens: 256,901
- Isolated anneal loss tokens: 5,243
- Schedule hash: `sha256:0477330cd06ea86f99f164c853751659987566a2f20f93ffe2bf515a49fe3382`

## Stage quotas

| Stage | Permission | Sequence | Loss tokens | Bands |
|---|---|---:|---:|---|
| seed | train | 256 | 13,107 | B0, B1 |
| general_foundation | train | 256 | 144,179 | B0, B1, B2 |
| reasoning_skill_build | train | 256 | 52,429 | B2, B3, B4 |
| long_context | train | 512 | 47,186 | B2, B3, B4 |
| anneal | anneal | 512 | 5,243 | B4, B5 |

## Pre-anneal lane targets

| Lane | Loss tokens |
|---|---:|
| general | 136,152 |
| science_math | 25,690 |
| code | 25,691 |
| reasoning | 25,691 |
| long_context | 21,837 |
| indic | 20,554 |
| agentic | 1,286 |

## Protected floors

- indic: 20,554 scheduled; 8.001% of pre-anneal loss tokens; passed.
- agentic: 1,286 scheduled; 0.501% of pre-anneal loss tokens; passed.

## Pre-anneal Indic tier targets

| Tier | Loss tokens |
|---|---:|
| verified_native | 8,221 |
| unverified_native | 4,111 |
| translated | 6,165 |
| synthetic | 2,057 |

Every stage/lane/tier target is below its eligible no-replacement supply. Validation and never-train shards contribute no scheduling supply. Anneal records are accessible only to the final reserve stage. The ordinary four-tier Indic ratio applies before annealing; the final high-difficulty reserve renormalizes those weights over tiers with eligible B4/B5 supply, so it does not fabricate a synthetic tier that is absent from the reserve.
