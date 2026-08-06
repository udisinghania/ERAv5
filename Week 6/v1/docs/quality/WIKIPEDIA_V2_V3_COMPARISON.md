# Wikipedia general: corpus-v2 versus corpus-v3

Corpus-v3 is the final Wikipedia policy candidate from the v2 human review. Corpus-v2 remains the reviewed structural baseline.

## Before and after

| Measure | corpus-v2 | corpus-v3 |
|---|---:|---:|
| Parents retained | 4,293 | 4,328 |
| Records retained | 5,144 | 5,183 |
| Text characters | 15,884,148 | 16,000,341 |
| Table-affected chunks salvaged | 0 | 84 |
| Stat-heavy compact lists capped | 0 | 1 |
| Human sensitive-context caps | 0 | 1 |

## Decision logic

- Raw wikitable blocks are removed rather than rendered. Any salvaged surrounding prose is conservatively placed in capped B0.
- Pelopas-style compact numeric/honours lists use a separate stat-heavy-list rule and capped B0.
- The Alachua County case is a hashed human-review override. A naive automated sensitive-name detector was rejected because it flagged 891 mostly ordinary public-reference chunks.
- Orphaned table footnotes remain hard rejections because the substantive table content is absent.

## v3 bands

| Band | Records |
|---|---:|
| B0 | 820 |
| B1 | 1 |
| B2 | 1,328 |
| B3 | 1,611 |
| B4 | 1,423 |

## Gate

The v3 verifier checks the reviewed Pelopas and Alachua decisions, table salvage, hard-rejection removal, lineage, hashes, and deterministic record counts.
