# Corpus acquisition and curation report

The Assignment 6 toy corpus is intentionally bounded but real. It contains
20,000 training candidates and 1,000 physically separate evaluation records.
Every remote source is pinned to a full Hub commit SHA; the Assignment 4 source
is pinned to its parent shard content hash.

| Primary lane | Source | Candidate records | License policy |
|---|---|---:|---|
| General | Wikimedia Wikipedia 20231101.en | 4,000 | CC-BY-SA-3.0 / GFDL |
| Long context | Wikimedia Wikipedia 20231101.en | 1,000 | CC-BY-SA-3.0 / GFDL |
| Science/math | OpenWebMath | 2,000 | ODC-By 1.0; original URL retained |
| Code | CodeParrot Clean | 3,000 | Per-row allowlist: Apache, MIT, BSD, ISC, Unlicense |
| Reasoning | GSM8K train | 3,000 | MIT |
| Indic | Sangraha verified Hindi | 2,000 | CC-BY-4.0 |
| Indic | Sangraha unverified Hindi | 1,000 | CC-BY-4.0 |
| Indic | Sangraha synthetic Hindi | 1,000 | CC-BY-4.0 |
| Indic | Assignment 4 Samanantar translations | 1,000 | CC-BY-NC-4.0 |
| Agentic | Hermes function calling / JSON / Glaive | 2,000 | Apache-2.0 |

GSM8K test contributes 1,000 `never_train` records to the evaluation registry.
It is not counted as a primary lane or as training supply.

## Curation outcome

- Input training candidates: 20,000
- Admitted after cross-source deduplication and evaluation firewall: 19,948
- Exact duplicates rejected: 48
- 13-gram evaluation overlaps rejected: 4
- Never-train evaluation records: 1,000
- Deterministic partitions: train 88%, validation 10%, anneal 2%
- Primary lanes with train, validation, and anneal supply: 7 of 7

The executable counts, content hashes, rejection totals, source lock hashes,
and registry hash are in `data/curated/curation_report.json`. Run
`python scripts/verify_corpus.py` for a complete offline integrity check.
