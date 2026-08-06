# Wikipedia general: corpus-v1 versus corpus-v2

Corpus-v2 is the versioned response to the corpus-v1 human review. Corpus-v1 remains immutable and reproducible.

## Before and after

| Measure | corpus-v1 | corpus-v2 |
|---|---:|---:|
| Parent documents retained | 4,336 | 4,293 |
| Output records | 5,198 | 5,144 |
| Output text characters | 16,095,811 | 15,884,148 |
| Raw-wikitable chunks retained | not measured | 0 |
| Raw-wikitable chunks rejected | 0 | 50 |

## corpus-v2 quality bands

| Band | Records | Meaning |
|---|---:|---|
| B0 | 737 | Retained low-prose, short, list, disambiguation, or category-tail material |
| B1 | 2 | Usable material with non-list structural quality warnings |
| B2 | 1,334 | Standard clean prose |
| B3 | 1,620 | Substantial multi-paragraph clean prose |
| B4 | 1,451 | Long, well-structured clean prose |

## New reviewed signals

| Flag | Records | Treatment |
|---|---:|---|
| category_tail | 27 | retain in capped B0 |
| linewise_list | 335 | retain in capped B0 |
| raw_wikitable_markup | 50 | reject chunk |
| orphaned_table_footnotes | 2 | reject chunk |

## Boundary behavior

| Boundary | Chunks |
|---|---:|
| line | 9 |
| paragraph | 5,135 |

The new `line` boundary preserves extracted list entries before sentence logic. Sentence splitting also refuses common abbreviations and single-letter initials.

## Decision gate

Review the v2 packet, including both beginning and ending previews. If the targeted errors are gone without unacceptable false positives, freeze this Wikipedia policy and begin lane-specific audits for the remaining data lanes.
