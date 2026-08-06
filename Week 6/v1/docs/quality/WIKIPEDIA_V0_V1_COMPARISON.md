# Wikipedia general: corpus-v0 versus corpus-v1

Corpus-v1 applies the six approved recommendations to the same cached raw candidates. Corpus-v0 remains unchanged through the `corpus-v0` Git tag.

## Before and after

| Measure | corpus-v0 | corpus-v1 |
|---|---:|---:|
| Raw candidates examined | 4,800 | 4,800 |
| Parent documents admitted | 4,000 | 4,336 |
| Output records/chunks | 4,000 | 5,198 |
| Output text characters | 11,440,810 | 16,095,811 |
| Mid-boundary character truncations | 434 | 0 |
| Records with PII redactions | 78 | 0 |

## corpus-v1 quality bands

| Band | Records | Meaning |
|---|---:|---|
| B0 | 428 | Short, disambiguation, list-dominant, or repetitive material |
| B1 | 30 | Usable material with structural quality warnings |
| B2 | 1,407 | Standard clean prose |
| B3 | 1,782 | Substantial multi-paragraph clean prose |
| B4 | 1,551 | Long, well-structured clean prose |

## Sampling cap groups

| Cap group | Records | Maximum scheduled share |
|---|---:|---:|
| general_disambiguation | 106 | 2.0% |
| general_short | 304 | 1.0% |
| general_structured_low_prose | 48 | 2.0% |

## What changed

1. Articles split at paragraph, sentence, or—only as a last resort—word boundaries.
2. Public-reference masking preserves bare identifiers and technical IP addresses.
3. Emails, secrets, explicitly labelled phones, and formatted international phones remain masked.
4. Useful 300–399 character parents are retained in B0.
5. Short, disambiguation, list-like, and repetitive material uses lower weights and explicit caps.
6. Every chunk records signals, flags, band, weight, caps, policy hash, boundary, parent, and PII counts.

## Remaining gate

This policy is Wikipedia-specific. Representative v1 samples and cap sizes must be reviewed before analogous lane-specific policies are applied elsewhere. Generate and complete `WIKIPEDIA_V1_REVIEW_PACKET.md`; it covers every band, every cap group, and all non-paragraph chunk boundaries.
