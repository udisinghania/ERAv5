# Remaining lanes v1 completed review record

## Evidence counts

- Numeric repairs: 31 samples — 27 valid, one true phone exposure, one financial-account exposure, and two unclear quality cases.
- Boundaries: 21 sampled parents and 57 displayed joins — all marked clean.
- Targeted fallback audit: all 13 word-boundary records inspected — no mid-word or mid-number cuts.

## Confirmed exceptions

| Source | Parent | Finding | v2 action |
|---|---|---|---|
| `sangraha_verified_hindi` | `5d8428585046edef47a2abb8630431b0e3b23fb3` | Two bank accounts tied to a named fraud victim | Mask both as `[BANK_ACCOUNT]` |
| `sangraha_unverified_hindi` | `607393391270fd15d3e1d1d6a252e9fb756f7bb42533639021f0788122718463` | Repeated real WhatsApp contact number | Preserve `[PHONE]` masking |
| `sangraha_synthetic_hindi` | `181fc620ce792c0efc0d69c7cad402c6c6716852c120c9e29b16b8046177c9d2` | Garbled duplicated digit strings | Exclude parent |
| `sangraha_verified_hindi` | `9951ed19a17f07076709a9fa53a23c36c4f373cd1dedf440c95f9f7cc7fa8c92` | OCR-merged verse numbering | Exclude pending source check |
| `assignment4_samanantar_translated` | `89530342c4770bf63ea47f65386b788c2e91b77eebccbc93745d4107903a0eb2` | `[PHONE]` masks the Hindi value supplied as 250,000 in the paired English state | Restore `250,000` by human adjudication |

## Final gate result

The v1 experiment is retained as reviewed evidence but is superseded by `remaining_lanes_v2`. The v2 verifier asserts that the sensitive numbers are absent, the two quarantined parents are absent, all artifact hashes and lineage are valid, and the reviewed boundary population has not changed.
