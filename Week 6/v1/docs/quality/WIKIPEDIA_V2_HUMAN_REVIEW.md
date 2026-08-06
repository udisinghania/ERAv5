# Wikipedia corpus-v2 human-review findings

The v2 deterministic packet contains 56 review slots representing 53 unique records. The checked decisions total **36 keep and 20 downweight across the 56 slots**, or **35 keep and 18 downweight across the 53 unique records**. No reviewed retained record was marked for structural-quality rejection.

## Confirmed fixes

1. All sampled category-only tails now receive B0 weight and the `general_category_tail` cap.
2. All nine non-paragraph chunks end on complete preserved lines; the v1 abbreviation and word-boundary failures did not recur.
3. Previously missed taxonomic and filmography lists now receive B0 weight through `linewise_list`.

## Remaining structural gap

`Pelopas Kiato F.C.` is a short, statistics-heavy honours list with only two prose sentences. It remains B1 at weight 0.50 because it narrowly misses the generic linewise-list thresholds. A diagnostic combination of alpha fraction below 0.45, digit fraction at least 0.15, short-line fraction at least 0.90, and at least eight nonempty lines identifies Pelopas and no other corpus-v2 record. This is a real but low-severity misband. It is recorded for the next general-lane policy version rather than silently changing the reviewed corpus-v2 artifact for one record.

## Sensitive-personal-data review

The reviewed Alachua County chunk names a jail inmate in connection with childbirth and an infant death. This is publicly available, public-interest reference material, but it combines a likely private individual with health, custody, and traumatic-event context. That is not equivalent to an ordinary phone/email PII pattern, and an automatic name-removal rule would risk suppressing victims, witnesses, historical subjects, and legitimate public reporting.

The record is therefore placed in `data/experiments/corpus_v2/manual_review_registry.json` and is ineligible for training until a privacy decision chooses among retaining the public-interest account, masking the name, or excluding the paragraph/record. No automatic corpus mutation was made.

## Hard-rejection validation

The ordinary review packet samples retained records, so it cannot contain chunks rejected for raw wikitable markup. `WIKIPEDIA_V2_REJECTION_AUDIT.md` separately reconstructs every hard rejection, checks the lowest-marker cases, and verifies that all four v1 human-reviewed rejects are caught.
