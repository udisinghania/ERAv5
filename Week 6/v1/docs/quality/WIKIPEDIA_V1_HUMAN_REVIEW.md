# Wikipedia corpus-v1 human-review findings

The deterministic corpus-v1 packet contained 50 review slots representing 47 unique records. The corrected review outcome was 33 keep, 13 downweight, and 4 reject decisions across the 50 slots. In this review, **keep** means retain at the record's existing band weight; it does not restore B0 material to full weight.

## Confirmed rejects

1. `Results of the 1994 Sri Lankan general election by electoral district`: the extracted result tables were absent and mostly repeated footnotes remained.
2. `Coach Trip (series 8)`: raw wikitable syntax survived extraction.
3. `1951 Ohio State Buckeyes baseball team`: raw wikitable syntax survived extraction.
4. `FC Luch Vladivostok`: extensive raw table syntax survived extraction and a chunk ended mid-entry.

## Repeated error patterns

| Pattern | Review evidence | corpus-v2 response |
|---|---|---|
| Category-only continuation tails | Caning, Saint Seiya Omega, Gérard Condé, Kenneth Leighton were B1 despite having less training value than clean B0 shorts | `category_tail` signal; retain at B0 weight with a dedicated cap |
| Raw wikitable leakage | Markup was hidden beyond the beginning-only previews in several chunks | `raw_wikitable_markup` signal; reject affected chunks and count them for a future salvage pass |
| Orphaned table footnotes | Sri Lankan election page retained repeated notes without its substantive tables | `orphaned_table_footnotes` signal; reject affected chunks |
| Unbulleted linewise lists | Taxonomic, filmography, and checklist records escaped bullet-based list detection | `linewise_list` signal; retain at B0 weight with a dedicated cap |
| Abbreviation-induced mid-entry splits | Scientific entries were split after `var.`, `subsp.`, initials, or at a word boundary | preserve line units before abbreviation-aware sentence splitting |
| Beginning-only review evidence | Reviewers could not validate a chunk's end or markup hidden later in the record | every review example now includes beginning and ending previews |

## Important interpretation

Repeated-trigram score was not a reliable list detector by itself. Reviewed list examples ranged from 0.154 to 0.435. The recurring failure was structural: one semantic item per newline without an explicit bullet marker. Corpus-v2 therefore adds a separate line-structure signal instead of lowering the repetition threshold globally.

PII review boxes mean no visible issue was found in the displayed review evidence. They are not a substitute for the deterministic source-aware PII scan applied to the complete record.

## Versioning decision

Corpus-v1 remains unchanged as the reviewed baseline. The corrections are implemented only in corpus-v2 so the effect of each learning cycle remains reproducible and auditable.
