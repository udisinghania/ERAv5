# Week 4 Local Evidence Audit

Audit target: `D:\ERAv5\Week_4`  
Authoritative run: `D:\ERAv5\Week_4\twm_pilot_500k_v1`  
Mode: read-only; Week 4 was not modified.

## What is present

The local folder contains **24,385 files** totaling **16,282,193,115 bytes**.

| Artifact class | Local finding |
|---|---|
| README | None |
| Git metadata | None; `script_git_hash` is null |
| Source | 17 Python files, `main.ipynb`, and one notebook checkpoint |
| Parquet | 24,322 files; 3,647,825,360 bytes |
| SQLite | Four files; 12,600,295,424 bytes |
| Main run | Complete Stage A-D run with state, report, manifest, debug samples, staged/final data, split data, and LSH index |
| Other runs | Interrupted `100k`; completed older `100k_v2`; initialized-only `500k`; unscoped `twm_output` |
| Tokenizer | 32k byte-level BPE JSON plus 11-language fertility report |
| Decontamination | 2,489 GSM8K/ARC-Challenge proxy questions and versioned manifest |
| Graphs/images/PDF | None found |
| HTML/site/Netlify | No config, URL, deployment metadata, or local dashboard bundle found |

The only relevant "dashboard" text is an orchestrator comment and an ordinary source sentence containing that word. Local and Netlify contents cannot be asserted identical because no local deployment artifact exists.

All **24,322** files with a `.parquet` extension were checked for the required `PAR1` header and footer magic; all passed. This is a container-integrity check, not an independent schema, row-level, or semantic validation.

### Main-run artifact layout

| Path under `twm_pilot_500k_v1` | Files | Bytes | Purpose |
|---|---:|---:|---|
| `spokes_partitioned` | 2,816 | 766,714,264 | Stage-A cleaned translation spokes |
| `twm_formatted` | 256 | 701,606,022 | Stage-B English-hub TWM records |
| `twm_final` | 256 | 700,448,686 | Stage-C deduplicated records |
| `pretraining_split` | 513 | 700,059,646 | 98/2 hub-safe train/validation split plus manifest |
| `twm_minhash.sqlite` | 1 | 8,598,130,688 | Stage-C disk-backed LSH index |
| `abandoned_attempts` | 3 | 32,345,320 | Recoverably quarantined intermediate attempts |
| reports/state/manifest/debug | 7 | small | Lineage and human-audit evidence |

The raw Samanantar corpus is not stored as a separate local input dump. `orchestrator.py` streams the Hugging Face dataset. Local inputs include the FastText `lid.176.ftz` model, golden proxy JSONL, source code, and run configuration.

## Run authority

| Run | Status | Use in inventory |
|---|---|---|
| `twm_pilot_100k` | Stage A completed for 5/11 languages only | Exclude |
| `twm_pilot_100k_v2` | Complete older run | Secondary cross-check only |
| `twm_pilot_500k` | Initialized; no data stages complete | Exclude |
| `twm_pilot_500k_v1` | 11/11 Stage A; 256/256 Stage B; Stage C/D complete | **Authoritative** |

The older notebook is historical evidence. It contains execution errors and references APIs no longer present in the current files. Its 1M-row Marathi experiment reported 11,922 normalized rows with length changes, 616,749 post-filter rows, and 614,361 post-dedup rows. Those figures must not be merged with the authoritative main run.

The local hardware query reports **NVIDIA GeForce RTX 3070 Laptop GPU, 8,192 MiB**, driver 546.30. The active shell Python lacks the Week 4 PyArrow stack and PyTorch, so the pipeline or micro-proxy should run in an isolated environment or Kaggle; no dependency installation belongs inside Week 4.

## Main-run measured inventory

### Per-language counts

`Raw` and `Stage A` are translation-pair units. `Stage B` is the count of target-language memberships after repeated `(English source, language)` records are deterministically consolidated.

The audited run records English-to-Indic pairs. It does not materialize reverse-direction examples, so no source pair is double-counted as two directions. Any future reverse example must inherit its parent-pair ID and remain a derived example.

| Language | Raw | Stage-A survivors | Stage-A retention | Stage-B translations | Stage-B/raw |
|---|---:|---:|---:|---:|---:|
| Assamese (as) | 141,227 | 120,593 | 85.389479% | 108,727 | 76.987403% |
| Bengali (bn) | 500,000 | 473,048 | 94.609600% | 411,562 | 82.312400% |
| Gujarati (gu) | 500,000 | 473,487 | 94.697400% | 390,098 | 78.019600% |
| Hindi (hi) | 500,000 | 485,376 | 97.075200% | 455,363 | 91.072600% |
| Kannada (kn) | 500,000 | 443,194 | 88.638800% | 372,702 | 74.540400% |
| Malayalam (ml) | 500,000 | 439,882 | 87.976400% | 370,087 | 74.017400% |
| Marathi (mr) | 500,000 | 468,831 | 93.766200% | 394,070 | 78.814000% |
| Odia (or) | 500,000 | 482,842 | 96.568400% | 346,219 | 69.243800% |
| Punjabi (pa) | 500,000 | 483,934 | 96.786800% | 428,798 | 85.759600% |
| Tamil (ta) | 500,000 | 443,366 | 88.673200% | 381,279 | 76.255800% |
| Telugu (te) | 500,000 | 449,443 | 89.888600% | 361,224 | 72.244800% |
| **Total** | **5,141,227** | **4,763,996** | **92.662627%** | **4,020,129** | **78.193960%** |

### Cleaning, privacy, and deduplication

| Metric | Value | Evidence |
|---|---:|---|
| Empty or malformed pairs | Not separately recorded | No dedicated main-run counter |
| Unicode-normalization changes | Not recorded for the main run | Historical notebook: 11,922/1M length-changing rows; do not merge |
| Script-validation failures | Not separately counted | No separate script counter |
| Length-ratio failures | 41,864 | Measured alignment drops |
| Language-validation failures | 37,927 | Measured FastText LID drops; not a script-failure count |
| Quality drops | 297,440 | Measured |
| "Too few words" rejections | 296,926 | Measured; 99.827192% of quality drops |
| Total Stage-A rejection | 377,231 (7.337373%) | Derived: raw - Stage-A survivors |
| Exact proxy-overlap drops | 0 | Measured; limited to configured proxy policy |
| PII redactions | 4,616 | Measured; not necessarily distinct records |
| Stage-B consolidation difference | 743,867 | Stage A - Stage B; exact duplicates and conflicting-key choices are not separated |
| Exact duplicate aligned pairs | Not separable from Stage-B consolidation | Exact duplicates and conflicting-key choices share one counter |
| English hubs before LSH | 2,977,126 | Measured |
| Average Stage-B languages/hub | 1.350339 | Derived |
| LSH candidate comparisons | 219,227,124 | Measured |
| Candidate-limit hits | 947 | Measured; audit before scale-up |
| Near-duplicate TWM hubs | 4,664 (0.156661%) | Measured LSH drops; hub-level |
| Final unique aligned-pair count | Not reconstructable from the hub-level final report | Post-LSH pair membership is not recorded |
| Final unique TWM hubs | 2,972,462 | Measured |
| Records without text signature | 0 | Measured |

`2,972,462 / 5,141,227` is **not** a retention rate: Stage B aggregates multiple language-pair records into English-centered hubs, changing the unit.

### Tokens, split, and tokenizer

| Metric | Value | Evidence class |
|---|---:|---|
| Final clean-token estimate | 215,294,331 | Measured estimator output |
| Source-side words or tokens | Not recorded | The main run did not emit this counter |
| Target-side words or tokens | Not recorded | The main run did not emit this counter |
| Combined formatted-sequence tokens | 215,294,331 | Heuristic estimator; includes source, targets, and TWM tags |
| Estimated tokens/final hub | 72.429633 | Derived |
| Train rows | 2,912,312 (97.97642493%) | Measured |
| Validation rows | 60,150 (2.02357507%) | Measured |
| Estimated train tokens | about 210,937,689 | Derived by row fraction; assumes equal average length |
| Estimated validation tokens | about 4,356,642 | Same derived assumption |
| Raw word count | Not recorded | The main run did not emit this counter |
| Raw token count | Not recorded | The main run did not emit this counter |
| Frozen-V5 clean tokens | Pending frozen-V5 retokenization | Local formatted data exists |
| Unique corpus token types | Not measured | No corpus-type inventory was produced |
| Week 4 tokenizer vocabulary | 32,000 entries | Measured; not unique corpus inventory |

Cleaning reports exist as JSON run state, detailed pipeline report, shard manifest, split manifest, benchmark-proxy manifest, and tokenizer fertility report. No cleaning chart or graph was found locally.

The clean-token estimator is `ceil(semantic characters / 3)` over formatted TWM text. It is not tokenizer execution. Applying the split row fraction is a planning estimate because exact train/validation token lengths were not recorded.

The Week 4 tokenizer used exactly 2,000,000 target semantic characters per language for training and 1,000 validation records per language for fertility. Measured tokens/word range from **3.279668 Punjabi** to **6.873155 Malayalam**. These figures are specific to the Week 4 tokenizer.

### Final target-language distribution

| as | bn | gu | hi | kn | ml | mr | or | pa | ta | te |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.700718% | 10.225724% | 9.706132% | 11.315938% | 9.279161% | 9.212864% | 9.806150% | 8.613128% | 10.666143% | 9.481838% | 8.992206% |

These percentages count target-language memberships in final hubs, not mutually exclusive documents.

## Pipeline and provenance

Reusable current components:

| File | Function | Session 5 use |
|---|---|---|
| `orchestrator.py` | Streaming cleaning, TWM aggregation, lineage | Reuse for translated-Indic expansion after revision/license audit |
| `normalizer.py` | NFC, HTML unescape, codepoint/tag/whitespace cleanup | Reuse across text lanes |
| `lang_filter.py` | FastText LID, length-ratio alignment, heuristic quality | Reuse after language-specific false-rejection audits |
| `pii_masker.py` | Email/phone/IPv4 masking; exact benchmark firewall | Reuse; expand PII coverage for web/native data |
| `deduplicator.py` | Disk-backed MinHash/LSH | Reuse for cleaning experiments |
| `test_lsh_recall.py` | Synthetic LSH recall test | Reuse once Week 4 dependencies are available |
| `train_val_splitter.py` | Deterministic hub-safe 98/2 split | Reuse to prevent translation siblings crossing splits |
| `fertility_trainer.py` | Balanced tokenizer training/fertility | Reuse for candidate tokenizer comparisons |
| `build_golden_proxies.py` | Versioned GSM8K/ARC references | Expand into a complete evaluation registry |

The new Week 5 `samanantar_cleaning_ablation.py` reuses the current normalizer, language/alignment/quality filters, PII/decontamination policy, and bounded MinHash clustering. It creates equal-size, cluster-disjoint A/B/C training sets and evaluates a compact multilingual translation checkpoint. It is an optional cleaning micro-proxy, not a 1B/3B run.

The main pipeline records:

- run ID and timestamps;
- source URL and revision string;
- FastText, golden-proxy, content, and script SHA-256 values;
- normalization/filter configuration hash;
- deterministic global ordering;
- hub-safe deterministic train/validation split;
- CC-BY-NC-4.0 / `ADMITTED-NC`.

Remaining provenance gaps:

- Samanantar is recorded as mutable `main`, not a resolved commit;
- the folder has no Git metadata and `script_git_hash` is null;
- an older notebook manifest says CC-BY, while the authoritative run says CC-BY-NC-4.0; the stricter main-run record controls;
- no human-verifier, native-author, translator, or synthetic-generator lineage is stored;
- no independently verified deployment artifact exists.

## Session 5 lane support

| Lane or tier | Current local evidence |
|---|---|
| Indic verified native | Missing local supply |
| Indic unverified native | Missing local supply |
| Indic translated | Partially supported by Week 4 AI4Bharat Samanantar |
| Indic synthetic | Missing local supply |
| General web | Missing local supply |
| Code | Missing local supply |
| Science and mathematics | Missing local supply |
| Agentic | Missing local supply |
| Explicit reasoning | Missing local supply |
| Long context | Missing local supply |

Samanantar is counted only in the Indic translated row. Its English source strings are not General-web inventory; benchmark proxy questions are not Reasoning training supply; and TWM structural tags are not Agentic trajectories. No held-back Anneal inventory is locally evidenced.

## Differentiation from the batchmate submission

No FineWeb-2 Hindi inventory, character-level repetition experiment, mixture percentage, wording, result, or conclusion is used as evidence here. This submission is based on Samanantar translated parallel data and proposes a pair-level cleaning-policy ablation with compact translation-model metrics.

## Gap analysis and recommended README edits

1. Use only `twm_pilot_500k_v1` as authoritative Week 4 inventory.
2. Keep translated pairs, target memberships, and final hubs as separate units.
3. Label 215,294,331 as a heuristic estimate and all public quantities provisional until frozen-V5 retokenization.
4. Classify Week 4 only as Translated Indic.
5. Do not infer a production token budget from the 240B-parameter production scale; show demand per 1T and scale algebraically.
6. Enforce Indic 8% and Agentic 0.5% in every pre-Anneal stage.
7. Hold Anneal outside the 98% pre-Anneal mix and protect its Tier-A Agentic and B5 Reasoning allocations.
8. Acquire in order: verified-native Indic; unverified-native Indic web; additional translated languages/domains; Tier-A Agentic; verifier-backed Reasoning; Long-context; then Code, Science/Math, and General web.
9. Before translated-tier expansion, audit the dominant "too few words" rule, LID ambiguity classes, low Stage-B/raw languages, and candidate-limit hits.
10. Run the optional A/B/C Samanantar ablation only as a Week 4 cleaning micro-proxy; keep 1B/3B mixture gates explicitly planned.
