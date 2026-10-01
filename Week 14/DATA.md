# Exact training inputs and provenance

This package reuses the prior assignment's frozen 50M-target corpus. The dataset is included under `data/`, so a fresh clone with Git LFS does not need Week 13 or any local original folder.

| Input | Count / storage |
|---|---|
| Training supervised targets | 50,000,000 |
| Packed training sequences | 104,863 × 512 |
| Physical training token slots | 53,689,856 |
| Non-padding training tokens | 53,689,750 |
| Held-out supervised targets | 5,049,456 |
| Held-out sequences | 11,623 × 512 |
| Frozen vocabulary | 8,192 IDs |

The four training arrays are memory-mapped files: little-endian uint16 token IDs, uint8 target loss masks, int16 document segment IDs, and uint16 document-relative positions. Validation has its own corresponding four arrays and source-boundary manifest. BOS/padding are excluded from target loss. Attention is causal and restricted to the same document segment. Position IDs reset per fragment. Targets at position j are predicted from logits at j-1.

The inherited seven lanes are agentic, code, general, Indic, long-context, reasoning, and science/math. The corpus adds FineWeb-Edu, English Wikipedia and Cosmopedia-v2. Validation contains ten source groups. Source revisions, URLs, configurations and recorded licenses are available in:

- [Inherited source locks](data/Corpus_20M_v1/baseline_source_locks.json)
- [FineWeb-Edu acquisition](data/Corpus_20M_v1/sources/fineweb_edu/lock.json)
- [Wikipedia acquisition](data/Corpus_20M_v1/sources/wikipedia_en/lock.json)
- [Cosmopedia acquisition](data/Corpus_20M_v1/sources/cosmopedia_v2/lock.json)
- [Catalog](data/Corpus_20M_v1/catalog.json) and [packing report](data/Corpus_20M_v1/packed_50m_ctx512/packing_report.json)
- [Overlap audit](data/Corpus_20M_v1/reports/overlap_audit.json) and [original verification](data/Corpus_20M_v1/reports/verification.json)

The corresponding upstream dataset cards are preserved next to the three acquisition locks. The source license metadata applies to the source material; the repository's code license does not relicense third-party text. This is a tokenized training subset, not a claim of ownership of the original content. Synthetic material can contain factual errors; heuristic deduplication and registered-evaluation overlap checks do not establish universal absence of contamination.

The catalog and historical corpus report also name raw/intermediate shards that are not bundled. These are provenance records, not missing runtime dependencies. The packed arrays, validation, tokenizer, loader and vendored tokenizer modules suffice to train and evaluate. `segments.jsonl.gz` preserves packed-fragment lineage. Corpus acquisition is not run during reproduction.

Training and continuation reuse exactly this dataset. The holdout receives no optimizer updates, but was examined during development and is not an untouched final test set. The model's reported loss is computed on all held-out supervised targets, weighted by their counts rather than averaging source losses equally.

All binary inputs and the three final checkpoints have relative SHA-256 entries in `MANIFEST.json`. `python run.py verify` validates the package after copying or cloning, including detection of unexpanded Git LFS pointers.
