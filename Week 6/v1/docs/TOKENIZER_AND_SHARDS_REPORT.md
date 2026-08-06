# Frozen corpus, tokenizer, and shard report

## Stage boundary

Tokenization begins only after cleaning, human review, deduplication, evaluation decontamination, and group-level partitioning are frozen. This ordering matters: changing text after tokenizer training would invalidate token counts, manifests, mixture accounting, and exact replay.

## Frozen corpus v1

- Reviewed training inputs: 23,895 records.
- Admitted after the final firewall: 23,842 records.
- Rejected at freeze: 49 exact duplicates and 4 GSM8K 13-gram overlaps.
- Never-train evaluation: 1,000 physically separate GSM8K records.
- Group leakage: zero.
- Evaluation overlap after freeze: zero.
- Corpus hash: `sha256:0b7be48f0803e32751e8b2bcf5091a30f5e1c99c99cce182c68082960c7cbafb`.

The old `data/curated` baseline remains unchanged. The reviewed freeze lives in `data/frozen_corpus_v1`.

## Tokenizer v1

The optional third-party tokenizer package is unavailable in the offline runtime, so the project uses a dependency-free deterministic tokenizer trained from the final training partition only.

- Algorithm: greedy learned unigram pieces with complete byte fallback.
- Vocabulary: 8,192 tokens.
- Training documents: 21,015 train-partition records; validation, anneal, and never-train records are excluded from tokenizer learning.
- Special tokens: role boundaries, state/action/observation boundaries, PII placeholders, and Indic language tags.
- Unknown tokens: zero by construction.
- Full-corpus encode/decode round-trip failures: zero.
- Tokenizer hash: `sha256:1ac8ae0f659f554dfa77e272b08aab33d552a91329a9715f9fdbd6034f9c01be`.

| Lane | Tokens per whitespace-delimited word |
|---|---:|
| Agentic | 3.051 |
| Code | 4.614 |
| General | 3.054 |
| Indic | 4.565 |
| Long context | 3.050 |
| Reasoning | 3.056 |
| Science/math | 3.149 |

Code and Indic fertility is higher because identifiers and unseen Indic word forms decompose into learned subpieces or UTF-8 byte fallbacks. The representation remains exact and unknown-free; tokenizer optimization is a future quality improvement rather than a correctness blocker for the laptop demonstration.

## Tokenized shards

The freeze produces 22 lane/permission shards containing 51,944,856 tokens. Of those, 46,860,382 tokens carry training or validation loss.

| Lane | Records | Tokens | Loss-bearing tokens |
|---|---:|---:|---:|
| Agentic | 2,021 | 5,242,126 | 917,710 |
| Code | 4,099 | 12,433,222 | 12,429,123 |
| General | 5,183 | 7,761,840 | 7,756,657 |
| Indic | 5,304 | 9,302,175 | 9,241,285 |
| Long context | 1,341 | 6,791,977 | 6,790,636 |
| Reasoning, including 1,000 never-train records | 3,996 | 1,213,203 | 527,556 |
| Science/math | 2,898 | 9,200,313 | 9,197,415 |

Each shard contains:

- `tokens.uint16.bin`: compact token IDs;
- `loss.uint8.bin`: one origin-aware loss permission per token;
- `index.jsonl.gz`: record, group, source, offset, length, weight, and loss-policy lineage;
- `manifest.json`: content, tokenizer, cleaning, mask, source-lock, language, permission, and parent-manifest hashes.

## Loss origin rules

- Ordinary pretraining documents: all content after BOS carries causal loss.
- GSM8K: question is context; reasoning and final answer carry loss.
- Agentic trajectories: system/user/tool output is context; assistant and tool-call output carries loss.
- Assignment 4 translations: source state and action are context; translated observation carries loss.
- Never-train evaluation: every loss bit is zero.

These masks are created before packing, so concatenation cannot accidentally turn evaluation or prompt tokens into optimization targets.
