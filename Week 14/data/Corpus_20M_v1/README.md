# Corpus for a 20M-parameter language model

This folder is a separate corpus expansion of `Assignment_6_v2`. It contains a
verified snapshot of the baseline documents, tokenizer and tokenized shards,
new educational/general English data, and a 50,000,000-loss-token training set.
No model training is performed by the corpus builder.

## Completed result

The build and independent overlap audit passed. The folder is approximately
626 MB and contains:

| Item | Measured count |
| --- | ---: |
| Added training loss tokens | 15,003,209 |
| Added held-out validation loss tokens | 753,287 |
| Total available training loss tokens | 52,693,755 |
| Packed training loss tokens | 50,000,000 |
| Packed sequences of length 512 | 104,863 |
| New training documents | 8,239 |
| New validation documents | 401 |

All 132,921 packed fragments were checked against their original token and mask
spans. Independent checks found no normalized exact duplicates among additions
or against the baseline, and no exact 13-word training/evaluation overlaps
involving the additions. These checks do not prove semantic uniqueness.
Original baseline files retained their original hashes.

The final measured counts are in `reports/build.json`, source counts and filtering
results in `reports/additions.json`, and verification results in
`reports/verification.json`. Treat the build as complete only when those reports
exist and the verification status is `PASS`.

## Addition targets

| Source | Configuration | Training loss tokens | New validation loss tokens |
| --- | --- | ---: | ---: |
| HuggingFaceFW/fineweb-edu | sample-10BT | at least 10,000,000 | at least 500,000 |
| wikimedia/wikipedia | 20231101.en | at least 3,000,000 | at least 150,000 |
| HuggingFaceTB/smollm-corpus | cosmopedia-v2 | at least 2,000,000 | at least 100,000 |

Whole documents are retained, so source quotas may be exceeded by the final
accepted document. The packed training stream stops at exactly 50M supervised
targets. Every count uses the original frozen 8,192-ID tokenizer, including EOS
but excluding BOS and any source-specific masked tokens. Published Hugging Face
token counts use other tokenizers and are not used for quota accounting.

## Files

- `catalog.json`: relative paths and counts for all baseline and new token shards;
  `train` and `anneal` are eligible for training. Validation and `never_train`
  remain excluded.
- `baseline/`: copied original corpus, tokenized shards and frozen tokenizer.
- `additions/<source>/<split>/`: cleaned document JSONL, token IDs, loss masks,
  document offsets, hashes and counts.
- `sources/<source>/`: pinned Hugging Face revision, configuration, sampling
  details, licensing metadata and the corresponding original dataset card.
- `packed_50m_ctx512/`: shuffled training stream and record-to-fragment lineage.
- `packed_dataset.py`: a memory-mapped reader, attention-mask helper and loss
  helper for a custom PyTorch Transformer.
- `reports/`: acquisition, inherited baseline identity, packing and verification.
- `audit_overlap.py`: independent exact overlap check of the exported text.
- `tests/test_pipeline.py`: small regression tests for packing and model input semantics.
- `build_corpus.py`: resumable acquisition and deterministic corpus builder.
- `vendor/`: exact copies of the small tokenizer modules needed to preserve the
  original tokenizer behavior. The baseline source code and environments are not
  modified.

## Training contract

The packed files contain arrays of shape `[sequences, 512]`:

| Array | Type | Meaning |
| --- | --- | --- |
| input_ids | little-endian uint16 | Original 8,192-vocabulary token IDs |
| loss_mask | uint8 | Whether this position is a supervised **target** |
| segment_ids | little-endian int16 | Document-fragment identity inside a sequence; -1 means padding |
| position_ids | little-endian uint16 | Positions reset to zero for each fragment |

Use causal attention **within the same non-padding segment only**. The supplied
`attention_mask()` follows PyTorch SDPA's convention: True means allowed. Supply
the saved `position_ids` to the model. A generic causal-only model call would
ignore document isolation and is not sufficient.

Predict `input_ids[:, 1:]` from `logits[:, :-1]` and weight the losses with
`loss_mask[:, 1:]`. The first token of each document or continuation is context
only. Continuations repeat exactly one preceding token so their next-token
targets are not lost or counted twice. Padding and BOS contribute no loss.

`segments.jsonl.gz` reconstructs every fragment from the source shard and record.
The last selected record may end early to achieve exactly 50M targets. Selection
is one deterministically shuffled pass across training-eligible records, not a
repetition of the earlier assignment's 10M-token schedule. The available pool is
larger than the selected stream. This is a single-stage corpus mix; the old OPUS
and five-stage curriculum are not executed.

## Quality and evaluation separation

Sampling uses seeded random blocks from across each source, with source revision
checked on every uncached response. Truncated API rows are rejected. Exact text,
source row index and revision are retained for accepted records.

Filters reject encoding damage, very short or non-prose documents, repeated
lines/words, obvious credential patterns, low FineWeb language/education scores,
and Wikipedia lists/disambiguation pages. Obvious email addresses are masked.
This is automated screening, not a claim that all text is error-free or manually
reviewed. Cosmopedia is synthetic and can contain factual mistakes.

New records are screened against all inherited records and earlier accepted new
records using normalized exact hashes and 128-permutation MinHash candidate
retrieval, followed by exact 5-word-shingle Jaccard comparison at 0.80.
Wikipedia parent article IDs prevent reusing existing articles or their chunks.
Near-duplicate detection is heuristic; it does not prove semantic uniqueness.
Inherited baseline records are preserved and not re-curated.

New holdouts are assigned deterministically by parent identity. New text matching
any registered evaluation 13-word span is rejected. New validation text is also
screened against training 13-word spans using a conservative Bloom filter;
false-positive rejections are possible. The overlap checks cover local evaluation
data, not every public benchmark. Never-train records remain entirely masked.

## Rebuild or verify

Use Python 3.12 with `numpy` and `tokenizers`. The existing baseline virtual
environment has those dependencies. The builder itself does not need a GPU.
The optional attention/loss helpers require PyTorch.

```powershell
& 'C:\Users\udisi\Documents\Codex\2026-08-01\now\outputs\Assignment_6_v2\.venv\Scripts\python.exe' -B .\build_corpus.py
& 'C:\Users\udisi\Documents\Codex\2026-08-01\now\outputs\Assignment_6_v2\.venv\Scripts\python.exe' -B .\build_corpus.py --verify-only
```

Downloads, resumable curation state and the overlap index are cached in the
current task's `work/corpus_20m_v1` directory. The delivered training files are
self-contained and do not require that cache. The builder checks the original
baseline path during verification; edit `BASELINE` if that original is moved.
Source revisions are locked; a changed upstream source stops acquisition rather
than silently mixing revisions.

Dataset cards and licenses are retained under `sources/`. Collection licenses
do not replace the rights or attribution attached to individual source content.
This package has not been uploaded or published.
