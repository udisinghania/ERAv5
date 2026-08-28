# Experiment history — supporting evidence, not Part 3

This appendix records how the submitted solution was reached. It does not add
an assignment part: the graded submission remains Part 1 (the observable loss
harness) and Part 2 (one `t+2` head). Session 9 was treated as architecture
reference material, while the assignment statement remained the requirement
source.

## What the Session-9-aligned rewrite changed

The residual width, layer count, attention-head count, vocabulary, tied `t+1`
head, independent `t+2` head, data, validation probe, and seed stayed fixed.
V1 used the implementation defaults—pre-norm LayerNorm and a two-matrix GELU
FFN of width 1,536. V2 uses pre-norm RMSNorm and a three-matrix, bias-free
SwiGLU FFN of width 1,024. The smaller SwiGLU width compensates for its third
matrix, leaving the complete two-head model almost the same size.

The Session 9 material also discusses `t+3` and `t+4`, but they are deliberately
not in this submission: the assignment asks for one extra head only.

## First recorded smoke tests

| Check | Initial H1 | Initial H2 | Final H1 | Final H2 | Meaning |
|---|---:|---:|---:|---:|---|
| Notebook Part 2, 80 updates | 9.040891 | 9.086224 | 4.244803 | 4.448154 | Both shifted objectives were finite and learnable. |
| First retained end-to-end trainer smoke, 4 steps | 9.226139 | 9.083124 | 8.022191 | 8.087437 | Data, optimization, validation, and checkpoint paths executed end to end. |

The later `smoke-model` command is intentionally non-persistent, so this record
does not invent a numeric result for it.

## Failure that motivated the controls

The first 97,995,264-parameter run used a very small held-out probe (802 H1 and
797 H2 targets), repeated the training corpus 29.9136 times,
and ran until the six-hour clock expired. Its best observed validation sum was
6.958633 at step 7,400 (7.9751
passes), but the final sum was 8.675194 at step
27,759: a 24.67%
increase. At that final step the training sum was still only
3.911201. Falling training loss beside
rising validation loss is the direct evidence that the run overtrained.

That run also exposed the checkpointing flaw: `latest.pt` was overwritten and
there was no preserved best or numbered history. The replacement controls are:
a 100k-target frozen validation probe, held-out early stopping, a separate
`best.pt`, a resumable `latest.pt`, a hard corpus-pass ceiling, and numbered
checkpoints.

## V1 versus submitted V2

| Measurement | V1: LayerNorm + GELU | V2: RMSNorm + SwiGLU | Observed change |
|---|---:|---:|---:|
| Parameters | 17,126,400 | 17,109,888 | -0.096% |
| Best H1 loss | 2.720188 | 2.595900 | -4.57% |
| Best H2 loss | 3.445438 | 3.372842 | -2.11% |
| Best sum | 6.165626 | 5.968742 | -3.19% |
| Corpus passes at stop | 6.0000 | 8.0749 | — |
| Stop reason | `maximum_corpus_passes_reached` | `validation_early_stopping` | — |
| Checkpoint history | best/latest only | best/latest + 16 numbered | — |

Negative percentages mean the submitted V2 value is lower. V2's best held-out
sum was 3.19% lower while
using 0.096% fewer
parameters.

This is a matched-data development comparison, not a clean architecture-only
ablation. V2 also used a longer schedule and reached 8.0749
corpus passes versus V1's 6.0000; therefore the improvement
cannot be attributed solely to RMSNorm and SwiGLU.

## Evidence provenance

The exact values above are machine-readable in `experiment_history.json`.
It also records SHA-256 hashes for the original smoke, V0, V1, and V2 configs,
summaries, and metric ledgers. The full V2 ledger and checkpoint index are
included in the submission; bulky raw exploratory ledgers remain local.

The compact original evidence retained in Git is:

- [First end-to-end trainer smoke summary](experiments/first_trainer_smoke/run_summary.json)
- [V0 98M resolved configuration](experiments/v0_98m_overtrained/resolved_config.json)
- [V0 98M final run summary](experiments/v0_98m_overtrained/run_summary.json)
- [V1 17M resolved configuration](experiments/v1_17m_baseline/resolved_config.json)
- [V1 17M final run summary](experiments/v1_17m_baseline/run_summary.json)
- [V2 full validation/training ledger](outputs/mtp_17m_session_run/metrics.jsonl)
