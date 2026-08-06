# Deterministic microbatches and auditable OPUS v1

## Outcome

The frozen packing artifact is now converted into a complete, deterministic
microbatch order. Every packed sequence is consumed exactly once, curriculum
stages never mix, the scheduled lane and Indic-tier totals remain exact, and
the full OPUS candidate/selection history is recorded for replay.

This stage changes **order only**. It cannot add data, remove scheduled loss
tokens, reuse a sequence, or change the frozen curriculum mixture.

## Microbatch shape

The physical microbatch budget is 1,024 tokens:

| Curriculum context | Sequence length | Sequences per full microbatch |
|---|---:|---:|
| seed, general foundation, reasoning build | 256 | 4 |
| long context, anneal | 512 | 2 |

Microbatches are homogeneous in sequence length and never cross a curriculum
stage boundary. The final microbatch of a stage may be partially filled.

## What auditable OPUS v1 does

At each decision, the controller deterministically proposes up to four
candidate microbatches from the unused sequences in the current stage. Each
candidate is evaluated using only frozen data metadata:

1. cumulative lane-mixture deviation from the stage target;
2. cumulative Indic-tier deviation from the stage target;
3. loss-bearing-token density;
4. packing utilization;
5. source diversity.

Lane and Indic-tier tracking dominate the raw proxy score. Density, utilization,
and source diversity break otherwise similar choices. Candidate generation uses
a pinned seed and SHA-256 ranks, so no ambient random-number state is involved.

The controller then records an explicit outcome for every proposal:

1. **Rejected:** a proposal below the frozen 300,000-parts-per-million useful
   loss-density gate is rejected. Rejection applies only to that proposed
   combination; its sequences remain available.
2. **Deferred:** a valid proposal that is not selected is deferred. Its
   sequences remain in the pool and may appear in later proposals.
3. **Accepted:** exactly one proposal becomes the executable microbatch.

Before accepting the raw proxy winner, the controller projects Indic and
agentic progress at the candidate's cumulative loss-token pace. If another
eligible proposal has a smaller protected-lane deficit, it overrides the raw
winner. The decision ledger stores the normal winner, accepted candidate,
per-lane required/observed/deficit values, reasons, and whether an override was
used. A deterministic sparse-final-batch fallback exists so scheduled data can
never be stranded, although this frozen run did not need it.

This remains a **metadata-proxy OPUS demonstration**. It is not a
gradient-norm, influence-function, or model-loss selector, because those
variants require model state and add substantial runtime. A model-aware OPUS
comparison is suitable for Assignment 6 v2.

## Zero-loss context handling

Packing contains 35 agentic sequences whose labels are entirely masked. A
microbatch containing only those sequences would have no training signal. The
batcher therefore spreads them across microbatches that also contain at least
one loss-bearing sequence. They remain in the audit trail and physical-token
accounting, but no standalone zero-loss microbatch is produced.

## Frozen result

| Measure | Result |
|---|---:|
| Packed sequences consumed | 1,081 / 1,081 |
| Unique sequence consumption | 1,081 |
| Microbatches / OPUS decisions | 302 / 302 |
| Physical tokens | 307,712 |
| Nonpadding tokens | 299,492 |
| Loss-bearing tokens | 262,144 |
| Zero-loss sequences paired | 35 |
| Standalone zero-loss microbatches | 0 |

Stage counts are 17 seed, 160 general-foundation, 64 reasoning-skill-build,
52 long-context, and 9 anneal microbatches.

The proposal audit contains 302 accepted, 880 deferred, and 6 rejected
candidate outcomes. The protected-floor controller overrode the normal proxy
winner 72 times. It did not merely accept the first proposal: 222 of 302
decisions selected candidate 1, 2, or 3. All four alternatives were distinct in
292 decisions; candidate sets naturally collapse near the end of each stage
when too few unused sequences remain. Every stage finishes at zero lane-mixture
and zero Indic-tier accounting error because the frozen totals are invariant.

The frozen batch-plan hash is:

`sha256:158ae9939c5502b256449cce5aaff4c026775a980c7b8241ede10cb11faab749`

## Artifacts and verification

- `configs/batching_v1.json` freezes shape, candidate count, proposal gate,
  protected-floor policy, proxy weights, seed, tie-break, and zero-loss policy.
- `data/batches_v1/batches.jsonl.gz` is the executable microbatch order.
- `data/batches_v1/opus_decisions.jsonl.gz` records every candidate, metric,
  score, and selected candidate.
- `data/batches_v1/batch_report.json` binds the batch plan to the schedule and
  packing hashes.
- `scripts/verify_batches_v1.py` independently reconstructs all 302 decisions
  and requires byte-equivalent logical records.

The verifier also requires real accepted/rejected/deferred populations and at
least one protected-floor override. It proves every rejection reason against
the configured threshold, every override reduces projected deficit, every
deferred proposal passed the gate, and every accepted proposal matches the
emitted batch. Exact single-use coverage, monotonic curriculum stage order,
homogeneous sequence length, exact lane/tier accounting, and the absence of
zero-loss microbatches remain enforced.

## Downstream regeneration boundary

Changing the OPUS policy changed the executable batch-plan hash. The tiny
decoder training, recovery, replay, and fork artifacts must therefore be
regenerated before they can be submitted. They will consume this fixed batch
order and write two independent ledgers:

- a **consumption ledger** stating exactly which data was read;
- a **learning ledger** recording loss, optimizer state, learning rate, and
  parameter/checkpoint hashes.

No prior training or recovery hash is valid evidence for this new plan.
