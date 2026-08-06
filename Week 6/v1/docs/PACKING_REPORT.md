# Deterministic selection and packing report

## Purpose

Packing converts the frozen schedule into model-ready fixed-length sequences. It must improve utilization without allowing one packed document to attend to another or changing which tokens are permitted to produce loss.

## Deterministic selection

- Selection is without replacement across all stages.
- Eligibility is fixed by permission, lane, Indic tier, and difficulty band.
- Human-reviewed quality weights influence a deterministic hash priority; they do not alter difficulty.
- The final record in each stage/lane/tier group may be clipped after its exact final loss-bearing token so integer quotas are met exactly.
- Selected records: 521, of which 49 are final-quota clips.

| Stage | Selected records |
|---|---:|
| Seed | 56 |
| General foundation | 330 |
| Reasoning skill build | 67 |
| Long context | 58 |
| Anneal | 10 |

## Boundary-safe packing

- Packed sequences: 1,081.
- Physical packed tokens: 307,712.
- Nonpadding tokens: 299,492.
- Padding tokens: 8,220.
- Packing utilization: 97.329%.
- Scheduled loss-bearing tokens: exactly 262,144.
- Loss density: 85.191% of physical packed tokens.

Every document receives a local segment ID. Attention is represented compactly as `causal_within_segment_only`: during training, token *i* may attend to token *j* only when both have the same nonnegative segment ID and *j* is not later than *i*. This avoids storing a large square attention matrix while proving cross-document isolation.

Position IDs begin at zero for every document. When a document exceeds a sequence boundary, the next fragment begins with one duplicate of the previous token as zero-loss context. The following original token therefore retains its causal predecessor. There are 1,032 such continuation-context tokens; none contributes loss or changes scheduled accounting.

Padding uses `<pad>`, segment `-1`, position `0`, and loss `0`.

## Data-type policy registry

Packing is grouped by curriculum stage, lane, Indic tier, and sequence length,
so incompatible data types never share a policy group. Every emitted sequence
records its policy ID, boundary unit, and loss-origin rule. The seven exercised
policies are:

| Lane | Policy | Sequences |
|---|---|---:|
| General | prose dense segmented | 494 |
| Science/math | technical dense segmented | 96 |
| Code | code-file segmented | 96 |
| Reasoning | prompt/output segmented | 166 |
| Long context | long-document continuation | 81 |
| Indic | multilingual tier segmented | 97 |
| Agentic | trajectory role-masked segmented | 51 |

The common segment-isolated causal mechanism remains deliberate; the policy
registry makes the data-specific boundary and loss-origin contracts executable
and independently verifiable rather than relying on descriptive prose.

## Exact accounting

The verifier reconstructs every fixed-length slice and proves:

- selected record IDs are unique;
- binary token, mask, segment, position, sequence-index, and selection hashes match;
- sequence offsets are contiguous;
- segment IDs are not reused non-contiguously;
- each segment begins at zero-loss position zero;
- positions increase sequentially inside a segment;
- padding cannot carry attention identity or loss;
- every stage, lane, and Indic-tier loss total equals its frozen schedule target;
- the global packed loss total is exactly 262,144.

Packing hash: `sha256:2baf444b2fccdec3702af6e3c317ca9e2e7110c9781399a5261c4b87d1fa57e7`.
