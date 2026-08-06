# Tiny deterministic Transformer training v1

## Outcome

The frozen 302-microbatch plan was executed end to end on the local NVIDIA
GeForce RTX 3070 Laptop GPU. The same execution path supports a deterministic
CPU fallback. Every packed payload was consumed exactly once,
origin-aware masks determined the learning positions, and two hash-chained
ledgers bind data consumption to learning events.

This is a real Transformer training execution at demonstration scale. It tests
correctness and replayability; it does not claim production model quality.

## Frozen environment

- Interpreter: the Python executable that launches `run_demo.py`
- PyTorch: `2.5.1+cu121`
- CUDA runtime bundled with PyTorch: 12.1
- Host NVIDIA driver: 546.30
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB
- Device policy: `auto` prefers CUDA; explicit `cuda` and `cpu` modes are supported
- Numerical precision: FP32
- Deterministic PyTorch algorithms: required
- TF32 and cuDNN benchmarking: disabled

`requirements-gpu.txt` specifies the direct requirements and CUDA wheel index.
`requirements-gpu.lock.txt` captures the successful environment exactly.
`requirements-cpu.txt` provides the evaluator fallback. A checkpoint records
the selected device type, device name and PyTorch version; exact resume rejects
an execution-backend change.

## Model

| Property | Value |
|---|---:|
| Architecture | decoder-only Transformer |
| Parameters | 1,509,888 |
| Vocabulary | 8,192 |
| Layers | 2 |
| Hidden size | 128 |
| Attention heads | 4 |
| Feed-forward size | 512 |
| Maximum position | 512 |
| Input/output embeddings | tied |
| Dropout | 0 |

Each block uses pre-layer normalization, multi-head causal self-attention, a
GELU feed-forward network, and residual connections.

## Attention correctness

The attention condition is:

`causal AND same packed segment AND nonpadding`

Consequently, a token may attend to earlier tokens in its own document segment
but cannot attend to a different document packed into the same sequence.
Position IDs reset at segment boundaries as established during packing.

## Objective and loss shift

The model predicts token `t` from the hidden state at `t-1`. A prediction is
included only when:

1. the origin-aware mask marks token `t` as learnable;
2. positions `t-1` and `t` belong to the same segment;
3. the preceding position is nonpadding.

This preserves the earlier policies: ordinary pretraining text learns on all
causal content; reasoning and agentic records learn only on intended outputs;
masked context remains visible but does not directly contribute loss.

## Optimizer execution

- AdamW, beta1 0.9, beta2 0.95
- Peak learning rate 0.001, minimum 0.0001
- Eight-update linear warmup followed by cosine decay
- Gradient clipping at global norm 1.0
- Four microbatches per accumulation group
- Accumulation is flushed at every curriculum-stage boundary

Microbatch gradients are accumulated as loss sums and divided by the exact
number of loss-bearing tokens before an optimizer update. Therefore, a small
microbatch cannot receive the same weight as a much denser one merely because
both are single microbatches.

The 302 microbatches produced 77 optimizer updates and exactly 262,144
loss-bearing tokens.

## Validation probe

One deterministic 128-token window was selected from each validation lane:
agentic, code, general, Indic, long context, reasoning, and science/math. The
probe contains 889 loss-bearing tokens. Its records carry `validation`
permission and never enter the optimizer or consumption ledger.

| Measurement | Before | After |
|---|---:|---:|
| Cross-entropy, nats | 9.048857 | 5.785802 |
| Perplexity | 8,508.80 | 325.64 |

The decrease confirms that the model, objective and optimizer learned a signal.
The probe is intentionally small and serves as an execution diagnostic. It is
not a robust estimate of generalization and must not be presented as benchmark
quality.

Weighted training cross-entropy across the run was 6.180160 nats. Individual
microbatch loss is expected to fluctuate because lanes and sequence genres
differ.

## Two-way ledgers

The consumption ledger has one entry per microbatch and records:

- batch and sequence identities;
- stage, physical tokens, and loss-bearing tokens;
- hashes of the actual token, mask, segment, and position tensors read;
- the corresponding learning-event ID;
- a predecessor hash.

The learning ledger records:

- the matching consumption-entry hash;
- cross-entropy and perplexity;
- accumulation state;
- optimizer update, learning rate, and pre-clip gradient norm;
- parameter hash after each optimizer update;
- its own predecessor hash.

Every learning entry also records one loss summary per packed sequence: the
sequence ID, valid shifted-loss token count, cross-entropy sum, and mean. The
verifier reconstructs each microbatch mean from those sequence summaries and
cross-links all 1,081 summaries to the matching consumption entry.

Changing, removing, inserting, or reordering a ledger entry invalidates its
chain and all following entries.

## Frozen results

- Training hash:
  `sha256:49f9f6d7dedb3fb22e97cb4540a3305a6b1abd60c2abe4bc7f5726a701806b02`
- Initial parameter hash:
  `sha256:ad2a20c43670bd17d2d846ead05cbb8e6d77983640ca6907d031379c818cbec1`
- Final parameter hash:
  `sha256:4dbf5b1d0695a8e1ffc8021544e869c044a12344766d22708031be505184c51f`
- Peak allocated GPU memory: 227,646,464 bytes (about 217.1 MiB)

## Reconstructable performance

The synchronized CUDA training interval includes batch loading, device
transfer, forward/backward computation, optimizer updates, parameter hashing,
and ledger construction. It excludes validation and final artifact writes.

| Measure | Final clean run |
|---|---:|
| Elapsed training interval | 4.937709 s |
| Physical tokens/s | 62,318.78 |
| Useful loss-bearing tokens/s | 53,090.21 |
| Packing utilization | 97.329% |
| Useful loss fraction | 85.191% |

`performance.json` retains the integer nanosecond denominator and all token
numerators. The verifier recomputes every rate. Performance timing is hashed
separately and excluded from deterministic model, ledger, checkpoint, and
training identity.

The verifier rehashes all 302 consumed payloads, verifies both chains and their
cross-links, checks exact curriculum accounting, reloads the checkpoint, and
reproduces the final parameter hash.

## Commands

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:ERA6_DEVICE = "auto"
python scripts\train_tiny_model_v1.py
python scripts\verify_training_v1.py
```

## Downstream status

The interruption-safe protocol, deliberate crash, explicit next-batch proof,
interval replay, fork, and final evidence packaging are complete. The 10–20M-
token scale-up remains reserved for Assignment 6 v2.
