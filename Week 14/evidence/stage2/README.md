# Session 14: extended learning and routing experiments

This is a separate extension of `Session_14_MoE`. The first experiment and its
checkpoints remain unchanged. Start with `REPORT.md` for measured findings once
the runs and diagnostic evaluations complete.

## Questions

1. Does more training continue to reduce held-out loss?
2. How does the MoE compare with the dense control at the same token exposure?
3. Does this trained MoE depend on learned expert selection or output weighting?

The fixed design is recorded in `design.json` before training. Both arms extend
their matching 10M-token checkpoints and AdamW states for another 40M supervised
tokens. This completes one full extra pass over the original 50M-token corpus:
50M original dense pretraining + 50M continuation = 100M token exposures per arm.
The second pass reuses existing training data; no new data were acquired.

## Training

- Architecture is unchanged: 20.17M-parameter dense control; 54.69M total / 31.68M
  active MoE, with 4 copied GELU experts per layer and top-2 routing in 9 layers.
- Matching optimizer states are preserved from the first continuation. Both
  models restart the same learning-rate schedule: 50-update warmup to 0.0002,
  cosine decay to 0.00002. This is a common schedule, not separate tuning per arm.
- Effective batch 32, microbatch 16, BF16 matmuls, FP32 router/weights/optimizer,
  gradient clipping 1.0, auxiliary balance coefficient 0.001 for the MoE.
- The saved seeded sequence permutation continues where the first run stopped.
  Its last sequence had 74 unused supervised targets. A complementary mask
  retains those targets and excludes the 437 already consumed ones.
- Across the two stages, every supervised target appears exactly once in the
  continuation pass. Both model types receive the identical targets in order.
- Held-out validation is unchanged: 5,049,456 supervised targets, excluded from
  updates. Intermediate probes use the same fixed fragments as the first run.
- Full held-out evaluations are taken near 10M/20M/30M extra tokens and exactly
  40M extra tokens. Intermediate points fall at optimizer-step boundaries;
  the JSON records their actual token counts.

## Router diagnostics

All diagnostics keep the trained weights fixed and execute two experts per token.
They are inference perturbations, not separately trained competing architectures.

| Policy | Expert selection | Output weighting |
|---|---|---|
| Learned | Original learned top-2 | Normalized learned scores |
| Uniform selected | Original learned top-2 | 1/2 each |
| Random, seeds 17/29/43 | Two distinct experts uniformly sampled per token | 1/2 each |

Every policy uses the complete validation set. Random routing is causal and uses
independent per-layer generators; it does not shuffle hidden states or scores
between tokens. Three routing seeds measure inference randomness, not training
seed variability. `diagnostics/learned.json` also records per-source expert use.
All fixed-prompt greedy outputs are retained in `diagnostics/samples.json`.

## Files

- `extend.py`: prepares the remainder of the second pass, loads model/optimizer
  state, and runs each 40M extension with resume support.
- `base_experiment.py`: copy of the first experiment's utilities.
- `dense_model.py`, `moe_model.py`, `packed_dataset.py`: unchanged architecture
  and data reader copies from the first experiment.
- `test_extension.py`: validates exhaustive, disjoint target accounting across
  the 10M/40M boundary and loads a real GPU batch.
- `test_diagnostics.py`: checks identical-copy equivalence, top-2 accounting,
  causality and restoration of the original routing policy.
- `diagnostics.py`: full-set ablations, per-source routing and greedy samples.
- `verify_extended.py`: audit of source integrity, final models and results.
- `make_report.py`: static charts and measured report.
- `moe/` and `dense/`: independent logs, configurations, evaluation files and
  last/best/final checkpoints. Best is selected by the fixed validation probe.

## Run locally

Use the previously verified environment; no package installation is required.
From this folder in PowerShell:

```powershell
$python = 'C:\Users\udisi\Documents\Codex\2026-08-01\now\outputs\Assignment_6_v2\.venv\Scripts\python.exe'
& $python -B .\extend.py prepare
& $python -B .\test_extension.py
& $python -B .\test_diagnostics.py
& $python -B .\extend.py moe
& $python -B .\extend.py dense
& $python -B .\diagnostics.py
& $python -B .\verify_extended.py
& 'C:\Users\udisi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\make_report.py
```

Completed training runs refuse to overwrite results. An interrupted run can use
`extend.py moe --resume` or `extend.py dense --resume`. The last committed model,
optimizer, cursor, counts and milestone history are restored. There are no
stochastic layers during training; the sample order is saved. Code/data identity
checks reject an incompatible resume. GPU operations are not promised bitwise
deterministic across devices or software versions.

For a fresh rerun, use a separate copy without the generated `moe/` and `dense/`
directories, keeping the sibling first experiment and original corpus available.
Source paths are near the top of `base_experiment.py` and `extend.py`.

## Recorded interruption and recovery

The MoE process stopped during the extension. Its last committed checkpoint was
update 1,500, after 22,902,306 extension targets. The checkpoint signature, code
hashes, finite weights, and token count implied by its data cursor were checked
before resuming. Progress logs for later, uncommitted updates were archived under
`moe/recovery_1/` and those updates were replayed from the committed optimizer
state. The archived `record.json` records the checkpoint hash and recovery point.
The reported run time excludes downtime and discarded uncommitted work.

## Interpretation limits

This is one trained run per architecture, using a validation set already examined
during development. The paired comparison uses equal target exposure, not equal
FLOPs, parameters or elapsed time. Lower held-out loss is evidence of continued
learning, not a guarantee of fluent text or a general architectural advantage.
Source-dependent expert usage alone does not prove semantic specialization.
