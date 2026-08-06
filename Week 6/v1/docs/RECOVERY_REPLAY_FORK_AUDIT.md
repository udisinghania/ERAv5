# Checkpoint, crash, resume, replay, and fork audit v1

## Outcome

The training system now survives a real process termination and resumes exactly
from a serialized checkpoint. A separate clean replay reproduces the original
run, while a declared hyperparameter fork preserves data consumption and
diverges only when its first different optimizer update is applied.

The independent audit status is **PASS**.

## Checkpoint contract

During the recovery run, a checkpoint is written every eight optimizer
updates. A checkpoint contains:

- all model parameters and buffers;
- complete AdamW first/second moments and step counters;
- CPU and CUDA random-number-generator states;
- selected execution backend (CPU or CUDA), device name and PyTorch version;
- the next microbatch cursor;
- completed optimizer-update count;
- accumulated training-loss accounting;
- initial parameter and validation state;
- consumption and learning ledger tail hashes;
- batch-plan and training-configuration identity;
- an exact proof for the next batch: ID, sequence IDs, packed-token spans,
  four tensor hashes, and a canonical proof hash;
- branch lineage.

Checkpoints are created only after an optimizer update, when gradient
accumulation has returned to zero. This avoids an ambiguous partially
accumulated resume state. Append-only plain JSONL ledgers are flushed and
synced before the checkpoint is written; deterministic gzip versions are
materialized after a run completes.

The recovery worker must use the same recorded backend as the checkpoint.
This preserves exact within-backend replay while still allowing the complete
demonstration to run on CPU when CUDA is unavailable.

## Deliberate crash

The first recovery process was instructed to terminate after optimizer update
32. At that point it had completed 125 microbatches. It wrote:

- `checkpoint_000032.pt`;
- `crash_event.json`;
- ledger prefixes ending at the checkpoint cursor.

The process then exited with planned code **86**. The orchestrator observed
that exact nonzero exit code and treated any other result as a failure.

Crash-checkpoint SHA-256:

`c6056499c144b8820281165cbf73606c92e0c528f33550c530b27cfd1e20495c`

Parameter hash at the crash:

`sha256:255f634412788441260defb3ac7acde009359027260d280c1a79bb1147f96b6e`

## Exact resume

A new Python process loaded the crash checkpoint, restored model, optimizer,
RNG states, cursor, counters, lineage and ledger tails, and continued at
microbatch 125. The completed resumed run matched the original frozen run in:

- final parameter hash;
- every consumption-ledger row and compressed ledger hash;
- every learning-ledger row and compressed ledger hash;
- final validation loss;
- 302-microbatch, 77-update, and 262,144-loss-token totals.

Before executing microbatch 125, the new process reconstructed its proof from
the frozen packing artifact. Batch ID 125, sequence IDs 501/96/307/122, their
four global packed-token spans, input/loss/segment/position tensor hashes, and
proof hash all matched the checkpoint. A mismatch in any field aborts before
training.

Next-batch proof hash:

`sha256:03c9f87e311619af8233886688b2b10d3336cacb80d2d9261af77edf85a5b391`

Final parameter hash for baseline and recovery:

`sha256:4dbf5b1d0695a8e1ffc8021544e869c044a12344766d22708031be505184c51f`

This proves that the interruption did not skip, repeat, or alter a batch and
did not lose optimizer state.

## Exact clean replay

A third process started from the pinned seed and repeated all 302
microbatches. It independently reproduced the same final parameter hash,
consumption ledger, learning ledger, and validation result as both baseline
and recovery.

This is stronger than comparing final loss alone: exact ledgers prove that
every learning event and parameter hash at every optimizer update matched.

The audit additionally materializes an explicit 28-batch historical interval,
batch indices 112–139. It crosses the crash boundary at 125 and compares every
original/replay batch ID, sequence ID, token span, tensor payload hash, and
reconstruction proof hash.

## Declared fork

A fourth process loaded the update-32 checkpoint and continued with a declared
learning-rate scale of 0.5. All other state and all data remained unchanged.

The fork consumed the exact same 302 batches, so its consumption ledger is
identical to the baseline. Microbatches 125–127 only accumulated gradients;
therefore their learning entries also remained identical. Divergence occurred
at microbatch index 128, exactly when optimizer update 33 applied the halved
learning rate:

| Field | Baseline | Fork |
|---|---:|---:|
| Update 33 learning rate | 0.0007386294 | 0.0003693147 |
| Final validation cross-entropy | 5.785802 | 5.893988 |

Fork final parameter hash:

`sha256:f5f6b895e311664b9240570fc40ad5df15f7dc8bf4c6fc97607cbf515cc836e7`

The different learning ledger and parameter hash are expected and required.
The fork records the parent checkpoint hash, fork cursor, parent parameter
hash, optimizer update and learning-rate scale.

## Independent verifier

`scripts/verify_recovery_v1.py` does not trust the orchestrator's PASS field.
It independently:

1. validates the audit hash and all bound report hashes;
2. reloads the crash checkpoint and verifies its cursor and parameter hash;
3. reconstructs and validates the checkpoint's exact next-batch proof;
4. verifies every ledger hash chain and tail;
5. compares baseline, recovery and replay ledgers row for row;
6. reconstructs all 28 declared replay-interval batch/span/payload proofs;
7. confirms fork consumption is unchanged;
8. finds the exact first fork divergence and requires it to be an optimizer
   update;
9. reloads recovery, replay and fork final checkpoints and reproduces their
   reported parameter hashes.

Audit hash:

`sha256:cf5006265cbb6ce9fbf63f0192b25e6591769b667b5e9241252d2629f0938e65`

## Commands

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:ERA6_DEVICE = "auto"
python scripts\run_recovery_demo_v1.py
python scripts\verify_recovery_v1.py
```

## Submission status

The required evidence bundle, run log, copied manifests, ledgers, checkpoints,
performance report, and one-command execution are complete. The 10–20M-token
expansion remains a v2 experiment.
