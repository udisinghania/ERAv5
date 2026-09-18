# Assignment 12: Data Parallelism and ZeRO on 32 Virtual GPUs

This project simulates distributed training with 32 CPU worker threads acting as virtual GPU ranks. I use a small linear regression model and compare ordinary data parallelism with ZeRO stages 1, 2, and 3.

The purpose is to show what each rank owns, how much persistent memory it uses, how optimizer computation changes, which collective operations are required, and why every stage still produces the same mathematical update.

## Files

- `assignment_12_zero_simulation.ipynb` — complete executable experiment and explanation.
- `requirements.txt` — minimal environment requirements.

## How to run

### Google Colab

1. Open Google Colab.
2. Upload `assignment_12_zero_simulation.ipynb`.
3. Select **Runtime → Run all**.

Colab already provides NumPy. A GPU runtime is not required because the notebook intentionally uses CPU threads as virtual ranks.

### Local Jupyter

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook assignment_12_zero_simulation.ipynb
```

## What I built

- 32 simultaneous CPU worker threads, one for each virtual rank.
- A deterministic 1,024-parameter linear model.
- A global batch split into 32 different microbatches.
- Separate forward and backward worker phases.
- A normal data-parallel implementation.
- ZeRO-1 with sharded FP32 master parameters and Adam states.
- ZeRO-2 with sharded optimizer state and gradients.
- ZeRO-3 with sharded parameters, gradients, and optimizer state.
- Explicit ZeRO-3 parameter all-gathers before forward and backward.
- Four Adam training steps with optimizer state carried between steps.
- FP32 correctness checks for master parameters and both Adam moments.
- Measured toy-model state memory and projected 30B-model memory.
- Exact 32-rank ring communication estimates and the rounded Session 12 values.

The controller emulates collectives, while each rank owns real NumPy arrays and performs model or optimizer work on a worker thread. This is a conceptual simulator, not a GPU performance benchmark.

## Memory assumptions

The Session 12 accounting uses 16 bytes per parameter:

| State | Format | Bytes per parameter |
|---|---:|---:|
| Compute parameter | FP16 | 2 |
| Gradient | FP16 | 2 |
| Master parameter | FP32 | 4 |
| Adam first moment | FP32 | 4 |
| Adam second moment | FP32 | 4 |
| **Total** |  | **16** |

With world size `N`, persistent bytes per parameter per rank are:

| Strategy | Persistent bytes per parameter per rank |
|---|---:|
| Data parallel | `16` |
| ZeRO-1 | `4 + 12/N` |
| ZeRO-2 | `2 + 14/N` |
| ZeRO-3 | `16/N` |

## My understanding of the stages

### Data parallelism

Every rank stores the entire training state. Each rank processes a different microbatch and produces a full local gradient. An all-reduce averages those gradients and returns the same result to every rank. Because all replicas start with the same parameters and apply the same update, they remain synchronized.

This is the easiest strategy to reason about, but it wastes memory by replicating all 16 bytes per parameter on every rank. It also repeats the full optimizer update on every rank.

### ZeRO-1

ZeRO-1 shards the 12 optimizer bytes: the FP32 master parameter and the two FP32 Adam moments. Parameters remain replicated and each rank still allocates a full-sized gradient buffer. Each rank updates only the parameter shard for which it owns optimizer state. The updated parameter shards are then all-gathered so that every rank again has a complete model.

The important result for me is that this first stage removes most of the redundant memory because optimizer state is 12 of the original 16 bytes.

### ZeRO-2

ZeRO-2 keeps optimizer sharding and also retains only each rank's gradient shard. The simulator uses reduce-scatter to average gradients while delivering rank `r` only the slice it owns. Parameters remain replicated, so forward and backward can still use a complete local model.

ZeRO-2 reduces memory further without increasing the simplified communication beyond roughly `2P` per step.

### ZeRO-3

ZeRO-3 shards every persistent state, including the FP16 parameters. A rank cannot perform the full model computation from its persistent state alone, so parameter shards must be all-gathered when they are needed.

The notebook explicitly gathers parameters before the forward pass, releases the temporary full copy, gathers again before backward, and releases it again. Gradients are reduce-scattered and each rank updates only its owned shard.

This provides the lowest persistent memory, but the extra parameter materialization raises communication from about `2P` to about `3P` per step. It also creates a transient memory peak equal to the gathered layer or bucket.

## Measured toy-model memory

The notebook creates actual per-rank arrays and sums their `nbytes` values:

| Strategy | Parameter bytes | Gradient bytes | Optimizer bytes | Persistent bytes per rank |
|---|---:|---:|---:|---:|
| Data parallel | 2,048 | 2,048 | 12,288 | 16,384 |
| ZeRO-1 | 2,048 | 2,048 | 384 | 4,480 |
| ZeRO-2 | 2,048 | 64 | 384 | 2,496 |
| ZeRO-3 | 64 | 64 | 384 | 512 |

Every strategy produces a temporary 2,048-byte full FP16 gradient before its collective. ZeRO-2 and ZeRO-3 can release non-owned gradient data after reduce-scatter; real implementations do this bucket by bucket during backward. ZeRO-3 additionally materializes a temporary 2,048-byte full FP16 parameter vector during forward and backward. These temporary buffers are not counted as persistent state.

## Projected memory for the 30B model

At world size 32:

| Strategy | Persistent state per rank | Fits in 74.5 GiB? |
|---|---:|---:|
| Data parallel | 447.0 GiB | No |
| ZeRO-1 | 122.2 GiB | No |
| ZeRO-2 | 68.1 GiB | Yes, before activations and overhead |
| ZeRO-3 | 14.0 GiB | Yes, before activations and overhead |

These are lower bounds for training state. Real peak memory also includes activations, communication buffers, the largest gathered layer, allocator fragmentation, framework overhead, and possibly uneven layer sizes.

## Computation changes

Forward and backward model work per rank remains approximately the same in all four strategies because every rank still processes its local microbatch.

The optimizer work changes:

| Strategy | Optimizer elements updated per rank | Optimizer elements updated globally |
|---|---:|---:|
| Data parallel | 1,024 | 32,768 |
| ZeRO-1 | 32 | 1,024 |
| ZeRO-2 | 32 | 1,024 |
| ZeRO-3 | 32 | 1,024 |

Data parallelism repeats the same full optimizer update 32 times. Every ZeRO stage assigns one shard to each rank, so optimizer work is `1/32` per rank and one complete model update globally.

## Communication changes

Let `P` be one complete FP16 parameter copy. With 32 ranks, the exact ring factor is `31/32`:

| Strategy | Collectives per step | Exact per-rank estimate | Session approximation |
|---|---|---:|---:|
| Data parallel | Gradient all-reduce | `1.9375P` | `2P` |
| ZeRO-1 | Gradient reduce-scatter + parameter all-gather | `1.9375P` | `2P` |
| ZeRO-2 | Gradient reduce-scatter + parameter all-gather | `1.9375P` | `2P` |
| ZeRO-3 | Two parameter all-gathers + gradient reduce-scatter | `2.90625P` | `3P` |

For the 30B example, `P = 60 GB`, so the rounded lesson estimates are 120 GB per rank per step for data parallelism, ZeRO-1, and ZeRO-2, and 180 GB for ZeRO-3.

## Correctness result

The experiment runs four Adam steps. It reconstructs the full FP32 master parameters and both Adam moment vectors from the sharded states and compares them with the data-parallel reference.

All stages produce zero maximum difference in the included run. The FP16 compute parameters are also identical, and the evaluation loss drops from `65.242112` to `62.032010`.

This shows that ZeRO changes storage and communication, not the intended optimization result.

## What I learned

My main takeaway is that ZeRO is a progression of ownership decisions. It is not a different optimizer or learning rule.

The result that stood out to me was how much memory ZeRO-1 removes immediately. Sharding the optimizer state attacks 12 of the 16 bytes stored per parameter. ZeRO-2 then removes non-owned gradient storage while keeping the same approximate communication volume.

ZeRO-3 is the strongest option when model state does not otherwise fit, but it is not automatically the fastest. It saves persistent memory by requiring parameter data to move during computation. Whether that trade is worthwhile depends on the model's activation memory, layer sizes, interconnect, bucket configuration, and how much communication overlaps with computation.

For the Session 12 30B example, I would prefer ZeRO-2 if 32 GPUs provide enough real headroom after measuring activations. I would choose ZeRO-3 if memory remains the binding constraint or if fewer GPUs must hold the model. I would make the final decision from measured step time and peak memory rather than from the state table alone.

## Limitations

- CPU thread timing does not predict CUDA, NVLink, or InfiniBand performance.
- Collectives are emulated by a central controller.
- The model has one flat parameter vector rather than many uneven transformer layers.
- Activation memory and framework overhead are explained but not fully allocated.
- Real ZeRO-3 gathers layer-sized or bucket-sized groups, not necessarily the entire model.

## Submission check

I can explain:

1. Why the four strategies produce the same update.
2. What additional state is sharded at each ZeRO stage.
3. Why ZeRO-1 and ZeRO-2 remain near `2P` communication.
4. Why ZeRO-3 reduces persistent memory but increases communication.
5. Which stage I would select for a stated memory and network constraint.
