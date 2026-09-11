# A tiny Transformer tells the truth about one training step

This repository is a measured, reproducible answer to the assignment. It trains a
136,960-parameter byte-level causal Transformer (2 layers, 4 heads, width 64) on a
small frozen text shard. The saved run used PyTorch 2.5.1, FP32, and an NVIDIA
GeForce RTX 3070 Laptop GPU. Both accumulation experiments start from identical
weights and see identical batches.

The headline result is simple: the numerics check out, the deliberately wrong
objective produces a visible gap, and this model is much too small to use the GPU
well.

## Reproduce

```bash
python -m venv .venv
# activate the environment, then:
pip install -r requirements.txt
python experiment.py --steps 40 --device auto
```

The experiment has a text fallback if the included corpus shard is removed. On a
CPU, use `--device cpu`; the script reports that the CUDA MFU measurement is
unavailable rather than inventing one.

## 1. Every tensor shape in the audited step

The audited micro-batch uses:

- `B=16`: independent sequences in the micro-batch
- `T=64`: byte-token positions per sequence
- `C=64`: model channels
- `H=4`: attention heads
- `D=16`: channels per head (`C/H`)
- `M=256`: expanded MLP channels (`4C`)
- `V=256`: possible UTF-8 byte values

The run printed **195 tensor entries**. This includes every named forward tensor in
our explicit attention/MLP implementation, scalar loss tensors, every parameter,
every gradient, and every tensor-valued AdamW state after the update. Each line
contains the literal shape, dtype, and the meaning of every dimension. The complete
printout is in [`artifacts/shape_ledger.txt`](artifacts/shape_ledger.txt) and is also
printed in full in the notebook.

A representative slice:

| Tensor | Shape | Dimensions mean |
|---|---:|---|
| `input_ids` | `(16, 64)` | batch, sequence |
| `token_embedding` | `(16, 64, 64)` | batch, sequence, model channel |
| `block_0.q` | `(16, 4, 64, 16)` | batch, attention head, query position, head channel |
| `block_0.attention_scores` | `(16, 4, 64, 64)` | batch, head, query position, key position |
| `block_0.mlp_up` | `(16, 64, 256)` | batch, sequence, expanded MLP channel |
| `logits` | `(16, 64, 256)` | batch, sequence, byte vocabulary |
| `per_token_cross_entropy` | `(1024,)` | batch-times-sequence |
| `mean_loss` | `()` | scalar |
| `gradient.lm_head.weight` | `(256, 64)` | output byte vocabulary, input model channel |
| `optimizer.exp_avg.lm_head.weight` | `(256, 64)` | output byte vocabulary, input model channel |

The shape trace is explicit code, not a profiler guess: see `record(...)`,
`Block.forward(...)`, and `add_backward_and_optimizer_shapes(...)` in
[`experiment.py`](experiment.py).

## 2. One gradient checked by hand

I checked `lm_head.weight[105, 0]` in float64 with a central difference and
`epsilon = 1e-5`:

\[
\frac{\partial L}{\partial w}\approx
\frac{L(w+\epsilon)-L(w-\epsilon)}{2\epsilon}
\]

Measured values:

| Quantity | Value |
|---|---:|
| `L(w + epsilon)` | 5.659547824005586 |
| `L(w - epsilon)` | 5.659547552736764 |
| finite difference | 0.01356344112579677 |
| `backward()` | 0.013563441054856254 |
| absolute error | 7.094051561462589e-11 |
| relative error | 5.230274158060207e-9 |

They agree through roughly ten decimal places. Central difference is used instead
of a one-sided nudge because its truncation error is quadratic in `epsilon`.

## 3. Breaking accumulation with an average of averages

There are two micro-batches per optimizer step: `(B=16, T=8)` and `(B=16, T=64)`.
The correct loss is

\[
L_{correct}=\frac{\sum L_{short}+\sum L_{long}}
{N_{short}+N_{long}}.
\]

The deliberately broken loss is

\[
L_{broken}=\tfrac12\operatorname{mean}(L_{short})+
\tfrac12\operatorname{mean}(L_{long}).
\]

The short source owns only `128 / 1152 = 11.11%` of the tokens, but the broken
objective gives it 50% of the gradient. Because the two lengths sample different
regions of the corpus, that is not just a rescaling: it changes the objective.
Evaluation uses the correct token weighting.

![Correct token weighting and broken mean-of-means curves](artifacts/accumulation_curves.svg)

After 40 steps, correct evaluation loss is **2.94407** and broken evaluation loss
is **2.98044**: an absolute gap of **0.03637**, or **1.235%** relative to the correct
run. Raw values for every step are in
[`artifacts/training_log.csv`](artifacts/training_log.csv).

## 4. Gradient norm at every step

The global L2 norm is computed after both micro-batches have accumulated and before
`optimizer.step()`. Every value is in the CSV above and plotted here:

![Gradient norm at every optimizer step](artifacts/grad_norms.svg)

One “moved before the loss did” example is step 37 to 38:

- evaluation loss: `2.963501 -> 2.958693` (both display as **2.96** at the plot's
  two-decimal reading precision; relative change 0.162%)
- gradient norm: `0.367548 -> 0.309012` (**15.926%** change)

This is deliberately phrased as a resolution-dependent diagnostic, not as a claim
that the loss was mathematically constant. The gradient norm exposed a large change
roughly 98 times larger in relative terms while the loss curve still looked flat at
that precision.

## 5. My MFU calculation

The benchmark uses a fixed `B=32, T=64` batch, 20 warm-up steps, then 60 individually
synchronized steps. TF32 is disabled, so I compare against an FP32 CUDA-core peak.

For one forward token, counting matrix-multiplication FLOPs only:

\[
F_{fwd/token}=L(24d^2+4Td)+2dV=262{,}144.
\]

I approximate backward as twice forward, so training is `3 x forward = 786,432`
FLOPs/token. The measured median step is 8.269 ms for 2,048 tokens, or 247,666
tokens/s, yielding **0.19477 TFLOP/s**.

The device reports 40 SMs and a 2.1 GHz maximum SM clock. For compute capability
8.6 I use 128 FP32 lanes/SM, giving this upper bound:

\[
F_{peak}=40\times128\times2\times2.1\text{ GHz}=21.504\text{ TFLOP/s}.
\]

Therefore:

\[
MFU=\frac{0.19477}{21.504}=\mathbf{0.906\%}.
\]

This is honest but approximate. The numerator excludes layer norm, softmax, GELU,
embedding lookup, AdamW, and memory traffic even though they consume wall time.
The denominator is a driver-clock upper bound, not a guaranteed sustained clock.

Why it is nowhere near 40%: the model's `64 x 64`-scale matrix multiplies are tiny;
Python launches many separate kernels; attention explicitly materializes small
`T x T` matrices; batch and sequence length are too small to fill the GPU; and this
audit-friendly FP32 implementation does not use fused attention, fused optimizer
kernels, compilation, mixed precision, or Tensor Cores. At 40% this denominator
would require 8.6016 TFLOP/s—about 44 times the achieved rate. The primary cost is
under-utilization and launch/memory overhead, not a shortage of arithmetic.

## 6. What decimal 0.1 looks like

Binary `0.1` repeats: `0.0001100110011..._2 = 1.100110011... x 2^-4`. Rounding that
repeating fraction gives:

| Format | sign · exponent · fraction | Hex | Stored decimal |
|---|---|---:|---:|
| fp32 | `0 01111011 10011001100110011001101` | `0x3DCCCCCD` | 0.10000000149011612 |
| bf16 | `0 01111011 1001101` | `0x3DCD` | 0.10009765625 |
| fp8 E4M3 | `0 0011 101` | `0x1D` | 0.1015625 |

For fp32 and bf16 the stored exponent is `-4 + 127 = 123 = 01111011`. For E4M3
the bias is 7, so it is `-4 + 7 = 3 = 0011`; the ideal fraction `0.6 x 8 = 4.8`
rounds to `5 = 101`, giving `(1 + 5/8) x 2^-4 = 0.1015625`. PyTorch's
`float8_e4m3fn` conversion independently returns the same value.

I would train this model in **bf16 for weights, activations, and gradients, with
FP32 optimizer states and sensitive reductions** on hardware that accelerates it.
BF16 keeps fp32's 8-bit exponent range, halves storage/bandwidth, and is much more
forgiving than FP8. I would not use raw E4M3 here: three fraction bits make 0.1 about
1.56% high and the narrow range needs a deliberate scaling recipe. This particular
audit run stays in FP32 so the gradient check and FP32 MFU denominator are easy to
interpret.

## Repository map

- [`small_model_truth.ipynb`](small_model_truth.ipynb) — executed review notebook
- [`experiment.py`](experiment.py) — model, training loops, checks, MFU, and plots
- [`verify_submission.py`](verify_submission.py) — fast consistency checks for saved artifacts
- [`artifacts/results.json`](artifacts/results.json) — machine-readable summary
- [`artifacts/shape_ledger.txt`](artifacts/shape_ledger.txt) — all 195 shape lines
- [`artifacts/training_log.csv`](artifacts/training_log.csv) — loss and grad norm at every step
- [`data/README.md`](data/README.md) — data provenance and fallback behavior

No claim here depends on a screenshot: the raw measurements, formulas, and code that
produced them are all in the repository.
