# Understanding every number in this experiment

This is the slow-reading companion to the submission. Start here if the code runs
but the output still feels like a wall of numbers.

## 1. What the model is trying to do

The corpus is converted to UTF-8 bytes. Given previous bytes, the model predicts the
next byte. There are 256 possible byte values, so the final layer produces 256 scores
for every position. Cross-entropy asks whether the score assigned to the true next
byte was high enough.

The model has 136,960 trainable numbers:

| Component | Arithmetic | Parameters |
|---|---:|---:|
| token embedding | `256 x 64` | 16,384 |
| position embedding | `64 x 64` | 4,096 |
| one Transformer block | norms + attention + MLP | 49,984 |
| two blocks | `2 x 49,984` | 99,968 |
| final layer norm | `64 + 64` | 128 |
| output head | `256 x 64` | 16,384 |
| **total** | | **136,960** |

## 2. How to read a shape

The long input is `(16, 64)`: 16 sequences, each 64 byte positions long. The token
embedding replaces each integer byte ID with 64 learned features, producing
`(16, 64, 64)` = batch, position, channel.

Four attention heads split the 64 channels into `64 / 4 = 16` channels per head:

```text
(B, T, C) = (16, 64, 64)
reshape     -> (16, 64, 4, 16)
transpose   -> (16, 4, 64, 16) = (B, H, T, D)
```

Every query position compares with every key position. That turns the last two
dimensions into `T x T`, so attention scores are `(16, 4, 64, 64)`. In the short
micro-batch they are `(16, 4, 8, 8)`. Sequence length therefore changes attention
storage quadratically: going from 8 to 64 positions multiplies this tensor by
`(64/8)^2 = 64`, not 8.

The final logits are `(16, 64, 256)`: one score for every possible next byte at every
position of every sequence. Flattening batch and position produces `(1024, 256)`.

Open `artifacts/shape_ledger.txt` and read each entry as a sentence. For example:

```text
microbatch_1_long.block_0.q:
shape=(16, 4, 64, 16)
dimensions=batch, attention head, query position, head channel
```

## 3. What one gradient means

For a loss `L` and one weight `w`, the gradient `dL/dw` is a local slope. Here
`backward()` reports:

```text
dL/dw = 0.013563441054856254
```

A positive value says increasing this weight alone would increase the loss locally,
so gradient descent moves it in the negative direction.

The independent nudge uses `epsilon = 0.00001`:

```text
L(w + epsilon) = 5.659547824005586
L(w - epsilon) = 5.659547552736764
difference     = 0.000000271268822

difference / (2 x epsilon)
= 0.000000271268822 / 0.00002
= 0.01356344112579677
```

That differs from `backward()` by only `7.09e-11`. The nudge cannot be arbitrarily
large because the loss surface curves, or arbitrarily small because floating-point
subtraction eventually loses the difference.

Try changing `epsilon` in `finite_difference_check` to `1e-1`, `1e-3`, `1e-5`,
`1e-7`, and `1e-9`. Predict where the error will be smallest before running it.

## 4. Why average-of-averages is wrong

The short micro-batch contains `16 x 8 = 128` target tokens. The long one contains
`16 x 64 = 1,024`. Together they contain 1,152.

Correct token weighting:

```text
short weight = 128 / 1152  = 11.11%
long weight  = 1024 / 1152 = 88.89%
```

Broken average-of-averages:

```text
short weight = 1 / 2 = 50%
long weight  = 1 / 2 = 50%
```

The broken calculation lets each *micro-batch* vote equally; the correct calculation
lets each *token* vote equally. If lengths were both 64, the two formulas would be
equivalent. That is a useful control experiment: change the short length from 8 to
64 and confirm that the curve gap nearly disappears.

## 5. What the grad norm means

There is one gradient per parameter. The global L2 norm compresses all of them into
one magnitude:

```text
grad_norm = sqrt(g1^2 + g2^2 + ... + g136960^2)
```

It says how long the complete update direction is before AdamW transforms it. It
does not say whether that direction is good.

At steps 38 to 39, evaluation loss continues its smooth decline by only 0.214%, from
2.958693 to 2.952354, while grad norm abruptly rises 53.994%, from 0.309012 to
0.475860. The relative gradient signal is about 252 times the relative loss movement.
The loss shows no corresponding warning at that step. This is a leading *visibility*
example, not proof of a future failure. A true predictive claim would require a longer
run and showing that a norm anomaly consistently precedes later loss degradation.

That distinction matters. The submitted result satisfies the requested observation,
but it should not be oversold.

## 6. What MFU means

Session 10 uses the approximation:

```text
training FLOPs per token = 6 x number of parameters
                         = 6 x 136,960
                         = 821,760
```

The timed loop processes 231,153 tokens/s, so:

```text
useful work = 821,760 x 231,153
            = 189,951,915,866 FLOPs/s
            = 0.18995 TFLOP/s
```

The hardware upper bound is estimated as:

```text
40 SMs x 128 FP32 lanes/SM x 2 FLOPs/cycle x 2.1 GHz
= 21.504 TFLOP/s
```

Therefore `MFU = 0.18995 / 21.504 = 0.883%`.

This does **not** mean 98.952% of wall time is literally idle. It means the rate of
model FLOPs credited by this convention is 0.883% of an ideal FP32 peak. Tiny matrix
multiplications, kernel launch latency, elementwise work, optimizer work, and memory
traffic all consume time without adding enough counted FLOPs.

Try increasing batch size in `benchmark_mfu` from 1 to 8, 32, and 128. MFU should
initially rise because the GPU receives more parallel work; eventually memory or
compute becomes the limit.

## 7. Why decimal 0.1 changes

In binary, 0.1 repeats forever:

```text
0.1 decimal = 0.00011001100110011... binary
            = 1.1001100110011... x 2^-4
```

No finite format stores the infinite tail, so each rounds it:

| Format | Bits: sign exponent fraction | Stored value | Relative error |
|---|---|---:|---:|
| fp32 | `0 01111011 10011001100110011001101` | 0.10000000149011612 | about 0.00000149% |
| bf16 | `0 01111011 1001101` | 0.10009765625 | about 0.09766% |
| fp8 E4M3 | `0 0011 101` | 0.1015625 | 1.5625% |

BF16 is the practical choice here for weights, activations, and gradients on
supporting hardware, while keeping optimizer states and sensitive reductions in
FP32. It keeps FP32's 8 exponent bits, which protects range. E4M3 needs an explicit
scaling recipe; its three fraction bits are too coarse to use casually.

## 8. A useful order for your own experiments

1. Change only sequence length and predict every affected shape.
2. Sweep finite-difference epsilon and plot error versus epsilon.
3. Make the micro-batch lengths equal, then unequal, and calculate token weights.
4. Change batch size and record tokens/s and MFU.
5. Add gradient clipping only after recording the *unclipped* norm; compare both.
6. Run BF16 and FP32 versions and compare loss, throughput, and memory.

Change one variable at a time. Write down your prediction before running—otherwise
the experiment becomes output-watching rather than learning.
