# 20M LLM reversible-training experiment - extended report

## Executive result

Nine independently verified full 50M-token checkpoints were completed: the five controlled main runs, one LR-retuned maximum-batch midpoint run, and three second-seed fixed-batch replications. Midpoint remained the best reversible method. The ordinary baseline remained fastest and achieved the lowest loss. Reversibility saved about 52% of allocated training memory at batch 32 and allowed roughly twice the real microbatch, at the cost of recomputation.

## Conceptual guide: what reversibility changes

A forward pass applies the current weights to tokens and produces intermediate layer outputs called activations. The final activation produces logits and a loss. A backward pass starts at that loss and uses the chain rule to calculate a gradient for every trainable weight. Ordinary backward needs many forward activations, so standard training stores them until their gradients have been calculated.

Reversible blocks change the memory strategy, not the meaning of training. Each block is designed so its input state can be reconstructed from its output state. The forward pass therefore keeps the final two-stream state instead of every internal stack activation. During backward, the program reconstructs one earlier state, reruns that block to obtain the local derivative information, calculates gradients, and discards the temporary reconstruction before moving to the preceding block.

For midpoint, (a,b) maps to (b, a + F(b)); therefore b_old = a_new and a_old = b_new - F(a_new). Symplectic Euler similarly reverses its two additive updates in reverse order. F, attention, and the MLP do not themselves need an inverse; they are reevaluated and the surrounding addition is undone by subtraction.

What is saved: final reversible state, masks, parameters, gradients, optimizer state, logits, and active workspace. What is avoided: the depth-growing collection of internal stack activations. This is why memory falls but does not become zero. The price is extra computation during backward. Parameters remain unchanged until the complete backward pass finishes; AdamW updates them afterward.

## Controlled five-run comparison

| Run | Batch | Updates | Final loss | PPL | Target tok/s | Peak GiB | Train min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline / fixed | 32 | 3,277 | 3.4963 | 32.99 | 62,173 | 3.93 | 13.40 |
| Midpoint / fixed | 32 | 3,277 | 3.5726 | 35.61 | 44,660 | 1.89 | 18.66 |
| Midpoint / max | 100 | 1,049 | 3.8140 | 45.33 | 49,444 | 5.36 | 16.85 |
| Euler / fixed | 32 | 3,277 | 3.7126 | 40.96 | 46,786 | 1.89 | 17.81 |
| Euler / max | 102 | 1,029 | 3.8980 | 49.30 | 48,131 | 5.46 | 17.31 |

Speed and memory comparisons use these original controlled runs only. Later replications ran under laptop Quiet Mode / thermal constraint, so their losses remain valid but their throughput is not pooled with the controlled measurements.

## Second-seed loss variability

| Variant | Seed 20260919 | Seed 20260920 | Mean | Sample SD |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 3.4963 | 3.4849 | 3.4906 | 0.0081 |
| Midpoint | 3.5726 | 3.5189 | 3.5458 | 0.0379 |
| Euler | 3.7126 | 3.6422 | 3.6774 | 0.0498 |

These n=2 sample standard deviations are exploratory error bars, not confidence intervals. Both seeds consumed exactly 50M targets and used batch 32.

## Maximum-batch LR retuning

Six peak learning rates were screened for 5M targets at each maximum batch. The midpoint pilot chose 9e-4, but its full 50M run ended at loss 3.8470 - worse than the original 6e-4 run at 3.8140. The Euler pilot selected 6e-4, so its original full run was retained. Short-pilot selection did not reliably predict the best 50M setting.

## Precision

BF16 matrix operations with FP32 parameters, normalization, loss and Adam state, plus FP64 reversible streams, was the fastest stable native policy. Default FP16 scaling failed; FP16 passed only with a smaller initial scale. Native FP8 matrix multiplication is unavailable on this Ampere RTX 3070.

## Memory and capacity

- Midpoint batch 32 saved 2.04 GiB (52.0%) allocated memory.
- Euler batch 32 saved 2.05 GiB (52.0%) allocated memory.
- Maximum real batches: baseline 54, midpoint 100, Euler 102.

## Cost, energy, and data scale

- Unique full-run training time: 2.90 GPU-hours; LR pilots add 0.32 hours.
- Measured wall time including validation/checkpoint work and LR pilots: 3.33 hours.
- Estimated whole-laptop energy including pilots: 0.53 kWh at the 160 W assumption.
- Illustrative local electricity: INR 4.26 at INR 8/kWh.
- Illustrative Google Cloud T4 GPU-only charge: US$1.17; VM CPU/RAM/storage are extra and T4 runtime may differ.
- 50M tokens equals 2.48 tokens/parameter. A rough 20 tokens/parameter reference would be about 403.3M tokens, not a quality guarantee.

## Qualitative check

Greedy completions from the three fixed-batch checkpoints are repetitive and frequently incoherent. This is expected: 20M parameters and 50M tokens are sufficient for a systems experiment, not for a capable assistant. See GENERATION_SAMPLES.md for all fifteen examples.

## Reproducibility and limitations

Environment: Windows-11-10.0.26200-SP0; Python 3.12.5; PyTorch 2.5.1+cu121; CUDA 12.1; GPU NVIDIA GeForce RTX 3070 Laptop GPU.
All nine checkpoints pass hashes, exact token counts, finite-weight checks, within-seed identical initialization, final/last equality, and reversible gradient/reconstruction audits. The notebook Reversibility_20M_50M_Colab.ipynb provides a portable checkpoint-aware Colab workflow.

Limitations: two seeds only; one laptop; one tokenizer/corpus mixture; short pretraining; and no matched-performance power instrumentation. Cost and emissions are scenarios, not invoices or audited measurements.

## External references

- Hoffmann et al., Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556
- Google Cloud GPU pricing: https://cloud.google.com/products/compute/gpus-pricing?hl=en
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- PyTorch AMP documentation: https://docs.pytorch.org/docs/stable/accelerator/amp.html
- NVIDIA Transformer Engine FP8 documentation: https://docs.nvidia.com/deeplearning/transformer-engine/getting_started/index.html
