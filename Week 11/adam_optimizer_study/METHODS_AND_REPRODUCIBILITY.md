# Methods, reproducibility, and limitations

This document contains supporting detail for the concise assignment [`README.md`](README.md).

## Experimental setup

- Dataset: 8,192 training, 2,048 validation, and 4,096 held-out test examples; 32 Gaussian input features; 10 classes assigned by a fixed nonlinear teacher network.
- Exact data: [`data/synthetic_teacher_v1.npz`](data/synthetic_teacher_v1.npz), approximately 1.7 MB. The splits are stored rather than regenerated on the instructor's machine.
- Student: a width-scalable residual MLP with `input`, `hidden_norm`, `hidden`, `output_norm`, and `output` parameter groups.
- Optimizer: AdamW with `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, and weight decay `1e-4`.
- Decay grouping: matrix weights receive decay; biases and normalization scales/shifts do not.
- Parameterization: standard fan-in initialization (`std = 1 / sqrt(fan-in)`), not muP.
- Batch size: 128.
- Reproducibility controls: deterministic algorithms, fixed seeds, TF32 disabled, fixed minibatch sequences, and paired initial states where comparisons are made.

## Data integrity

Before training, the loader verifies the stored arrays against this tensor-content hash:

```text
sha256:5482356c445d8bb9e29f969dc2da0958cac7740a44c62382d89df71af422bbb5
```

The distributed `.npz` file has this SHA-256:

```text
2bcf97e67b6db2d7e29bb0d8102875863841bf9d3656290f69c5de1e4333fdc6
```

Run `python experiment.py --verify-data` to check the tensor content. The schema and checksums are also documented in [`data/README.md`](data/README.md) and [`data/SHA256SUMS.txt`](data/SHA256SUMS.txt).

## Bias-correction criterion

“Stops mattering” is defined as the first step at which the instantaneous correction-scale difference is below the selected tolerance and stays below it thereafter. The sensitivity analysis is:

| tolerated difference | first permanent step |
|---:|---:|
| 10% | 1,660 |
| 5% | 2,327 |
| 1% | 3,916 |
| 0.1% | 6,213 |

This concerns the current update scale. Earlier differences accumulated in the weight trajectory do not automatically disappear.

## Matched warmup control

The direct LR multiplier reaches 1.0 at step 20. To measure whether warmup still affects the realized trajectory, training was repeated with and without warmup over five paired seeds for 200 steps. Initialization, minibatches, model, optimizer, and peak LR were matched.

The relative gap between each condition's mean update-to-weight ratio is required to stay below 5% for all remaining measured steps. Four layers meet that rule at step 19; `output_norm` meets it at step 41. Consequently, step 20 is the exact schedule boundary and step 41 is the empirical all-layer convergence point.

Evidence: [`artifacts/warmup_control_raw.csv`](artifacts/warmup_control_raw.csv), [`artifacts/warmup_control_summary.csv`](artifacts/warmup_control_summary.csv), and [`artifacts/warmup_control_gap.csv`](artifacts/warmup_control_gap.csv).

## Fair cosine/WSD comparison

Both schedules have 300 planned steps and are stopped at step 200. Each receives the same seven candidate peak LRs, the same three tuning seeds, the same validation set, and the same 200-step tuning budget. Both independently select `7.5e-4`.

Final evaluation uses five new paired seeds. Within each pair, initial-weight and minibatch-sequence hashes match. The test set is untouched during tuning.

| schedule | test loss mean ± SD | bootstrap 95% CI |
|---|---:|---:|
| cosine | 0.55906 ± 0.01254 | [0.54981, 0.56964] |
| WSD | 0.57866 ± 0.01347 | [0.56936, 0.59043] |

The paired cosine-minus-WSD difference is `-0.01961 ± 0.00259`, bootstrap 95% interval `[-0.02156, -0.01765]`. The entire paired interval favors cosine at the required stop point.

This does not claim that cosine would also win after both schedules complete step 300; WSD does not start its final decay until step 270.

## Width-sweep fitting and uncertainty

Each width uses seven local geometric LRs, five seeds per LR, 10 warmup steps, and 80 training steps: 105 runs total. A quadratic is fitted to mean validation loss as a function of log LR. Every fitted minimum is interior to its tested range.

| width | fitted minimum | bootstrap 95% interval |
|---:|---:|---:|
| 256 | 2.669e-3 | [2.562e-3, 2.807e-3] |
| 512 | 1.078e-3 | [1.024e-3, 1.155e-3] |
| 1,024 | 4.298e-4 | [4.100e-4, 4.501e-4] |

A log-log fit gives `optimal LR(width) = C × width^-1.317`. Five thousand seed bootstraps give an exponent interval `[-1.367, -1.272]` and a width-4,096 prediction of `6.94e-5`, interval `[6.23e-5, 7.68e-5]`.

The internal interval is fairly narrow, but external confidence remains low because only three widths, one synthetic task, and one short training horizon were measured. A practical width-4,096 confirmation grid is `3.5e-5`, `7e-5`, and `1.4e-4`.

## Software environment and numerical reproducibility

The reference snapshot-backed run used:

- Python 3.10.14
- PyTorch 2.5.1
- NumPy 2.2.5
- Matplotlib 3.10.8
- CUDA 12.1
- NVIDIA GeForce RTX 3070 Laptop GPU

Its runtime was approximately 140 seconds. Dependency versions are pinned in [`requirements.txt`](requirements.txt).

The same files, inputs, seeds, initializations, minibatches, and algorithms are used on another system. Floating-point reductions can nevertheless differ slightly between CPU/GPU models and operating systems. Reproduction should therefore be judged using normal numerical tolerance and identical conclusions, not bitwise identity of every final loss.

## Evidence and audit trail

- [`experiment.py`](experiment.py): complete experiment implementation.
- [`test_experiment.py`](test_experiment.py): eleven numerical, schedule, data-integrity, and reproducibility tests.
- [`artifacts/summary.json`](artifacts/summary.json): machine-readable configuration, hashes, and headline results.
- `artifacts/*.csv`: raw and aggregated measurements.
- `artifacts/*.png`: report figures.
- `artifacts/*_step_200.pt`: representative interrupted checkpoints.

## Scope limits

The claims apply to this controlled synthetic experiment. The assignment does not require real-data replication, direct width-4,096 training, completing both schedules to step 300, or a true muP comparison. Those are useful follow-up studies, not missing deliverables.
