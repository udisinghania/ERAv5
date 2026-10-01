# Session 14: Dense-to-Mixture-of-Experts Upcycling

This assignment continues the existing 20,166,912-parameter dense language model,
converts its trained feed-forward networks into routed experts, and measures
whether the converted model continues learning. The assignment's term "Linear
model" is interpreted as the ordinary dense model discussed in the session's
upcycling section. The starting model is a nonlinear Transformer, not a linear
regression model.

## Experiment

- Starting point: last week's completed dense checkpoint, trained on exactly
  50,000,000 supervised tokens. Its recorded validation cross-entropy fell from
  9.1182 to 3.5142 nats. The original run's result is copied into
  `dense_pretraining_result.json`.
- Backbone: 9 layers, hidden width 384, 6 attention heads, FFN width 1,664,
  GELU activations, tied embeddings, 8,192-token vocabulary, context length 512.
- Conversion: replace each dense FFN with 4 full copies of its trained weights.
  A new FP32 router selects 2 experts for each non-padding token and normalizes
  their weights to sum to one. Attention, embeddings, layer norms, and positions
  are copied from the trained dense checkpoint.
- Size: 54,685,440 total parameters; 31,682,304 active per token under the
  convention that includes the full tied embedding table. Two full experts mean
  more active FFN computation than the dense baseline. This is a simple upcycling
  demonstration, not a compute-matched efficiency experiment.
- Balancing: auxiliary load-balancing loss with coefficient 0.001, averaged over
  layers, measured over non-padding tokens in each microbatch. No token dropping,
  shared experts, expert parallelism, or forced subject-specific routing.
- Continuation: exactly 10,000,000 additional supervised tokens per arm, sampled
  without replacement from the existing 50M-token training stream. These are
  previously seen training data, not newly acquired tokens. Both arms receive
  exactly the same sequence order and final target mask.
- Control: continue the original dense checkpoint for the same 10M tokens and
  with the same optimizer settings. Equal token exposure is not equal FLOPs.
- Optimizer: fresh AdamW state for both arms, peak LR 0.0001, 30-update warmup,
  cosine decay to 0.00001, matrix weight decay 0.1, gradient clipping at 1.0,
  effective batch 32 sequences. BF16 matmuls, FP32 weights and optimizer states.
- Evaluation: the original 5,049,456 held-out targets, with the unchanged
  tokenizer, segmented causal attention, document-relative positions, and
  target-aligned loss masks. Validation never enters gradient updates.

## Read the result

After completion, `REPORT.md` contains the measured outcome and comparisons.
`loss_curves.png` shows the loss evidence; `expert_usage.png` shows final routing.
The source-specific full validation results are in each arm's
`final_validation.json`. A smaller fixed probe is logged during training; its
aggregate must not be compared directly with the full validation aggregate.

`conversion_check.json` tests the dense and copied-expert outputs in full FP32
with TF32 disabled. Training/evaluation use mixed precision, which can introduce
small numerical differences. No exact bitwise GPU reproducibility is claimed.

## Files and isolation

All new code, logs, reports, and checkpoints are contained in this folder.
The existing dataset, tokenizer, dense checkpoint, and GPU environment are read
from their original folders. No upstream files are overwritten or retrained.
`data_plan.json` records those input paths and hashes, plus the exact selection
of reused training sequences. `training_sequence_indices.npy` records that order.
The original corpus's verification and overlap reports remain upstream.

- `dense_model.py`: exact copy of the original dense architecture.
- `moe_model.py`: sparse dispatch, copied experts, router, balancing, and usage.
- `packed_dataset.py`: original memory-mapped packed-corpus reader.
- `experiment.py`: input verification, conversion check, GPU benchmark,
  continuation, validation, and checkpoints.
- `test_moe.py`: conversion/gradient equivalence, document and causal isolation,
  masked loss, sparse dispatch, router gradient, and optimizer restoration tests.
- `verify_results.py`: post-training checkpoint and evidence audit.
- `make_report.py`: regenerate the figures and measured report.
- `environment.json`: exact Python interpreter and core training package versions.
- `moe/` and `dense/`: separate runs. Each has configuration, event log,
  initial/final validation, last/best/final checkpoints, status, and result.

## Local environment and commands

The existing GPU environment was verified as PyTorch 2.5.1+cu121 on an NVIDIA
GeForce RTX 3070 Laptop GPU with 8 GiB VRAM. `benchmark.json` records measured
batch memory and throughput. No new environment or package download is needed.

From this folder in PowerShell:

```powershell
$python = 'C:\Users\udisi\Documents\Codex\2026-08-01\now\outputs\Assignment_6_v2\.venv\Scripts\python.exe'
& $python -B .\test_moe.py
& $python -B .\experiment.py prepare
& $python -B .\experiment.py moe
& $python -B .\experiment.py dense
& $python -B .\verify_results.py
```

The report generator uses the Codex bundled ReportLab and Node/Sharp runtimes
for static charts. On this computer, regenerate the report with:

```powershell
& 'C:\Users\udisi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\make_report.py
```

Completed runs refuse to overwrite themselves. Use a separate copy of this
project without the `moe/` and `dense/` result directories for a fresh run.
To resume an interrupted run, use `experiment.py moe --resume` or
`experiment.py dense --resume`. The committed checkpoint restores model,
optimizer, data cursor and update count. Uncommitted steps may be replayed.
The implementation has no stochastic layers after initialization; the sequence
order is saved. Resume checks bind the experiment code and data plan.

The input paths are defined near the top of `experiment.py`. To move this
assignment to another computer, transfer the referenced corpus and original
dense run, then update that source path and use a compatible GPU environment.

## Limits

This is one continuation run per model type, with reused data and a modest
token budget. A decrease in held-out loss demonstrates continued learning;
it does not establish improved reasoning, fluent generation, or a general MoE
advantage. The starting model's saved generations were often incoherent.
The separate dense control helps distinguish continued training from a benefit
specific to the architecture. Router balance is measured, not assumed.
