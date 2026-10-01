# 20M-parameter model trained on 50M tokens

This is a separate training run using the prepared `Corpus_20M_v1` dataset and
the PyTorch GPU environment from `Assignment_6_v2`. The original assignment is
unchanged. `status.json` records progress; `result.json` is written only after
training, full validation and generation samples complete.

## Completed results

- Finished 3,277 optimizer updates on exactly 50,000,000 supervised target tokens.
- Full held-out validation: 5,049,456 target tokens, excluded from training.
- Validation cross-entropy fell from 9.1182 to 3.5142 nats; final perplexity 33.59.
- Recorded active run time: 634 seconds (10.6 minutes), excluding the interruption
  and initial full validation. This includes checkpointing and final evaluation.
- Peak allocated GPU memory in the resumed session: 3.93 GiB. The batch-size
  benchmark observed a higher 4.12 GiB allocated peak.
- All four model tests passed. All three checkpoints load successfully, contain
  finite weights, match the run signature and recorded SHA-256 hashes, and hold
  identical final model weights at update 3,277. The saved data cursor is 104,863.
- Standalone GPU generation from `final.pt` passed its smoke test.

Output quality remains weak: continuations are repetitive and often incoherent.
For example, greedy decoding of `Water is` produced `Water is a most of the most
of the most of the most of the most of`. This is a successful training experiment,
not a usable chatbot. The lower validation loss should not be mistaken for strong
language or reasoning capability.

## Model and training

- 20,166,912 parameters; decoder-only Transformer with tied input/output embeddings.
- 9 pre-LayerNorm blocks, hidden size 384, 6 attention heads, GELU MLP size 1,664.
- Frozen original tokenizer: 8,192 token IDs; context length 512.
- Learned position embeddings; positions reset for each document fragment.
- Segmented causal scaled-dot-product attention respects the dataset's document boundaries.
- One pass over exactly 50,000,000 supervised target tokens.
- BF16 mixed precision, FP32 model/optimizer states, fused AdamW on one RTX 3070 Laptop GPU.
- Batch size 32 sequences; learning rate warms up to 0.0006 then decays to 0.00006.
- Gradient clipping at 1.0; weight decay 0.1 on matrix parameters.

Batch size was chosen from measurements in `benchmark.json`, with a 6 GiB
reserved-memory ceiling. No tokenizer or model is downloaded during training.
The training stream and source masks are preserved, including response-only
loss for the relevant inherited reasoning/agentic records.

## Checkpoints

- `checkpoints/final.pt`: final model and full held-out validation results.
- `checkpoints/best.pt`: model with the lowest fixed validation-probe loss.
- `checkpoints/last.pt`: model, optimizer, random states and exact data cursor
  for resuming an interrupted run.

The run was resumed from update 2,750 after 42,007,341 supervised tokens.
Checkpoint identities bind the model code, training code, tokenizer, corpus
catalog and packing manifest. A resume refuses mismatched identities.

## Evaluation

`initial_validation.json` and `final_validation.json` use the same complete
validation set, with results by data source. `latest_probe.json` uses a smaller
fixed sample for checkpoint selection and is not directly comparable with the
full-validation aggregate. Validation is excluded from gradient updates.

This is a small base model trained on approximately 2.48 tokens per parameter.
It is intended for experiments and text continuation. Lower validation loss
does not establish factual accuracy, reasoning ability or instruction following.
`samples.json` retains the unedited final continuations, including weak outputs.

## Try a prompt

From this folder, in PowerShell:

```powershell
& 'C:\Users\udisi\Documents\Codex\2026-08-01\now\outputs\Assignment_6_v2\.venv\Scripts\python.exe' -B .\generate.py --prompt 'The sun is' --max-new-tokens 80
```

Use `--temperature 0` for greedy decoding, or `--device cpu` for CPU inference.
The script caps output at the trained 512-token context. Keep `Corpus_20M_v1`
beside this folder so the frozen tokenizer can be loaded.

## Resume and test

```powershell
& 'C:\Users\udisi\Documents\Codex\2026-08-01\now\outputs\Assignment_6_v2\.venv\Scripts\python.exe' -B .\train.py train --resume
& 'C:\Users\udisi\Documents\Codex\2026-08-01\now\outputs\Assignment_6_v2\.venv\Scripts\python.exe' -B .\test_model.py
```

Resume is for an interrupted run; a completed run refuses to train again in the
same directory. The tests cover causal/document isolation, label shifting,
token-weighted gradient accumulation and optimizer-state restoration.
