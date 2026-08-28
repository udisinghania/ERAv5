# Assignment 9: Observable Cross-Entropy and Multi-Token Prediction

This repository is the submission for Assignment 9. It contains one executed
notebook for the observable loss harness, the training implementation, and the
retained JSON/JSONL evidence from the production 17M-parameter run. Google
Colab is not required for this submission.

## Submission artifacts

- [Executed notebook](Assignment_9_Loss_Harness.ipynb)
- [Exact final metrics](assignment9_final_metrics.json)
- [Short write-up](ASSIGNMENT_WRITEUP.md)
- [Development history: smoke tests, failures, V1 versus V2](EXPERIMENT_HISTORY.md)
- [Machine-readable development history](experiment_history.json)
- [Original smoke, V0, and V1 configs/summaries](experiments/)
- [Production validation and training ledger](outputs/mtp_17m_session_run/metrics.jsonl)
- [Production run summary](outputs/mtp_17m_session_run/run_summary.json)
- [Resolved production configuration](outputs/mtp_17m_session_run/resolved_config.json)
- [Frozen validation probe](outputs/mtp_17m_session_run/validation_probe.json)
- [Numbered checkpoint index](outputs/mtp_17m_session_run/checkpoint_index.jsonl)
- [Training implementation](outputs/train_mtp_98m.py)
- [17M Session-9 entry point](outputs/train_mtp_17m_session.py)
- [Reproduction data manifest](repro_data/README.md)
- [Python requirements](requirements.txt)

The large `.pt` checkpoint files are intentionally excluded from Git because
each full checkpoint is about 196 MiB, above GitHub's ordinary 100 MiB file
limit. Their SHA-256 hashes and the complete checkpoint history are preserved
in the run summary and checkpoint index.

## Development history (supporting evidence, not Part 3)

The repository records the first smoke tests, the overtrained 98M attempt, the
17M V1 baseline, and the Session-9-aligned V2 in
[EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md). The headline matched-data
comparison is:

| Best held-out value | V1 | Submitted V2 | Change |
|---|---:|---:|---:|
| H1 loss | 2.720188 | 2.595900 | -4.57% |
| H2 loss | 3.445438 | 3.372842 | -2.11% |
| Sum | 6.165626 | 5.968742 | -3.19% |

This is not presented as an architecture-only ablation because V2 also trained
for more corpus passes. Parts 1 and 2 below remain the assignment submission.

## Production run

| Item | Value |
|---|---:|
| Parameters with both heads | 17,109,888 |
| Architecture | 6 layers, D=384, 6 heads |
| Normalization / FFN | RMSNorm / SwiGLU |
| Validation targets | 100,000 requested; 102,019 H1 / 101,860 H2 measured |
| Best step | 6,500 |
| Stopping step | 7,500 |
| Corpus passes at stop | 8.0749 |
| Stop reason | `validation_early_stopping` |

## Measured submission results

## Part 1 — the loss harness

| Requirement | Measured result |
|---|---|
| Tensor shapes | tokens `(2, 256)`; hidden `(2, 256, 128)`; logits `(2, 256, 8192)`; shifted logits `(2, 255, 8192)`; targets `(2, 255)` |
| String-level shift | `h[t] predicts x[t+1]`; observed pair `<eos> -> {"` |
| Padding mask | contributing targets `126 -> 109`; removed `17`; loss `8.976094 -> 9.013454` |
| Packed boundary mask | pair `<eos> -> {"`; contributing targets `252 -> 251`; loss `9.055383 -> 9.054023` |
| Untrained perplexity | `8,357.53` for vocabulary `8,192` (`2.02%` relative difference) |
| Tied versus untied | full model `1,509,888` tied versus `2,558,464` untied; saved `1,048,576` parameters |
| Ordinary versus chunked CE memory | `2,048.00 MiB` versus `76.00 MiB`; reduction `26.95x` |

Every tensor dimension is printed in the notebook: `B` is batch size, `T` is
sequence length, `D` is hidden width, and `V` is vocabulary size. The notebook
prints decoded token strings beside their shifted targets, not only token IDs.
The packed transition is removed because predicting the first token of an
independent document from the previous document is not a valid language-model
target. Masking can raise or lower the mean loss; correctness is determined by
which targets contribute.

The untrained-model check passes because perplexity is close to vocabulary size.
The ordinary and chunked losses differ by only
`0.0000009537`, confirming that chunking changes
memory use rather than the objective.

## Part 2 — the second prediction head

The shared hidden state at position `t` feeds Head 1 for `x[t+1]` and an
independent Head 2 for `x[t+2]`. The optimized objective is exactly
`L_total = L_head1 + L_head2`.

| Held-out metric | Best retained checkpoint | Early-stopping point |
|---|---:|---:|
| Optimizer step | 6,500 | 7,500 |
| Head 1 loss (`t+1`) | 2.595900 | 2.635009 |
| Head 2 loss (`t+2`) | 3.372842 | 3.394982 |
| Sum | 5.968742 | 6.029991 |
| Head 1 perplexity | 13.41 | 13.94 |
| Head 2 perplexity | 29.16 | 29.81 |

Head 2 is expected to remain harder: it must predict `x[t+2]` without observing
the intervening `x[t+1]`, so its conditional uncertainty and loss are higher.
This is also what the run measured: Head 2 was higher than Head 1 at all
15/15
periodic held-out checks. At the first periodic check (step
500), the losses were
H1=4.074287 and
H2=4.390004; at the retained best step,
both had improved to H1=2.595900 and
H2=3.372842, with Head 2 still harder.
The best checkpoint is selected on the held-out sum, not training loss. Training
stops after 2 consecutive
validations fail to improve that sum by at least
`0.001000`, and `best.pt` remains
separate from the later `latest.pt` resume checkpoint. A hard limit of
12.00 corpus passes provides an additional
guard against repeated-data overtraining.

Checkpoint integrity: `recomputed_and_matched`. The hashes are
recorded in `assignment9_final_metrics.json` and the production run summary.


## Reproduction note

The minimum Session 6 tokenizer, packed corpus, masks, positions, segment IDs,
and frozen batch plan are included under `repro_data/Assignment_6_v2`. From the
repository root, install the dependencies and run the checks below. None of
these three checks starts a long training run.

```powershell
python -m pip install -r requirements.txt
python outputs/train_mtp_17m_session.py inspect-data
python outputs/train_mtp_17m_session.py inspect-validation
python outputs/train_mtp_17m_session.py smoke-model
```

Open or execute `Assignment_9_Loss_Harness.ipynb` to reproduce every Part 1
measurement and the Part 2 smoke run. The notebook also executes a production
results section that reads the retained JSONL ledger, prints all 16 held-out
checks, and verifies the best and early-stopping steps without launching a new
training run. The production run itself is reproducible with the retained
configuration and training implementation; its multi-hour allowance was not
needed because held-out early stopping fired after two non-improving
validations. The excluded `.pt` files are not required to audit the submitted
results because the complete metric ledger, configuration, checkpoint index,
hashes, and final summary are retained as text artifacts.
