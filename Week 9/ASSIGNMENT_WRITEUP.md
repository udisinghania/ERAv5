# Assignment 9 final measured values

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
