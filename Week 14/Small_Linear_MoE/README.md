# Session 14: Linear model to Mixture of Experts

This is the **standalone small linear experiment**. The existing 20M Transformer experiment remains in the parent Week 14 folder. This folder needs none of its code, data or checkpoints.

The assignment asks: train a linear model, convert it to an MoE, and show continued training with reduced loss. We train one affine next-byte predictor, copy its trained weights into four experts, select two experts per example, and continue training. A matched dense continuation controls for improvement due simply to more training.

| Stage | Held-out byte loss |
|---|---:|
| Randomly initialized linear model | 5.541114 |
| Trained linear model, 400 updates | 2.393201 |
| Continued MoE, 600 additional updates | **2.233570** |
| Continued dense control, same 600 batches | 2.352456 |

![Learning curves](figures/linear_learning.png)

Evaluation uses all **79,116 bundled held-out byte targets**. Lower loss is better. Byte losses must not be compared numerically with the parent Transformer's BPE-token losses. The holdout has been examined during development; it is not an untouched test set.

## Architecture

| Property | Value |
|---|---|
| Input | Four previous bytes; fixed one-hot features |
| Input width | 4 × 257 = 1,028, including a padding category |
| Output | 256 next-byte logits |
| Dense model | One affine layer: 1,028 → 256, with bias |
| Dense parameters | 263,424 |
| MoE stages / routed experts | One stage / four experts |
| Each expert | An independent copy of the trained affine layer |
| Router | 1,028 → 4, no bias; 4,112 parameters |
| Routing | Softmax, top-2, renormalized selected probabilities |
| MoE total / active parameters | 1,057,808 / 530,960 |
| Inactive expert parameters per example | 526,848 |
| Shared experts / expert parallelism | None / none; one GPU |

The baseline's **logits are affine** in the fixed features; classification probabilities use softmax. The routed MoE can have nonlinear logits. There are no learned embeddings, hidden layers, attention or GELU. Active-parameter counts describe selected expert blocks, not a measured FLOP or memory saving. All expert weights remain resident in memory.

Copied experts preserve the original predictions: measured maximum conversion logit difference **9.54e-7**. Experts subsequently receive different routed examples and develop different weights.

## Reproduce

See [SETUP.md](SETUP.md) for the tested environment and installation commands. From this folder:

The [review notebook](Small_Linear_MoE.ipynb) includes executed result-inspection cells and an optional full-training cell, disabled by default.

```powershell
python run.py verify
python run.py evaluate
python run.py reproduce --output runs/reproduction
```

`verify` requires only standard Python and checks every packaged file. `evaluate` requires CUDA and recomputes the three shipped checkpoint losses. `reproduce` trains a fresh baseline, runs both continuation arms, checks conversion and affine semantics, evaluates the new checkpoints, and reruns all deep-dive interventions and balancing arms. Use a fresh output folder for each training run; shipped evidence is preserved.

The input dataset is bundled; no Hugging Face login, tokenizer, Week 13 folder or Transformer package is needed. See [DATA.md](DATA.md) for provenance and the boundary between reproducing this experiment and rebuilding the upstream corpus.

## Deep dive: challenge the mechanism

The [full illustrated report](deep_dive/REPORT.md) covers all architecture counts, expert divergence, router gradients, active/inactive parameters, shutdown, top-k, and the balancing concepts from sections 12–14.

![Routing and expert shutdown](deep_dive/routing_and_shutdown.png)

- **22 intervention cases** use the complete selected byte holdout: learned/equal/random routing, fixed averaging, individual expert removal, rerouting, single-expert prediction and inference top-k changes.
- **Six matched short continuation arms** compare no balancing, auxiliary coefficients .001/.01, z-loss, microbatch/whole-batch scope and selection-only bias balancing.
- All six arms save checkpoints, including selection-bias state, and verify final loss after reloading.

Normal routing scores 2.2336, versus 2.3719 for equal averaging of all experts and approximately 2.4091 for random pairs. Rerouting reduces the harm of expert removal. These are measurements of this trained checkpoint, not universal guarantees. Inference top-k tests do not establish the best top-k after retraining. Balancing runs use one training seed and only 204,800 extra exposures per arm; tiny loss gaps do not establish a winning method.

## Files and verification

- `src/`: runnable standalone model, training and diagnostic implementations.
- `data/`: frozen byte arrays and source provenance.
- `checkpoints/`: trained linear baseline, MoE and dense continuation.
- `results/`: historical training logs, configuration and measured losses.
- `deep_dive/`: report, figures, raw results and six balancing checkpoints.
- `validation/`: independent-copy reproduction evidence.
- `MANIFEST.json`: package hashes; `REPRODUCIBILITY.md`: exact validation scope.

The small checkpoints use ordinary Git; this folder explicitly overrides the parent folder's checkpoint LFS rule. A fresh clone can run this experiment without downloading the parent 20M dataset/checkpoints. Nothing in this folder modifies the original 20M study.

For the verified copy and scoped commit commands, see [PUBLISH.md](PUBLISH.md).
