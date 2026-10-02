# Reproducibility audit — 2 October 2026

The complete numerical experiment was rerun from a separate copy in a directory containing spaces, launched from an unrelated working directory. The copy had no sibling Transformer package. It used only this package's source, bundled byte dataset and the installed Python dependencies.

## Fresh training, not just checkpoint evaluation

| Stage | Historical loss | Fresh-run loss | Difference |
|---|---:|---:|---:|
| Linear baseline, 400 updates | 2.393201353841788 | 2.393201353841788 | 0 |
| MoE continuation, 600 updates | 2.2335704091984003 | 2.2335704091984003 | 0 |
| Dense continuation, 600 updates | 2.3524556857424934 | 2.3524556857424934 | 0 |

The starting baseline was randomly initialized, trained, and then converted. The freshly trained MoE checkpoint—not the shipped reference checkpoint—was used for the new deep-dive run. All 22 intervention losses and all six final balancing losses matched the historical references exactly on the tested machine. Each balancing arm was trained for 200 updates; all six saved checkpoints reproduced their final losses after reloading, including selection-bias state.

Additional checks cover dataset shape/range/hash, finite model parameters, exact parameter counts, affine semantics, copied-expert output parity, expert-output removal, bias selection with original probability weights, and whole-batch auxiliary-gradient parity. The package manifest checks every immutable delivered file. Training outputs go to separate run folders.

Saved evidence is in validation/reproduction/. REPRODUCTION.json records the full-run result and environment; evaluation.json records the three checkpoint losses; deep_dive/reference_comparison.json records all 28 loss comparisons. Per-arm training histories and console logs are included.

The PowerShell copier was tested for repeated identical copies and refusal to overwrite conflicting files. A sentinel in the parent 20M folder remained unchanged. The package was also committed and cloned through a local Git repository with the parent checkpoint LFS rule present. All small-model checkpoint filters were unset as intended; checkpoint evaluation from that clone reproduced all three reference losses exactly. This was a local Git round trip, not a remote GitHub clone. Evidence: validation/packaging.json and validation/git_clone_evaluation.json.

The review notebook's four code cells were executed with Python and captured output, with optional retraining disabled. The notebook UI itself was not independently tested.

## What is and is not reproduced

- Reproduced: baseline training, dense/MoE continuation, conversion, full selected byte-holdout evaluation, all shutdown/routing interventions, six balancing continuations and checkpoint reloads.
- Inputs: the frozen bundled 1M-target training subset and 79,116-byte holdout. Upstream dataset acquisition, the original 50M corpus construction and tokenization are not rerun. Their provenance is retained.
- Environment: Windows, Python 3.12.5, PyTorch 2.5.1+cu121, NumPy 1.26.4, RTX 3070 Laptop GPU. Dependencies came from an existing environment with those versions; this audit did not perform a new internet package installation or test a different operating system/GPU.
- Numerical checks use tolerance 1e-4 for loss parity on other runs. Same-machine exact results are evidence, not a promise of bitwise equality on other hardware.
- The six balancing methods are single-seed, short continuation experiments. Reproducibility does not establish statistical superiority, semantic expert roles or multi-GPU scaling.

Run `python run.py reproduce --output runs/reproduction` to repeat the complete workflow. Choose another new output folder for subsequent runs. No original 20M files are used or modified.
