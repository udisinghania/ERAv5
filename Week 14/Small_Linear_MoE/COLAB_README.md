# Colab reproduction workflow

[Open Small_Linear_MoE_Colab.ipynb](https://colab.research.google.com/github/udisinghania/ERAv5/blob/main/Week%2014/Small_Linear_MoE/Small_Linear_MoE_Colab.ipynb), select **Runtime → Change runtime type → GPU**, then **Run all**.

This notebook is separate from Small_Linear_MoE.ipynb, which is a review of recorded results. The Colab workflow contains no prefilled training outputs. It is intended to execute the actual experiment on the selected GPU.

## What Run all does

1. Creates a new session folder and optionally mounts your Google Drive for persistent outputs.
2. Clones the public repository with sparse checkout and LFS downloads disabled, selecting Week 14/Small_Linear_MoE. The parent 20M data and checkpoints are unnecessary.
3. Creates an isolated Python 3.12 environment, installing torch 2.5.1+cu121 and NumPy 1.26.4. The notebook kernel's packages are not replaced. If its interpreter is not Python 3.12, uv obtains a compatible one.
4. Verifies all manifest hashes, checks CUDA, and reevaluates all three shipped checkpoints.
5. Trains a new linear baseline from random initialization, copies it to MoE, continues matched dense and MoE arms, and evaluates both on all 79,116 held-out byte targets.
6. Runs the 22 routing/shutdown cases and six short balancing experiments from the newly trained MoE, saving and reloading all balancing checkpoints.
7. Runs the two-seed speed/memory study in separate processes for each stage.
8. Creates a ZIP of actual outputs for download. If Drive is enabled, those outputs also remain in the selected Drive folder.

Set RUN_FRESH_TRAINING or RUN_TWO_SEED_STUDY to False only if deliberately skipping those experiments. The notebook clearly labels skipped stages. SAVE_TO_DRIVE is False by default; changing it to True invokes Colab's normal Drive authorization. Without Drive or downloading the archive, runtime resets can remove outputs. Each execution chooses a fresh session name and does not overwrite a previous run.

## Numerical differences

The reference scores were obtained on an RTX 3070 Laptop GPU using Windows and Python 3.12.5. A different GPU or platform can yield different floating-point results. The notebook records its actual environment and uses --allow-reference-drift: a difference exceeding 1e-4 is recorded as **REFERENCE_DRIFT**, not PASS. Correctness assertions and continued-loss-reduction checks still apply. Default local commands remain strict.

## Validation scope

The notebook's evaluation, fresh-training, deep-dive, study and archive orchestration is exercised locally with explicit local-package/interpreter overrides. The public Git clone/dependency-install branches are inspected and separately checked where possible; they are not described as a hosted Colab GPU execution. The published validation record distinguishes real GPU calculations from bootstrap inspection.

**A complete hosted Colab GPU session has not yet been independently verified.** GPU availability is controlled by Colab. The notebook performs a CUDA preflight rather than assuming a particular GPU was allocated. No paid runtime, subscription or billing action is configured.

References for the bootstrap behavior: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [uv Python management](https://docs.astral.sh/uv/guides/install-python/), [uv environments](https://docs.astral.sh/uv/pip/environments/).

## If a run stops

- CUDA unavailable: select a GPU runtime and rerun from the beginning.
- Missing extended_study.py: the GitHub folder is still the earlier release; push the complete README update before using this new notebook.
- Hash mismatch: the copied package is incomplete or altered. Do not bypass verification; obtain the package again.
- REFERENCE_DRIFT: inspect evaluation.json and reference_comparison.json. It reports numerical differences, not necessarily a failed learning experiment.
- Interrupted training: checkpoints are saved at stage completion. This runner does not resume optimizer state mid-stage; use a fresh run folder to repeat a stage or workflow.
- Dependency/network error: rerun in a new session after access is restored. Installation is not part of the reported GPU timing.
