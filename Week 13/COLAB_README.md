# Google Colab reproduction

Use `Reversibility_20M_50M_Colab.ipynb` for a fresh-clone, checkpoint-aware
rerun. The notebook clones the submitted GitHub repository into `/content` and
copies its Python sources into a separate Google Drive run directory. Training
outputs therefore survive a runtime disconnect without dirtying the Git clone.

## Dataset

The submission repository can include the minimal 402 MiB training-ready data
under `data/` using Git LFS. The notebook detects that layout and copies the
required arrays into the persistent Drive workspace. Raw documents and corpus
construction intermediates are not needed for training.

As a fallback, preserve this layout directly in Google Drive:

```text
MyDrive/ERA/outputs/
  Corpus_20M_v1/
    packed_dataset.py
    packed_50m_ctx512/
    baseline/artifacts/tokenizer_v2/tokenizer.json
  Run_20M_50M_v1/
    validation/
```

Do not upload or commit the local experiment checkpoints before starting a fresh
Colab run.

## What the notebook does

1. Mounts Drive, installs Git LFS, and asks for the GitHub HTTPS clone URL.
2. Installs the small Python dependency file without replacing Colab's CUDA
   PyTorch build.
3. Copies repository data into Drive when it is not already present, then
   verifies CUDA, BF16 capability, source files, and corpus/validation files.
4. Runs reconstruction, gradient, masking, and activation-storage tests.
5. Trains equal 5M-token midpoint and Euler pilots and selects the lower-loss
   reversible variant.
6. Searches maximum batch on the assigned GPU instead of reusing the RTX 3070
   result.
7. Trains the baseline fixed batch, reversible fixed batch, and reversible
   maximum batch for exactly 50M targets each.
8. Validates token counts/audits and displays loss, perplexity, tokens/s, peak
   memory, and training time.

The definitive precision policy requires native BF16 support. If the assigned
GPU does not support BF16, request a different accelerator or use the documented
local CUDA environment; silently changing to FP16 would no longer reproduce the
controlled experiment.

The measured local run took about 1.4 GPU-hours for the original controlled
suite. Colab speed, maximum batch, availability, and runtime duration vary by
the assigned accelerator. Rerunning the full-training cell resumes incomplete
checkpoints and skips completed runs.
