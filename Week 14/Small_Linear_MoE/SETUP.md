# Standalone setup

Tested: Windows, Python 3.12.5, PyTorch 2.5.1+cu121, NumPy 1.26.4, RTX 3070 Laptop GPU (8 GiB). Training uses FP32 with TF32 disabled; BF16 support is not required. The implementation requires a CUDA GPU for training/evaluation. CPU-only package verification works without PyTorch or NumPy.

From the Small_Linear_MoE folder, create an environment:

```powershell
py -3.12 -m venv .venv
$Python = Join-Path $PWD '.venv\Scripts\python.exe'
& $Python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
& $Python -m pip install numpy==1.26.4
& $Python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
& $Python run.py verify
& $Python run.py evaluate
& $Python run.py reproduce --output runs/reproduction
```

No activation or PowerShell execution-policy change is required. Installing the dependencies needs network access. The tested CUDA wheel needs a compatible NVIDIA driver; CUDA toolkit compilation is not part of this experiment. Linux and different GPU/software combinations were not independently tested. Reference checks allow 1e-4 loss difference; numerical divergence beyond that is reported rather than silently labeled reproduced.

For individual stages:

```powershell
& $Python run.py tests
& $Python run.py train --output runs/fresh_training
& $Python run.py evaluate --checkpoints runs/fresh_training --output runs/fresh_evaluation
& $Python run.py deep-dive --checkpoints runs/fresh_training --output runs/fresh_deep_dive
```

Paths passed to --output and --checkpoints are relative to your current directory, or may be absolute. The script locates bundled inputs relative to its own file, so launching run.py from another working directory also works. Training requires a new empty output directory. New runs do not alter the bundled checkpoints or reference scores.

The bundled figures and Markdown report are already generated. Optional figure regeneration uses ReportLab plus Node/sharp and is not required for training, evaluation or reproducibility verification. See tools/original_figure_generator.py; set MOE_NODE and MOE_SHARP for your machine. That helper regenerates deep-dive figures from shipped JSON and intentionally does not replace the report.

For a clone without parent-model LFS downloads:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/udisinghania/ERAv5.git
Remove-Item Env:\GIT_LFS_SKIP_SMUDGE
Set-Location 'ERAv5\Week 14\Small_Linear_MoE'
```

The small model uses regular Git blobs. Parent-model LFS pointers can remain unexpanded when only this experiment is being run.
