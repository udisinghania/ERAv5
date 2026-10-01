# Portable setup and reproduction

Use Python 3.12 and the pinned `requirements.txt`. The recorded GPU runs used PyTorch 2.5.1 with CUDA 12.1. Install that wheel first from `https://download.pytorch.org/whl/cu121`, then the requirements. CPU review can use a CPU PyTorch build. Full evaluation and training require CUDA with BF16 support; the tested device has 8 GiB. Training deliberately stops on an unsupported GPU instead of silently changing precision. The notebook's inspection-only cells work without a GPU.

After cloning the ERAv5 repository, change into **Week 14**, run `git lfs install`, then `git lfs pull --include="Week 14/**/*.bin,Week 14/**/*.pt"`. LFS placeholders are not data: `python run.py verify` checks sizes and SHA-256 and fails if they have not been downloaded. The package is self-contained under Week 14. No personal machine paths are needed by `run.py`.

## Quick review

```text
python run.py verify
python run.py summary
python run.py tests
python run.py check --device cpu
python run.py evaluate --kind moe
python run.py evaluate --kind dense
python run.py evaluate --kind pretrained
```

The first command needs only Python's standard library. Tests and loading need PyTorch and NumPy. Tokenization uses the bundled frozen byte-BPE implementation and tokenizer JSON; it is not a generic Hugging Face tokenizer JSON. `check` verifies round-trip and token-ID parity, strict state loading, exact parameter counts, finite weights and forward execution. `evaluate` uses the complete saved holdout, not a small sample, and records the difference from the historical score.

## Reproduce the continuation from the supplied dense checkpoint

Each command is run from Week 14; paths also work from another current directory when `run.py` is passed as an absolute path. These are real training commands, not placeholders.

```text
python run.py tests --output runs/reproduction
python run.py train --phase stage1 --action prepare --output runs/reproduction
python run.py train --phase stage1 --action train --kind moe --output runs/reproduction
python run.py train --phase stage1 --action train --kind dense --output runs/reproduction
python run.py train --phase stage1 --action audit --output runs/reproduction
python run.py train --phase stage2 --action prepare --output runs/reproduction
python run.py train --phase stage2 --action train --kind moe --output runs/reproduction
python run.py train --phase stage2 --action train --kind dense --output runs/reproduction
python run.py train --phase stage2 --action diagnostics --output runs/reproduction
```

Stage 1 preparation verifies input hashes and initial dense/MoE equivalence, saves the exact token plan and benchmarks microbatches 8 and 16. The historical run selected 16 for both arms. Stage 2 uses 16; it requires that memory capacity. Report any microbatch changes as a changed experiment because the router balance objective is microbatch-scoped. `stage1 audit` requires successful semantic tests and validates the two completed parents before extension. The extension preserves AdamW state and exactly complements the 10M boundary mask.

Add `--resume` to an interrupted **train** command to restore its last committed optimizer checkpoint. Completed runs refuse to overwrite. Keep the same output directory, data paths and code while resuming. Choose a new output directory to repeat an experiment. The original run has a detailed recovery record; a new run has its own logs.

## Optional dense pretraining from scratch

The provided 50M dense pretraining checkpoint is sufficient to reproduce the upcycling assignment. Its original source, logs and results are in `evidence/pretraining`. To also repeat the original pretraining:

```text
python run.py train --phase pretrain --action prepare --output runs/from_scratch
python run.py train --phase pretrain --action train --output runs/from_scratch
```

This benchmarks batch capacity and trains for all 50M targets. The historical dense microbatch was 32, global batch 32. Validation uses the packaged, already segmented arrays; raw-source acquisition is unnecessary.

To upcycle **this newly trained dense model**, add `--prior runs/from_scratch/pretrain` to every continuation command above (and use `--output runs/from_scratch`). The supplied validation remains common to all runs. The runner does not silently switch to newly generated weights. It uses the recorded dense checkpoint unless `--prior` is explicitly supplied.

## Portable paths and outputs

| Option / location | Meaning |
|---|---|
| `--data-root PATH` | Folder containing `Corpus_20M_v1` and `Run_20M_50M_v1`; defaults to packaged `data` |
| `--prior PATH` | Dense run containing `config.json`, `result.json`, `checkpoints/final.pt`; defaults to packaged original dense run |
| `--output PATH` | New writable working folder; defaults to `runs/reproduction` |
| `src/` | Portable source copies; only path handling changed from training originals |
| `evidence/` | Original recorded source and results, including historical absolute paths and manifests |
| `MANIFEST.json` | Current package's relative file hashes; use this after relocation |
| `data/.../catalog.json` | Historical shard provenance; raw shard files referenced there are not required or bundled |

The runner copies source into the chosen output directory, where the original scripts write checkpoints/logs. It refuses to replace modified code during resume. The dense pretraining's validation loader reads packaged arrays; tokenizer code is extracted unchanged from the original builder. `source_adaptations.json` identifies exactly which training files had path changes. Architectures and numerical training algorithms were not rewritten for packaging.

The package contains all training-ready inputs and final weights. It does not contain every raw document/intermediate shard or historical AdamW checkpoint. The corpus acquisition script is retained as historical evidence, not advertised as a fresh downloadable dataset pipeline. All source dataset cards and acquisition revision records available in the corpus are retained under `data/Corpus_20M_v1/sources`.

## Colab

Open `Session_14_MoE.ipynb`. Its setup clones the repository with automatic LFS downloads disabled, selects `Week 14`, then downloads only that folder's data and checkpoints. Inspection outputs saved in the notebook were generated locally, not on Colab. Optional training cells are disabled by default because they run full experiments. Set the switches to true to run them. For persistence, set the output folder to a mounted Google Drive directory before training. A T4 runtime does not support the selected BF16 workflow; select a compatible GPU. No full Colab training rerun is claimed.
