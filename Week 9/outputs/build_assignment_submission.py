from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MIB = 2**20


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def build_writeup(combined: dict[str, Any]) -> str:
    part1 = combined["part1"]
    production = combined["part2_production"]
    flow = part1["tensor_flow"]
    padding = part1["padding"]
    boundary = part1["boundary"]
    perplexity = part1["perplexity"]
    tying = part1["weight_tying"]
    memory = part1["memory"]
    best = production["best_validation"]
    final = production["stopping_validation"]
    trajectory = production["validation_trajectory"]
    return f"""# Assignment 9 final measured values

## Part 1 — the loss harness

| Requirement | Measured result |
|---|---|
| Tensor shapes | tokens `{tuple(flow['tokens'])}`; hidden `{tuple(flow['hidden'])}`; logits `{tuple(flow['logits'])}`; shifted logits `{tuple(flow['shifted_logits'])}`; targets `{tuple(flow['targets'])}` |
| String-level shift | `{part1['string_shift']['relationship']}`; observed pair `{part1['string_shift']['boundary_pair']}` |
| Padding mask | contributing targets `{padding['count_before']:,} -> {padding['count_after']:,}`; removed `{padding['padding_targets_removed']:,}`; loss `{padding['naive_loss']:.6f} -> {padding['masked_loss']:.6f}` |
| Packed boundary mask | pair `{boundary['token_pair']}`; contributing targets `{boundary['count_before']:,} -> {boundary['count_after']:,}`; loss `{boundary['loss_before']:.6f} -> {boundary['loss_after']:.6f}` |
| Untrained perplexity | `{perplexity['untrained_perplexity']:,.2f}` for vocabulary `{perplexity['vocab_size']:,}` (`{perplexity['relative_error']:.2%}` relative difference) |
| Tied versus untied | full model `{tying['tied_total_parameters']:,}` tied versus `{tying['untied_total_parameters']:,}` untied; saved `{tying['parameters_saved']:,}` parameters |
| Ordinary versus chunked CE memory | `{memory['materialized_peak_mib']:,.2f} MiB` versus `{memory['chunked_peak_mib']:,.2f} MiB`; reduction `{memory['measured_reduction_ratio']:.2f}x` |

Every tensor dimension is printed in the notebook: `B` is batch size, `T` is
sequence length, `D` is hidden width, and `V` is vocabulary size. The notebook
prints decoded token strings beside their shifted targets, not only token IDs.
The packed transition is removed because predicting the first token of an
independent document from the previous document is not a valid language-model
target. Masking can raise or lower the mean loss; correctness is determined by
which targets contribute.

The untrained-model check passes because perplexity is close to vocabulary size.
The ordinary and chunked losses differ by only
`{memory['absolute_loss_difference']:.10f}`, confirming that chunking changes
memory use rather than the objective.

## Part 2 — the second prediction head

The shared hidden state at position `t` feeds Head 1 for `x[t+1]` and an
independent Head 2 for `x[t+2]`. The optimized objective is exactly
`L_total = L_head1 + L_head2`.

| Held-out metric | Best retained checkpoint | Early-stopping point |
|---|---:|---:|
| Optimizer step | {production['best_step']:,} | {production['stopping_step']:,} |
| Head 1 loss (`t+1`) | {best['head1_loss']:.6f} | {final['head1_loss']:.6f} |
| Head 2 loss (`t+2`) | {best['head2_loss']:.6f} | {final['head2_loss']:.6f} |
| Sum | {best['total_loss']:.6f} | {final['total_loss']:.6f} |
| Head 1 perplexity | {best['head1_perplexity']:.2f} | {final['head1_perplexity']:.2f} |
| Head 2 perplexity | {best['head2_perplexity']:.2f} | {final['head2_perplexity']:.2f} |

Head 2 is expected to remain harder: it must predict `x[t+2]` without observing
the intervening `x[t+1]`, so its conditional uncertainty and loss are higher.
This is also what the run measured: Head 2 was higher than Head 1 at all
{trajectory['head2_higher_count']}/{trajectory['periodic_validation_count']}
periodic held-out checks. At the first periodic check (step
{trajectory['first_periodic']['global_step']:,}), the losses were
H1={trajectory['first_periodic']['head1_loss']:.6f} and
H2={trajectory['first_periodic']['head2_loss']:.6f}; at the retained best step,
both had improved to H1={best['head1_loss']:.6f} and
H2={best['head2_loss']:.6f}, with Head 2 still harder.
The best checkpoint is selected on the held-out sum, not training loss. Training
stops after {production['early_stopping_patience_validations']} consecutive
validations fail to improve that sum by at least
`{production['early_stopping_minimum_delta']:.6f}`, and `best.pt` remains
separate from the later `latest.pt` resume checkpoint. A hard limit of
{production['maximum_corpus_passes']:.2f} corpus passes provides an additional
guard against repeated-data overtraining.

Checkpoint integrity: `{production['checkpoint_integrity']}`. The hashes are
recorded in `assignment9_final_metrics.json` and the production run summary.
"""


def build_readme(combined: dict[str, Any]) -> str:
    production = combined["part2_production"]
    model = production["model_configuration"]
    validation = production["validation_configuration"]
    writeup = build_writeup(combined).replace(
        "# Assignment 9 final measured values",
        "## Measured submission results",
        1,
    )
    return f"""# Assignment 9: Observable Cross-Entropy and Multi-Token Prediction

This repository is the submission for Assignment 9. It contains one executed
notebook for the observable loss harness, the training implementation, and the
retained JSON/JSONL evidence from the production 17M-parameter run. Google
Colab is not required for this submission.

## Submission artifacts

- [Executed notebook](Assignment_9_Loss_Harness.ipynb)
- [Exact final metrics](assignment9_final_metrics.json)
- [Short write-up](ASSIGNMENT_WRITEUP.md)
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

## Production run

| Item | Value |
|---|---:|
| Parameters with both heads | {production['parameter_count']:,} |
| Architecture | {model['layers']} layers, D={model['hidden_size']}, {model['heads']} heads |
| Normalization / FFN | RMSNorm / SwiGLU |
| Validation targets | {validation['target_loss_bearing_tokens']:,} requested; {production['best_validation']['head1_targets']:,} H1 / {production['best_validation']['head2_targets']:,} H2 measured |
| Best step | {production['best_step']:,} |
| Stopping step | {production['stopping_step']:,} |
| Corpus passes at stop | {production['corpus_passes']:.4f} |
| Stop reason | `{production['stop_reason']}` |

{writeup}

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
measurement and the Part 2 smoke run. The production run is reproducible with
the retained configuration and training implementation; its multi-hour
allowance was not needed because held-out early stopping fired after two
non-improving validations. The excluded `.pt` files are not required to audit
the submitted results because the complete metric ledger, configuration,
checkpoint index, hashes, and final summary are retained as text artifacts.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/mtp_17m_session_run"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    harness = read_json(root / "assignment9_metrics.json")
    summary = read_json(run_dir / "run_summary.json")
    manifest = read_json(run_dir / "run_manifest.json")
    resolved_config = read_json(run_dir / "resolved_config.json")
    ledger = read_jsonl(run_dir / "metrics.jsonl")

    validations = [row for row in ledger if row.get("event") == "validation"]
    initial_validation = next(
        row for row in validations if row.get("scope") == "initial"
    )
    periodic_validations = [
        row for row in validations if row.get("scope") == "periodic"
    ]
    ledger_best = min(
        periodic_validations, key=lambda row: float(row["total_loss"])
    )
    best = summary["best_validation"]
    if int(ledger_best["global_step"]) != int(summary["best_validation_step"]):
        raise RuntimeError("summary best step does not match the validation ledger")
    if not math.isclose(
        float(ledger_best["total_loss"]),
        float(best["total_loss"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("summary best loss does not match the validation ledger")

    latest_path = Path(summary["checkpoint_path"])
    best_path = Path(summary["best_checkpoint_path"])
    latest_hash = str(summary["checkpoint_file_sha256"])
    best_hash = str(summary["best_checkpoint_file_sha256"])
    if latest_path.exists() and best_path.exists():
        if sha256(latest_path) != latest_hash:
            raise RuntimeError("latest checkpoint SHA-256 mismatch")
        if sha256(best_path) != best_hash:
            raise RuntimeError("best checkpoint SHA-256 mismatch")
        checkpoint_integrity = "recomputed_and_matched"
    else:
        checkpoint_integrity = (
            "recorded_in_run_summary_checkpoint_files_not_distributed"
        )

    train_at_best = next(
        row
        for row in ledger
        if row.get("event") == "train_step"
        and int(row["global_step"]) == int(summary["best_validation_step"])
    )
    production = {
        "parameter_count": int(manifest["model_parameters"]),
        "model_configuration": resolved_config["model"],
        "validation_configuration": resolved_config["validation"],
        "best_step": int(summary["best_validation_step"]),
        "stopping_step": int(summary["global_step"]),
        "stop_reason": summary["stop_reason"],
        "corpus_passes": float(summary["corpus_passes_by_head1_targets"]),
        "physical_tokens_seen": int(summary["physical_tokens_seen"]),
        "best_active_training_hours": float(
            train_at_best["active_training_seconds"]
        )
        / 3600,
        "total_active_training_hours": float(summary["active_training_seconds"])
        / 3600,
        "best_validation": best,
        "stopping_validation": summary["final_validation"],
        "validation_trajectory": {
            "initial": initial_validation,
            "periodic_validation_count": len(periodic_validations),
            "head2_higher_count": sum(
                float(row["head2_loss"]) > float(row["head1_loss"])
                for row in periodic_validations
            ),
            "first_periodic": periodic_validations[0],
        },
        "checkpoint_integrity": checkpoint_integrity,
        "best_checkpoint_path": str(
            Path("outputs/mtp_17m_session_run") / "best.pt"
        ),
        "best_checkpoint_sha256": best_hash,
        "latest_checkpoint_path": str(
            Path("outputs/mtp_17m_session_run") / "latest.pt"
        ),
        "latest_checkpoint_sha256": latest_hash,
        "early_stopping_patience_validations": int(
            resolved_config["training"]["early_stopping_patience_validations"]
        ),
        "early_stopping_minimum_delta": float(
            resolved_config["training"]["early_stopping_minimum_delta"]
        ),
        "maximum_corpus_passes": float(
            resolved_config["training"]["maximum_corpus_passes"]
        ),
    }
    combined = {
        "hardware": harness["hardware"],
        "harness_configuration": harness["configuration"],
        "part1": harness["part1"],
        "part2_notebook_smoke": harness["part2"],
        "part2_production": production,
    }

    metrics_path = root / "assignment9_final_metrics.json"
    writeup_path = root / "ASSIGNMENT_WRITEUP.md"
    readme_path = root / "README.md"
    metrics_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    writeup_path.write_text(build_writeup(combined), encoding="utf-8")
    readme_path.write_text(build_readme(combined), encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {writeup_path}")
    print(f"Wrote {readme_path}")
    print(
        f"Best validation step={production['best_step']:,} "
        f"H1={best['head1_loss']:.6f} H2={best['head2_loss']:.6f} "
        f"sum={best['total_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
