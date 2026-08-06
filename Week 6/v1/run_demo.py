#!/usr/bin/env python3
"""Run the complete Assignment 6 training-data execution demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "submission_artifacts"
PYTHON = Path(sys.executable).resolve()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def reset_bundle() -> None:
    resolved = BUNDLE.resolve()
    if resolved.parent != ROOT.resolve() or resolved.name != "submission_artifacts":
        raise RuntimeError("refusing to reset an unexpected evidence directory")
    if resolved.exists():
        shutil.rmtree(resolved)
    for name in ("manifests", "ledgers", "checkpoints"):
        (resolved / name).mkdir(parents=True, exist_ok=True)


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="\n")
        self.event_index = 0

    def write(self, message: str) -> None:
        self.handle.write(message.rstrip("\r\n") + "\n")
        self.handle.flush()

    def event(self, status: str, name: str, detail: str = "") -> None:
        self.event_index += 1
        suffix = f" {detail}" if detail else ""
        line = f"[{status}] {name}{suffix}"
        self.write(f"[{self.event_index:04d}] {line}")
        print(line, flush=True)

    def close(self) -> None:
        self.handle.close()


def run_command(log: RunLog, name: str, arguments: list[str]) -> None:
    command = [str(PYTHON), *arguments]
    log.event("RUN", name, "command=" + " ".join(["python", *arguments]))
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        log.write("[OUTPUT] " + line.rstrip("\r\n"))
    return_code = process.wait()
    if return_code:
        log.event("FAIL", name, f"exit_code={return_code}")
        raise RuntimeError(f"{name} failed with exit code {return_code}")
    log.event("PASS", name, "exit_code=0")


def copy(relative: str, destination: Path) -> None:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(f"required generated artifact is missing: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def package_supporting_artifacts() -> dict[str, Any]:
    tokenized = load_json("data/tokenized_v1/tokenized_report.json")
    copy("data/frozen_corpus_v1/freeze_report.json", BUNDLE / "manifests/freeze_report.json")
    copy("artifacts/tokenizer_v1/tokenizer.json", BUNDLE / "manifests/tokenizer.json")
    copy("data/tokenized_v1/tokenized_report.json", BUNDLE / "manifests/tokenized_report.json")
    for shard in tokenized["shards"]:
        copy(
            shard["manifest_path"],
            BUNDLE
            / "manifests/shards"
            / shard["lane"]
            / shard["permission"]
            / "manifest.json",
        )
    supporting = {
        "configs/training_v1.json": "manifests/training_config.json",
        "artifacts/schedule_v1/schedule.json": "manifests/schedule.json",
        "data/packed_v1/packing_report.json": "manifests/packing_report.json",
        "data/batches_v1/batch_report.json": "manifests/batch_report.json",
        "data/batches_v1/batches.jsonl.gz": "manifests/batches.jsonl.gz",
        "data/batches_v1/opus_decisions.jsonl.gz": "manifests/opus_decisions.jsonl.gz",
        "artifacts/training_v1/training_report.json": "manifests/training_report.json",
        "artifacts/recovery_v1/crash_event.json": "manifests/crash_event.json",
        "artifacts/recovery_v1/resume_event.json": "manifests/resume_event.json",
        "artifacts/recovery_audit_v1/replay_interval_report.json": "manifests/replay_interval_report.json",
        "artifacts/recovery_audit_v1/audit_report.json": "manifests/recovery_audit.json",
    }
    for source, destination in supporting.items():
        copy(source, BUNDLE / destination)

    for run_name, root in {
        "baseline": "artifacts/training_v1",
        "recovery": "artifacts/recovery_v1",
        "replay": "artifacts/replay_v1",
        "fork": "artifacts/fork_v1",
    }.items():
        copy(f"{root}/consumption_ledger.jsonl.gz", BUNDLE / f"ledgers/{run_name}_consumption.jsonl.gz")
        copy(f"{root}/learning_ledger.jsonl.gz", BUNDLE / f"ledgers/{run_name}_learning.jsonl.gz")

    checkpoints = {
        "artifacts/training_v1/final_checkpoint.pt": "baseline_final.pt",
        "artifacts/recovery_v1/checkpoint_000032.pt": "crash_update_000032.pt",
        "artifacts/recovery_v1/final_checkpoint.pt": "recovery_final.pt",
        "artifacts/replay_v1/final_checkpoint.pt": "replay_final.pt",
        "artifacts/fork_v1/final_checkpoint.pt": "fork_final.pt",
    }
    for source, name in checkpoints.items():
        copy(source, BUNDLE / "checkpoints" / name)
    copy("artifacts/training_v1/performance.json", BUNDLE / "performance.json")

    artifact_paths = [BUNDLE / "run.log", BUNDLE / "performance.json"]
    for directory in (BUNDLE / "manifests", BUNDLE / "ledgers", BUNDLE / "checkpoints"):
        artifact_paths.extend(path for path in directory.rglob("*") if path.is_file())
    entries = [
        {
            "path": path.relative_to(BUNDLE).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(artifact_paths, key=lambda item: item.relative_to(BUNDLE).as_posix())
    ]
    manifest: dict[str, Any] = {
        "schema_version": "assignment6-submission-artifact-manifest-v1",
        "artifacts": entries,
    }
    manifest["manifest_hash"] = sha256_bytes(canonical_bytes(manifest))
    atomic_json(BUNDLE / "manifests/submission_artifact_manifest.json", manifest)
    return manifest


def requirement(
    identifier: str,
    title: str,
    passed: bool,
    evidence: list[str],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "requirement": title,
        "result": "PASS" if passed else "FAIL",
        "evidence": evidence,
        "details": details or {},
    }


def build_evidence(artifact_manifest: dict[str, Any]) -> dict[str, Any]:
    frozen = load_json("data/frozen_corpus_v1/freeze_report.json")
    tokenized = load_json("data/tokenized_v1/tokenized_report.json")
    schedule = load_json("artifacts/schedule_v1/schedule.json")
    packing = load_json("data/packed_v1/packing_report.json")
    batches = load_json("data/batches_v1/batch_report.json")
    training = load_json("artifacts/training_v1/training_report.json")
    performance = load_json("artifacts/training_v1/performance.json")
    recovery = load_json("artifacts/recovery_audit_v1/audit_report.json")
    interval = load_json("artifacts/recovery_audit_v1/replay_interval_report.json")
    resume = load_json("artifacts/recovery_v1/resume_event.json")
    training_config = load_json("configs/training_v1.json")
    opus = batches["opus_summary"]
    comparisons = recovery["comparisons"]
    outcomes = opus["candidate_outcome_distribution"]
    requirements = [
        requirement(
            "portable_execution",
            "One-command execution uses the launching interpreter with CUDA-preferred CPU fallback",
            training_config["runtime"]["interpreter_policy"]
            == "launching_python_executable"
            and set(training_config["runtime"]["supported_devices"]) == {"cuda", "cpu"}
            and training["backend"]["device_type"]
            in training_config["runtime"]["supported_devices"],
            ["run.log", "manifests/training_config.json", "manifests/training_report.json"],
            {
                "selected_device_type": training["backend"]["device_type"],
                "selected_device_name": training["backend"]["device_name"],
                "supported_devices": training_config["runtime"]["supported_devices"],
                "interpreter_policy": training_config["runtime"]["interpreter_policy"],
            },
        ),
        requirement(
            "tokenizer_integrity",
            "Frozen tokenizer and immutable tokenized shards with manifests",
            tokenized["status"] == "FROZEN"
            and len(tokenized["shards"]) > 0
            and all(row["manifest_hash"].startswith("sha256:") for row in tokenized["shards"]),
            ["manifests/tokenizer.json", "manifests/tokenized_report.json", "manifests/shards/"],
            {"tokenizer_hash": tokenized["tokenizer_hash"], "shards": len(tokenized["shards"])},
        ),
        requirement(
            "evaluation_firewall",
            "Evaluation and validation data are blocked from training",
            frozen["status"] == "FROZEN"
            and frozen["rejection_counts"].get("evaluation_13gram_overlap", 0) > 0
            and training["initial_validation"]["loss_bearing_tokens"] > 0,
            ["manifests/freeze_report.json", "manifests/training_report.json", "run.log"],
            {
                "never_train_records": frozen["input_never_train_records"],
                "evaluation_13gram_rejections": frozen["rejection_counts"].get(
                    "evaluation_13gram_overlap", 0
                ),
            },
        ),
        requirement(
            "packing_correctness",
            "Packing, loss masks, attention isolation and position IDs",
            packing["status"] == "FROZEN"
            and packing["loss_bearing_tokens"] == schedule["schedule"]["total_loss_token_budget"]
            and packing["packing_utilization"] > 0
            and len(packing.get("data_type_policies", {})) == 7
            and all(int(value) > 0 for value in packing.get("policy_sequence_counts", {}).values()),
            ["manifests/packing_report.json", "manifests/batches.jsonl.gz"],
            {
                "packing_hash": packing["packing_hash"],
                "packing_utilization": packing["packing_utilization"],
                "loss_bearing_tokens": packing["loss_bearing_tokens"],
                "data_type_policies": packing.get("data_type_policies", {}),
                "policy_sequence_counts": packing.get("policy_sequence_counts", {}),
            },
        ),
        requirement(
            "mixture_compliance",
            "Curriculum stages, lane targets and protected floors",
            schedule["status"] == "FROZEN"
            and all(row["passed"] for row in schedule["schedule"]["protected_floors"].values()),
            ["manifests/schedule.json", "manifests/batch_report.json"],
            {
                "schedule_hash": schedule["schedule_hash"],
                "protected_floors": schedule["schedule"]["protected_floors"],
            },
        ),
        requirement(
            "opus_audit_trail",
            "OPUS acceptance, rejection, deferral and protected-floor override",
            all(int(outcomes.get(name, 0)) > 0 for name in ("accepted", "rejected", "deferred"))
            and int(opus["protected_floor_overrides"]) > 0,
            ["manifests/opus_decisions.jsonl.gz", "manifests/batch_report.json"],
            {
                "candidate_outcomes": outcomes,
                "protected_floor_overrides": opus["protected_floor_overrides"],
            },
        ),
        requirement(
            "consumption_ledger",
            "Complete hash-chained training consumption ledger",
            training["microbatches"] == batches["microbatches"]
            and training["loss_bearing_tokens"] == batches["loss_bearing_tokens"],
            ["ledgers/baseline_consumption.jsonl.gz", "manifests/training_report.json"],
            {"microbatches": training["microbatches"], "ledger_tail": training["ledger_tails"]["consumption"]},
        ),
        requirement(
            "learning_trace",
            "Learning ledger and sequence-level loss linked to source consumption",
            training.get("loss_tracking_granularity") == "microbatch_and_sequence"
            and training["ledger_tails"]["learning"].startswith("sha256:"),
            ["ledgers/baseline_learning.jsonl.gz", "ledgers/baseline_consumption.jsonl.gz"],
            {"granularity": training.get("loss_tracking_granularity"), "sequences": batches["sequences"]},
        ),
        requirement(
            "checkpoint_ledger_binding",
            "Checkpoints are tied to batch cursor and ledger offsets",
            recovery["planned_crash"]["completed_microbatches"] > 0
            and recovery["planned_crash"]["checkpoint_sha256"],
            ["checkpoints/crash_update_000032.pt", "manifests/crash_event.json"],
            recovery["planned_crash"],
        ),
        requirement(
            "crash_recovery",
            "Deliberate crash and resume without skipped or repeated batches",
            comparisons["planned_crash_exit_code_observed"]
            and comparisons["recovery_consumption_ledger_exact"]
            and comparisons["recovery_learning_ledger_exact"],
            ["manifests/crash_event.json", "manifests/resume_event.json", "ledgers/recovery_consumption.jsonl.gz"],
            {"crash_exit_code": recovery["planned_crash"]["observed_exit_code"]},
        ),
        requirement(
            "resume_next_batch",
            "Resumed next batch ID, sequence IDs, token spans and payload hashes",
            resume["status"] == "PASS"
            and all(
                resume[key]
                for key in (
                    "batch_id_matched",
                    "sequence_ids_matched",
                    "token_spans_matched",
                    "payload_hashes_matched",
                    "proof_hash_matched",
                )
            ),
            ["manifests/resume_event.json", "checkpoints/crash_update_000032.pt"],
            {"next_batch_index": resume["checkpoint_next_batch_index"], "proof_hash": resume["expected_next_batch"]["proof_hash"]},
        ),
        requirement(
            "historical_replay",
            "Replay of historical batch IDs, token spans and hashes",
            interval["status"] == "PASS"
            and interval["batches_replayed"] > 0
            and all(all(row["comparisons"].values()) for row in interval["entries"]),
            ["manifests/replay_interval_report.json", "ledgers/replay_consumption.jsonl.gz"],
            {
                "start": interval["interval_start_batch_index"],
                "end_exclusive": interval["interval_end_batch_index_exclusive"],
                "batches": interval["batches_replayed"],
            },
        ),
        requirement(
            "checkpoint_fork",
            "Fork from an earlier checkpoint with preserved consumption lineage",
            comparisons["fork_consumption_ledger_exact"]
            and comparisons["fork_learning_ledger_different"]
            and comparisons["fork_final_parameters_different"],
            ["checkpoints/fork_final.pt", "ledgers/fork_consumption.jsonl.gz", "manifests/recovery_audit.json"],
            recovery["fork"],
        ),
        requirement(
            "throughput",
            "Packing efficiency and useful loss-bearing tokens per second",
            performance["status"] == "PASS"
            and performance["useful_loss_bearing_tokens_per_second"] > 0
            and performance["loss_bearing_tokens"] == training["loss_bearing_tokens"],
            ["performance.json", "manifests/training_report.json"],
            {
                "device_type": performance["device_type"],
                "device": performance["device"],
                "packing_utilization": performance["packing_utilization"],
                "useful_loss_bearing_tokens_per_second": performance[
                    "useful_loss_bearing_tokens_per_second"
                ],
                "elapsed_nanoseconds": performance["elapsed_nanoseconds"],
            },
        ),
        requirement(
            "audit",
            "Independent end-to-end audit and checkpoint reload",
            recovery["status"] == "PASS" and all(comparisons.values()),
            ["manifests/recovery_audit.json", "run.log"],
            {"audit_hash": recovery["audit_hash"], "comparisons": comparisons},
        ),
    ]
    evidence: dict[str, Any] = {
        "schema_version": "assignment6-evidence-v1",
        "status": "PASS" if all(row["result"] == "PASS" for row in requirements) else "FAIL",
        "requirements": requirements,
        "artifact_manifest": "manifests/submission_artifact_manifest.json",
        "artifact_manifest_hash": artifact_manifest["manifest_hash"],
        "immutable_identifiers": {
            "corpus_hash": frozen["corpus_hash"],
            "tokenizer_hash": tokenized["tokenizer_hash"],
            "schedule_hash": schedule["schedule_hash"],
            "packing_hash": packing["packing_hash"],
            "batch_plan_hash": batches["batch_plan_hash"],
            "training_hash": training["training_hash"],
            "final_parameter_hash": training["final_parameter_hash"],
            "recovery_audit_hash": recovery["audit_hash"],
        },
    }
    evidence["evidence_hash"] = sha256_bytes(canonical_bytes(evidence))
    return evidence


def evidence_markdown(evidence: dict[str, Any]) -> str:
    rows = [
        "# Assignment 6 generated evidence",
        "",
        "This report is generated from the manifests, ledgers, checkpoints and performance report produced by `python run_demo.py`.",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---|---|",
    ]
    for item in evidence["requirements"]:
        rows.append(
            f"| {item['requirement']} | **{item['result']}** | "
            + ", ".join(f"`{path}`" for path in item["evidence"])
            + " |"
        )
    rows.extend(
        [
            "",
            "## Run identity",
            "",
            *[
                f"- {name.replace('_', ' ').title()}: `{value}`"
                for name, value in evidence["immutable_identifiers"].items()
            ],
            "",
            "See `run.log` for the complete event sequence and command output. Every supporting file is hashed by `manifests/submission_artifact_manifest.json`.",
            "",
        ]
    )
    return "\n".join(rows)


def execute_pipeline(log: RunLog) -> None:
    commands = [
        ("runtime_preflight", ["scripts/verify_runtime_v1.py"]),
        ("foundation_tests", ["-m", "unittest", "discover", "-s", "tests", "-q"]),
        ("source_documents_curated", ["scripts/curate_corpus.py"]),
        ("source_documents_verified", ["scripts/verify_corpus.py"]),
        ("wikipedia_review_gate_verified", ["scripts/verify_wikipedia_v3.py"]),
        ("remaining_lanes_review_gate_verified", ["scripts/verify_remaining_lanes_v2.py"]),
        ("corpus_frozen", ["scripts/freeze_corpus_v1.py"]),
        ("frozen_corpus_verified", ["scripts/verify_frozen_corpus_v1.py"]),
        ("tokenizer_trained", ["scripts/train_tokenizer_v1.py"]),
        ("tokenized_shards_created", ["scripts/tokenize_shards_v1.py"]),
        ("tokenized_shards_verified", ["scripts/verify_tokenized_v1.py"]),
        ("mixture_schedule_compiled", ["scripts/build_curriculum_schedule_v1.py"]),
        ("mixture_schedule_verified", ["scripts/verify_curriculum_schedule_v1.py"]),
        ("batches_packed", ["scripts/build_packed_v1.py"]),
        ("packing_verified", ["scripts/verify_packed_v1.py"]),
        ("opus_batches_built", ["scripts/build_batches_v1.py"]),
        ("opus_batches_verified", ["scripts/verify_batches_v1.py"]),
        ("model_training_executed", ["scripts/train_tiny_model_v1.py"]),
        ("training_ledgers_verified", ["scripts/verify_training_v1.py"]),
        ("crash_resume_replay_fork_executed", ["scripts/run_recovery_demo_v1.py"]),
        ("recovery_audit_verified", ["scripts/verify_recovery_v1.py"]),
    ]
    for name, command in commands:
        run_command(log, name, command)
        if name == "runtime_preflight":
            log.event(
                "PASS",
                "portable_runtime_verified",
                f"python={PYTHON.name} device_request={os.environ['ERA6_DEVICE']}",
            )
        elif name == "frozen_corpus_verified":
            log.event("PASS", "eval_shard_blocked")
            log.event("PASS", "evaluation_data_blocked")
        elif name == "tokenized_shards_verified":
            tokenized = load_json("data/tokenized_v1/tokenized_report.json")
            log.event("PASS", "shards_created", f"count={len(tokenized['shards'])}")
            log.event("PASS", "manifests_validated", f"count={len(tokenized['shards'])}")
            log.event("PASS", "tokenizer_hash_verified", tokenized["tokenizer_hash"])
        elif name == "mixture_schedule_verified":
            log.event("PASS", "mixture_compiled")
        elif name == "packing_verified":
            packing = load_json("data/packed_v1/packing_report.json")
            log.event("PASS", "batches_packed", f"sequences={packing['sequences']}")
        elif name == "opus_batches_verified":
            report = load_json("data/batches_v1/batch_report.json")
            log.event("PASS", "OPUS_decisions_recorded", f"decisions={report['opus_decisions']}")
        elif name == "training_ledgers_verified":
            training = load_json("artifacts/training_v1/training_report.json")
            performance = load_json("artifacts/training_v1/performance.json")
            log.event("PASS", "checkpoint_saved", training["paths"]["final_checkpoint"])
            log.event(
                "PASS",
                "performance_measured",
                f"useful_loss_tokens_per_second={performance['useful_loss_bearing_tokens_per_second']:.6f}",
            )
        elif name == "recovery_audit_verified":
            audit = load_json("artifacts/recovery_audit_v1/audit_report.json")
            log.event("PASS", "crash_simulated", f"exit_code={audit['planned_crash']['observed_exit_code']}")
            log.event("PASS", "run_resumed")
            log.event("PASS", "resume_next_batch_matched", f"batch_index={audit['resume_next_batch']['batch_index']}")
            log.event("PASS", "historical_stream_replayed", f"batches={audit['replay_interval']['batches']}")
            log.event("PASS", "replay_hash_matched", audit["replay_interval"]["report_hash"])
            log.event("PASS", "branch_forked", f"first_divergent_microbatch={audit['fork']['first_divergent_microbatch']}")
            log.event("PASS", "audit_completed", audit["audit_hash"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and verify the complete Assignment 6 demonstration."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device; auto prefers CUDA and falls back to CPU.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["ERA6_DEVICE"] = args.device
    try:
        reset_bundle()
        log = RunLog(BUNDLE / "run.log")
        try:
            log.event("START", "assignment6_complete_demonstration")
            execute_pipeline(log)
            log.event("PASS", "complete_execution_sequence")
        finally:
            log.close()
        artifact_manifest = package_supporting_artifacts()
        evidence = build_evidence(artifact_manifest)
        atomic_json(BUNDLE / "evidence.json", evidence)
        atomic_text(BUNDLE / "evidence.md", evidence_markdown(evidence))
        if evidence["status"] != "PASS":
            raise RuntimeError("generated evidence contains a failed requirement")
        result = subprocess.run(
            [str(PYTHON), "scripts/verify_submission_bundle.py"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode:
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
            return result.returncode
        print("[PASS] complete Assignment 6 demonstration and evidence bundle generated")
        return 0
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
