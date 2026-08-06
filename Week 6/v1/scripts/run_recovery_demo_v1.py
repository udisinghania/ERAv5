from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import atomic_write_json, canonical_json_bytes, read_jsonl_gz, sha256_bytes, sha256_file  # noqa: E402
from era6.training import PackedBatchStore, batch_reconstruction_proof  # noqa: E402


def run_worker(arguments: list[str], expected_code: int = 0) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    command = [sys.executable, str(ROOT / "scripts" / "resumable_train_v1.py"), *arguments]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(result.stdout, end="", flush=True)
    if result.returncode != expected_code:
        raise RuntimeError(
            f"worker returned {result.returncode}, expected {expected_code}: {' '.join(arguments)}"
        )
    return result


def load_report(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    recovery_config_path = ROOT / "configs" / "recovery_v1.json"
    config = json.loads(recovery_config_path.read_text(encoding="utf-8"))
    crash_update = int(config["deliberate_crash_after_optimizer_update"])
    exit_code = int(config["planned_crash_exit_code"])
    checkpoint_every = int(config["checkpoint_every_optimizer_updates"])
    crash_checkpoint = f"artifacts/recovery_v1/checkpoint_{crash_update:06d}.pt"

    crash_result = run_worker(
        [
            "--output", "artifacts/recovery_v1",
            "--run-kind", "recovery",
            "--fresh",
            "--checkpoint-every", str(checkpoint_every),
            "--crash-after-update", str(crash_update),
            "--preserve-update", str(crash_update),
            "--planned-exit-code", str(exit_code),
        ],
        expected_code=exit_code,
    )
    run_worker(
        [
            "--output", "artifacts/recovery_v1",
            "--run-kind", "recovery",
            "--resume-from", crash_checkpoint,
            "--checkpoint-every", str(checkpoint_every),
            "--preserve-update", str(crash_update),
        ]
    )
    run_worker(
        [
            "--output", "artifacts/replay_v1",
            "--run-kind", "replay",
            "--fresh",
        ]
    )
    fork_scale = float(config["fork"]["learning_rate_scale_after_fork"])
    run_worker(
        [
            "--output", "artifacts/fork_v1",
            "--run-kind", "fork",
            "--fresh",
            "--fork-from", crash_checkpoint,
            "--learning-rate-scale", str(fork_scale),
        ]
    )

    baseline = load_report("artifacts/training_v1/training_report.json")
    recovery = load_report("artifacts/recovery_v1/execution_report.json")
    replay = load_report("artifacts/replay_v1/execution_report.json")
    fork = load_report("artifacts/fork_v1/execution_report.json")
    crash_event = load_report("artifacts/recovery_v1/crash_event.json")
    resume_event = load_report("artifacts/recovery_v1/resume_event.json")
    batch_report = load_report("data/batches_v1/batch_report.json")
    packing_report = load_report("data/packed_v1/packing_report.json")
    baseline_learning = list(read_jsonl_gz(ROOT / baseline["paths"]["learning_ledger"]))
    baseline_consumption = list(
        read_jsonl_gz(ROOT / baseline["paths"]["consumption_ledger"])
    )
    replay_consumption = list(
        read_jsonl_gz(ROOT / replay["paths"]["consumption_ledger"])
    )
    fork_learning = list(read_jsonl_gz(ROOT / fork["paths"]["learning_ledger"]))
    common_learning_prefix = 0
    for base_row, fork_row in zip(baseline_learning, fork_learning, strict=True):
        if base_row != fork_row:
            break
        common_learning_prefix += 1
    replay_config = config["replay_requirements"]
    interval_start = int(replay_config["interval_start_batch_index"])
    interval_end = int(replay_config["interval_end_batch_index_exclusive"])
    batches = list(read_jsonl_gz(ROOT / batch_report["paths"]["batches"]))
    store = PackedBatchStore(ROOT, packing_report)
    interval_entries = []
    for batch_index in range(interval_start, interval_end):
        batch = batches[batch_index]
        original = baseline_consumption[batch_index]
        replayed = replay_consumption[batch_index]
        reconstructed = batch_reconstruction_proof(batch, store)
        entry_comparisons = {
            "batch_id_matched": original["batch_index"]
            == replayed["batch_index"]
            == reconstructed["batch_index"],
            "sequence_ids_matched": original["sequence_indices"]
            == replayed["sequence_indices"]
            == reconstructed["sequence_indices"],
            "token_spans_matched": original["token_spans"]
            == replayed["token_spans"]
            == reconstructed["token_spans"],
            "payload_hashes_matched": original["payload_hashes"]
            == replayed["payload_hashes"]
            == reconstructed["payload_hashes"],
            "proof_hashes_matched": original["batch_reconstruction_proof_hash"]
            == replayed["batch_reconstruction_proof_hash"]
            == reconstructed["proof_hash"],
        }
        interval_entries.append(
            {
                "batch_index": batch_index,
                "status": "PASS" if all(entry_comparisons.values()) else "FAIL",
                "sequence_indices": reconstructed["sequence_indices"],
                "token_spans": reconstructed["token_spans"],
                "original_payload_hashes": original["payload_hashes"],
                "replay_payload_hashes": replayed["payload_hashes"],
                "reconstruction_proof_hash": reconstructed["proof_hash"],
                "comparisons": entry_comparisons,
            }
        )
    interval_report = {
        "schema_version": 1,
        "status": "PASS"
        if interval_entries and all(row["status"] == "PASS" for row in interval_entries)
        else "FAIL",
        "interval_start_batch_index": interval_start,
        "interval_end_batch_index_exclusive": interval_end,
        "batches_replayed": len(interval_entries),
        "crosses_crash_boundary": interval_start
        < int(crash_event["completed_microbatches"])
        < interval_end,
        "entries": interval_entries,
    }
    interval_report["interval_report_hash"] = f"sha256:{sha256_bytes(canonical_json_bytes(interval_report))}"
    output = ROOT / "artifacts" / "recovery_audit_v1"
    output.mkdir(parents=True, exist_ok=True)
    interval_path = output / "replay_interval_report.json"
    atomic_write_json(interval_path, interval_report)

    comparisons = {
        "planned_crash_exit_code_observed": crash_result.returncode == exit_code,
        "crash_checkpoint_hash_valid": sha256_file(ROOT / crash_event["checkpoint"])
        == crash_event["checkpoint_sha256"],
        "recovery_final_parameters_exact": recovery["final_parameter_hash"]
        == baseline["final_parameter_hash"],
        "recovery_consumption_ledger_exact": recovery["component_hashes"]["consumption_ledger"]
        == baseline["component_hashes"]["consumption_ledger"],
        "recovery_learning_ledger_exact": recovery["component_hashes"]["learning_ledger"]
        == baseline["component_hashes"]["learning_ledger"],
        "replay_final_parameters_exact": replay["final_parameter_hash"]
        == baseline["final_parameter_hash"],
        "replay_consumption_ledger_exact": replay["component_hashes"]["consumption_ledger"]
        == baseline["component_hashes"]["consumption_ledger"],
        "replay_learning_ledger_exact": replay["component_hashes"]["learning_ledger"]
        == baseline["component_hashes"]["learning_ledger"],
        "fork_consumption_ledger_exact": fork["component_hashes"]["consumption_ledger"]
        == baseline["component_hashes"]["consumption_ledger"],
        "fork_learning_ledger_different": fork["component_hashes"]["learning_ledger"]
        != baseline["component_hashes"]["learning_ledger"],
        "fork_final_parameters_different": fork["final_parameter_hash"]
        != baseline["final_parameter_hash"],
        "fork_prefix_preserved_through_checkpoint": common_learning_prefix
        >= int(crash_event["completed_microbatches"]),
        "fork_diverges_at_next_optimizer_update": common_learning_prefix < len(fork_learning)
        and bool(fork_learning[common_learning_prefix]["optimizer_update_applied"])
        and int(fork_learning[common_learning_prefix]["optimizer_update_index"])
        == int(crash_event["optimizer_updates"]) + 1,
        "resume_next_batch_id_matched": bool(resume_event["batch_id_matched"]),
        "resume_next_batch_sequence_ids_matched": bool(
            resume_event["sequence_ids_matched"]
        ),
        "resume_next_batch_token_spans_matched": bool(
            resume_event["token_spans_matched"]
        ),
        "resume_next_batch_payload_hashes_matched": bool(
            resume_event["payload_hashes_matched"]
        ),
        "resume_next_batch_proof_hash_matched": bool(resume_event["proof_hash_matched"]),
        "replay_interval_batch_ids_matched": all(
            row["comparisons"]["batch_id_matched"] for row in interval_entries
        ),
        "replay_interval_sequence_ids_matched": all(
            row["comparisons"]["sequence_ids_matched"] for row in interval_entries
        ),
        "replay_interval_token_spans_matched": all(
            row["comparisons"]["token_spans_matched"] for row in interval_entries
        ),
        "replay_interval_payload_hashes_matched": all(
            row["comparisons"]["payload_hashes_matched"] for row in interval_entries
        ),
        "replay_interval_proof_hashes_matched": all(
            row["comparisons"]["proof_hashes_matched"] for row in interval_entries
        ),
    }
    if not all(comparisons.values()):
        raise RuntimeError(f"recovery audit comparison failed: {comparisons}")
    report_hashes = {
        "baseline_training_report": sha256_file(ROOT / "artifacts/training_v1/training_report.json"),
        "recovery_execution_report": sha256_file(ROOT / "artifacts/recovery_v1/execution_report.json"),
        "replay_execution_report": sha256_file(ROOT / "artifacts/replay_v1/execution_report.json"),
        "fork_execution_report": sha256_file(ROOT / "artifacts/fork_v1/execution_report.json"),
        "crash_event": sha256_file(ROOT / "artifacts/recovery_v1/crash_event.json"),
        "resume_event": sha256_file(ROOT / "artifacts/recovery_v1/resume_event.json"),
        "replay_interval_report": sha256_file(interval_path),
    }
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "recovery_config_sha256": sha256_file(recovery_config_path),
        "planned_crash": {
            "optimizer_update": crash_update,
            "observed_exit_code": crash_result.returncode,
            "completed_microbatches": crash_event["completed_microbatches"],
            "checkpoint": crash_event["checkpoint"],
            "checkpoint_sha256": crash_event["checkpoint_sha256"],
        },
        "baseline_final_parameter_hash": baseline["final_parameter_hash"],
        "recovery_final_parameter_hash": recovery["final_parameter_hash"],
        "replay_final_parameter_hash": replay["final_parameter_hash"],
        "fork_final_parameter_hash": fork["final_parameter_hash"],
        "fork": {
            "learning_rate_scale": fork_scale,
            "common_learning_prefix_microbatches": common_learning_prefix,
            "first_divergent_microbatch": common_learning_prefix,
            "parent_checkpoint_sha256": fork["lineage"]["parent_checkpoint_sha256"],
        },
        "resume_next_batch": {
            "batch_index": resume_event["checkpoint_next_batch_index"],
            "proof_hash": resume_event["expected_next_batch"]["proof_hash"],
            "event": "artifacts/recovery_v1/resume_event.json",
        },
        "replay_interval": {
            "start_batch_index": interval_start,
            "end_batch_index_exclusive": interval_end,
            "batches": len(interval_entries),
            "crosses_crash_boundary": interval_report["crosses_crash_boundary"],
            "report_hash": interval_report["interval_report_hash"],
            "report": interval_path.relative_to(ROOT).as_posix(),
        },
        "comparisons": comparisons,
        "report_hashes": report_hashes,
    }
    audit["audit_hash"] = f"sha256:{sha256_bytes(canonical_json_bytes(audit))}"
    atomic_write_json(output / "audit_report.json", audit)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
