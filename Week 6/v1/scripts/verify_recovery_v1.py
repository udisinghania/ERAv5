from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import canonical_json_bytes, read_jsonl_gz, sha256_bytes, sha256_file  # noqa: E402
from era6.model_torch import TinyDecoderTransformer, state_dict_hash  # noqa: E402
from era6.training import verify_hash_chain  # noqa: E402
from era6.training import PackedBatchStore, batch_reconstruction_proof  # noqa: E402


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def checkpoint_parameter_hash(report: dict, config: dict) -> str:
    checkpoint = torch.load(ROOT / report["paths"]["final_checkpoint"], map_location="cpu", weights_only=False)
    model = TinyDecoderTransformer(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    value = state_dict_hash(model)
    if value != report["final_parameter_hash"]:
        raise AssertionError(f"checkpoint parameter hash mismatch: {report['run_kind']}")
    if checkpoint["execution_state"]["next_batch_index"] != report["microbatches"]:
        raise AssertionError("final checkpoint batch cursor mismatch")
    if checkpoint["execution_state"]["optimizer_updates"] != report["optimizer_updates"]:
        raise AssertionError("final checkpoint optimizer cursor mismatch")
    if checkpoint.get("next_batch_proof") is not None:
        raise AssertionError("completed checkpoint unexpectedly has a next-batch proof")
    if checkpoint["execution_state"].get("execution_backend") != report["backend"]:
        raise AssertionError("final checkpoint execution backend mismatch")
    if "rng_state" not in checkpoint:
        raise AssertionError("final checkpoint is missing portable RNG state")
    return value


def main() -> int:
    audit_path = ROOT / "artifacts" / "recovery_audit_v1" / "audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_body = {key: value for key, value in audit.items() if key != "audit_hash"}
    if audit["audit_hash"] != f"sha256:{sha256_bytes(canonical_json_bytes(audit_body))}":
        raise AssertionError("recovery audit hash mismatch")
    if audit["status"] != "PASS" or not all(audit["comparisons"].values()):
        raise AssertionError("recovery audit contains a failed comparison")
    report_paths = {
        "baseline_training_report": "artifacts/training_v1/training_report.json",
        "recovery_execution_report": "artifacts/recovery_v1/execution_report.json",
        "replay_execution_report": "artifacts/replay_v1/execution_report.json",
        "fork_execution_report": "artifacts/fork_v1/execution_report.json",
        "crash_event": "artifacts/recovery_v1/crash_event.json",
        "resume_event": "artifacts/recovery_v1/resume_event.json",
        "replay_interval_report": "artifacts/recovery_audit_v1/replay_interval_report.json",
    }
    for name, relative in report_paths.items():
        if sha256_file(ROOT / relative) != audit["report_hashes"][name]:
            raise AssertionError(f"audited report changed: {name}")
    config = load("configs/training_v1.json")
    recovery_config = load("configs/recovery_v1.json")
    baseline = load(report_paths["baseline_training_report"])
    recovery = load(report_paths["recovery_execution_report"])
    replay = load(report_paths["replay_execution_report"])
    fork = load(report_paths["fork_execution_report"])
    crash = load(report_paths["crash_event"])
    resume_event = load(report_paths["resume_event"])
    interval_report = load(report_paths["replay_interval_report"])
    baseline_backend = baseline["backend"]
    for name, report in {"recovery": recovery, "replay": replay, "fork": fork}.items():
        for key in ("device_type", "device_name", "torch_version"):
            if report["backend"].get(key) != baseline_backend.get(key):
                raise AssertionError(f"{name} execution backend differs from baseline: {key}")
    if crash["planned_exit_code"] != recovery_config["planned_crash_exit_code"]:
        raise AssertionError("crash exit-code contract mismatch")
    if crash["optimizer_updates"] != recovery_config["deliberate_crash_after_optimizer_update"]:
        raise AssertionError("crash optimizer-update contract mismatch")
    if sha256_file(ROOT / crash["checkpoint"]) != crash["checkpoint_sha256"]:
        raise AssertionError("crash checkpoint hash mismatch")
    crash_checkpoint = torch.load(ROOT / crash["checkpoint"], map_location="cpu", weights_only=False)
    crash_state = crash_checkpoint["execution_state"]
    if crash_state.get("execution_backend") != recovery["backend"]:
        raise AssertionError("crash checkpoint execution backend mismatch")
    if "rng_state" not in crash_checkpoint:
        raise AssertionError("crash checkpoint is missing portable RNG state")
    if crash_state["next_batch_index"] != crash["completed_microbatches"]:
        raise AssertionError("crash checkpoint batch cursor mismatch")
    if crash_state["optimizer_updates"] != crash["optimizer_updates"]:
        raise AssertionError("crash checkpoint optimizer cursor mismatch")
    crash_model = TinyDecoderTransformer(config)
    crash_model.load_state_dict(crash_checkpoint["model_state"], strict=True)
    if state_dict_hash(crash_model) != crash["parameter_hash"]:
        raise AssertionError("crash checkpoint parameter hash mismatch")
    batch_report = load("data/batches_v1/batch_report.json")
    packing_report = load("data/packed_v1/packing_report.json")
    batches = list(read_jsonl_gz(ROOT / batch_report["paths"]["batches"]))
    store = PackedBatchStore(ROOT, packing_report)
    expected_next = batch_reconstruction_proof(
        batches[int(crash["completed_microbatches"])], store
    )
    if crash_checkpoint.get("next_batch_proof") != expected_next:
        raise AssertionError("crash checkpoint next-batch proof mismatch")
    if crash.get("expected_next_batch") != expected_next:
        raise AssertionError("crash event next-batch proof mismatch")
    if resume_event["status"] != "PASS" or not all(
        bool(resume_event[key])
        for key in [
            "batch_id_matched",
            "sequence_ids_matched",
            "token_spans_matched",
            "payload_hashes_matched",
            "proof_hash_matched",
        ]
    ):
        raise AssertionError("resume next-batch validation failed")
    if resume_event["expected_next_batch"] != expected_next:
        raise AssertionError("resume expected next batch differs from crash checkpoint")
    if resume_event["reconstructed_next_batch"] != expected_next:
        raise AssertionError("resume reconstructed next batch differs")

    run_reports = {"recovery": recovery, "replay": replay, "fork": fork}
    ledgers = {}
    for name, report in run_reports.items():
        if report["status"] != "COMPLETE":
            raise AssertionError(f"incomplete run: {name}")
        for component, expected in report["component_hashes"].items():
            if sha256_file(ROOT / report["paths"][component]) != expected:
                raise AssertionError(f"component changed: {name} {component}")
        consumption = list(read_jsonl_gz(ROOT / report["paths"]["consumption_ledger"]))
        learning = list(read_jsonl_gz(ROOT / report["paths"]["learning_ledger"]))
        verify_hash_chain(consumption)
        verify_hash_chain(learning)
        if len(consumption) != report["microbatches"] or len(learning) != report["microbatches"]:
            raise AssertionError(f"ledger length mismatch: {name}")
        if consumption[-1]["entry_hash"] != report["ledger_tails"]["consumption"]:
            raise AssertionError(f"consumption tail mismatch: {name}")
        if learning[-1]["entry_hash"] != report["ledger_tails"]["learning"]:
            raise AssertionError(f"learning tail mismatch: {name}")
        checkpoint_parameter_hash(report, config)
        ledgers[name] = (consumption, learning)

    baseline_consumption = list(read_jsonl_gz(ROOT / baseline["paths"]["consumption_ledger"]))
    baseline_learning = list(read_jsonl_gz(ROOT / baseline["paths"]["learning_ledger"]))
    if ledgers["recovery"] != (baseline_consumption, baseline_learning):
        raise AssertionError("resumed run logical ledgers differ from baseline")
    if ledgers["replay"] != (baseline_consumption, baseline_learning):
        raise AssertionError("replay logical ledgers differ from baseline")
    fork_consumption, fork_learning = ledgers["fork"]
    if fork_consumption != baseline_consumption:
        raise AssertionError("fork changed data consumption")
    prefix = int(audit["fork"]["common_learning_prefix_microbatches"])
    if fork_learning[:prefix] != baseline_learning[:prefix]:
        raise AssertionError("fork learning prefix is not exact")
    if fork_learning[prefix] == baseline_learning[prefix]:
        raise AssertionError("fork did not diverge at audited batch")
    if not fork_learning[prefix]["optimizer_update_applied"]:
        raise AssertionError("fork diverged before an optimizer update")
    if fork["lineage"]["parent_checkpoint_sha256"] != crash["checkpoint_sha256"]:
        raise AssertionError("fork parent checkpoint lineage mismatch")
    if float(fork["lineage"]["fork_learning_rate_scale"]) != float(
        recovery_config["fork"]["learning_rate_scale_after_fork"]
    ):
        raise AssertionError("fork learning-rate lineage mismatch")
    if not (
        recovery["final_parameter_hash"]
        == replay["final_parameter_hash"]
        == baseline["final_parameter_hash"]
    ):
        raise AssertionError("recovery/replay final parameters are not exact")
    if fork["final_parameter_hash"] == baseline["final_parameter_hash"]:
        raise AssertionError("fork final parameters did not diverge")

    interval_body = {
        key: value for key, value in interval_report.items() if key != "interval_report_hash"
    }
    if interval_report["interval_report_hash"] != f"sha256:{sha256_bytes(canonical_json_bytes(interval_body))}":
        raise AssertionError("replay interval report hash mismatch")
    start = int(interval_report["interval_start_batch_index"])
    end = int(interval_report["interval_end_batch_index_exclusive"])
    if len(interval_report["entries"]) != end - start or interval_report["status"] != "PASS":
        raise AssertionError("replay interval coverage mismatch")
    replay_consumption = ledgers["replay"][0]
    for expected_index, entry in zip(range(start, end), interval_report["entries"], strict=True):
        if int(entry["batch_index"]) != expected_index or entry["status"] != "PASS":
            raise AssertionError("replay interval batch order mismatch")
        proof = batch_reconstruction_proof(batches[expected_index], store)
        original = baseline_consumption[expected_index]
        replayed = replay_consumption[expected_index]
        if not (
            entry["sequence_indices"]
            == original["sequence_indices"]
            == replayed["sequence_indices"]
            == proof["sequence_indices"]
        ):
            raise AssertionError("replay interval sequence IDs mismatch")
        if not (
            entry["token_spans"]
            == original["token_spans"]
            == replayed["token_spans"]
            == proof["token_spans"]
        ):
            raise AssertionError("replay interval token spans mismatch")
        if not (
            entry["original_payload_hashes"]
            == original["payload_hashes"]
            == entry["replay_payload_hashes"]
            == replayed["payload_hashes"]
            == proof["payload_hashes"]
        ):
            raise AssertionError("replay interval payload hashes mismatch")
        if not all(entry["comparisons"].values()):
            raise AssertionError("replay interval comparison flag failed")
    if audit["resume_next_batch"]["proof_hash"] != expected_next["proof_hash"]:
        raise AssertionError("audit next-batch proof lineage mismatch")
    if audit["replay_interval"]["report_hash"] != interval_report["interval_report_hash"]:
        raise AssertionError("audit replay-interval lineage mismatch")

    print(
        json.dumps(
            {
                "status": "PASS",
                "audit_hash": audit["audit_hash"],
                "planned_crash_exit_code": audit["planned_crash"]["observed_exit_code"],
                "crash_optimizer_update": crash["optimizer_updates"],
                "crash_completed_microbatches": crash["completed_microbatches"],
                "recovery_final_parameter_hash": recovery["final_parameter_hash"],
                "replay_final_parameter_hash": replay["final_parameter_hash"],
                "fork_final_parameter_hash": fork["final_parameter_hash"],
                "exact_recovery_ledgers": True,
                "exact_replay_ledgers": True,
                "fork_consumption_exact": True,
                "fork_first_divergent_microbatch": prefix,
                "fork_divergent_optimizer_update": fork_learning[prefix]["optimizer_update_index"],
                "all_final_checkpoints_reloaded": True,
                "resume_next_batch_index": resume_event["checkpoint_next_batch_index"],
                "resume_next_batch_proof_hash": expected_next["proof_hash"],
                "resume_next_batch_matched": True,
                "replay_interval_start": start,
                "replay_interval_end_exclusive": end,
                "replay_interval_batches": end - start,
                "replay_batch_ids_token_spans_payload_hashes": "exact",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
