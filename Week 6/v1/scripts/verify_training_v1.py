from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import canonical_json_bytes, read_jsonl_gz, sha256_bytes, sha256_file  # noqa: E402
from era6.model_torch import TinyDecoderTransformer, parameter_count, state_dict_hash  # noqa: E402
from era6.training import (  # noqa: E402
    PackedBatchStore,
    batch_reconstruction_proof,
    total_optimizer_updates,
    verify_hash_chain,
)
from era6.performance import reconstructable_throughput  # noqa: E402


def main() -> int:
    report_path = ROOT / "artifacts" / "training_v1" / "training_report.json"
    config_path = ROOT / "configs" / "training_v1.json"
    batch_report_path = ROOT / "data" / "batches_v1" / "batch_report.json"
    packing_report_path = ROOT / "data" / "packed_v1" / "packing_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    batch_report = json.loads(batch_report_path.read_text(encoding="utf-8"))
    packing_report = json.loads(packing_report_path.read_text(encoding="utf-8"))
    if report["status"] != "COMPLETE":
        raise AssertionError("training report is not complete")
    if report["training_config_sha256"] != sha256_file(config_path):
        raise AssertionError("training configuration hash mismatch")
    if report["batch_report_sha256"] != sha256_file(batch_report_path):
        raise AssertionError("batch report hash mismatch")
    if report["batch_plan_hash"] != batch_report["batch_plan_hash"]:
        raise AssertionError("training used the wrong batch plan")
    if report["packing_hash"] != packing_report["packing_hash"]:
        raise AssertionError("training used the wrong packing artifact")
    for name, expected in report["component_hashes"].items():
        if sha256_file(ROOT / report["paths"][name]) != expected:
            raise AssertionError(f"training component hash mismatch: {name}")
    expected_training_hash = f"sha256:{sha256_bytes(canonical_json_bytes(report['component_hashes']))}"
    if report["training_hash"] != expected_training_hash:
        raise AssertionError("training hash mismatch")

    consumption = list(read_jsonl_gz(ROOT / report["paths"]["consumption_ledger"]))
    learning = list(read_jsonl_gz(ROOT / report["paths"]["learning_ledger"]))
    batches = list(read_jsonl_gz(ROOT / batch_report["paths"]["batches"]))
    if not (len(consumption) == len(learning) == len(batches) == report["microbatches"]):
        raise AssertionError("batch and ledger lengths differ")
    verify_hash_chain(consumption)
    verify_hash_chain(learning)
    if consumption[-1]["entry_hash"] != report["ledger_tails"]["consumption"]:
        raise AssertionError("consumption ledger tail mismatch")
    if learning[-1]["entry_hash"] != report["ledger_tails"]["learning"]:
        raise AssertionError("learning ledger tail mismatch")

    store = PackedBatchStore(ROOT, packing_report)
    total_loss = 0
    stage_loss: Counter[str] = Counter()
    optimizer_updates = 0
    last_update_hash = None
    for index, (batch, consumed, learned) in enumerate(zip(batches, consumption, learning, strict=True)):
        if consumed["consumption_index"] != index or consumed["batch_index"] != batch["batch_index"]:
            raise AssertionError(f"consumption order mismatch at {index}")
        if consumed["sequence_indices"] != batch["sequence_indices"]:
            raise AssertionError(f"sequence consumption mismatch at {index}")
        if consumed["stage"] != batch["stage"] or learned["stage"] != batch["stage"]:
            raise AssertionError(f"stage mismatch at {index}")
        if consumed["loss_bearing_tokens"] != batch["loss_bearing_tokens"]:
            raise AssertionError(f"consumption loss mismatch at {index}")
        if learned["loss_bearing_tokens"] != batch["loss_bearing_tokens"]:
            raise AssertionError(f"learning loss mismatch at {index}")
        if learned["consumption_entry_hash"] != consumed["entry_hash"]:
            raise AssertionError(f"ledger cross-link mismatch at {index}")
        if learned["learning_event_id"] != consumed["learning_event_id"]:
            raise AssertionError(f"learning event mismatch at {index}")
        if not math.isfinite(float(learned["cross_entropy_nats"])):
            raise AssertionError(f"non-finite training loss at {index}")
        sample_losses = learned.get("sample_losses", [])
        if [int(row["sequence_index"]) for row in sample_losses] != [
            int(value) for value in batch["sequence_indices"]
        ]:
            raise AssertionError(f"sample loss identities differ from consumption at {index}")
        if sum(int(row["loss_bearing_tokens"]) for row in sample_losses) != int(
            learned["loss_bearing_tokens"]
        ):
            raise AssertionError(f"sample loss-token accounting mismatch at {index}")
        reconstructed_loss = sum(
            float(row["cross_entropy_sum_nats"]) for row in sample_losses
        ) / int(learned["loss_bearing_tokens"])
        if not math.isclose(
            reconstructed_loss,
            float(learned["cross_entropy_nats"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise AssertionError(f"sample losses do not reconstruct microbatch loss at {index}")
        payload = store.numpy_batch(batch)
        if store.payload_hashes(payload) != consumed["payload_hashes"]:
            raise AssertionError(f"consumed payload mismatch at {index}")
        batch_proof = batch_reconstruction_proof(batch, store)
        if consumed.get("token_spans") != batch_proof["token_spans"]:
            raise AssertionError(f"consumed token spans mismatch at {index}")
        if consumed.get("batch_reconstruction_proof_hash") != batch_proof["proof_hash"]:
            raise AssertionError(f"consumed batch proof mismatch at {index}")
        total_loss += int(batch["loss_bearing_tokens"])
        stage_loss[batch["stage"]] += int(batch["loss_bearing_tokens"])
        if learned["optimizer_update_applied"]:
            optimizer_updates += 1
            if learned["optimizer_update_index"] != optimizer_updates:
                raise AssertionError("optimizer update indices are not contiguous")
            if learned["gradient_norm_before_clipping"] is None or learned["learning_rate"] is None:
                raise AssertionError("optimizer update lacks diagnostics")
            last_update_hash = learned["parameter_hash_after_update"]
    if total_loss != batch_report["loss_bearing_tokens"] or total_loss != report["loss_bearing_tokens"]:
        raise AssertionError("training loss-token total mismatch")
    if report.get("loss_tracking_granularity") != "microbatch_and_sequence":
        raise AssertionError("training report does not declare sequence-level loss tracking")
    expected_stage_loss = Counter()
    for key, value in batch_report["loss_by_stage_lane"].items():
        stage, _lane = key.split("|", 1)
        expected_stage_loss[stage] += value
    if stage_loss != expected_stage_loss:
        raise AssertionError("training stage totals mismatch")
    expected_updates = total_optimizer_updates(
        batches, int(config["optimizer"]["gradient_accumulation_microbatches"])
    )
    if optimizer_updates != expected_updates or optimizer_updates != report["optimizer_updates"]:
        raise AssertionError("optimizer update total mismatch")
    if last_update_hash != report["final_parameter_hash"]:
        raise AssertionError("learning ledger final parameter hash mismatch")

    performance = json.loads((ROOT / report["paths"]["performance"]).read_text(encoding="utf-8"))
    if performance != report["performance"] or performance["status"] != "PASS":
        raise AssertionError("training performance report mismatch")
    if sha256_file(ROOT / report["paths"]["performance"]) != report["performance_report_sha256"]:
        raise AssertionError("training performance report hash mismatch")
    reconstructed_performance = reconstructable_throughput(
        elapsed_nanoseconds=int(performance["elapsed_nanoseconds"]),
        physical_tokens=sum(int(batch["physical_tokens"]) for batch in batches),
        nonpadding_tokens=sum(int(batch["nonpadding_tokens"]) for batch in batches),
        loss_bearing_tokens=total_loss,
    )
    for key, expected in reconstructed_performance.items():
        observed = performance[key]
        if isinstance(expected, float):
            if not math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError(f"performance rate mismatch: {key}")
        elif observed != expected:
            raise AssertionError(f"performance count mismatch: {key}")
    if performance["batch_plan_hash"] != report["batch_plan_hash"]:
        raise AssertionError("performance report used the wrong batch plan")
    if performance["device_type"] != report["backend"]["device_type"]:
        raise AssertionError("performance and training backend device types differ")
    if performance["device"] != report["backend"]["device_name"]:
        raise AssertionError("performance and training backend device names differ")
    if bool(performance["cuda_synchronized_at_boundaries"]) != (
        performance["device_type"] == "cuda"
    ):
        raise AssertionError("CUDA synchronization flag does not match selected device")

    probe = json.loads((ROOT / report["paths"]["validation_probe"]).read_text(encoding="utf-8"))
    probe_body = {key: value for key, value in probe.items() if key != "probe_hash"}
    if probe["probe_hash"] != f"sha256:{sha256_bytes(canonical_json_bytes(probe_body))}":
        raise AssertionError("validation probe hash mismatch")
    if probe["probe_hash"] != report["validation_probe_hash"]:
        raise AssertionError("training report validation-probe hash mismatch")
    if any(row["permission"] != "validation" for row in probe["selected"]):
        raise AssertionError("validation probe contains non-validation data")
    if report["final_validation"]["cross_entropy_nats"] >= report["initial_validation"]["cross_entropy_nats"]:
        raise AssertionError("validation loss did not improve")

    checkpoint = torch.load(
        ROOT / report["paths"]["final_checkpoint"], map_location="cpu", weights_only=False
    )
    model = TinyDecoderTransformer(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if parameter_count(model) != report["model_parameters"]:
        raise AssertionError("checkpoint model parameter count mismatch")
    if state_dict_hash(model) != report["final_parameter_hash"]:
        raise AssertionError("checkpoint parameter hash mismatch")
    metadata = checkpoint["metadata"]
    checkpoint_backend = metadata.get("execution_backend", {})
    for key in ("device_type", "device_name", "torch_version"):
        if checkpoint_backend.get(key) != report["backend"].get(key):
            raise AssertionError(f"checkpoint execution backend mismatch: {key}")
    for key in [
        "batch_plan_hash",
        "initial_parameter_hash",
        "final_parameter_hash",
        "consumption_ledger_tail",
        "learning_ledger_tail",
    ]:
        expected = {
            "consumption_ledger_tail": report["ledger_tails"]["consumption"],
            "learning_ledger_tail": report["ledger_tails"]["learning"],
        }.get(key, report.get(key))
        if metadata[key] != expected:
            raise AssertionError(f"checkpoint metadata mismatch: {key}")
    optimizer_steps = {
        int(value["step"].item())
        for value in checkpoint["optimizer_state"]["state"].values()
        if "step" in value
    }
    if optimizer_steps != {optimizer_updates}:
        raise AssertionError("checkpoint optimizer step mismatch")

    print(
        json.dumps(
            {
                "status": "PASS",
                "training_hash": report["training_hash"],
                "checkpoint_parameter_hash": report["final_parameter_hash"],
                "microbatches_cross_linked": len(batches),
                "optimizer_updates": optimizer_updates,
                "loss_bearing_tokens": total_loss,
                "consumption_payloads_rehashed": len(consumption),
                "ledger_hash_chains": "verified",
                "validation_permission_isolation": "verified",
                "validation_cross_entropy_before": report["initial_validation"]["cross_entropy_nats"],
                "validation_cross_entropy_after": report["final_validation"]["cross_entropy_nats"],
                "checkpoint_reload": "exact",
                "useful_loss_bearing_tokens_per_second": performance[
                    "useful_loss_bearing_tokens_per_second"
                ],
                "performance_reconstruction": "exact",
                "sample_losses_cross_linked": sum(
                    len(row["sample_losses"]) for row in learning
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
