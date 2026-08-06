from __future__ import annotations

import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    read_jsonl_gz,
    sha256_bytes,
    sha256_file,
    write_jsonl_gz,
)
from era6.model_torch import (  # noqa: E402
    TinyDecoderTransformer,
    configure_determinism,
    parameter_count,
    state_dict_hash,
)
from era6.tokenizer import MultilaneTokenizer  # noqa: E402
from era6.performance import reconstructable_throughput  # noqa: E402
from era6.runtime import (  # noqa: E402
    peak_accelerator_memory,
    reset_peak_memory,
    resolve_execution_device,
    synchronize,
)
from era6.training import (  # noqa: E402
    PackedBatchStore,
    batch_reconstruction_proof,
    build_validation_probe,
    chained_entry,
    learning_rate,
    total_optimizer_updates,
)


OUTPUT_ROOT = ROOT / "artifacts" / "training_v1"


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
) -> None:
    buffer = io.BytesIO()
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": metadata,
        },
        buffer,
    )
    atomic_write_bytes(path, buffer.getvalue())


def evaluate(
    model: TinyDecoderTransformer,
    payload: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, float | int]:
    tensors = PackedBatchStore.to_device(payload, device)
    model.eval()
    with torch.no_grad():
        loss, count = model.loss(**tensors)
    model.train()
    value = float(loss.item())
    return {
        "cross_entropy_nats": value,
        "perplexity": math.exp(min(value, 20.0)),
        "loss_bearing_tokens": count,
    }


def main() -> int:
    config_path = ROOT / "configs" / "training_v1.json"
    batch_report_path = ROOT / "data" / "batches_v1" / "batch_report.json"
    packing_report_path = ROOT / "data" / "packed_v1" / "packing_report.json"
    tokenized_report_path = ROOT / "data" / "tokenized_v1" / "tokenized_report.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    batch_report = json.loads(batch_report_path.read_text(encoding="utf-8"))
    packing_report = json.loads(packing_report_path.read_text(encoding="utf-8"))
    tokenized_report = json.loads(tokenized_report_path.read_text(encoding="utf-8"))
    device, runtime = resolve_execution_device(config)
    configure_determinism(config)
    tokenizer = MultilaneTokenizer.load(ROOT / "artifacts" / "tokenizer_v1" / "tokenizer.json")
    probe_payload, probe_manifest = build_validation_probe(
        ROOT, tokenized_report, tokenizer, config["validation_probe"]
    )
    batches = list(read_jsonl_gz(ROOT / batch_report["paths"]["batches"]))
    store = PackedBatchStore(ROOT, packing_report)
    model = TinyDecoderTransformer(config).to(device)
    initial_parameter_hash = state_dict_hash(model)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    accumulation_limit = int(optimizer_config["gradient_accumulation_microbatches"])
    total_updates = total_optimizer_updates(batches, accumulation_limit)
    initial_validation = evaluate(model, probe_payload, device)
    optimizer.zero_grad(set_to_none=True)
    consumption_rows = []
    learning_rows = []
    previous_consumption = None
    previous_learning = None
    accumulated_microbatches = 0
    accumulated_loss_tokens = 0
    optimizer_updates = 0
    training_loss_sum = 0.0
    training_loss_tokens = 0
    processed_physical_tokens = 0
    processed_nonpadding_tokens = 0
    reset_peak_memory(device)
    synchronize(device)
    training_start_ns = time.perf_counter_ns()
    for batch_index, batch in enumerate(batches):
        payload = store.numpy_batch(batch)
        payload_hashes = store.payload_hashes(payload)
        batch_proof = batch_reconstruction_proof(batch, store)
        consumption = chained_entry(
            {
                "schema_version": 1,
                "consumption_index": batch_index,
                "batch_index": batch["batch_index"],
                "batch_plan_hash": batch_report["batch_plan_hash"],
                "stage": batch["stage"],
                "sequence_indices": batch["sequence_indices"],
                "sequence_count": batch["sequence_count"],
                "physical_tokens": batch["physical_tokens"],
                "loss_bearing_tokens": batch["loss_bearing_tokens"],
                "payload_hashes": payload_hashes,
                "token_spans": batch_proof["token_spans"],
                "batch_reconstruction_proof_hash": batch_proof["proof_hash"],
                "learning_event_id": f"learning-{batch_index:06d}",
            },
            previous_consumption,
        )
        previous_consumption = consumption["entry_hash"]
        consumption_rows.append(consumption)
        tensors = store.to_device(payload, device)
        loss, effective_count, sample_losses = model.loss(
            **tensors, return_sequence_details=True
        )
        if effective_count != int(batch["loss_bearing_tokens"]):
            raise RuntimeError(f"shifted loss count mismatch in batch {batch_index}")
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite loss in batch {batch_index}")
        (loss * effective_count).backward()
        accumulated_microbatches += 1
        accumulated_loss_tokens += effective_count
        training_loss_sum += loss_value * effective_count
        training_loss_tokens += effective_count
        processed_physical_tokens += int(batch["physical_tokens"])
        processed_nonpadding_tokens += int(batch["nonpadding_tokens"])
        next_stage = batches[batch_index + 1]["stage"] if batch_index + 1 < len(batches) else None
        update_now = accumulated_microbatches == accumulation_limit or next_stage != batch["stage"]
        update_details: dict[str, Any] = {
            "optimizer_update_applied": update_now,
            "optimizer_update_index": None,
            "learning_rate": None,
            "gradient_norm_before_clipping": None,
            "parameter_hash_after_update": None,
        }
        if update_now:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_loss_tokens)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optimizer_config["gradient_clip_norm"])
            )
            optimizer_updates += 1
            rate = learning_rate(optimizer_updates, total_updates, optimizer_config)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_details = {
                "optimizer_update_applied": True,
                "optimizer_update_index": optimizer_updates,
                "learning_rate": rate,
                "gradient_norm_before_clipping": float(norm.item()),
                "parameter_hash_after_update": state_dict_hash(model),
            }
            accumulated_microbatches = 0
            accumulated_loss_tokens = 0
        learning = chained_entry(
            {
                "schema_version": 1,
                "learning_event_id": f"learning-{batch_index:06d}",
                "batch_index": batch["batch_index"],
                "stage": batch["stage"],
                "consumption_entry_hash": consumption["entry_hash"],
                "loss_bearing_tokens": effective_count,
                "cross_entropy_nats": loss_value,
                "perplexity": math.exp(min(loss_value, 20.0)),
                "sample_losses": [
                    {**detail, "sequence_index": int(sequence_index)}
                    for detail, sequence_index in zip(
                        sample_losses, batch["sequence_indices"], strict=True
                    )
                ],
                "accumulation_microbatches_after": accumulated_microbatches,
                **update_details,
            },
            previous_learning,
        )
        previous_learning = learning["entry_hash"]
        learning_rows.append(learning)
        if (batch_index + 1) % 25 == 0 or batch_index + 1 == len(batches):
            print(
                json.dumps(
                    {
                        "microbatches": batch_index + 1,
                        "total_microbatches": len(batches),
                        "optimizer_updates": optimizer_updates,
                        "latest_loss": loss_value,
                    }
                ),
                flush=True,
            )
    synchronize(device)
    training_elapsed_ns = time.perf_counter_ns() - training_start_ns
    if accumulated_microbatches or optimizer_updates != total_updates:
        raise RuntimeError("optimizer accumulation did not finish exactly")
    final_validation = evaluate(model, probe_payload, device)
    final_parameter_hash = state_dict_hash(model)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    consumption_path = OUTPUT_ROOT / "consumption_ledger.jsonl.gz"
    learning_path = OUTPUT_ROOT / "learning_ledger.jsonl.gz"
    checkpoint_path = OUTPUT_ROOT / "final_checkpoint.pt"
    performance_path = OUTPUT_ROOT / "performance.json"
    consumption_stats = write_jsonl_gz(consumption_path, consumption_rows)
    learning_stats = write_jsonl_gz(learning_path, learning_rows)
    atomic_write_json(OUTPUT_ROOT / "validation_probe.json", probe_manifest)
    performance = {
        "schema_version": 1,
        "status": "PASS",
        "run_id": config["run_id"],
        "batch_plan_hash": batch_report["batch_plan_hash"],
        "timing_scope": f"synchronized_{device.type}_training_loop_including_batch_load_forward_backward_optimizer_and_ledger_construction_excluding_validation_and_artifact_writes",
        "clock": "time.perf_counter_ns",
        "accelerator_synchronized_at_boundaries": True,
        "cuda_synchronized_at_boundaries": device.type == "cuda",
        "device_type": device.type,
        "device": runtime["device_name"],
        "microbatches": len(batches),
        "optimizer_updates": optimizer_updates,
        **reconstructable_throughput(
            elapsed_nanoseconds=training_elapsed_ns,
            physical_tokens=processed_physical_tokens,
            nonpadding_tokens=processed_nonpadding_tokens,
            loss_bearing_tokens=training_loss_tokens,
        ),
        "peak_accelerator_memory_bytes": peak_accelerator_memory(device),
        "peak_gpu_memory_bytes": peak_accelerator_memory(device),
    }
    atomic_write_json(performance_path, performance)
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        metadata={
            "run_id": config["run_id"],
            "completed_microbatches": len(batches),
            "optimizer_updates": optimizer_updates,
            "batch_plan_hash": batch_report["batch_plan_hash"],
            "execution_backend": runtime,
            "initial_parameter_hash": initial_parameter_hash,
            "final_parameter_hash": final_parameter_hash,
            "consumption_ledger_tail": previous_consumption,
            "learning_ledger_tail": previous_learning,
        },
    )
    component_hashes = {
        "consumption_ledger": sha256_file(consumption_path),
        "learning_ledger": sha256_file(learning_path),
        "validation_probe": sha256_file(OUTPUT_ROOT / "validation_probe.json"),
        "final_checkpoint": sha256_file(checkpoint_path),
    }
    training_hash = f"sha256:{sha256_bytes(canonical_json_bytes(component_hashes))}"
    report = {
        "schema_version": 1,
        "status": "COMPLETE",
        "run_id": config["run_id"],
        "training_hash": training_hash,
        "batch_plan_hash": batch_report["batch_plan_hash"],
        "packing_hash": packing_report["packing_hash"],
        "tokenizer_hash": packing_report["tokenizer_hash"],
        "training_config_sha256": sha256_file(config_path),
        "batch_report_sha256": sha256_file(batch_report_path),
        "backend": {
            **runtime,
            "cuda_runtime": torch.version.cuda,
            "device": runtime["device_name"],
            "precision": config["determinism"]["float_dtype"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else False,
        },
        "model_parameters": parameter_count(model),
        "initial_parameter_hash": initial_parameter_hash,
        "final_parameter_hash": final_parameter_hash,
        "microbatches": len(batches),
        "optimizer_updates": optimizer_updates,
        "loss_bearing_tokens": training_loss_tokens,
        "training_cross_entropy_nats": training_loss_sum / training_loss_tokens,
        "loss_tracking_granularity": "microbatch_and_sequence",
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "validation_loss_change": final_validation["cross_entropy_nats"]
        - initial_validation["cross_entropy_nats"],
        "validation_probe_hash": probe_manifest["probe_hash"],
        "peak_accelerator_memory_bytes": peak_accelerator_memory(device),
        "peak_gpu_memory_bytes": peak_accelerator_memory(device),
        "performance": performance,
        "performance_report_sha256": sha256_file(performance_path),
        "ledger_tails": {
            "consumption": previous_consumption,
            "learning": previous_learning,
        },
        "component_hashes": component_hashes,
        "paths": {
            "consumption_ledger": consumption_path.relative_to(ROOT).as_posix(),
            "learning_ledger": learning_path.relative_to(ROOT).as_posix(),
            "validation_probe": (OUTPUT_ROOT / "validation_probe.json").relative_to(ROOT).as_posix(),
            "final_checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
            "performance": performance_path.relative_to(ROOT).as_posix(),
        },
        "consumption_index": consumption_stats,
        "learning_index": learning_stats,
    }
    atomic_write_json(OUTPUT_ROOT / "training_report.json", report)
    print(json.dumps({key: report[key] for key in ["status", "training_hash", "model_parameters", "microbatches", "optimizer_updates", "loss_bearing_tokens", "training_cross_entropy_nats", "initial_validation", "final_validation", "final_parameter_hash"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
