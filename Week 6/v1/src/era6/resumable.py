from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from .canonical import atomic_write_bytes, atomic_write_json, canonical_json_bytes, read_jsonl_gz, sha256_file, write_jsonl_gz
from .model_torch import TinyDecoderTransformer, configure_determinism, parameter_count, state_dict_hash
from .runtime import capture_rng_state, resolve_execution_device, restore_rng_state
from .tokenizer import MultilaneTokenizer
from .training import (
    PackedBatchStore,
    build_validation_probe,
    batch_reconstruction_proof,
    chained_entry,
    learning_rate,
    total_optimizer_updates,
    verify_hash_chain,
)


def read_plain_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON ledger row {path}:{line_number}") from error
    return rows


def write_plain_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    atomic_write_bytes(path, payload)


def append_plain_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_json_bytes(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _checkpoint_bytes(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def save_checkpoint(
    path: Path,
    *,
    model: TinyDecoderTransformer,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    lineage: dict[str, Any],
    next_batch_proof: dict[str, Any] | None,
) -> None:
    device = next(model.parameters()).device
    rng_state = capture_rng_state(device)
    payload = {
        "schema_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": rng_state,
        "execution_state": state,
        "lineage": lineage,
        "next_batch_proof": next_batch_proof,
    }
    atomic_write_bytes(path, _checkpoint_bytes(payload))


def _clean_generated_output(output: Path, artifacts_root: Path) -> None:
    resolved = output.resolve()
    if not resolved.is_relative_to(artifacts_root.resolve()) or resolved == artifacts_root.resolve():
        raise ValueError("refusing to clean an output outside the scoped artifacts directory")
    output.mkdir(parents=True, exist_ok=True)
    names = {
        "consumption_ledger.jsonl",
        "learning_ledger.jsonl",
        "consumption_ledger.jsonl.gz",
        "learning_ledger.jsonl.gz",
        "execution_report.json",
        "validation_probe.json",
        "final_checkpoint.pt",
        "crash_event.json",
        "resume_event.json",
    }
    for path in output.iterdir():
        if path.name in names or (path.name.startswith("checkpoint_") and path.suffix == ".pt"):
            if path.is_file():
                path.unlink()


def execute(
    root: Path,
    *,
    output: Path,
    run_kind: str,
    fresh: bool,
    resume_from: Path | None,
    fork_from: Path | None,
    learning_rate_scale: float,
    checkpoint_every: int,
    crash_after_update: int | None,
    preserve_update: int | None,
    planned_exit_code: int,
) -> int:
    if sum(value is not None for value in (resume_from, fork_from)) > 1:
        raise ValueError("resume and fork are mutually exclusive")
    artifacts_root = root / "artifacts"
    output = output.resolve()
    if fresh:
        _clean_generated_output(output, artifacts_root)
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / "configs" / "training_v1.json"
    batch_report_path = root / "data" / "batches_v1" / "batch_report.json"
    packing_report_path = root / "data" / "packed_v1" / "packing_report.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    batch_report = json.loads(batch_report_path.read_text(encoding="utf-8"))
    packing_report = json.loads(packing_report_path.read_text(encoding="utf-8"))
    tokenized_report = json.loads(
        (root / "data" / "tokenized_v1" / "tokenized_report.json").read_text(encoding="utf-8")
    )
    configure_determinism(config)
    device, runtime = resolve_execution_device(config)
    tokenizer = MultilaneTokenizer.load(root / "artifacts" / "tokenizer_v1" / "tokenizer.json")
    probe_payload, probe_manifest = build_validation_probe(
        root, tokenized_report, tokenizer, config["validation_probe"]
    )
    batches = list(read_jsonl_gz(root / batch_report["paths"]["batches"]))
    store = PackedBatchStore(root, packing_report)
    model = TinyDecoderTransformer(config).to(device)
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
    consumption_path = output / "consumption_ledger.jsonl"
    learning_path = output / "learning_ledger.jsonl"
    checkpoint_source = resume_from or fork_from
    lineage: dict[str, Any]
    resume_event: dict[str, Any] | None = None
    if checkpoint_source is None:
        initial_parameter_hash = state_dict_hash(model)
        model.eval()
        with torch.no_grad():
            probe_tensors = store.to_device(probe_payload, device)
            initial_loss, initial_probe_tokens = model.loss(**probe_tensors)
        model.train()
        state = {
            "next_batch_index": 0,
            "optimizer_updates": 0,
            "training_loss_sum": 0.0,
            "training_loss_tokens": 0,
            "initial_parameter_hash": initial_parameter_hash,
            "initial_validation_loss": float(initial_loss.item()),
            "initial_validation_tokens": initial_probe_tokens,
            "previous_consumption_hash": None,
            "previous_learning_hash": None,
            "batch_plan_hash": batch_report["batch_plan_hash"],
            "training_config_sha256": sha256_file(config_path),
            "execution_backend": runtime,
        }
        consumption_rows: list[dict[str, Any]] = []
        learning_rows: list[dict[str, Any]] = []
        write_plain_jsonl(consumption_path, consumption_rows)
        write_plain_jsonl(learning_path, learning_rows)
        lineage = {"run_kind": run_kind, "parent_checkpoint_sha256": None, "fork_learning_rate_scale": 1.0}
    else:
        checkpoint_source = checkpoint_source.resolve()
        checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        state = checkpoint["execution_state"]
        checkpoint_backend = state.get("execution_backend")
        if checkpoint_backend is None:
            raise RuntimeError("checkpoint is missing its execution backend contract")
        for key in ("device_type", "device_name", "torch_version"):
            if checkpoint_backend.get(key) != runtime.get(key):
                raise RuntimeError(f"checkpoint execution backend mismatch: {key}")
        restore_rng_state(checkpoint["rng_state"], device)
        if state["batch_plan_hash"] != batch_report["batch_plan_hash"]:
            raise RuntimeError("checkpoint batch plan mismatch")
        source_output = checkpoint_source.parent
        source_consumption = read_plain_jsonl(source_output / "consumption_ledger.jsonl")
        source_learning = read_plain_jsonl(source_output / "learning_ledger.jsonl")
        prefix = int(state["next_batch_index"])
        expected_next_batch = checkpoint.get("next_batch_proof")
        actual_next_batch = (
            batch_reconstruction_proof(batches[prefix], store)
            if prefix < len(batches)
            else None
        )
        if expected_next_batch != actual_next_batch:
            raise RuntimeError("checkpoint next-batch reconstruction proof mismatch")
        consumption_rows = source_consumption[:prefix]
        learning_rows = source_learning[:prefix]
        verify_hash_chain(consumption_rows)
        verify_hash_chain(learning_rows)
        if len(consumption_rows) != prefix or len(learning_rows) != prefix:
            raise RuntimeError("checkpoint ledger prefix is incomplete")
        if consumption_rows[-1]["entry_hash"] != state["previous_consumption_hash"]:
            raise RuntimeError("checkpoint consumption tail mismatch")
        if learning_rows[-1]["entry_hash"] != state["previous_learning_hash"]:
            raise RuntimeError("checkpoint learning tail mismatch")
        if fork_from is not None:
            write_plain_jsonl(consumption_path, consumption_rows)
            write_plain_jsonl(learning_path, learning_rows)
            lineage = {
                "run_kind": run_kind,
                "parent_checkpoint_sha256": sha256_file(checkpoint_source),
                "parent_checkpoint": checkpoint_source.relative_to(root).as_posix(),
                "parent_parameter_hash": state_dict_hash(model),
                "fork_batch_index": prefix,
                "fork_optimizer_update": int(state["optimizer_updates"]),
                "fork_learning_rate_scale": learning_rate_scale,
            }
        else:
            existing_consumption = read_plain_jsonl(consumption_path)
            existing_learning = read_plain_jsonl(learning_path)
            if existing_consumption != consumption_rows or existing_learning != learning_rows:
                raise RuntimeError("resume output ledger differs from checkpoint prefix")
            lineage = checkpoint["lineage"]
            resume_event = {
                "schema_version": 1,
                "event": "resume_next_batch_validation",
                "status": "PASS",
                "checkpoint": checkpoint_source.relative_to(root).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint_source),
                "checkpoint_next_batch_index": prefix,
                "expected_next_batch": expected_next_batch,
                "reconstructed_next_batch": actual_next_batch,
                "batch_id_matched": expected_next_batch["batch_index"]
                == actual_next_batch["batch_index"],
                "sequence_ids_matched": expected_next_batch["sequence_indices"]
                == actual_next_batch["sequence_indices"],
                "token_spans_matched": expected_next_batch["token_spans"]
                == actual_next_batch["token_spans"],
                "payload_hashes_matched": expected_next_batch["payload_hashes"]
                == actual_next_batch["payload_hashes"],
                "proof_hash_matched": expected_next_batch["proof_hash"]
                == actual_next_batch["proof_hash"],
            }
            atomic_write_json(output / "resume_event.json", resume_event)
    optimizer.zero_grad(set_to_none=True)
    accumulated_microbatches = 0
    accumulated_loss_tokens = 0
    latest_periodic: Path | None = None
    for batch_index in range(int(state["next_batch_index"]), len(batches)):
        batch = batches[batch_index]
        payload = store.numpy_batch(batch)
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
                "payload_hashes": store.payload_hashes(payload),
                "token_spans": batch_proof["token_spans"],
                "batch_reconstruction_proof_hash": batch_proof["proof_hash"],
                "learning_event_id": f"learning-{batch_index:06d}",
            },
            state["previous_consumption_hash"],
        )
        append_plain_jsonl(consumption_path, consumption)
        consumption_rows.append(consumption)
        state["previous_consumption_hash"] = consumption["entry_hash"]
        tensors = store.to_device(payload, device)
        loss, effective_count, sample_losses = model.loss(
            **tensors, return_sequence_details=True
        )
        if effective_count != int(batch["loss_bearing_tokens"]):
            raise RuntimeError("shifted loss count mismatch")
        loss_value = float(loss.item())
        (loss * effective_count).backward()
        accumulated_microbatches += 1
        accumulated_loss_tokens += effective_count
        state["training_loss_sum"] += loss_value * effective_count
        state["training_loss_tokens"] += effective_count
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
            state["optimizer_updates"] += 1
            rate = learning_rate(state["optimizer_updates"], total_updates, optimizer_config)
            if fork_from is not None or float(lineage.get("fork_learning_rate_scale", 1.0)) != 1.0:
                rate *= float(lineage["fork_learning_rate_scale"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_details = {
                "optimizer_update_applied": True,
                "optimizer_update_index": state["optimizer_updates"],
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
            state["previous_learning_hash"],
        )
        append_plain_jsonl(learning_path, learning)
        learning_rows.append(learning)
        state["previous_learning_hash"] = learning["entry_hash"]
        state["next_batch_index"] = batch_index + 1
        if update_now and checkpoint_every and state["optimizer_updates"] % checkpoint_every == 0:
            periodic = output / f"checkpoint_{state['optimizer_updates']:06d}.pt"
            next_proof = (
                batch_reconstruction_proof(batches[int(state["next_batch_index"])], store)
                if int(state["next_batch_index"]) < len(batches)
                else None
            )
            save_checkpoint(
                periodic,
                model=model,
                optimizer=optimizer,
                state=state,
                lineage=lineage,
                next_batch_proof=next_proof,
            )
            if latest_periodic and latest_periodic.exists() and latest_periodic.name != f"checkpoint_{preserve_update or -1:06d}.pt":
                latest_periodic.unlink()
            latest_periodic = periodic
        if update_now and crash_after_update == state["optimizer_updates"]:
            crash_checkpoint = output / f"checkpoint_{state['optimizer_updates']:06d}.pt"
            if not crash_checkpoint.exists():
                next_proof = (
                    batch_reconstruction_proof(batches[int(state["next_batch_index"])], store)
                    if int(state["next_batch_index"]) < len(batches)
                    else None
                )
                save_checkpoint(
                    crash_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    state=state,
                    lineage=lineage,
                    next_batch_proof=next_proof,
                )
            crash_checkpoint_payload = torch.load(
                crash_checkpoint, map_location="cpu", weights_only=False
            )
            crash_event = {
                "schema_version": 1,
                "event": "deliberate_training_crash",
                "planned_exit_code": planned_exit_code,
                "completed_microbatches": state["next_batch_index"],
                "optimizer_updates": state["optimizer_updates"],
                "checkpoint": crash_checkpoint.relative_to(root).as_posix(),
                "checkpoint_sha256": sha256_file(crash_checkpoint),
                "parameter_hash": state_dict_hash(model),
                "consumption_ledger_tail": state["previous_consumption_hash"],
                "learning_ledger_tail": state["previous_learning_hash"],
                "expected_next_batch": crash_checkpoint_payload["next_batch_proof"],
            }
            atomic_write_json(output / "crash_event.json", crash_event)
            print(json.dumps(crash_event, indent=2), flush=True)
            return planned_exit_code
    if accumulated_microbatches or int(state["optimizer_updates"]) != total_updates:
        raise RuntimeError("execution did not finish on an optimizer boundary")
    model.eval()
    with torch.no_grad():
        final_loss, final_probe_tokens = model.loss(**store.to_device(probe_payload, device))
    model.train()
    final_parameter_hash = state_dict_hash(model)
    final_checkpoint = output / "final_checkpoint.pt"
    save_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        state=state,
        lineage=lineage,
        next_batch_proof=None,
    )
    consumption_gz = output / "consumption_ledger.jsonl.gz"
    learning_gz = output / "learning_ledger.jsonl.gz"
    consumption_stats = write_jsonl_gz(consumption_gz, consumption_rows)
    learning_stats = write_jsonl_gz(learning_gz, learning_rows)
    atomic_write_json(output / "validation_probe.json", probe_manifest)
    component_hashes = {
        "consumption_ledger": sha256_file(consumption_gz),
        "learning_ledger": sha256_file(learning_gz),
        "final_checkpoint": sha256_file(final_checkpoint),
        "validation_probe": sha256_file(output / "validation_probe.json"),
    }
    paths = {
        "consumption_ledger": consumption_gz.relative_to(root).as_posix(),
        "learning_ledger": learning_gz.relative_to(root).as_posix(),
        "final_checkpoint": final_checkpoint.relative_to(root).as_posix(),
        "validation_probe": (output / "validation_probe.json").relative_to(root).as_posix(),
    }
    if resume_event is not None:
        component_hashes["resume_event"] = sha256_file(output / "resume_event.json")
        paths["resume_event"] = (output / "resume_event.json").relative_to(root).as_posix()
    report = {
        "schema_version": 1,
        "status": "COMPLETE",
        "run_kind": run_kind,
        "lineage": lineage,
        "batch_plan_hash": batch_report["batch_plan_hash"],
        "training_config_sha256": sha256_file(config_path),
        "backend": runtime,
        "model_parameters": parameter_count(model),
        "initial_parameter_hash": state["initial_parameter_hash"],
        "final_parameter_hash": final_parameter_hash,
        "microbatches": len(consumption_rows),
        "optimizer_updates": state["optimizer_updates"],
        "loss_bearing_tokens": state["training_loss_tokens"],
        "training_cross_entropy_nats": state["training_loss_sum"] / state["training_loss_tokens"],
        "initial_validation": {
            "cross_entropy_nats": state["initial_validation_loss"],
            "loss_bearing_tokens": state["initial_validation_tokens"],
        },
        "final_validation": {
            "cross_entropy_nats": float(final_loss.item()),
            "loss_bearing_tokens": final_probe_tokens,
        },
        "ledger_tails": {
            "consumption": state["previous_consumption_hash"],
            "learning": state["previous_learning_hash"],
        },
        "resume_next_batch_validation": resume_event,
        "component_hashes": component_hashes,
        "paths": paths,
        "consumption_index": consumption_stats,
        "learning_index": learning_stats,
    }
    atomic_write_json(output / "execution_report.json", report)
    print(json.dumps({key: report[key] for key in ["status", "run_kind", "final_parameter_hash", "microbatches", "optimizer_updates", "loss_bearing_tokens", "final_validation"]}, indent=2), flush=True)
    return 0
