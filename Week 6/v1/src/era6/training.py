from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .canonical import canonical_json_bytes, read_jsonl_gz, sha256_bytes


def chained_entry(payload: dict[str, Any], previous_hash: str | None) -> dict[str, Any]:
    body = {**payload, "previous_entry_hash": previous_hash}
    return {**body, "entry_hash": f"sha256:{sha256_bytes(canonical_json_bytes(body))}"}


def verify_hash_chain(rows: list[dict[str, Any]]) -> None:
    previous = None
    for index, row in enumerate(rows):
        if row["previous_entry_hash"] != previous:
            raise AssertionError(f"ledger predecessor mismatch at row {index}")
        body = {key: value for key, value in row.items() if key != "entry_hash"}
        expected = f"sha256:{sha256_bytes(canonical_json_bytes(body))}"
        if row["entry_hash"] != expected:
            raise AssertionError(f"ledger entry hash mismatch at row {index}")
        previous = row["entry_hash"]


class PackedBatchStore:
    def __init__(self, root: Path, packing_report: dict[str, Any]) -> None:
        self.root = root
        self.sequences = {
            int(row["sequence_index"]): row
            for row in read_jsonl_gz(root / packing_report["paths"]["sequences"])
        }
        self.tokens = np.fromfile(root / packing_report["paths"]["input_ids"], dtype="<u2")
        self.loss = np.fromfile(root / packing_report["paths"]["loss_mask"], dtype="u1")
        self.segments = np.fromfile(root / packing_report["paths"]["segment_ids"], dtype="<i2")
        self.positions = np.fromfile(root / packing_report["paths"]["position_ids"], dtype="<u2")

    def numpy_batch(self, batch: dict[str, Any]) -> dict[str, np.ndarray]:
        def stack(source: np.ndarray) -> np.ndarray:
            rows = []
            for sequence_index in batch["sequence_indices"]:
                sequence = self.sequences[int(sequence_index)]
                start = int(sequence["global_token_offset"])
                end = start + int(sequence["sequence_length"])
                rows.append(source[start:end])
            return np.ascontiguousarray(np.stack(rows))

        return {
            "input_ids": stack(self.tokens),
            "loss_mask": stack(self.loss),
            "segment_ids": stack(self.segments),
            "position_ids": stack(self.positions),
        }

    @staticmethod
    def payload_hashes(payload: dict[str, np.ndarray]) -> dict[str, str]:
        return {
            name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for name, value in sorted(payload.items())
        }

    @staticmethod
    def to_device(payload: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.as_tensor(payload["input_ids"], dtype=torch.long, device=device),
            "loss_mask": torch.as_tensor(payload["loss_mask"], dtype=torch.uint8, device=device),
            "segment_ids": torch.as_tensor(payload["segment_ids"], dtype=torch.int16, device=device),
            "position_ids": torch.as_tensor(payload["position_ids"], dtype=torch.long, device=device),
        }


def batch_reconstruction_proof(
    batch: dict[str, Any], store: PackedBatchStore
) -> dict[str, Any]:
    payload = store.numpy_batch(batch)
    token_spans = []
    for sequence_index in batch["sequence_indices"]:
        sequence = store.sequences[int(sequence_index)]
        start = int(sequence["global_token_offset"])
        length = int(sequence["sequence_length"])
        token_spans.append(
            {
                "sequence_index": int(sequence_index),
                "global_token_start": start,
                "global_token_end_exclusive": start + length,
            }
        )
    body = {
        "batch_index": int(batch["batch_index"]),
        "stage": batch["stage"],
        "sequence_indices": [int(value) for value in batch["sequence_indices"]],
        "token_spans": token_spans,
        "payload_hashes": store.payload_hashes(payload),
    }
    return {**body, "proof_hash": f"sha256:{sha256_bytes(canonical_json_bytes(body))}"}


def _rank_record(row: dict[str, Any], seed: str, lane: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{seed}|{lane}|{row['record_id']}".encode("utf-8")).digest()
    return digest, row["record_id"]


def build_validation_probe(
    root: Path,
    tokenized_report: dict[str, Any],
    tokenizer: Any,
    probe_config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    length = int(probe_config["sequence_length"])
    minimum_loss = int(probe_config["minimum_loss_tokens_per_sequence"])
    seed = probe_config["selection_seed"]
    lanes = sorted({row["lane"] for row in tokenized_report["shards"] if row["permission"] == "validation"})
    selected = []
    token_rows = []
    loss_rows = []
    segment_rows = []
    position_rows = []
    for lane in lanes:
        shard = next(
            row
            for row in tokenized_report["shards"]
            if row["permission"] == "validation" and row["lane"] == lane
        )
        manifest = json.loads((root / shard["manifest_path"]).read_text(encoding="utf-8"))["manifest"]
        extra = manifest["extra"]
        index_rows = sorted(
            read_jsonl_gz(root / extra["index_path"]),
            key=lambda row: _rank_record(row, seed, lane),
        )
        shard_tokens = np.fromfile(root / extra["tokens_path"], dtype="<u2")
        shard_loss = np.fromfile(root / extra["loss_path"], dtype="u1")
        choice = None
        for row in index_rows:
            start = int(row["token_offset"])
            end = start + int(row["token_count"])
            record_tokens = shard_tokens[start:end]
            record_loss = shard_loss[start:end]
            possible_starts = list(range(0, max(1, len(record_tokens) - length + 1), max(1, length // 2)))
            possible_starts.append(max(0, len(record_tokens) - length))
            best_start = max(
                sorted(set(possible_starts)),
                key=lambda value: (int(record_loss[value + 1 : value + length].sum()), -value),
            )
            window_end = min(len(record_tokens), best_start + length)
            effective_loss = int(record_loss[best_start + 1 : window_end].sum())
            if effective_loss >= minimum_loss:
                choice = (row, record_tokens[best_start:window_end], record_loss[best_start:window_end], best_start)
                break
        if choice is None:
            raise RuntimeError(f"no validation probe window with enough loss for lane {lane}")
        row, window_tokens, window_loss, record_start = choice
        nonpadding = len(window_tokens)
        ids = np.full(length, tokenizer.special_token_ids["<pad>"], dtype="<u2")
        mask = np.zeros(length, dtype="u1")
        segments = np.full(length, -1, dtype="<i2")
        positions = np.zeros(length, dtype="<u2")
        ids[:nonpadding] = window_tokens
        mask[:nonpadding] = window_loss
        mask[0] = 0
        segments[:nonpadding] = 0
        positions[:nonpadding] = np.arange(nonpadding, dtype="<u2")
        token_rows.append(ids)
        loss_rows.append(mask)
        segment_rows.append(segments)
        position_rows.append(positions)
        selected.append(
            {
                "lane": lane,
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "record_token_start": record_start,
                "record_token_end": record_start + nonpadding,
                "nonpadding_tokens": nonpadding,
                "loss_bearing_tokens": int(mask[1:].sum()),
                "permission": "validation",
            }
        )
    payload = {
        "input_ids": np.ascontiguousarray(np.stack(token_rows)),
        "loss_mask": np.ascontiguousarray(np.stack(loss_rows)),
        "segment_ids": np.ascontiguousarray(np.stack(segment_rows)),
        "position_ids": np.ascontiguousarray(np.stack(position_rows)),
    }
    component_hashes = PackedBatchStore.payload_hashes(payload)
    manifest = {
        "schema_version": 1,
        "permission": "validation",
        "selection_seed": seed,
        "sequence_length": length,
        "lanes": lanes,
        "sequences": len(selected),
        "loss_bearing_tokens": int(payload["loss_mask"][:, 1:].sum()),
        "selected": selected,
        "component_hashes": component_hashes,
    }
    manifest["probe_hash"] = f"sha256:{sha256_bytes(canonical_json_bytes(manifest))}"
    return payload, manifest


def total_optimizer_updates(batches: list[dict[str, Any]], accumulation: int) -> int:
    counts: Counter[str] = Counter(batch["stage"] for batch in batches)
    return sum(math.ceil(count / accumulation) for count in counts.values())


def learning_rate(update_index: int, total_updates: int, optimizer: dict[str, Any]) -> float:
    maximum = float(optimizer["learning_rate"])
    minimum = float(optimizer["minimum_learning_rate"])
    warmup = int(optimizer["warmup_optimizer_updates"])
    if update_index <= warmup:
        return maximum * update_index / max(1, warmup)
    progress = (update_index - warmup) / max(1, total_updates - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return minimum + (maximum - minimum) * cosine
