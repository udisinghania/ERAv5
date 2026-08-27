from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import gc
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


MIB = 2**20


STAGE_ORDER = (
    "seed",
    "general_foundation",
    "reasoning_skill_build",
    "long_context",
    "anneal",
)


@dataclass(frozen=True)
class Superbatch:
    index: int
    stage: str
    sequence_length: int
    base_batches: tuple[dict[str, Any], ...]

    @property
    def sequence_indices(self) -> list[int]:
        return [
            int(sequence_index)
            for batch in self.base_batches
            for sequence_index in batch["sequence_indices"]
        ]

    @property
    def physical_tokens(self) -> int:
        return sum(int(batch["physical_tokens"]) for batch in self.base_batches)

    @property
    def reported_head1_targets(self) -> int:
        return sum(int(batch["loss_bearing_tokens"]) for batch in self.base_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assignment 9 resumable MTP trainer"
    )
    parser.add_argument(
        "command",
        choices=(
            "inspect-data",
            "inspect-validation",
            "smoke-model",
            "train",
            "compare-checkpoints",
        ),
        help="Run one implementation gate; neither command starts timed training.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("mtp_17m_config.json"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override the run directory from the configuration.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop at this absolute global step; intended for controlled tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume exactly from run-dir/latest.pt.",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Explicitly authorize the configured multi-hour wall-clock run.",
    )
    parser.add_argument("--checkpoint-a", type=Path, default=None)
    parser.add_argument("--checkpoint-b", type=Path, default=None)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    if args.max_steps is not None and args.full_run:
        parser.error("choose either --max-steps or --full-run, not both")
    if args.command == "train" and args.max_steps is None and not args.full_run:
        parser.error("train requires --max-steps for a test or --full-run")
    if args.command != "train" and (args.resume or args.full_run or args.max_steps):
        parser.error("training flags can only be used with the train command")
    if args.command == "compare-checkpoints":
        if args.checkpoint_a is None or args.checkpoint_b is None:
            parser.error("compare-checkpoints requires --checkpoint-a and --checkpoint-b")
    elif args.checkpoint_a is not None or args.checkpoint_b is not None:
        parser.error("checkpoint comparison paths require compare-checkpoints")
    return args


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def payload_hashes(payload: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in sorted(payload.items())
    }


class FrozenSession6Store:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.packed_root = self.root / "data" / "packed_v2"
        self.batch_root = self.root / "data" / "batches_v2"
        self.packing_report = read_json(self.packed_root / "packing_report.json")
        self.batch_report = read_json(self.batch_root / "batch_report.json")
        self.tokenizer_payload = read_json(
            self.root / "artifacts" / "tokenizer_v2" / "tokenizer.json"
        )
        self.batches = read_jsonl_gz(
            self.root / self.batch_report["paths"]["batches"]
        )
        sequence_rows = read_jsonl_gz(
            self.root / self.packing_report["paths"]["sequences"]
        )
        self.sequences = {
            int(row["sequence_index"]): row for row in sequence_rows
        }
        self.tokens = np.memmap(
            self.root / self.packing_report["paths"]["input_ids"],
            mode="r",
            dtype="<u2",
        )
        self.loss_mask = np.memmap(
            self.root / self.packing_report["paths"]["loss_mask"],
            mode="r",
            dtype="u1",
        )
        self.segment_ids = np.memmap(
            self.root / self.packing_report["paths"]["segment_ids"],
            mode="r",
            dtype="<i2",
        )
        self.position_ids = np.memmap(
            self.root / self.packing_report["paths"]["position_ids"],
            mode="r",
            dtype="<u2",
        )
        self._validate_lineage()

    def _validate_lineage(self) -> None:
        if self.packing_report["status"] != "FROZEN":
            raise RuntimeError("Session 6 packing report is not frozen")
        if self.batch_report["status"] != "FROZEN":
            raise RuntimeError("Session 6 batch report is not frozen")
        if self.batch_report["packing_hash"] != self.packing_report["packing_hash"]:
            raise RuntimeError("batch plan and packed tensors have different lineage")
        if self.packing_report["tokenizer_hash"] != self.tokenizer_payload["tokenizer_hash"]:
            raise RuntimeError("packed tensors and tokenizer have different lineage")
        if len(self.batches) != int(self.batch_report["microbatches"]):
            raise RuntimeError("batch index record count does not match its report")
        observed_stages = tuple(dict.fromkeys(row["stage"] for row in self.batches))
        if observed_stages != STAGE_ORDER:
            raise RuntimeError(
                f"unexpected curriculum order: {observed_stages!r}"
            )

    def payload_for_base_batch(
        self, batch: dict[str, Any], verify_hash: bool
    ) -> dict[str, np.ndarray]:
        sources = {
            "input_ids": self.tokens,
            "loss_mask": self.loss_mask,
            "segment_ids": self.segment_ids,
            "position_ids": self.position_ids,
        }
        payload: dict[str, np.ndarray] = {}
        expected_length = int(batch["sequence_length"])
        for name, source in sources.items():
            rows = []
            for sequence_index in batch["sequence_indices"]:
                sequence = self.sequences[int(sequence_index)]
                length = int(sequence["sequence_length"])
                if length != expected_length:
                    raise RuntimeError("mixed sequence lengths inside a frozen batch")
                start = int(sequence["global_token_offset"])
                rows.append(source[start : start + length])
            payload[name] = np.ascontiguousarray(np.stack(rows))
        if verify_hash:
            observed = payload_hashes(payload)
            if observed != batch["tensor_hashes"]:
                raise RuntimeError(
                    f"tensor hash mismatch in base batch {batch['batch_index']}"
                )
        return payload

    def superbatches(self, combine: int) -> Iterator[Superbatch]:
        if combine < 1:
            raise ValueError("combine must be positive")
        pending: list[dict[str, Any]] = []
        index = 0
        for batch in self.batches:
            if pending and (
                batch["stage"] != pending[0]["stage"]
                or int(batch["sequence_length"])
                != int(pending[0]["sequence_length"])
            ):
                yield Superbatch(
                    index,
                    str(pending[0]["stage"]),
                    int(pending[0]["sequence_length"]),
                    tuple(pending),
                )
                index += 1
                pending = []
            pending.append(batch)
            if len(pending) == combine:
                yield Superbatch(
                    index,
                    str(pending[0]["stage"]),
                    int(pending[0]["sequence_length"]),
                    tuple(pending),
                )
                index += 1
                pending = []
        if pending:
            yield Superbatch(
                index,
                str(pending[0]["stage"]),
                int(pending[0]["sequence_length"]),
                tuple(pending),
            )

    def payload_for_superbatch(
        self, superbatch: Superbatch, verify_hashes: bool
    ) -> dict[str, np.ndarray]:
        pieces = [
            self.payload_for_base_batch(batch, verify_hashes)
            for batch in superbatch.base_batches
        ]
        return {
            name: np.ascontiguousarray(
                np.concatenate([piece[name] for piece in pieces], axis=0)
            )
            for name in pieces[0]
        }

    def token_display(self, token_id: int) -> str:
        token = self.tokenizer_payload["tokens"][int(token_id)]
        if int(token["token_id"]) != int(token_id):
            raise RuntimeError("tokenizer token table is not indexed by token ID")
        return str(token["display"])


class SegmentCausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden size must be divisible by attention heads")
        self.heads = heads
        self.head_size = hidden_size // heads
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        query, key, value = self.query_key_value(hidden).chunk(3, dim=-1)
        query = query.view(batch, length, self.heads, self.head_size).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_size).transpose(1, 2)
        value = value.view(batch, length, self.heads, self.head_size).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_size)

        valid = segment_ids >= 0
        same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
        causal = torch.ones(
            (length, length), dtype=torch.bool, device=hidden.device
        ).tril()
        allowed = same_segment & valid[:, :, None] & valid[:, None, :] & causal
        identity = torch.eye(length, dtype=torch.bool, device=hidden.device)[None]
        allowed = allowed | ((~valid)[:, :, None] & identity)
        scores = scores.masked_fill(
            ~allowed[:, None, :, :], torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        attended = attended * valid[:, :, None]
        return self.output(attended)


class RMSNorm(nn.Module):
    """RMS-only pre-normalization used by modern decoder-only LLMs."""

    def __init__(self, hidden_size: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = float(epsilon)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden.dtype
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon
        )
        return (normalized * self.weight.float()).to(input_dtype)


def build_norm(model_config: dict[str, Any]) -> nn.Module:
    normalization = str(model_config.get("normalization", "layernorm")).lower()
    hidden_size = int(model_config["hidden_size"])
    if normalization == "layernorm":
        return nn.LayerNorm(hidden_size)
    if normalization == "rmsnorm":
        return RMSNorm(
            hidden_size,
            float(model_config.get("normalization_epsilon", 1e-6)),
        )
    raise ValueError(f"unsupported normalization: {normalization}")


class DecoderBlock(nn.Module):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()
        hidden_size = int(model_config["hidden_size"])
        heads = int(model_config["heads"])
        feed_forward_size = int(model_config["feed_forward_size"])
        self.ffn_type = str(model_config.get("ffn_type", "gelu")).lower()
        self.attention_norm = build_norm(model_config)
        self.attention = SegmentCausalSelfAttention(hidden_size, heads)
        self.ffn_norm = build_norm(model_config)
        if self.ffn_type == "gelu":
            self.ffn_in = nn.Linear(hidden_size, feed_forward_size)
            self.ffn_out = nn.Linear(feed_forward_size, hidden_size)
        elif self.ffn_type == "swiglu":
            linear_bias = bool(model_config.get("ffn_bias", False))
            self.ffn_gate = nn.Linear(
                hidden_size, feed_forward_size, bias=linear_bias
            )
            self.ffn_up = nn.Linear(
                hidden_size, feed_forward_size, bias=linear_bias
            )
            self.ffn_down = nn.Linear(
                feed_forward_size, hidden_size, bias=linear_bias
            )
        else:
            raise ValueError(f"unsupported FFN type: {self.ffn_type}")

    def forward(self, hidden: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), segment_ids)
        normalized = self.ffn_norm(hidden)
        if self.ffn_type == "gelu":
            branch = self.ffn_out(F.gelu(self.ffn_in(normalized)))
        else:
            branch = self.ffn_down(
                F.silu(self.ffn_gate(normalized)) * self.ffn_up(normalized)
            )
        hidden = hidden + branch
        return hidden


class MTPDecoder(nn.Module):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.hidden_size = int(model_config["hidden_size"])
        self.activation_checkpointing = bool(
            model_config["activation_checkpointing"]
        )
        self.token_embedding = nn.Embedding(
            int(model_config["vocab_size"]), self.hidden_size
        )
        self.position_embedding = nn.Embedding(
            int(model_config["maximum_context_length"]), self.hidden_size
        )
        self.blocks = nn.ModuleList(
            DecoderBlock(model_config)
            for _ in range(int(model_config["layers"]))
        )
        self.final_norm = build_norm(model_config)
        self.head2 = nn.Linear(
            self.hidden_size, int(model_config["vocab_size"]), bias=False
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    @property
    def head1_weight(self) -> torch.Tensor:
        return self.token_embedding.weight

    def hidden_states(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        position_ids: torch.Tensor,
        trace: bool,
    ) -> torch.Tensor:
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        hidden = token_embeddings + position_embeddings
        if trace:
            trace_tensor("token_embeddings", token_embeddings, "[B,T,D]")
            trace_tensor("position_embeddings", position_embeddings, "[B,T,D]")
            trace_tensor("embeddings", hidden, "[B,T,D]")
        for index, block in enumerate(self.blocks):
            if self.activation_checkpointing and self.training:
                hidden = checkpoint(block, hidden, segment_ids, use_reentrant=False)
            else:
                hidden = block(hidden, segment_ids)
            if trace:
                trace_tensor(f"block_{index:02d}_hidden", hidden, "[B,T,D]")
        hidden = self.final_norm(hidden)
        if trace:
            trace_tensor("final_hidden", hidden, "[B,T,D]")
        return hidden


def unique_parameters(module: nn.Module) -> list[nn.Parameter]:
    output: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in module.parameters():
        identity = id(parameter)
        if parameter.requires_grad and identity not in seen:
            seen.add(identity)
            output.append(parameter)
    return output


def cuda_memory_mib() -> tuple[float, float]:
    return (
        torch.cuda.memory_allocated() / MIB,
        torch.cuda.memory_reserved() / MIB,
    )


def trace_tensor(name: str, tensor: torch.Tensor, dimensions: str) -> None:
    allocated, reserved = cuda_memory_mib()
    tensor_mib = tensor.numel() * tensor.element_size() / MIB
    print(
        f"TRACE {name:<22} shape={str(tuple(tensor.shape)):<18} "
        f"{dimensions:<9} dtype={str(tensor.dtype):<13} "
        f"tensor={tensor_mib:>7.2f} MiB allocated={allocated:>8.2f} MiB "
        f"reserved={reserved:>8.2f} MiB"
    )


def numpy_to_cuda(payload: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.as_tensor(
            payload["input_ids"], dtype=torch.long, device="cuda"
        ),
        "loss_mask": torch.as_tensor(
            payload["loss_mask"], dtype=torch.bool, device="cuda"
        ),
        "segment_ids": torch.as_tensor(
            payload["segment_ids"], dtype=torch.long, device="cuda"
        ),
        "position_ids": torch.as_tensor(
            payload["position_ids"], dtype=torch.long, device="cuda"
        ),
    }


def torch_target_masks(
    loss_mask: torch.Tensor, segment_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    head1 = (
        loss_mask[:, 1:]
        & (segment_ids[:, :-1] >= 0)
        & (segment_ids[:, 1:] >= 0)
        & (segment_ids[:, :-1] == segment_ids[:, 1:])
    )
    segment_t = segment_ids[:, :-2]
    segment_t1 = segment_ids[:, 1:-1]
    segment_t2 = segment_ids[:, 2:]
    head2 = (
        loss_mask[:, 2:]
        & (segment_t >= 0)
        & (segment_t1 >= 0)
        & (segment_t2 >= 0)
        & (segment_t == segment_t1)
        & (segment_t1 == segment_t2)
    )
    return head1, head2


def backward_chunked_head(
    active_hidden_graph: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int,
    loss_weight: float,
    scaler: torch.amp.GradScaler,
    trace: bool,
) -> tuple[torch.Tensor, float]:
    hidden_leaf = active_hidden_graph.detach().requires_grad_(True)
    token_count = int(targets.numel())
    if token_count == 0:
        raise RuntimeError("MTP head received zero active targets")
    loss_sum = 0.0
    for chunk_index, start in enumerate(range(0, token_count, chunk_size)):
        end = min(start + chunk_size, token_count)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = F.linear(hidden_leaf[start:end], weight)
            chunk_loss_sum = F.cross_entropy(
                logits, targets[start:end], reduction="sum"
            )
        if trace and chunk_index == 0:
            trace_tensor("first_chunk_logits", logits, "[C,V]")
            trace_tensor("first_chunk_targets", targets[start:end], "[C]")
        scaled_mean = loss_weight * chunk_loss_sum / token_count
        scaler.scale(scaled_mean).backward()
        loss_sum += float(chunk_loss_sum.detach().float())
    if hidden_leaf.grad is None:
        raise RuntimeError("chunked head did not produce a hidden-state gradient")
    return hidden_leaf.grad, loss_sum / token_count


def cuda_training_step(
    model: MTPDecoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    parameters: list[nn.Parameter],
    payload: dict[str, np.ndarray],
    training_config: dict[str, Any],
    trace: bool,
) -> dict[str, float | int]:
    tensors = numpy_to_cuda(payload)
    input_ids = tensors["input_ids"]
    loss_mask = tensors["loss_mask"]
    segment_ids = tensors["segment_ids"]
    position_ids = tensors["position_ids"]
    optimizer.zero_grad(set_to_none=True)
    if trace:
        trace_tensor("input_ids", input_ids, "[B,T]")
        trace_tensor("loss_mask", loss_mask, "[B,T]")
        trace_tensor("segment_ids", segment_ids, "[B,T]")
        trace_tensor("position_ids", position_ids, "[B,T]")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        hidden = model.hidden_states(
            input_ids, segment_ids, position_ids, trace=trace
        )
        head1_mask, head2_mask = torch_target_masks(loss_mask, segment_ids)
        head1_hidden = hidden[:, :-1, :][head1_mask]
        head2_hidden = hidden[:, :-2, :][head2_mask]
        head1_targets = input_ids[:, 1:][head1_mask]
        head2_targets = input_ids[:, 2:][head2_mask]
    if trace:
        trace_tensor("head1_active_hidden", head1_hidden, "[N1,D]")
        trace_tensor("head1_targets", head1_targets, "[N1]")
        trace_tensor("head2_active_hidden", head2_hidden, "[N2,D]")
        trace_tensor("head2_targets", head2_targets, "[N2]")

    gradient1, loss1 = backward_chunked_head(
        head1_hidden,
        head1_targets,
        model.head1_weight,
        int(training_config["chunk_size"]),
        float(training_config["head1_loss_weight"]),
        scaler,
        trace,
    )
    gradient2, loss2 = backward_chunked_head(
        head2_hidden,
        head2_targets,
        model.head2.weight,
        int(training_config["chunk_size"]),
        float(training_config["head2_loss_weight"]),
        scaler,
        False,
    )
    torch.autograd.backward(
        tensors=(head1_hidden, head2_hidden),
        grad_tensors=(gradient1, gradient2),
    )
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters, float(training_config["gradient_clip_norm"])
    )
    before = model.head2.weight[0, 0].detach().clone()
    scaler.step(optimizer)
    scaler.update()
    parameter_delta = float((model.head2.weight[0, 0].detach() - before).abs())
    return {
        "head1_loss": loss1,
        "head2_loss": loss2,
        "total_loss": (
            float(training_config["head1_loss_weight"]) * loss1
            + float(training_config["head2_loss_weight"]) * loss2
        ),
        "head1_targets": int(head1_targets.numel()),
        "head2_targets": int(head2_targets.numel()),
        "gradient_norm": float(gradient_norm),
        "parameter_delta": parameter_delta,
    }


def config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * MIB), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def trim_metrics_to_checkpoint(path: Path, global_step: int) -> int:
    if not path.exists():
        return 0
    kept: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if int(row.get("global_step", 0)) <= global_step:
            kept.append(json.dumps(row, sort_keys=True))
        else:
            removed += 1
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    os.replace(temporary, path)
    return removed


def scheduled_learning_rate(step: int, training_config: dict[str, Any]) -> float:
    maximum = float(training_config["learning_rate"])
    minimum = float(training_config["minimum_learning_rate"])
    warmup = int(training_config["warmup_steps"])
    planned = int(training_config["planned_optimizer_steps"])
    if step <= warmup:
        return maximum * step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, planned - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (maximum - minimum) * cosine


def update_early_stopping_state(
    state: dict[str, Any],
    validation: dict[str, float | int],
    training_config: dict[str, Any],
) -> bool:
    """Update validation tracking and return True only for a new best score."""
    metric = str(training_config["early_stopping_metric"])
    if metric != "total_loss":
        raise ValueError("early stopping currently supports total_loss only")
    current = float(validation[metric])
    minimum_delta = float(training_config["early_stopping_minimum_delta"])
    previous_best = float(state.get("best_validation_total_loss", math.inf))
    improved = current < previous_best - minimum_delta
    if improved:
        state["best_validation_total_loss"] = current
        state["best_validation_step"] = int(state["global_step"])
        state["best_validation"] = dict(validation)
        state["validations_without_improvement"] = 0
    else:
        state["validations_without_improvement"] = int(
            state.get("validations_without_improvement", 0)
        ) + 1
    return improved


def load_validation_payload(
    config: dict[str, Any], store: FrozenSession6Store
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frozen_probe_path = config["validation"].get("frozen_probe_path")
    frozen_manifest_path = config["validation"].get(
        "frozen_probe_manifest_path"
    )
    if frozen_probe_path and frozen_manifest_path:
        archive_path = store.root / str(frozen_probe_path)
        manifest_path = store.root / str(frozen_manifest_path)
        with np.load(archive_path, allow_pickle=False) as archive:
            payload = {
                name: np.ascontiguousarray(archive[name])
                for name in (
                    "input_ids",
                    "loss_mask",
                    "segment_ids",
                    "position_ids",
                )
            }
        manifest = read_json(manifest_path)
        if payload_hashes(payload) != manifest["component_hashes"]:
            raise RuntimeError("frozen validation probe hash mismatch")
        return payload, manifest

    source_root = store.root / "src"
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from era6.tokenizer_candidates import load_lossless_tokenizer

    tokenizer = load_lossless_tokenizer(
        store.root / "artifacts" / "tokenizer_v2" / "tokenizer.json"
    )
    tokenized_report = read_json(
        store.root / "data" / "tokenized_v2" / "tokenized_report.json"
    )
    if "target_loss_bearing_tokens" not in config["validation"]:
        from era6.training import build_validation_probe

        return build_validation_probe(
            store.root, tokenized_report, tokenizer, config["validation"]
        )
    return build_balanced_validation_probe(
        store.root,
        tokenized_report,
        int(tokenizer.special_token_ids["<pad>"]),
        config["validation"],
    )


def validation_record_rank(
    row: dict[str, Any], selection_seed: str, lane: str
) -> tuple[bytes, str]:
    digest = hashlib.sha256(
        f"{selection_seed}|{lane}|{row['record_id']}".encode("utf-8")
    ).digest()
    return digest, str(row["record_id"])


def build_balanced_validation_probe(
    root: Path,
    tokenized_report: dict[str, Any],
    pad_token_id: int,
    probe_config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select distinct validation documents until each lane reaches its quota."""
    length = int(probe_config["sequence_length"])
    minimum_loss = int(probe_config["minimum_loss_tokens_per_sequence"])
    selection_seed = str(probe_config["selection_seed"])
    target_loss_tokens = int(probe_config["target_loss_bearing_tokens"])
    maximum_per_lane = int(probe_config["maximum_sequences_per_lane"])
    if length < 3:
        raise ValueError("validation sequence_length must be at least 3")
    if target_loss_tokens < 1:
        raise ValueError("target_loss_bearing_tokens must be positive")
    if maximum_per_lane < 1:
        raise ValueError("maximum_sequences_per_lane must be positive")

    validation_shards = {
        str(row["lane"]): row
        for row in tokenized_report["shards"]
        if row["permission"] == "validation"
    }
    lanes = sorted(validation_shards)
    if not lanes:
        raise RuntimeError("tokenized report has no validation lanes")
    per_lane_target = math.ceil(target_loss_tokens / len(lanes))

    selected: list[dict[str, Any]] = []
    token_rows: list[np.ndarray] = []
    loss_rows: list[np.ndarray] = []
    segment_rows: list[np.ndarray] = []
    position_rows: list[np.ndarray] = []
    lane_loss_tokens: dict[str, int] = {}
    lane_sequences: dict[str, int] = {}

    for lane in lanes:
        shard = validation_shards[lane]
        manifest = read_json(root / shard["manifest_path"])["manifest"]
        extra = manifest["extra"]
        index_rows = sorted(
            read_jsonl_gz(root / extra["index_path"]),
            key=lambda row: validation_record_rank(row, selection_seed, lane),
        )
        shard_tokens = np.memmap(root / extra["tokens_path"], mode="r", dtype="<u2")
        shard_loss = np.memmap(root / extra["loss_path"], mode="r", dtype="u1")
        accumulated = 0
        lane_count = 0

        for row in index_rows:
            start = int(row["token_offset"])
            end = start + int(row["token_count"])
            record_tokens = shard_tokens[start:end]
            record_loss = shard_loss[start:end]
            if len(record_tokens) < 2:
                continue
            possible_starts = list(
                range(
                    0,
                    max(1, len(record_tokens) - length + 1),
                    max(1, length // 2),
                )
            )
            possible_starts.append(max(0, len(record_tokens) - length))
            best_start = max(
                sorted(set(possible_starts)),
                key=lambda value: (
                    int(record_loss[value + 1 : value + length].sum()),
                    -value,
                ),
            )
            window_end = min(len(record_tokens), best_start + length)
            window_tokens = record_tokens[best_start:window_end]
            window_loss = record_loss[best_start:window_end]
            effective_loss = int(window_loss[1:].sum())
            if effective_loss < minimum_loss:
                continue

            nonpadding = len(window_tokens)
            ids = np.full(length, pad_token_id, dtype="<u2")
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
                    "record_token_start": best_start,
                    "record_token_end": best_start + nonpadding,
                    "nonpadding_tokens": nonpadding,
                    "loss_bearing_tokens": int(mask[1:].sum()),
                    "permission": "validation",
                }
            )
            accumulated += int(mask[1:].sum())
            lane_count += 1
            if accumulated >= per_lane_target:
                break
            if lane_count >= maximum_per_lane:
                break

        if accumulated < per_lane_target:
            raise RuntimeError(
                f"validation lane {lane!r} supplied {accumulated:,} targets; "
                f"need {per_lane_target:,}"
            )
        lane_loss_tokens[lane] = accumulated
        lane_sequences[lane] = lane_count

    payload = {
        "input_ids": np.ascontiguousarray(np.stack(token_rows)),
        "loss_mask": np.ascontiguousarray(np.stack(loss_rows)),
        "segment_ids": np.ascontiguousarray(np.stack(segment_rows)),
        "position_ids": np.ascontiguousarray(np.stack(position_rows)),
    }
    actual_head1_targets = int(payload["loss_mask"][:, 1:].sum())
    actual_head2_targets = int(payload["loss_mask"][:, 2:].sum())
    if actual_head1_targets < target_loss_tokens:
        raise RuntimeError("constructed validation probe missed its target")
    manifest = {
        "schema_version": 2,
        "permission": "validation",
        "selection_seed": selection_seed,
        "sequence_length": length,
        "target_loss_bearing_tokens": target_loss_tokens,
        "head1_loss_bearing_tokens": actual_head1_targets,
        "head2_loss_bearing_tokens": actual_head2_targets,
        "balanced_target_per_lane": per_lane_target,
        "lanes": lanes,
        "sequences": len(selected),
        "sequences_by_lane": lane_sequences,
        "loss_bearing_tokens_by_lane": lane_loss_tokens,
        "selected": selected,
        "component_hashes": payload_hashes(payload),
    }
    manifest["probe_hash"] = f"sha256:{config_sha256(manifest)}"
    return payload, manifest


def inspect_validation(config: dict[str, Any]) -> None:
    store = FrozenSession6Store(Path(config["session6_root"]))
    payload, manifest = load_validation_payload(config, store)
    expected = int(config["validation"]["target_loss_bearing_tokens"])
    observed = int(manifest["head1_loss_bearing_tokens"])
    if observed < expected:
        raise RuntimeError("validation target count is below the configured minimum")
    if len({row["record_id"] for row in manifest["selected"]}) != int(
        manifest["sequences"]
    ):
        raise RuntimeError("validation probe reused a document")
    print("VALIDATION INSPECTION PASSED")
    print(f"  payload shape : {list(payload['input_ids'].shape)} [B, T]")
    print(f"  unique documents: {manifest['sequences']:,}")
    print(f"  H1 targets: {manifest['head1_loss_bearing_tokens']:,}")
    print(f"  H2 targets: {manifest['head2_loss_bearing_tokens']:,}")
    print(f"  per-lane H1 targets: {manifest['loss_bearing_tokens_by_lane']}")
    print(f"  probe hash: {manifest['probe_hash']}")


def chunked_loss_sum_no_grad(
    active_hidden: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int,
) -> float:
    total = 0.0
    for start in range(0, int(targets.numel()), chunk_size):
        end = min(start + chunk_size, int(targets.numel()))
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = F.linear(active_hidden[start:end], weight)
            loss_sum = F.cross_entropy(logits, targets[start:end], reduction="sum")
        total += float(loss_sum.float())
    return total


def evaluate_mtp(
    model: MTPDecoder,
    payload: dict[str, np.ndarray],
    training_config: dict[str, Any],
    validation_config: dict[str, Any],
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    chunk_size = int(training_config["chunk_size"])
    evaluation_batch_size = int(
        validation_config.get("evaluation_batch_size", payload["input_ids"].shape[0])
    )
    if evaluation_batch_size < 1:
        raise ValueError("validation evaluation_batch_size must be positive")
    loss1_sum = 0.0
    loss2_sum = 0.0
    count1 = 0
    count2 = 0
    sequences = int(payload["input_ids"].shape[0])
    for start in range(0, sequences, evaluation_batch_size):
        end = min(start + evaluation_batch_size, sequences)
        tensors = numpy_to_cuda(
            {name: value[start:end] for name, value in payload.items()}
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            hidden = model.hidden_states(
                tensors["input_ids"],
                tensors["segment_ids"],
                tensors["position_ids"],
                trace=False,
            )
            head1_mask, head2_mask = torch_target_masks(
                tensors["loss_mask"], tensors["segment_ids"]
            )
            head1_hidden = hidden[:, :-1, :][head1_mask]
            head2_hidden = hidden[:, :-2, :][head2_mask]
            head1_targets = tensors["input_ids"][:, 1:][head1_mask]
            head2_targets = tensors["input_ids"][:, 2:][head2_mask]
        loss1_sum += chunked_loss_sum_no_grad(
            head1_hidden, head1_targets, model.head1_weight, chunk_size
        )
        loss2_sum += chunked_loss_sum_no_grad(
            head2_hidden, head2_targets, model.head2.weight, chunk_size
        )
        count1 += int(head1_targets.numel())
        count2 += int(head2_targets.numel())
    loss1 = loss1_sum / count1
    loss2 = loss2_sum / count2
    if was_training:
        model.train()
    return {
        "head1_loss": loss1,
        "head2_loss": loss2,
        "total_loss": (
            float(training_config["head1_loss_weight"]) * loss1
            + float(training_config["head2_loss_weight"]) * loss2
        ),
        "head1_perplexity": math.exp(min(loss1, 20.0)),
        "head2_perplexity": math.exp(min(loss2, 20.0)),
        "head1_targets": count1,
        "head2_targets": count2,
    }


def save_training_checkpoint(
    path: Path,
    *,
    model: MTPDecoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    state: dict[str, Any],
    configuration_hash: str,
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    started = time.perf_counter()
    torch.save(
        {
            "schema_version": 1,
            "configuration_sha256": configuration_hash,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "training_state": state,
            "rng_state": {
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
        },
        temporary,
    )
    os.replace(temporary, path)
    return time.perf_counter() - started


def archive_training_checkpoint(
    source: Path,
    archive_directory: Path,
    global_step: int,
) -> dict[str, Any] | None:
    """Retain an immutable, step-numbered copy of a completed checkpoint."""
    archive_directory.mkdir(parents=True, exist_ok=True)
    archived = archive_directory / f"step_{global_step:08d}.pt"
    if archived.exists():
        return None
    method = "hardlink"
    try:
        os.link(source, archived)
    except OSError:
        method = "copy"
        temporary = archived.with_suffix(archived.suffix + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, archived)
    record = {
        "schema_version": 1,
        "global_step": int(global_step),
        "path": str(archived.resolve()),
        "bytes": int(archived.stat().st_size),
        "retention_method": method,
        "created_unix_time": time.time(),
    }
    append_jsonl(archive_directory.parent / "checkpoint_index.jsonl", record)
    return record


def load_training_checkpoint(
    path: Path,
    *,
    model: MTPDecoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    configuration_hash: str,
) -> dict[str, Any]:
    checkpoint_payload = torch.load(path, map_location="cuda", weights_only=False)
    if checkpoint_payload["configuration_sha256"] != configuration_hash:
        raise RuntimeError("checkpoint configuration hash does not match this run")
    model.load_state_dict(checkpoint_payload["model_state"])
    optimizer.load_state_dict(checkpoint_payload["optimizer_state"])
    scaler.load_state_dict(checkpoint_payload["scaler_state"])
    rng = checkpoint_payload["rng_state"]
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in rng["torch_cuda"]])
    return dict(checkpoint_payload["training_state"])


def update_structural_hash(hasher: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"tensor\0")
        hasher.update(str(tensor.dtype).encode("utf-8") + b"\0")
        hasher.update(json.dumps(list(tensor.shape)).encode("utf-8") + b"\0")
        hasher.update(memoryview(tensor.numpy()))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"ndarray\0")
        hasher.update(str(array.dtype).encode("utf-8") + b"\0")
        hasher.update(json.dumps(list(array.shape)).encode("utf-8") + b"\0")
        hasher.update(memoryview(array))
    elif isinstance(value, dict):
        hasher.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            update_structural_hash(hasher, key)
            update_structural_hash(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(type(value).__name__.encode("utf-8") + b"\0")
        for item in value:
            update_structural_hash(hasher, item)
    elif value is None:
        hasher.update(b"none\0")
    elif isinstance(value, (str, int, float, bool)):
        hasher.update(type(value).__name__.encode("utf-8") + b"\0")
        hasher.update(repr(value).encode("utf-8") + b"\0")
    else:
        raise TypeError(f"unsupported checkpoint value for hashing: {type(value)!r}")


def structural_sha256(value: Any) -> str:
    hasher = hashlib.sha256()
    update_structural_hash(hasher, value)
    return hasher.hexdigest()


def fingerprint_checkpoint(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    payload = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    logical_state = {
        key: value
        for key, value in payload["training_state"].items()
        if key != "active_training_seconds"
    }
    result = {
        "path": str(path.resolve()),
        "size_bytes": path.resolve().stat().st_size,
        "configuration_sha256": payload["configuration_sha256"],
        "model_sha256": structural_sha256(payload["model_state"]),
        "optimizer_sha256": structural_sha256(payload["optimizer_state"]),
        "scaler_sha256": structural_sha256(payload["scaler_state"]),
        "rng_sha256": structural_sha256(payload["rng_state"]),
        "logical_training_state": logical_state,
        "logical_training_state_sha256": structural_sha256(logical_state),
    }
    result["fingerprint_seconds"] = time.perf_counter() - started
    del payload
    gc.collect()
    return result


def compare_checkpoints(path_a: Path, path_b: Path) -> None:
    print("FINGERPRINTING CHECKPOINT A", flush=True)
    fingerprint_a = fingerprint_checkpoint(path_a)
    print("FINGERPRINTING CHECKPOINT B", flush=True)
    fingerprint_b = fingerprint_checkpoint(path_b)
    fields = (
        "configuration_sha256",
        "model_sha256",
        "optimizer_sha256",
        "scaler_sha256",
        "rng_sha256",
        "logical_training_state_sha256",
    )
    print("CHECKPOINT EQUIVALENCE")
    all_equal = True
    for field in fields:
        equal = fingerprint_a[field] == fingerprint_b[field]
        all_equal &= equal
        print(f"  {field:<31}: {'MATCH' if equal else 'DIFFER'}")
    print(
        "  logical state A               : "
        f"{json.dumps(fingerprint_a['logical_training_state'], sort_keys=True)}"
    )
    print(
        "  logical state B               : "
        f"{json.dumps(fingerprint_b['logical_training_state'], sort_keys=True)}"
    )
    if not all_equal:
        raise RuntimeError("checkpoint equivalence failed")
    print("CHECKPOINT EQUIVALENCE PASSED")
    print("  Interrupted/resumed and uninterrupted training are bitwise equivalent.")


def build_target_masks(
    loss_mask: np.ndarray, segment_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid1 = (
        (loss_mask[:, 1:] != 0)
        & (segment_ids[:, :-1] >= 0)
        & (segment_ids[:, 1:] >= 0)
        & (segment_ids[:, :-1] == segment_ids[:, 1:])
    )
    segment_t = segment_ids[:, :-2]
    segment_t1 = segment_ids[:, 1:-1]
    segment_t2 = segment_ids[:, 2:]
    valid2 = (
        (loss_mask[:, 2:] != 0)
        & (segment_t >= 0)
        & (segment_t1 >= 0)
        & (segment_t2 >= 0)
        & (segment_t == segment_t1)
        & (segment_t1 == segment_t2)
    )
    return valid1, valid2


def find_boundary_window(
    segment_ids: np.ndarray, head1_mask: np.ndarray
) -> tuple[int, int, int]:
    for row_index, segments in enumerate(segment_ids):
        boundary_positions = np.flatnonzero(
            (segments[:-1] >= 0)
            & (segments[1:] >= 0)
            & (segments[:-1] != segments[1:])
        )
        if boundary_positions.size:
            boundary = int(boundary_positions[0])
            return row_index, max(0, boundary - 5), min(len(segments) - 2, boundary + 7)
    for row_index, mask in enumerate(head1_mask):
        active = np.flatnonzero(mask)
        if active.size:
            anchor = int(active[0])
            return row_index, max(0, anchor - 2), min(len(mask) - 1, anchor + 10)
    raise RuntimeError("inspection superbatch has no boundary or active target")


def inspect_data(config: dict[str, Any]) -> None:
    root = Path(config["session6_root"])
    combine = int(config["data"]["base_batches_per_superbatch"])
    verify_hashes = bool(config["data"]["verify_base_batch_hashes"])
    store = FrozenSession6Store(root)
    superbatches = list(store.superbatches(combine))

    stage_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for superbatch in superbatches:
        stats = stage_stats[superbatch.stage]
        stats["superbatches"] += 1
        stats["base_batches"] += len(superbatch.base_batches)
        stats["sequences"] += len(superbatch.sequence_indices)
        stats["physical_tokens"] += superbatch.physical_tokens
        stats["reported_head1_targets"] += superbatch.reported_head1_targets
        stats["sequence_length"] = superbatch.sequence_length

    print("SESSION 6 LINEAGE")
    print(f"  packing status : {store.packing_report['status']}")
    print(f"  batching status: {store.batch_report['status']}")
    print(f"  packing hash   : {store.packing_report['packing_hash']}")
    print(f"  tokenizer hash : {store.packing_report['tokenizer_hash']}")
    print(f"  vocabulary V   : {store.tokenizer_payload['vocab_size']:,}")
    print()
    print("CURRICULUM-PRESERVING SUPERBATCH PLAN")
    print(
        "stage                      T  base batches  supersteps  sequences  "
        "physical tokens  head1 targets"
    )
    print("-" * 104)
    for stage in STAGE_ORDER:
        stats = stage_stats[stage]
        print(
            f"{stage:<27} {stats['sequence_length']:>3,} "
            f"{stats['base_batches']:>13,} {stats['superbatches']:>11,} "
            f"{stats['sequences']:>10,} {stats['physical_tokens']:>16,} "
            f"{stats['reported_head1_targets']:>14,}"
        )
    print("-" * 104)
    print(
        f"{'one corpus pass':<31} {len(store.batches):>13,} "
        f"{len(superbatches):>11,} {len(store.sequences):>10,} "
        f"{store.packing_report['physical_tokens']:>16,} "
        f"{store.packing_report['loss_bearing_tokens']:>14,}"
    )

    first = superbatches[0]
    payload = store.payload_for_superbatch(first, verify_hashes)
    head1_mask, head2_mask = build_target_masks(
        payload["loss_mask"], payload["segment_ids"]
    )
    if int(head1_mask.sum()) != first.reported_head1_targets:
        raise RuntimeError(
            "head-1 shifted target count disagrees with the frozen batch reports"
        )

    batch_size, length = payload["input_ids"].shape
    print()
    print("FIRST SUPERBATCH TENSOR CONTRACT")
    print(f"  input_ids   : {payload['input_ids'].shape} [B, T] uint16 token IDs")
    print(f"  loss_mask   : {payload['loss_mask'].shape} [B, T] target eligibility")
    print(f"  segment_ids : {payload['segment_ids'].shape} [B, T] document identity")
    print(f"  position_ids: {payload['position_ids'].shape} [B, T] within-document position")
    print(f"  B={batch_size}: sequences | T={length}: positions | B*T={batch_size * length:,} tokens")
    print(f"  head 1 t+1 targets: {int(head1_mask.sum()):,}")
    print(f"  head 2 t+2 targets: {int(head2_mask.sum()):,}")
    print(f"  verified source hashes: {len(first.base_batches)} / {len(first.base_batches)}")

    row_index, start, end = find_boundary_window(payload["segment_ids"], head1_mask)
    tokens = payload["input_ids"][row_index]
    segments = payload["segment_ids"][row_index]
    print()
    print("STRING-LEVEL MTP ALIGNMENT")
    print(f"  sequence row {row_index}; logits from h[t] feed both heads")
    print(
        "  t | INPUT x[t]             | HEAD 1 x[t+1]          | "
        "HEAD 2 x[t+2]          | segments     | H1  H2"
    )
    print("-" * 104)
    for position in range(start, end + 1):
        input_token = repr(store.token_display(int(tokens[position])))
        target1 = repr(store.token_display(int(tokens[position + 1])))
        target2 = repr(store.token_display(int(tokens[position + 2])))
        segment_triplet = (
            f"{int(segments[position])}->{int(segments[position + 1])}"
            f"->{int(segments[position + 2])}"
        )
        print(
            f"{position:>3} | {input_token:<22} | {target1:<22} | "
            f"{target2:<22} | {segment_triplet:<12} | "
            f"{str(bool(head1_mask[row_index, position])):<3} "
            f"{str(bool(head2_mask[row_index, position])):<3}"
        )

    expected_targets = int(
        config["data"]["expected_loss_bearing_targets_per_pass"]
    )
    if int(store.packing_report["loss_bearing_tokens"]) != expected_targets:
        raise RuntimeError("configured and packed loss-target totals disagree")
    expected_supersteps = sum(
        math.ceil(int(summary["microbatches"]) / combine)
        for summary in store.batch_report["stage_summaries"]
    )
    if len(superbatches) != expected_supersteps:
        raise RuntimeError("superbatch count does not reconcile by stage")
    print()
    print("DATA INSPECTION PASSED")
    print("  Frozen order, tensor hashes, t+1 masks, t+2 masks, and strings agree.")
    print("  No CUDA model has been allocated in this inspection step.")


def resolve_run_directory(config: dict[str, Any], override: Path | None) -> Path:
    value = override if override is not None else Path(config["output_dir"])
    if not value.is_absolute():
        value = Path(__file__).resolve().parent.parent / value
    return value.resolve()


def run_training(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MTP training")
    seed = int(config["training"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    run_dir = resolve_run_directory(config, args.run_dir)
    checkpoint_path = run_dir / "latest.pt"
    best_checkpoint_path = run_dir / "best.pt"
    numbered_checkpoint_directory = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    configuration_hash = config_sha256(config)

    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    if not args.resume and (
        checkpoint_path.exists() or metrics_path.exists() or manifest_path.exists()
    ):
        raise RuntimeError(
            f"run directory already contains state: {run_dir}; use --resume or a new directory"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    store = FrozenSession6Store(Path(config["session6_root"]))
    combine = int(config["data"]["base_batches_per_superbatch"])
    superbatches = list(store.superbatches(combine))
    validation_payload, validation_manifest = load_validation_payload(config, store)
    training_config = config["training"]
    model = MTPDecoder(config["model"]).to(device)
    parameters = unique_parameters(model)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count != int(config["model"]["parameter_count_with_head2"]):
        raise RuntimeError("training model parameter count changed")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training_config["learning_rate"]),
        betas=(
            float(training_config["beta1"]),
            float(training_config["beta2"]),
        ),
        eps=float(training_config["epsilon"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if args.resume:
        state = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            configuration_hash=configuration_hash,
        )
        removed = trim_metrics_to_checkpoint(metrics_path, int(state["global_step"]))
        if not best_checkpoint_path.exists():
            raise FileNotFoundError(
                f"best checkpoint is missing from resumable run: {best_checkpoint_path}"
            )
        print(
            f"RESUMED step={state['global_step']:,} epoch={state['epoch_index']} "
            f"cursor={state['superbatch_cursor']:,}; trimmed log rows={removed}"
        )
        manifest = read_json(manifest_path)
        manifest["status"] = "RUNNING"
        manifest["last_resumed_unix_time"] = time.time()
        atomic_write_json(manifest_path, manifest)
    else:
        state = {
            "global_step": 0,
            "epoch_index": 0,
            "superbatch_cursor": 0,
            "active_training_seconds": 0.0,
            "head1_targets_seen": 0,
            "head2_targets_seen": 0,
            "physical_tokens_seen": 0,
            "maximum_peak_allocated_mib": 0.0,
            "best_validation_total_loss": math.inf,
            "best_validation_step": -1,
            "best_validation": None,
            "validations_without_improvement": 0,
        }
        manifest = {
            "schema_version": 1,
            "status": "RUNNING",
            "run_name": config["run_name"],
            "configuration_sha256": configuration_hash,
            "created_unix_time": time.time(),
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "model_parameters": parameter_count,
            "supersteps_per_corpus_pass": len(superbatches),
            "loss_bearing_targets_per_corpus_pass": int(
                store.packing_report["loss_bearing_tokens"]
            ),
            "packing_hash": store.packing_report["packing_hash"],
            "tokenizer_hash": store.packing_report["tokenizer_hash"],
            "validation_probe_hash": validation_manifest["probe_hash"],
        }
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(run_dir / "resolved_config.json", config)
        atomic_write_json(run_dir / "validation_probe.json", validation_manifest)
        initial_validation = evaluate_mtp(
            model, validation_payload, training_config, config["validation"]
        )
        initial_is_best = update_early_stopping_state(
            state, initial_validation, training_config
        )
        if not initial_is_best:
            raise RuntimeError("initial validation did not initialize best tracking")
        append_jsonl(
            metrics_path,
            {
                "event": "validation",
                "scope": "initial",
                "global_step": 0,
                "epoch_index": 0,
                "created_unix_time": time.time(),
                "is_best": True,
                "validations_without_improvement": 0,
                **initial_validation,
            },
        )
        checkpoint_seconds = save_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=state,
            configuration_hash=configuration_hash,
        )
        if bool(training_config.get("retain_numbered_checkpoints", False)):
            archive_training_checkpoint(
                checkpoint_path,
                numbered_checkpoint_directory,
                int(state["global_step"]),
            )
        best_checkpoint_seconds = save_training_checkpoint(
            best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=state,
            configuration_hash=configuration_hash,
        )
        print(
            f"NEW RUN initialized; validation H1={initial_validation['head1_loss']:.6f} "
            f"H2={initial_validation['head2_loss']:.6f}; "
            f"step-0 latest={checkpoint_seconds:.2f}s best={best_checkpoint_seconds:.2f}s"
        )

    maximum_steps = args.max_steps
    target_training_seconds = (
        float(training_config["target_wall_clock_hours"]) * 3_600
        if args.full_run
        else None
    )
    log_every = 1 if maximum_steps is not None and maximum_steps <= 10 else int(
        training_config["log_every_steps"]
    )
    checkpoint_every = int(training_config["checkpoint_every_steps"])
    validation_every = int(training_config["validation_every_steps"])
    early_stopping_patience = int(
        training_config["early_stopping_patience_validations"]
    )
    maximum_corpus_passes = float(
        training_config.get("maximum_corpus_passes", math.inf)
    )
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience_validations must be positive")
    if maximum_corpus_passes <= 0:
        raise ValueError("maximum_corpus_passes must be positive")
    stop_reason = "unknown"
    session_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    print("TRAINING CONTROL")
    print(f"  run directory       : {run_dir}")
    print(f"  starting step       : {state['global_step']:,}")
    print(f"  current epoch/cursor: {state['epoch_index']} / {state['superbatch_cursor']:,}")
    print(
        f"  early stopping      : combined validation loss, patience "
        f"{early_stopping_patience} checks"
    )
    if math.isfinite(maximum_corpus_passes):
        print(f"  maximum corpus passes: {maximum_corpus_passes:.2f}")
    if maximum_steps is not None:
        print(f"  bounded stop        : absolute global step {maximum_steps:,}")
    else:
        print(
            f"  authorized full run : {float(training_config['target_wall_clock_hours']):.2f} "
            "active training hours"
        )

    try:
        while True:
            if maximum_steps is not None and int(state["global_step"]) >= maximum_steps:
                stop_reason = "max_steps_reached"
                break
            if (
                target_training_seconds is not None
                and float(state["active_training_seconds"]) >= target_training_seconds
            ):
                stop_reason = "active_training_time_reached"
                break
            corpus_passes = (
                int(state["head1_targets_seen"])
                / int(store.packing_report["loss_bearing_tokens"])
            )
            if math.isfinite(maximum_corpus_passes) and corpus_passes >= maximum_corpus_passes:
                stop_reason = "maximum_corpus_passes_reached"
                break

            cursor = int(state["superbatch_cursor"])
            superbatch = superbatches[cursor]
            next_step = int(state["global_step"]) + 1
            learning_rate = scheduled_learning_rate(next_step, training_config)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            payload = store.payload_for_superbatch(
                superbatch,
                verify_hashes=bool(config["data"]["verify_base_batch_hashes"]),
            )

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            result = cuda_training_step(
                model,
                optimizer,
                scaler,
                parameters,
                payload,
                training_config,
                trace=False,
            )
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_started
            peak_mib = torch.cuda.max_memory_allocated(device) / MIB

            state["global_step"] = next_step
            state["active_training_seconds"] = float(
                state["active_training_seconds"]
            ) + step_seconds
            state["head1_targets_seen"] = int(state["head1_targets_seen"]) + int(
                result["head1_targets"]
            )
            state["head2_targets_seen"] = int(state["head2_targets_seen"]) + int(
                result["head2_targets"]
            )
            state["physical_tokens_seen"] = int(state["physical_tokens_seen"]) + int(
                payload["input_ids"].size
            )
            state["maximum_peak_allocated_mib"] = max(
                float(state.get("maximum_peak_allocated_mib", 0.0)), peak_mib
            )
            cursor += 1
            if cursor == len(superbatches):
                cursor = 0
                state["epoch_index"] = int(state["epoch_index"]) + 1
            state["superbatch_cursor"] = cursor

            metric = {
                "event": "train_step",
                "global_step": next_step,
                "epoch_index": int(state["epoch_index"]),
                "next_superbatch_cursor": cursor,
                "stage": superbatch.stage,
                "sequence_length": superbatch.sequence_length,
                "batch_size": int(payload["input_ids"].shape[0]),
                "learning_rate": learning_rate,
                "step_seconds": step_seconds,
                "peak_allocated_mib": peak_mib,
                "head1_targets_per_second": int(result["head1_targets"]) / step_seconds,
                "active_training_seconds": float(state["active_training_seconds"]),
                "created_unix_time": time.time(),
                **result,
            }
            append_jsonl(metrics_path, metric)

            if next_step % log_every == 0:
                print(
                    f"step={next_step:>5,} epoch={state['epoch_index']} "
                    f"stage={superbatch.stage:<23} T={superbatch.sequence_length:<3} "
                    f"H1={float(result['head1_loss']):.5f} "
                    f"H2={float(result['head2_loss']):.5f} "
                    f"sum={float(result['total_loss']):.5f} "
                    f"lr={learning_rate:.2e} sec={step_seconds:.2f} "
                    f"peak={peak_mib:.0f}MiB",
                    flush=True,
                )

            if next_step % validation_every == 0:
                validation = evaluate_mtp(
                    model,
                    validation_payload,
                    training_config,
                    config["validation"],
                )
                is_best = update_early_stopping_state(
                    state, validation, training_config
                )
                append_jsonl(
                    metrics_path,
                    {
                        "event": "validation",
                        "scope": "periodic",
                        "global_step": next_step,
                        "epoch_index": int(state["epoch_index"]),
                        "created_unix_time": time.time(),
                        "is_best": is_best,
                        "best_total_loss": float(
                            state["best_validation_total_loss"]
                        ),
                        "validations_without_improvement": int(
                            state["validations_without_improvement"]
                        ),
                        **validation,
                    },
                )
                print(
                    f"validation step={next_step:,} H1={validation['head1_loss']:.5f} "
                    f"H2={validation['head2_loss']:.5f} "
                    f"sum={validation['total_loss']:.5f} "
                    f"best={state['best_validation_total_loss']:.5f} "
                    f"bad={state['validations_without_improvement']}/"
                    f"{early_stopping_patience}",
                    flush=True,
                )
                if is_best:
                    best_checkpoint_seconds = save_training_checkpoint(
                        best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        state=state,
                        configuration_hash=configuration_hash,
                    )
                    print(
                        f"new best step={next_step:,} wrote {best_checkpoint_path.name} "
                        f"in {best_checkpoint_seconds:.2f}s",
                        flush=True,
                    )
                if int(state["validations_without_improvement"]) >= early_stopping_patience:
                    stop_reason = "validation_early_stopping"
                    print(
                        f"early stopping at step={next_step:,}; retained best step="
                        f"{state['best_validation_step']:,}",
                        flush=True,
                    )
                    break

            if next_step % checkpoint_every == 0:
                checkpoint_seconds = save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    state=state,
                    configuration_hash=configuration_hash,
                )
                archived_record = None
                if bool(training_config.get("retain_numbered_checkpoints", False)):
                    archived_record = archive_training_checkpoint(
                        checkpoint_path,
                        numbered_checkpoint_directory,
                        next_step,
                    )
                print(
                    f"checkpoint step={next_step:,} wrote {checkpoint_path.name} "
                    f"in {checkpoint_seconds:.2f}s",
                    flush=True,
                )
                if archived_record is not None:
                    print(
                        f"retained {Path(archived_record['path']).name} "
                        f"via {archived_record['retention_method']}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("Keyboard interrupt received; saving the last completed step.", flush=True)

    final_validation = evaluate_mtp(
        model, validation_payload, training_config, config["validation"]
    )
    final_step = int(state["global_step"])
    final_was_already_validated = (
        final_step > 0 and final_step % validation_every == 0
    )
    if final_was_already_validated:
        final_is_best = final_step == int(state["best_validation_step"])
    else:
        final_is_best = update_early_stopping_state(
            state, final_validation, training_config
        )
    append_jsonl(
        metrics_path,
        {
            "event": "validation",
            "scope": "final",
            "global_step": int(state["global_step"]),
            "epoch_index": int(state["epoch_index"]),
            "created_unix_time": time.time(),
            "is_best": final_is_best,
            "best_total_loss": float(state["best_validation_total_loss"]),
            "validations_without_improvement": int(
                state["validations_without_improvement"]
            ),
            **final_validation,
        },
    )
    if final_is_best and not final_was_already_validated:
        best_checkpoint_seconds = save_training_checkpoint(
            best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=state,
            configuration_hash=configuration_hash,
        )
        print(
            f"new final best step={final_step:,} wrote {best_checkpoint_path.name} "
            f"in {best_checkpoint_seconds:.2f}s",
            flush=True,
        )
    checkpoint_seconds = save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        state=state,
        configuration_hash=configuration_hash,
    )
    if bool(training_config.get("retain_numbered_checkpoints", False)):
        archive_training_checkpoint(
            checkpoint_path,
            numbered_checkpoint_directory,
            final_step,
        )
    checkpoint_hash = file_sha256(checkpoint_path)
    best_checkpoint_hash = file_sha256(best_checkpoint_path)
    numbered_checkpoints = sorted(numbered_checkpoint_directory.glob("step_*.pt"))
    summary = {
        "schema_version": 1,
        "status": "STOPPED_SAFELY",
        "stop_reason": stop_reason,
        "global_step": int(state["global_step"]),
        "epoch_index": int(state["epoch_index"]),
        "next_superbatch_cursor": int(state["superbatch_cursor"]),
        "active_training_seconds": float(state["active_training_seconds"]),
        "session_wall_seconds": time.perf_counter() - session_started,
        "head1_targets_seen": int(state["head1_targets_seen"]),
        "head2_targets_seen": int(state["head2_targets_seen"]),
        "physical_tokens_seen": int(state["physical_tokens_seen"]),
        "corpus_passes_by_head1_targets": int(state["head1_targets_seen"])
        / int(store.packing_report["loss_bearing_tokens"]),
        "maximum_peak_allocated_mib": float(
            state.get("maximum_peak_allocated_mib", 0.0)
        ),
        "final_validation": final_validation,
        "best_validation": dict(state["best_validation"]),
        "best_validation_step": int(state["best_validation_step"]),
        "validations_without_improvement": int(
            state["validations_without_improvement"]
        ),
        "checkpoint_seconds": checkpoint_seconds,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_sha256": checkpoint_hash,
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_file_sha256": best_checkpoint_hash,
        "numbered_checkpoint_directory": str(numbered_checkpoint_directory),
        "numbered_checkpoint_count": len(numbered_checkpoints),
    }
    atomic_write_json(run_dir / "run_summary.json", summary)
    manifest = read_json(manifest_path)
    manifest["status"] = "STOPPED_SAFELY"
    manifest["latest_summary"] = summary
    atomic_write_json(manifest_path, manifest)
    print("TRAINING PROCESS STOPPED SAFELY")
    print(json.dumps(summary, indent=2))


def smoke_model(config: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the model smoke test")
    seed = int(config["training"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    device_properties = torch.cuda.get_device_properties(device)
    total_mib = device_properties.total_memory / MIB

    store = FrozenSession6Store(Path(config["session6_root"]))
    combine = int(config["data"]["base_batches_per_superbatch"])
    superbatches = list(store.superbatches(combine))
    selected = []
    for length in (256, 512):
        selected.append(
            next(
                superbatch
                for superbatch in superbatches
                if superbatch.sequence_length == length
                and len(superbatch.base_batches) == combine
            )
        )

    training_config = config["training"]

    def build_fresh_components() -> tuple[
        MTPDecoder,
        list[nn.Parameter],
        torch.optim.Optimizer,
        torch.amp.GradScaler,
    ]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        fresh_model = MTPDecoder(config["model"]).to(device)
        fresh_parameters = unique_parameters(fresh_model)
        fresh_optimizer = torch.optim.AdamW(
            fresh_parameters,
            lr=float(training_config["learning_rate"]),
            betas=(
                float(training_config["beta1"]),
                float(training_config["beta2"]),
            ),
            eps=float(training_config["epsilon"]),
            weight_decay=float(training_config["weight_decay"]),
        )
        fresh_scaler = torch.amp.GradScaler("cuda", enabled=True)
        return fresh_model, fresh_parameters, fresh_optimizer, fresh_scaler

    model, parameters, optimizer, scaler = build_fresh_components()
    parameter_count = sum(parameter.numel() for parameter in parameters)
    expected_parameters = int(config["model"]["parameter_count_with_head2"])
    if parameter_count != expected_parameters:
        raise RuntimeError(
            f"model has {parameter_count:,} parameters; expected {expected_parameters:,}"
        )
    expected_uniform_loss = math.log(int(config["model"]["vocab_size"]))

    model_config = config["model"]
    print(f"{config['run_name']} CUDA SMOKE TEST")
    print(f"  GPU              : {torch.cuda.get_device_name(device)}")
    print(f"  GPU capacity     : {total_mib:,.2f} MiB")
    print(f"  parameters       : {parameter_count:,}")
    print(f"  FP32 weights     : {parameter_count * 4 / MIB:,.2f} MiB")
    print(
        f"  dimensions       : B=sequences, T=positions, "
        f"D={int(model_config['hidden_size']):,} hidden features, "
        f"V={int(model_config['vocab_size']):,} vocabulary scores"
    )
    print(
        f"  decoder block    : pre-norm "
        f"{str(model_config.get('normalization', 'layernorm')).upper()} + "
        f"{str(model_config.get('ffn_type', 'gelu')).upper()} FFN"
    )
    print(f"  reference ln(V)  : {expected_uniform_loss:.6f} nats")
    print("  head 1 weight    : tied to token_embedding.weight [V,D]")
    print("  head 2 weight    : independent [V,D]")

    results = []
    for smoke_index, superbatch in enumerate(selected, start=1):
        payload = store.payload_for_superbatch(superbatch, verify_hashes=True)
        batch_size, length = payload["input_ids"].shape
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        print()
        print(
            f"SMOKE STEP {smoke_index}: B={batch_size}, T={length}, "
            f"physical tokens={batch_size * length:,}, stage={superbatch.stage}"
        )
        print("  model state: fresh deterministic initialization")
        result = cuda_training_step(
            model,
            optimizer,
            scaler,
            parameters,
            payload,
            training_config,
            trace=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_mib = torch.cuda.max_memory_allocated(device) / MIB
        peak_fraction = peak_mib / total_mib
        result = {
            **result,
            "batch_size": batch_size,
            "sequence_length": length,
            "elapsed_seconds": elapsed,
            "peak_mib": peak_mib,
            "peak_fraction": peak_fraction,
        }
        results.append(result)
        print("  STEP RESULT")
        print(
            f"    Head 1 loss / PPL : {result['head1_loss']:.6f} / "
            f"{math.exp(float(result['head1_loss'])):,.2f}"
        )
        print(
            f"    Head 2 loss / PPL : {result['head2_loss']:.6f} / "
            f"{math.exp(float(result['head2_loss'])):,.2f}"
        )
        print(f"    Combined loss     : {result['total_loss']:.6f}")
        print(
            f"    Active targets    : H1={result['head1_targets']:,}, "
            f"H2={result['head2_targets']:,}"
        )
        print(f"    Gradient norm     : {result['gradient_norm']:.6f}")
        print(f"    Parameter change  : {result['parameter_delta']:.9f}")
        print(
            f"    CUDA peak         : {peak_mib:,.2f} MiB "
            f"({peak_fraction:.1%} of VRAM)"
        )
        print(f"    Step time         : {elapsed:.3f} s")
        if smoke_index < len(selected):
            del model, optimizer, scaler, parameters
            gc.collect()
            torch.cuda.empty_cache()
            model, parameters, optimizer, scaler = build_fresh_components()

    for result in results:
        for name in ("head1_loss", "head2_loss", "gradient_norm"):
            if not math.isfinite(float(result[name])):
                raise RuntimeError(f"non-finite {name} in CUDA smoke test")
        if abs(float(result["head1_loss"]) - expected_uniform_loss) >= 0.75:
            raise RuntimeError("untrained head-1 loss is implausibly far from ln(V)")
        if abs(float(result["head2_loss"]) - expected_uniform_loss) >= 0.75:
            raise RuntimeError("untrained head-2 loss is implausibly far from ln(V)")
        if float(result["parameter_delta"]) <= 0:
            raise RuntimeError("AdamW step did not change the sampled head-2 parameter")
        if float(result["peak_fraction"]) > 0.80:
            raise RuntimeError("smoke step exceeded the 80% VRAM safety ceiling")
    print()
    print("MODEL SMOKE TEST PASSED")
    print("  Both supported tensor shapes completed forward, chunked CE backward,")
    print("  gradient clipping, and a real AdamW update below the VRAM ceiling.")
    print("  The transient smoke-test model is not saved; timed training did not start.")

    del model, optimizer, scaler, parameters
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    if args.command == "inspect-data":
        inspect_data(config)
    elif args.command == "inspect-validation":
        inspect_validation(config)
    elif args.command == "smoke-model":
        smoke_model(config)
    elif args.command == "train":
        run_training(args, config)
    elif args.command == "compare-checkpoints":
        compare_checkpoints(args.checkpoint_a, args.checkpoint_b)


if __name__ == "__main__":
    main()
