"""Make a tiny causal Transformer report what actually happens in one training step.

The script is deliberately dependency-light: PyTorch plus the Python standard library.
It writes JSON/CSV/SVG artifacts used by the accompanying executed notebook and README.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data" / "general_anneal.jsonl.gz"


def seed_everything(seed: int = 1729) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bytes(path: Path = DATA, limit: int = 1_000_000) -> torch.Tensor:
    """Load UTF-8 text bytes from the small, frozen JSONL corpus shard."""
    if not path.exists():
        # A reviewer can still run the repo without the supplied data file.
        fallback = (
            "A small model is useful when every operation can be inspected. "
            "The loss is a scalar, but its gradient is a structured report about change. "
            "Token weighting matters whenever sequence lengths differ.\n"
        ) * 4096
        raw = fallback.encode("utf-8")
    else:
        chunks: List[bytes] = []
        total = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                text = record.get("text", "")
                if text:
                    piece = (text + "\n").encode("utf-8")
                    chunks.append(piece)
                    total += len(piece)
                if total >= limit:
                    break
        raw = b"".join(chunks)[:limit]
    if len(raw) < 20_000:
        raw = (raw * (20_000 // max(1, len(raw)) + 1))[:20_000]
    return torch.tensor(list(raw), dtype=torch.long)


ShapeLedger = MutableMapping[str, Dict[str, object]]


def record(ledger: ShapeLedger | None, name: str, tensor: torch.Tensor, dims: str) -> torch.Tensor:
    if ledger is not None:
        ledger[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "dims": dims,
        }
    return tensor


@dataclass(frozen=True)
class Config:
    vocab_size: int = 256
    max_seq_len: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.attn_out = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.mlp_ratio * cfg.d_model
        self.mlp_up = nn.Linear(cfg.d_model, hidden)
        self.mlp_down = nn.Linear(hidden, cfg.d_model)

    def forward(self, x: torch.Tensor, ledger: ShapeLedger | None, prefix: str) -> torch.Tensor:
        bsz, seq, channels = x.shape
        heads, head_dim = self.cfg.n_heads, self.cfg.head_dim
        norm1 = record(ledger, f"{prefix}.ln1", self.ln1(x), "batch, sequence, model channel")
        qkv = record(ledger, f"{prefix}.qkv", self.qkv(norm1), "batch, sequence, concatenated Q/K/V channel")
        q, k, v = qkv.chunk(3, dim=-1)
        q = record(ledger, f"{prefix}.q", q.view(bsz, seq, heads, head_dim).transpose(1, 2), "batch, attention head, query position, head channel")
        k = record(ledger, f"{prefix}.k", k.view(bsz, seq, heads, head_dim).transpose(1, 2), "batch, attention head, key position, head channel")
        v = record(ledger, f"{prefix}.v", v.view(bsz, seq, heads, head_dim).transpose(1, 2), "batch, attention head, value position, head channel")
        scores = record(ledger, f"{prefix}.attention_scores", (q @ k.transpose(-2, -1)) / math.sqrt(head_dim), "batch, attention head, query position, key position")
        causal = record(ledger, f"{prefix}.causal_mask", torch.ones(seq, seq, device=x.device, dtype=torch.bool).tril().view(1, 1, seq, seq), "broadcast batch, broadcast head, query position, key position")
        masked_scores = record(ledger, f"{prefix}.masked_scores", scores.masked_fill(~causal, float("-inf")), "batch, attention head, query position, key position")
        probs = record(ledger, f"{prefix}.attention_probabilities", F.softmax(masked_scores, dim=-1), "batch, attention head, query position, key position")
        attended = record(ledger, f"{prefix}.attended_values", probs @ v, "batch, attention head, query position, head channel")
        merged = record(ledger, f"{prefix}.merged_heads", attended.transpose(1, 2).contiguous().view(bsz, seq, channels), "batch, sequence, model channel")
        projected = record(ledger, f"{prefix}.attention_output", self.attn_out(merged), "batch, sequence, model channel")
        after_attn = record(ledger, f"{prefix}.residual_after_attention", x + projected, "batch, sequence, model channel")
        norm2 = record(ledger, f"{prefix}.ln2", self.ln2(after_attn), "batch, sequence, model channel")
        up = record(ledger, f"{prefix}.mlp_up", self.mlp_up(norm2), "batch, sequence, expanded MLP channel")
        activated = record(ledger, f"{prefix}.gelu", F.gelu(up), "batch, sequence, expanded MLP channel")
        down = record(ledger, f"{prefix}.mlp_down", self.mlp_down(activated), "batch, sequence, model channel")
        return record(ledger, f"{prefix}.residual_after_mlp", after_attn + down, "batch, sequence, model channel")


class TinyTransformer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, ledger: ShapeLedger | None = None) -> torch.Tensor:
        bsz, seq = input_ids.shape
        record(ledger, "input_ids", input_ids, "batch, sequence")
        positions = record(ledger, "position_indices", torch.arange(seq, device=input_ids.device), "sequence")
        tok = record(ledger, "token_embedding", self.token_embedding(input_ids), "batch, sequence, model channel")
        pos = record(ledger, "position_embedding", self.position_embedding(positions), "sequence, model channel")
        x = record(ledger, "embedding_sum", tok + pos, "batch, sequence, model channel")
        for index, block in enumerate(self.blocks):
            x = block(x, ledger, f"block_{index}")
        x = record(ledger, "final_norm", self.final_norm(x), "batch, sequence, model channel")
        return record(ledger, "logits", self.lm_head(x), "batch, sequence, byte vocabulary")


def sample_batch(stream: torch.Tensor, batch_size: int, seq_len: int, generator: torch.Generator, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(stream) - seq_len - 1, (batch_size,), generator=generator)
    rows = [stream[start : start + seq_len + 1] for start in starts.tolist()]
    joined = torch.stack(rows)
    return joined[:, :-1].to(device, non_blocking=True), joined[:, 1:].to(device, non_blocking=True)


def loss_parts(model: TinyTransformer, x: torch.Tensor, y: torch.Tensor, ledger: ShapeLedger | None = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
    record(ledger, "targets", y, "batch, sequence")
    logits = model(x, ledger)
    flat_logits = record(ledger, "flat_logits", logits.reshape(-1, logits.size(-1)), "batch-times-sequence, byte vocabulary")
    flat_targets = record(ledger, "flat_targets", y.reshape(-1), "batch-times-sequence")
    token_loss = record(ledger, "per_token_cross_entropy", F.cross_entropy(flat_logits, flat_targets, reduction="none"), "batch-times-sequence")
    loss_sum = record(ledger, "loss_sum", token_loss.sum(), "scalar")
    valid_tokens = record(ledger, "valid_token_count", torch.tensor(token_loss.numel(), device=x.device), "scalar")
    mean_loss = record(ledger, "mean_loss", loss_sum / valid_tokens, "scalar")
    return loss_sum, mean_loss, token_loss.numel()


def grad_norm(model: nn.Module, ledger: ShapeLedger | None = None) -> float:
    squares = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squares += parameter.grad.detach().float().pow(2).sum()
    record(ledger, "accumulation.gradient_squared_sum", squares, "scalar")
    norm = record(ledger, "accumulation.global_gradient_l2_norm", squares.sqrt(), "scalar")
    return norm.item()


def parameter_dims(name: str, tensor: torch.Tensor) -> str:
    if "token_embedding.weight" in name:
        return "byte vocabulary, model channel"
    if "position_embedding.weight" in name:
        return "maximum sequence position, model channel"
    if name == "lm_head.weight":
        return "output byte vocabulary, input model channel"
    if name.endswith(".weight") and tensor.ndim == 2:
        return "output channel, input channel"
    if name.endswith(".bias"):
        return "output channel"
    return "model channel"


def add_backward_and_optimizer_shapes(ledger: ShapeLedger, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    for name, parameter in model.named_parameters():
        dims = parameter_dims(name, parameter)
        record(ledger, f"parameter.{name}", parameter, dims)
        if parameter.grad is not None:
            record(ledger, f"gradient.{name}", parameter.grad, dims)
        state = optimizer.state.get(parameter, {})
        for state_name, state_value in state.items():
            if torch.is_tensor(state_value):
                state_dims = "scalar optimizer step" if state_value.ndim == 0 else dims
                record(ledger, f"optimizer.{state_name}.{name}", state_value, state_dims)


def finite_difference_check(cfg: Config, stream: torch.Tensor) -> Dict[str, object]:
    """Compare autograd to an independent central finite difference in float64."""
    seed_everything(991)
    model = TinyTransformer(cfg).double().cpu()
    generator = torch.Generator().manual_seed(23)
    x, y = sample_batch(stream, batch_size=2, seq_len=8, generator=generator, device=torch.device("cpu"))
    model.zero_grad(set_to_none=True)
    _, loss, _ = loss_parts(model, x, y)
    loss.backward()
    target_row = int(y[0, 0])
    parameter = model.lm_head.weight
    index = (target_row, 0)
    analytic = parameter.grad[index].item()
    original = parameter[index].item()
    epsilon = 1e-5
    with torch.no_grad():
        parameter[index] = original + epsilon
        _, plus, _ = loss_parts(model, x, y)
        parameter[index] = original - epsilon
        _, minus, _ = loss_parts(model, x, y)
        parameter[index] = original
    numeric = (plus.item() - minus.item()) / (2 * epsilon)
    absolute_error = abs(analytic - numeric)
    relative_error = absolute_error / max(abs(analytic), abs(numeric), 1e-15)
    return {
        "parameter": f"lm_head.weight[{index[0]}, {index[1]}]",
        "epsilon": epsilon,
        "loss_plus": plus.item(),
        "loss_minus": minus.item(),
        "autograd": analytic,
        "finite_difference": numeric,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "agreement_decimals": max(0, int(-math.log10(max(absolute_error, 1e-16)))),
    }


@torch.no_grad()
def evaluate(model: TinyTransformer, batches: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> float:
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in batches:
        loss_sum, _, count = loss_parts(model, x, y)
        total_loss += loss_sum.item()
        total_tokens += count
    model.train(was_training)
    return total_loss / total_tokens


def make_batches(short_stream: torch.Tensor, long_stream: torch.Tensor, batch_size: int, step: int, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(10_000 + step)
    return [
        sample_batch(short_stream, batch_size, 8, generator, device),
        sample_batch(long_stream, batch_size, 64, generator, device),
    ]


def train_pair(cfg: Config, stream: torch.Tensor, steps: int, device: torch.device) -> Tuple[List[Dict[str, float]], ShapeLedger]:
    # Deliberately use different corpus regions so incorrect 50/50 source weighting is visible.
    split = len(stream) // 3
    short_stream, long_stream = stream[:split], stream[split:]
    seed_everything(7)
    correct = TinyTransformer(cfg).to(device)
    broken = copy.deepcopy(correct)
    optim_correct = torch.optim.AdamW(correct.parameters(), lr=2e-3, weight_decay=0.01)
    optim_broken = torch.optim.AdamW(broken.parameters(), lr=2e-3, weight_decay=0.01)

    eval_generator = torch.Generator().manual_seed(4242)
    eval_batches = [
        sample_batch(short_stream, 24, 8, eval_generator, device),
        sample_batch(long_stream, 24, 64, eval_generator, device),
    ]
    rows: List[Dict[str, float]] = []
    ledger: ShapeLedger = {}

    for step in range(steps):
        batches = make_batches(short_stream, long_stream, batch_size=16, step=step, device=device)
        total_tokens = sum(x.numel() for x, _ in batches)
        if step == 0:
            record(ledger, "accumulation.total_token_count", torch.tensor(total_tokens, device=device), "scalar")

        optim_correct.zero_grad(set_to_none=True)
        correct_pre_loss = 0.0
        for micro_index, (x, y) in enumerate(batches):
            this_ledger: ShapeLedger | None = {} if step == 0 else None
            loss_sum, _, count = loss_parts(correct, x, y, this_ledger)
            contribution = loss_sum / total_tokens
            record(this_ledger, "token_weighted_loss_contribution", contribution, "scalar")
            contribution.backward()
            if this_ledger is not None:
                length_label = "short" if micro_index == 0 else "long"
                for tensor_name, item in this_ledger.items():
                    ledger[f"microbatch_{micro_index}_{length_label}.{tensor_name}"] = item
            correct_pre_loss += loss_sum.item()
        norm_correct = grad_norm(correct, ledger if step == 0 else None)
        optim_correct.step()
        if step == 0:
            add_backward_and_optimizer_shapes(ledger, correct, optim_correct)

        optim_broken.zero_grad(set_to_none=True)
        broken_objective = 0.0
        for x, y in batches:
            _, loss_mean, _ = loss_parts(broken, x, y)
            (loss_mean / len(batches)).backward()
            broken_objective += loss_mean.item() / len(batches)
        norm_broken = grad_norm(broken)
        optim_broken.step()

        eval_correct = evaluate(correct, eval_batches)
        eval_broken = evaluate(broken, eval_batches)
        rows.append({
            "step": step + 1,
            "correct_batch_loss": correct_pre_loss / total_tokens,
            "broken_batch_objective": broken_objective,
            "correct_eval_loss": eval_correct,
            "broken_eval_loss": eval_broken,
            "correct_grad_norm": norm_correct,
            "broken_grad_norm": norm_broken,
            "correct_perplexity": math.exp(min(20, eval_correct)),
            "broken_perplexity": math.exp(min(20, eval_broken)),
        })
        print(
            f"step={step + 1:02d} correct_eval={eval_correct:.5f} broken_eval={eval_broken:.5f} "
            f"correct_grad_norm={norm_correct:.5f} broken_grad_norm={norm_broken:.5f}"
        )
    return rows, ledger


def find_grad_lead(rows: Sequence[Dict[str, float]]) -> Dict[str, object]:
    """Find the clearest abrupt grad-norm move not mirrored by the loss trend."""
    candidates: List[Tuple[float, Dict[str, object]]] = []
    for previous, current in zip(rows, rows[1:]):
        loss_delta = current["correct_eval_loss"] - previous["correct_eval_loss"]
        norm_delta = current["correct_grad_norm"] - previous["correct_grad_norm"]
        loss_relative = abs(loss_delta) / max(abs(previous["correct_eval_loss"]), 1e-12)
        norm_relative = abs(norm_delta) / max(abs(previous["correct_grad_norm"]), 1e-12)
        item = {
            "step": int(current["step"]),
            "previous_step": int(previous["step"]),
            "loss_before": previous["correct_eval_loss"],
            "loss_after": current["correct_eval_loss"],
            "loss_absolute_change": loss_delta,
            "loss_relative_change": loss_relative,
            "grad_norm_before": previous["correct_grad_norm"],
            "grad_norm_after": current["correct_grad_norm"],
            "grad_norm_absolute_change": norm_delta,
            "grad_norm_relative_change": norm_relative,
            "relative_signal_ratio": norm_relative / max(loss_relative, 1e-12),
        }
        candidates.append((item["relative_signal_ratio"], item))
    return max(candidates, key=lambda pair: pair[0])[1]


def get_max_clock_ghz() -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return float(output.splitlines()[0].strip()) / 1000.0
    except Exception:
        return None


def benchmark_mfu(cfg: Config, stream: torch.Tensor, device: torch.device) -> Dict[str, object]:
    if device.type != "cuda":
        return {"available": False, "reason": "MFU benchmark requires CUDA in this report."}
    seed_everything(31337)
    model = TinyTransformer(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(77)
    x, y = sample_batch(stream, 32, cfg.max_seq_len, generator, device)

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = loss_parts(model, x, y)
        loss.backward()
        optimizer.step()

    for _ in range(20):
        one_step()
    torch.cuda.synchronize()
    durations: List[float] = []
    for _ in range(60):
        start = time.perf_counter()
        one_step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    median_step = statistics.median(durations)
    tokens_per_step = x.numel()
    tokens_per_second = tokens_per_step / median_step

    # Report both the course's conventional 6N estimate and an architecture-aware
    # matmul count. One multiply-add is two FLOPs. Backward is approximated as 2x
    # forward. LayerNorm, softmax, GELU, optimizer, and memory traffic are excluded.
    d, layers, seq, vocab = cfg.d_model, cfg.n_layers, cfg.max_seq_len, cfg.vocab_size
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    standard_6n_flops_per_token = 6 * parameter_count
    forward_flops_per_token = layers * (24 * d * d + 4 * seq * d) + 2 * d * vocab
    train_flops_per_token = 3 * forward_flops_per_token
    standard_achieved_flops = tokens_per_second * standard_6n_flops_per_token
    attention_aware_achieved_flops = tokens_per_second * train_flops_per_token

    props = torch.cuda.get_device_properties(device)
    clock_ghz = get_max_clock_ghz()
    cuda_cores_per_sm = 128 if (props.major, props.minor) == (8, 6) else None
    peak_flops = None
    if clock_ghz is not None and cuda_cores_per_sm is not None:
        peak_flops = props.multi_processor_count * cuda_cores_per_sm * 2 * clock_ghz * 1e9
    standard_mfu = standard_achieved_flops / peak_flops if peak_flops else None
    attention_aware_mfu = attention_aware_achieved_flops / peak_flops if peak_flops else None
    return {
        "available": True,
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "streaming_multiprocessors": props.multi_processor_count,
        "cuda_cores_per_sm_assumption": cuda_cores_per_sm,
        "driver_reported_max_clock_ghz": clock_ghz,
        "peak_fp32_tflops_upper_bound": peak_flops / 1e12 if peak_flops else None,
        "batch_size": x.shape[0],
        "sequence_length": x.shape[1],
        "tokens_per_step": tokens_per_step,
        "median_step_seconds": median_step,
        "tokens_per_second": tokens_per_second,
        "parameter_count": parameter_count,
        "standard_6n_flops_per_token": standard_6n_flops_per_token,
        "standard_6n_achieved_tflops": standard_achieved_flops / 1e12,
        "standard_6n_mfu": standard_mfu,
        "standard_6n_mfu_percent": 100 * standard_mfu if standard_mfu is not None else None,
        "forward_matmul_flops_per_token": forward_flops_per_token,
        "training_matmul_flops_per_token": train_flops_per_token,
        "attention_aware_achieved_tflops": attention_aware_achieved_flops / 1e12,
        "attention_aware_mfu": attention_aware_mfu,
        "attention_aware_mfu_percent": 100 * attention_aware_mfu if attention_aware_mfu is not None else None,
        # Primary aliases deliberately use the Session 10 convention.
        "achieved_tflops": standard_achieved_flops / 1e12,
        "mfu": standard_mfu,
        "mfu_percent": 100 * standard_mfu if standard_mfu is not None else None,
        "target_40_percent_tflops": 0.4 * peak_flops / 1e12 if peak_flops else None,
        "counting_scope": "Primary MFU uses the Session 10 6N approximation. The architecture-aware comparison counts explicit matmuls; both include optimizer and elementwise time in the denominator but not numerator.",
    }


def float_representations() -> Dict[str, Dict[str, object]]:
    fp32 = torch.tensor(0.1, dtype=torch.float32)
    fp32_bits = fp32.view(torch.int32).item() & 0xFFFFFFFF
    bf16 = torch.tensor(0.1, dtype=torch.bfloat16)
    bf16_bits = bf16.view(torch.int16).item() & 0xFFFF
    # E4M3: 0.1 = 1.6 * 2^-4. Mantissa 0.6*8=4.8 rounds to 5 -> 1.625*2^-4.
    fp8_bits = 0b0_0011_101
    fp8_value = (1.0 + 5 / 8) * 2 ** -4
    result = {
        "fp32": {
            "bits": f"{fp32_bits:032b}", "grouped": f"{fp32_bits >> 31:b} {(fp32_bits >> 23) & 0xFF:08b} {fp32_bits & 0x7FFFFF:023b}",
            "hex": f"0x{fp32_bits:08X}", "stored_value": float(fp32), "fields": "sign | 8-bit exponent (bias 127) | 23-bit fraction",
        },
        "bf16": {
            "bits": f"{bf16_bits:016b}", "grouped": f"{bf16_bits >> 15:b} {(bf16_bits >> 7) & 0xFF:08b} {bf16_bits & 0x7F:07b}",
            "hex": f"0x{bf16_bits:04X}", "stored_value": float(bf16), "fields": "sign | 8-bit exponent (bias 127) | 7-bit fraction",
        },
        "fp8_e4m3": {
            "bits": f"{fp8_bits:08b}", "grouped": f"{fp8_bits >> 7:b} {(fp8_bits >> 3) & 0xF:04b} {fp8_bits & 0x7:03b}",
            "hex": f"0x{fp8_bits:02X}", "stored_value": fp8_value, "fields": "sign | 4-bit exponent (bias 7) | 3-bit fraction",
        },
    }
    if hasattr(torch, "float8_e4m3fn"):
        result["fp8_e4m3"]["torch_check"] = float(torch.tensor(0.1).to(torch.float8_e4m3fn).float())
    return result


def polyline_svg(rows: Sequence[Dict[str, float]], out: Path, fields: Sequence[Tuple[str, str, str]], title: str, y_label: str) -> None:
    width, height = 900, 520
    left, right, top, bottom = 85, 25, 55, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    all_y = [row[field] for field, _, _ in fields for row in rows]
    y_min, y_max = min(all_y), max(all_y)
    margin = max((y_max - y_min) * 0.08, 1e-6)
    y_min, y_max = y_min - margin, y_max + margin

    def sx(i: int) -> float:
        return left + i * plot_w / max(1, len(rows) - 1)

    def sy(value: float) -> float:
        return top + (y_max - value) * plot_h / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="system-ui" font-size="20" font-weight="600">{title}</text>',
    ]
    for tick in range(6):
        frac = tick / 5
        y = top + frac * plot_h
        value = y_max - frac * (y_max - y_min)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dedbd3" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="monospace" font-size="12" fill="#444">{value:.3f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    for tick in range(0, len(rows), max(1, len(rows)//8)):
        parts.append(f'<text x="{sx(tick):.2f}" y="{height-bottom+22}" text-anchor="middle" font-family="monospace" font-size="12">{rows[tick]["step"]:.0f}</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" font-family="system-ui" font-size="14">optimizer step</text>')
    parts.append(f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle" font-family="system-ui" font-size="14">{y_label}</text>')
    legend_x = left + 18
    for index, (field, label, color) in enumerate(fields):
        points = " ".join(f"{sx(i):.2f},{sy(row[field]):.2f}" for i, row in enumerate(rows))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
        ly = top + 18 + index * 24
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+30}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+38}" y="{ly+4}" font-family="system-ui" font-size="13">{label}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def write_artifacts(rows: Sequence[Dict[str, float]], ledger: ShapeLedger, summary: Dict[str, object]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ARTIFACTS / "shape_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    with (ARTIFACTS / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ARTIFACTS / "shape_ledger.txt").open("w", encoding="utf-8") as handle:
        for name, item in ledger.items():
            line = f"{name}: shape={tuple(item['shape'])}, dtype={item['dtype']} | dimensions: {item['dims']}"
            print(line)
            handle.write(line + "\n")
    polyline_svg(rows, ARTIFACTS / "accumulation_curves.svg", [
        ("correct_eval_loss", "correct token-weighted accumulation", "#176B87"),
        ("broken_eval_loss", "BROKEN mean-of-means", "#C44536"),
    ], "Unequal micro-batches: correct weighting vs mean-of-means", "token-weighted evaluation loss")
    polyline_svg(rows, ARTIFACTS / "grad_norms.svg", [
        ("correct_grad_norm", "correct run", "#176B87"),
        ("broken_grad_norm", "broken run", "#C44536"),
    ], "Global L2 gradient norm at every step", "gradient norm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    seed_everything()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    stream = load_bytes()
    cfg = Config()

    gradient_check = finite_difference_check(cfg, stream)
    print("finite_difference", json.dumps(gradient_check, indent=2))
    rows, ledger = train_pair(cfg, stream, args.steps, device)
    grad_lead = find_grad_lead(rows)
    mfu = benchmark_mfu(cfg, stream, device)
    floats = float_representations()
    parameter_count = sum(p.numel() for p in TinyTransformer(cfg).parameters())
    final = rows[-1]
    summary: Dict[str, object] = {
        "environment": {
            "torch": torch.__version__, "device": str(device),
            "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "seed": 1729, "tf32_enabled": False,
        },
        "model": {**asdict(cfg), "parameter_count": parameter_count, "tokenization": "UTF-8 bytes"},
        "data": {"path": str(DATA.relative_to(ROOT)), "bytes_loaded": len(stream), "short_microbatch": [16, 8], "long_microbatch": [16, 64]},
        "gradient_check": gradient_check,
        "accumulation": {
            "correct_formula": "(sum loss in short + sum loss in long) / (tokens_short + tokens_long)",
            "broken_formula": "0.5 * mean(loss_short) + 0.5 * mean(loss_long)",
            "short_token_weight_correct": 8 / 72,
            "short_token_weight_broken": 0.5,
            "final_correct_eval_loss": final["correct_eval_loss"],
            "final_broken_eval_loss": final["broken_eval_loss"],
            "final_absolute_gap": final["broken_eval_loss"] - final["correct_eval_loss"],
            "final_relative_gap_percent": 100 * (final["broken_eval_loss"] / final["correct_eval_loss"] - 1),
        },
        "grad_norm_leads_loss": grad_lead,
        "mfu": mfu,
        "float_representations": floats,
        "artifacts": {
            "shape_ledger_entries": len(ledger), "training_steps": len(rows),
            "training_log": "artifacts/training_log.csv", "shape_ledger": "artifacts/shape_ledger.txt",
            "accumulation_plot": "artifacts/accumulation_curves.svg", "grad_norm_plot": "artifacts/grad_norms.svg",
        },
    }
    write_artifacts(rows, ledger, summary)
    print("summary", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
