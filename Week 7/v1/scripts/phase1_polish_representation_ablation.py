#!/usr/bin/env python3
"""Phase-1 Polish: controlled representation-geometry ablation.

This standalone script is the final local experiment for the Fourier-Kronecker
Phase-1 study.  It trains three small causal language models on the frozen Week
6 standard-BPE corpus and audits the geometry that enters the Transformer:

    Arm A -- Kronecker V1, one-hot byte x one-hot position, cutoff at byte 32.
    Arm B -- Fourier-Kronecker V3, data-aware continuous Euler phases.
    Arm C -- one-layer character CNN, an order-sensitive local control.

The experiment is intentionally diagnostic rather than a claim of large-scale
pretraining.  It measures, before and after training:

* projected embedding norm and per-dimension variance;
* cosine distributions for random versus edit-similar word-token pairs;
* cosine sensitivity to reversing byte order;
* first-step gradient norms, active parameter counts, and nonzero gradients;
* short-corpus train/validation loss for the three representations.

Strict initialization control
-----------------------------
Every arm contains the same trainable parameter names, shapes, and step-zero
values: the 8192->d_model projection, a 64-channel CNN branch, Transformer, and
LM head.  V1 and V3 deliberately leave the CNN branch inert.  CNN activates it.
The complete trainable SHA-256 must match across all arms.  The report separately
records which tensors/elements actually received gradients, so this convenience
is not misrepresented as equal effective capacity.

The CNN is a lightweight missing-baseline control, not a parameter-matched claim.
It embeds at most 32 bytes, applies one width-3 convolution, preserves absolute
order by flattening [position, channel], zero-pads to 8192 features, and uses the
same projection and z-normalization as V1/V3.

Default RTX 3070 run (three arms x 500 updates):

    python phase1_polish_representation_ablation.py

Fast end-to-end verification:

    python phase1_polish_representation_ablation.py --smoke-test

Outputs include metrics.json, five PNG figures, and README_PHASE1_POLISH.md.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_DATA_ROOT = PACKAGE_ROOT / "data" / "week6_v2"


@dataclass(frozen=True)
class ModelPreset:
    n_layers: int
    n_heads: int
    d_model: int
    mlp_ratio: int


PRESETS = {
    "mid": ModelPreset(n_layers=6, n_heads=6, d_model=384, mlp_ratio=8),
    "smoke": ModelPreset(n_layers=1, n_heads=2, d_model=64, mlp_ratio=2),
}


@dataclass
class Config:
    preset: str = "mid"
    block_size: int = 256
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 500
    val_batches: int = 20
    log_interval: int = 50
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    warmup_steps: int = 25
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float = 0.0
    d_char: int = 256
    d_pos: int = 32
    cnn_width: int = 64
    z_norm_eps: float = 1.0e-5
    geometry_tokens: int = 2_048
    pair_count: int = 512
    order_probe_count: int = 512
    seed: int = 7_007
    num_workers: int = 2
    device: str = "auto"

    @property
    def codec_dim(self) -> int:
        return self.d_char * self.d_pos


@dataclass(frozen=True)
class DataPaths:
    tokenizer_json: Path
    train_tokens: Path
    val_tokens: Path
    source: str


@dataclass(frozen=True)
class Vocabulary:
    token_bytes: Tuple[bytes, ...]
    source: str

    def __len__(self) -> int:
        return len(self.token_bytes)

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray([len(value) for value in self.token_bytes], dtype=np.int64)


def json_dump(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def discover_data(args: argparse.Namespace) -> DataPaths:
    """Prefer the Week-6 V2 standard-BPE artifacts used in Phase 1."""
    if args.tokenizer_json or args.train_tokens or args.val_tokens:
        if not (args.tokenizer_json and args.train_tokens and args.val_tokens):
            raise ValueError("Explicit tokenizer/train/validation paths must be supplied together")
        answer = DataPaths(
            args.tokenizer_json.resolve(), args.train_tokens.resolve(),
            args.val_tokens.resolve(), "explicit CLI paths",
        )
    else:
        answer = DataPaths(
            BUNDLED_DATA_ROOT / "tokenizer.json",
            BUNDLED_DATA_ROOT / "train/tokens.uint16.bin",
            BUNDLED_DATA_ROOT / "validation/tokens.uint16.bin",
            "bundled Week 6 V2 standard-BPE general corpus",
        )
    for path in (answer.tokenizer_json, answer.train_tokens, answer.val_tokens):
        if not path.is_file():
            raise FileNotFoundError(path)
    return answer


def load_vocabulary(path: Path) -> Vocabulary:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("tokens")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Missing tokens array in {path}")
    by_id = {int(row["token_id"]): bytes.fromhex(row["bytes_hex"]) for row in records}
    if sorted(by_id) != list(range(len(by_id))):
        raise ValueError("Tokenizer IDs must be unique and contiguous from zero")
    values = tuple(by_id[index] for index in range(len(by_id)))
    if any(not value for value in values):
        raise ValueError("Empty byte tokens are unsupported")
    return Vocabulary(values, str(path))


class MemmapCausalDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Read-only fixed-block next-token examples; worker-safe on Windows."""
    def __init__(self, path: Path, block_size: int, stride: Optional[int] = None) -> None:
        self.path = path
        self.block_size = block_size
        self.stride = stride or block_size
        self.n_tokens = path.stat().st_size // 2
        self.n_blocks = 1 + (self.n_tokens - block_size - 1) // self.stride
        if self.n_blocks <= 0:
            raise ValueError(f"Token stream is shorter than block size: {path}")
        self._array: Optional[np.memmap] = None

    def _tokens(self) -> np.memmap:
        if self._array is None:
            self._array = np.memmap(self.path, mode="r", dtype="<u2")
        return self._array

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        values = np.array(
            self._tokens()[start : start + self.block_size + 1],
            dtype=np.int64, copy=True,
        )
        tensor = torch.from_numpy(values)
        return tensor[:-1], tensor[1:]

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_array"] = None
        return state


def make_loader(dataset: Dataset[Any], batch_size: int, shuffle: bool, seed: int, workers: int, pin: bool) -> DataLoader[Any]:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
        num_workers=workers, pin_memory=pin, persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def cycle(loader: DataLoader[Any]) -> Iterator[Any]:
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Data-aware Fourier schedule.  This is the same deterministic search family as
# the final Phase-1 proof; it uses only the training token-length distribution.
# ---------------------------------------------------------------------------

def empirical_profile(vocabulary: Vocabulary, train_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    tokens = np.memmap(train_path, mode="r", dtype="<u2")
    if int(tokens.max()) >= len(vocabulary):
        raise ValueError("Token stream contains an ID outside the vocabulary")
    counts = np.bincount(tokens, minlength=len(vocabulary)).astype(np.int64)
    occurrence_hist = np.bincount(vocabulary.lengths, weights=counts).astype(np.int64)
    total = float(occurrence_hist.sum())
    masses = np.asarray(
        [occurrence_hist[1:9].sum(), occurrence_hist[9:17].sum(), occurrence_hist[17:].sum()],
        dtype=np.float64,
    ) / total
    weights = 0.85 * masses + 0.15 / 3.0
    return counts, {
        "token_occurrences": int(total),
        "occurrence_histogram": occurrence_hist.tolist(),
        "bucket_mass_le8_9to16_17plus": masses.tolist(),
        "search_weights_8_16_32": weights.tolist(),
    }


def frequency_metrics(omegas: np.ndarray, window: int) -> Dict[str, float]:
    positions = np.arange(window, dtype=np.float64)[:, None]
    waves = np.empty((window, 32), dtype=np.float64)
    waves[:, 0::2] = np.cos(positions * omegas[None, :])
    waves[:, 1::2] = np.sin(positions * omegas[None, :])
    waves /= 4.0
    gram = waves @ waves.T
    off = gram - np.eye(window)
    singular = np.linalg.svd(waves, compute_uv=False)
    probability = np.square(singular) / np.square(singular).sum()
    effective_rank = float(np.exp(-np.sum(probability * np.log(probability + 1e-300))))
    return {
        "window": float(window),
        "effective_rank": effective_rank,
        "rank_ceiling": float(min(window, 32)),
        "condition_number": float(singular.max() / singular[singular > 1e-12].min()),
        "rms_off_diagonal_cosine": float(np.sqrt(np.mean(np.square(off)))),
        "max_abs_off_diagonal_cosine": float(np.max(np.abs(off))),
    }


def alias_score(omegas: np.ndarray) -> float:
    lags = np.arange(32, 257, dtype=np.float64)[:, None]
    return float(np.mean(np.cos(lags * omegas[None, :]), axis=1).max())


def search_frequencies(profile: Mapping[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    weights = np.asarray(profile["search_weights_8_16_32"], dtype=np.float64)
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    indices = np.arange(16, dtype=np.float64)
    best: Optional[Tuple[float, float, float, np.ndarray, List[Dict[str, float]], float]] = None
    for strength in np.linspace(0.05, 0.35, 61):
        for phase in np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False):
            offsets = strength * np.sin(2.0 * math.pi * golden * (indices + 1.0) + phase)
            omegas = math.pi * (indices + 0.5 + offsets) / 16.0
            metrics = [frequency_metrics(omegas, window) for window in (8, 16, 32)]
            score = 0.0
            for weight, metric in zip(weights, metrics):
                deficit = 1.0 - metric["effective_rank"] / metric["rank_ceiling"]
                score += float(weight) * (
                    metric["rms_off_diagonal_cosine"]
                    + 0.20 * metric["max_abs_off_diagonal_cosine"]
                    + 0.50 * deficit
                )
            recurrence = alias_score(omegas)
            score += 0.02 * recurrence
            candidate = (score, float(strength), float(phase), omegas, metrics, recurrence)
            if best is None or score < best[0]:
                best = candidate
    assert best is not None
    score, strength, phase, omegas, metrics, recurrence = best
    artifact = {
        "method": "golden-ratio perturbation of 16 DCT midpoint frequencies",
        "selected_strength": strength,
        "selected_phase_radians": phase,
        "objective_value": score,
        "max_positive_cosine_lag_32_to_256": recurrence,
        "omegas_radians_per_byte": omegas.tolist(),
        "metrics": metrics,
        "profile": dict(profile),
    }
    return torch.tensor(omegas, dtype=torch.float64), artifact


def phase_waves(length: int, omegas: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float64)[:, None]
    phase = positions * omegas[None, :]
    waves = torch.empty((length, 32), dtype=torch.float64)
    waves[:, 0::2] = torch.cos(phase)
    waves[:, 1::2] = torch.sin(phase)
    return waves


Arm = Literal["v1", "v3", "cnn"]


def build_codec_table(vocabulary: Vocabulary, arm: Arm, omegas: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Build V1/V3 raw rows; CNN uses its live convolution and needs no table."""
    if arm == "cnn":
        return torch.empty((0, 256, 32), dtype=dtype)
    table = torch.zeros((len(vocabulary), 256, 32), dtype=torch.float32)
    for token_id, complete in enumerate(vocabulary.token_bytes):
        value = complete[:32] if arm == "v1" else complete
        byte_ids = torch.tensor(list(value), dtype=torch.long)
        if arm == "v1":
            table[token_id, byte_ids, torch.arange(len(value))] = 1.0
        else:
            table[token_id].index_add_(0, byte_ids, phase_waves(len(value), omegas).float())
        table[token_id] *= 1.0 / math.sqrt(float(len(value)))
    return table.to(dtype)


def build_padded_byte_table(vocabulary: Vocabulary, limit: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    pad = 256
    ids = torch.full((len(vocabulary), limit), pad, dtype=torch.int16)
    lengths = torch.empty(len(vocabulary), dtype=torch.int16)
    for token_id, complete in enumerate(vocabulary.token_bytes):
        value = complete[:limit]
        ids[token_id, : len(value)] = torch.tensor(list(value), dtype=torch.int16)
        lengths[token_id] = len(value)
    return ids, lengths


class AuditedStructuredEmbedding(nn.Module):
    """Three forward paths with one common trainable state.

    The V1/V3 arms use deterministic raw codec rows.  The CNN arm embeds padded
    bytes, applies one trainable local convolution, masks padding, flattens
    position-specific channels, and zero-pads to the same 8,192 input features.
    Every path then receives identical per-token z-normalization and projection.
    """

    def __init__(
        self,
        arm: Arm,
        raw_table: torch.Tensor,
        padded_bytes: torch.Tensor,
        byte_lengths: torch.Tensor,
        omegas: torch.Tensor,
        d_model: int,
        config: Config,
    ) -> None:
        super().__init__()
        self.arm = arm
        self.eps = config.z_norm_eps
        self.codec_dim = config.codec_dim
        self.cnn_width = config.cnn_width
        self.register_buffer("raw_codec_table", raw_table, persistent=False)
        self.register_buffer("padded_byte_table", padded_bytes, persistent=False)
        self.register_buffer("byte_lengths", byte_lengths, persistent=False)
        self.register_buffer("omegas", omegas.float(), persistent=False)

        # Present and identically initialized in all arms.  V1/V3 leave these
        # parameters outside the graph; CNN activates them.
        self.char_embedding = nn.Embedding(257, config.cnn_width, padding_idx=256)
        self.char_conv = nn.Conv1d(
            config.cnn_width, config.cnn_width, kernel_size=3, padding=1, bias=True
        )
        self.projection = nn.Linear(config.codec_dim, d_model, bias=False)

    def _cnn_vectors(self, byte_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        chars = self.char_embedding(byte_ids.long())               # [U, 32, C]
        local = F.gelu(self.char_conv(chars.transpose(1, 2)))      # [U, C, 32]
        positions = torch.arange(byte_ids.shape[1], device=byte_ids.device)[None, :]
        mask = positions < lengths.long()[:, None]
        local = local * mask[:, None, :]
        compact = local.transpose(1, 2).flatten(start_dim=1)       # [U, 32*C]
        return F.pad(compact, (0, self.codec_dim - compact.shape[1]))

    def _normalize_project(self, vectors: torch.Tensor) -> torch.Tensor:
        centered = vectors - vectors.mean(dim=1, keepdim=True)
        normalized = centered * torch.rsqrt(centered.square().mean(dim=1, keepdim=True) + self.eps)
        return self.projection(normalized)

    def embed_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        flat = token_ids.reshape(-1)
        unique, inverse = torch.unique(flat, sorted=False, return_inverse=True)
        if self.arm == "cnn":
            byte_ids = self.padded_byte_table.index_select(0, unique)
            lengths = self.byte_lengths.index_select(0, unique)
            vectors = self._cnn_vectors(byte_ids, lengths)
        else:
            vectors = self.raw_codec_table.index_select(0, unique).float().flatten(start_dim=1)
        embedded = self._normalize_project(vectors).index_select(0, inverse)
        return embedded.view(*token_ids.shape, -1)

    def embed_arbitrary_bytes(self, values: Sequence[bytes]) -> torch.Tensor:
        """Embed strings outside the fixed vocabulary for reversal probes."""
        device = self.projection.weight.device
        if self.arm == "cnn":
            padded = torch.full((len(values), 32), 256, dtype=torch.long, device=device)
            lengths = torch.empty(len(values), dtype=torch.long, device=device)
            for index, complete in enumerate(values):
                value = complete[:32]
                padded[index, : len(value)] = torch.tensor(list(value), device=device)
                lengths[index] = len(value)
            vectors = self._cnn_vectors(padded, lengths)
        else:
            vectors = torch.zeros((len(values), 256, 32), dtype=torch.float32, device=device)
            for index, complete in enumerate(values):
                value = complete[:32] if self.arm == "v1" else complete
                byte_ids = torch.tensor(list(value), dtype=torch.long, device=device)
                if self.arm == "v1":
                    vectors[index, byte_ids, torch.arange(len(value), device=device)] = 1.0
                else:
                    positions = torch.arange(len(value), device=device, dtype=torch.float32)[:, None]
                    phase = positions * self.omegas[None, :]
                    waves = torch.empty((len(value), 32), device=device)
                    waves[:, 0::2], waves[:, 1::2] = torch.cos(phase), torch.sin(phase)
                    vectors[index].index_add_(0, byte_ids, waves)
                vectors[index] *= 1.0 / math.sqrt(float(len(value)))
            vectors = vectors.flatten(start_dim=1)
        return self._normalize_project(vectors)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_ids(token_ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0, is_causal=True,
        )
        return self.output(attended.transpose(1, 2).contiguous().view(batch, steps, channels))


class TransformerBlock(nn.Module):
    def __init__(self, preset: ModelPreset, dropout: float) -> None:
        super().__init__()
        hidden = preset.mlp_ratio * preset.d_model
        self.ln1 = nn.LayerNorm(preset.d_model)
        self.attention = CausalSelfAttention(preset.d_model, preset.n_heads, dropout)
        self.ln2 = nn.LayerNorm(preset.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(preset.d_model, hidden, bias=False), nn.GELU(approximate="tanh"),
            nn.Linear(hidden, preset.d_model, bias=False), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.ln1(x)))
        return x + self.mlp(self.ln2(x))


class CausalTransformer(nn.Module):
    def __init__(
        self, preset: ModelPreset, vocab_size: int, arm: Arm, raw_table: torch.Tensor,
        padded_bytes: torch.Tensor, byte_lengths: torch.Tensor, omegas: torch.Tensor,
        config: Config,
    ) -> None:
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding = AuditedStructuredEmbedding(
            arm, raw_table, padded_bytes, byte_lengths, omegas,
            preset.d_model, config,
        )
        self.sequence_position = nn.Embedding(config.block_size, preset.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(preset, config.dropout) for _ in range(preset.n_layers)])
        self.final_norm = nn.LayerNorm(preset.d_model)
        self.lm_head = nn.Linear(preset.d_model, vocab_size, bias=False)
        self.apply(self._initialize)
        nn.init.normal_(self.token_embedding.projection.weight, 0.0, 1.0 / math.sqrt(config.codec_dim))
        # The generic Transformer embedding initialization (std=0.02) makes the
        # CNN's sparse pre-z-norm variance so small that global normalization can
        # amplify its first backward pass under FP16 loss scaling.  Unit-fan-in
        # initialization calibrates CNN feature RMS to the fixed codec paths and
        # makes the comparison numerically meaningful rather than epsilon-led.
        nn.init.normal_(
            self.token_embedding.char_embedding.weight,
            0.0,
            1.0 / math.sqrt(config.cnn_width),
        )
        # PyTorch initializes padding rows to zero; make that invariant explicit.
        with torch.no_grad():
            self.token_embedding.char_embedding.weight[256].zero_()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, steps = tokens.shape
        positions = torch.arange(steps, device=tokens.device)
        x = self.token_embedding(tokens) + self.sequence_position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten()) if targets is not None else None
        return logits, loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_device(request: str) -> torch.device:
    device = torch.device("cuda" if request == "auto" and torch.cuda.is_available() else "cpu" if request == "auto" else request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def amp_context(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()


def trainable_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def optimizer_for(model: nn.Module, config: Config, device: torch.device) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for value in model.parameters():
        (decay if value.ndim >= 2 else no_decay).append(value)
    groups = [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"lr": config.learning_rate, "betas": (config.beta1, config.beta2), "eps": 1e-8}
    try:
        return torch.optim.AdamW(groups, fused=device.type == "cuda", **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, **kwargs)


def lr_at_step(step: int, config: Config) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader[Any], device: torch.device, batches: int) -> Dict[str, float]:
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for index, (tokens, targets) in enumerate(loader):
        if index >= batches:
            break
        tokens, targets = tokens.to(device), targets.to(device)
        with amp_context(device):
            logits, loss = model(tokens, targets)
        assert loss is not None
        n = targets.numel()
        loss_sum += float(loss) * n
        correct += int((logits.argmax(dim=-1) == targets).sum())
        count += n
    model.train()
    return {"loss": loss_sum / count, "accuracy": correct / count, "tokens": float(count)}


# ---------------------------------------------------------------------------
# Geometry probes.
# ---------------------------------------------------------------------------

def normalize_word_bytes(value: bytes) -> Optional[bytes]:
    """Return simple ASCII word content; skip punctuation and binary tokens."""
    if value.startswith(b" "):
        value = value[1:]
    if 3 <= len(value) <= 20 and all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in value):
        return value.lower()
    return None


def levenshtein(left: bytes, right: bytes) -> int:
    previous = list(range(len(right) + 1))
    for row, lbyte in enumerate(left, start=1):
        current = [row]
        for column, rbyte in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (lbyte != rbyte)))
        previous = current
    return previous[-1]


def build_pair_sets(
    vocabulary: Vocabulary, counts: np.ndarray, pair_count: int, seed: int
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Dict[str, Any]]:
    """Find edit-similar word tokens and length-matched random controls."""
    eligible = [
        token_id for token_id in np.argsort(-counts)
        if normalize_word_bytes(vocabulary.token_bytes[int(token_id)]) is not None
    ]
    words = {token_id: normalize_word_bytes(vocabulary.token_bytes[token_id]) for token_id in eligible}

    # Substitution signatures: words one character apart share at least one
    # deleted-character signature.  This avoids an O(V^2) vocabulary search.
    signature_groups: Dict[Tuple[int, bytes], List[int]] = {}
    for token_id in eligible:
        word = words[token_id]
        assert word is not None
        for index in range(len(word)):
            signature_groups.setdefault((len(word), word[:index] + word[index + 1 :]), []).append(token_id)

    candidates: set[Tuple[int, int]] = set()
    for group in signature_groups.values():
        unique = list(dict.fromkeys(group))
        for left_index in range(len(unique)):
            for right_index in range(left_index + 1, len(unique)):
                left, right = unique[left_index], unique[right_index]
                if words[left] != words[right] and levenshtein(words[left] or b"", words[right] or b"") <= 1:
                    candidates.add((min(left, right), max(left, right)))

    # If the vocabulary has fewer exact one-edit pairs than requested, add
    # two-edit candidates inside compact prefix/length buckets.
    used_distance_two = len(candidates) < pair_count
    if used_distance_two:
        buckets: Dict[Tuple[int, bytes], List[int]] = {}
        for token_id in eligible:
            word = words[token_id]
            assert word is not None
            buckets.setdefault((len(word), word[:1]), []).append(token_id)
        for group in buckets.values():
            for left_index in range(min(len(group), 80)):
                for right_index in range(left_index + 1, min(len(group), 80)):
                    left, right = group[left_index], group[right_index]
                    if words[left] != words[right] and levenshtein(words[left] or b"", words[right] or b"") <= 2:
                        candidates.add((min(left, right), max(left, right)))
                    if len(candidates) >= 4 * pair_count:
                        break

    rng = random.Random(seed)
    similar = list(candidates)
    rng.shuffle(similar)
    similar = similar[: min(pair_count, len(similar))]
    if not similar:
        raise RuntimeError("No edit-similar word-token pairs were found")

    by_length: Dict[int, List[int]] = {}
    for token_id in eligible:
        word = words[token_id]
        assert word is not None
        by_length.setdefault(len(word), []).append(token_id)
    random_pairs: List[Tuple[int, int]] = []
    similar_set = set(similar)
    for left, _ in similar:
        word = words[left]
        assert word is not None
        pool = by_length[len(word)]
        for _attempt in range(100):
            a, b = rng.sample(pool, 2)
            pair = (min(a, b), max(a, b))
            if pair not in similar_set and (words[a] or b"") != (words[b] or b""):
                random_pairs.append(pair)
                break
    return similar, random_pairs, {
        "eligible_word_tokens": len(eligible),
        "similar_pair_count": len(similar),
        "random_pair_count": len(random_pairs),
        "similar_edit_distance_max": 2 if used_distance_two else 1,
        "random_pairs_length_matched": True,
    }


def distribution_summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "q05": float(np.quantile(values, 0.05)), "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
    }


@torch.no_grad()
def collect_embeddings(
    model: CausalTransformer, token_ids: Sequence[int], device: torch.device, chunk: int = 256
) -> torch.Tensor:
    model.eval()
    pieces = []
    for start in range(0, len(token_ids), chunk):
        ids = torch.tensor(token_ids[start : start + chunk], dtype=torch.long, device=device)
        pieces.append(model.token_embedding.embed_ids(ids).float().cpu())
    model.train()
    return torch.cat(pieces)


def pair_cosines(embedding_by_id: Mapping[int, torch.Tensor], pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    values = []
    for left, right in pairs:
        values.append(float(F.cosine_similarity(embedding_by_id[left][None], embedding_by_id[right][None])))
    return np.asarray(values, dtype=np.float64)


@torch.no_grad()
def geometry_audit(
    model: CausalTransformer,
    vocabulary: Vocabulary,
    geometry_ids: Sequence[int],
    similar_pairs: Sequence[Tuple[int, int]],
    random_pairs: Sequence[Tuple[int, int]],
    order_values: Sequence[bytes],
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    required = sorted(set(geometry_ids) | {item for pair in similar_pairs + random_pairs for item in pair})
    embeddings = collect_embeddings(model, required, device)
    embedding_by_id = {token_id: embeddings[index] for index, token_id in enumerate(required)}
    geometry = torch.stack([embedding_by_id[token_id] for token_id in geometry_ids])
    norms = torch.linalg.vector_norm(geometry, dim=1).numpy()
    per_dimension_variance = geometry.var(dim=0, unbiased=False).numpy()
    similar_cosines = pair_cosines(embedding_by_id, similar_pairs)
    random_cosines = pair_cosines(embedding_by_id, random_pairs)

    model.eval()
    original_parts, reversed_parts = [], []
    for start in range(0, len(order_values), 128):
        batch = order_values[start : start + 128]
        original_parts.append(model.token_embedding.embed_arbitrary_bytes(batch).float().cpu())
        reversed_parts.append(model.token_embedding.embed_arbitrary_bytes([value[::-1] for value in batch]).float().cpu())
    model.train()
    original = torch.cat(original_parts)
    reversed_embedding = torch.cat(reversed_parts)
    reversal_cosines = F.cosine_similarity(original, reversed_embedding, dim=1).numpy()

    centered = geometry - geometry.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square() / singular.square().sum()
    effective_rank = float(torch.exp(-(energy * torch.log(energy + 1e-30)).sum()))
    metrics = {
        "sampled_tokens": len(geometry_ids),
        "mean_embedding_norm": float(norms.mean()),
        "std_embedding_norm": float(norms.std()),
        "mean_per_dimension_variance": float(per_dimension_variance.mean()),
        "std_per_dimension_variance": float(per_dimension_variance.std()),
        "min_per_dimension_variance": float(per_dimension_variance.min()),
        "max_per_dimension_variance": float(per_dimension_variance.max()),
        "embedding_effective_rank": effective_rank,
        "similar_word_cosine": distribution_summary(similar_cosines),
        "random_word_cosine": distribution_summary(random_cosines),
        "similar_minus_random_mean_cosine": float(similar_cosines.mean() - random_cosines.mean()),
        "reversal_cosine": distribution_summary(reversal_cosines),
        "reversal_sensitivity_one_minus_cosine": float(1.0 - reversal_cosines.mean()),
    }
    arrays = {
        "norms": norms, "dimension_variance": per_dimension_variance,
        "similar_cosines": similar_cosines, "random_cosines": random_cosines,
        "reversal_cosines": reversal_cosines,
    }
    return metrics, arrays


def gradient_audit(model: CausalTransformer) -> Dict[str, Any]:
    groups: Dict[str, List[Tuple[str, nn.Parameter]]] = {
        "codec_projection": [], "cnn_byte_embedding": [], "cnn_convolution": [],
        "sequence_position": [], "attention": [], "mlp": [], "normalization": [], "lm_head": [],
    }
    for name, parameter in model.named_parameters():
        if "token_embedding.projection" in name:
            key = "codec_projection"
        elif "token_embedding.char_embedding" in name:
            key = "cnn_byte_embedding"
        elif "token_embedding.char_conv" in name:
            key = "cnn_convolution"
        elif "sequence_position" in name:
            key = "sequence_position"
        elif ".attention." in name:
            key = "attention"
        elif ".mlp." in name:
            key = "mlp"
        elif "norm" in name or ".ln" in name:
            key = "normalization"
        elif "lm_head" in name:
            key = "lm_head"
        else:
            raise KeyError(f"Unclassified trainable parameter: {name}")
        groups[key].append((name, parameter))

    result: Dict[str, Any] = {}
    total_active, total_nonzero = 0, 0
    for key, members in groups.items():
        squared_norm, active, nonzero = 0.0, 0, 0
        for _name, parameter in members:
            if parameter.grad is not None:
                gradient = parameter.grad.detach().float()
                squared_norm += float(gradient.square().sum())
                active += parameter.numel()
                nonzero += int(torch.count_nonzero(gradient))
        total_active += active
        total_nonzero += nonzero
        result[key] = {
            "l2_norm": math.sqrt(squared_norm),
            "active_parameter_elements": active,
            "nonzero_gradient_elements": nonzero,
        }
    result["all_modules"] = {
        "l2_norm": math.sqrt(sum(value["l2_norm"] ** 2 for value in result.values())),
        "active_parameter_elements": total_active,
        "nonzero_gradient_elements": total_nonzero,
    }
    return result


def train_arm(
    arm: Arm,
    config: Config,
    preset: ModelPreset,
    vocabulary: Vocabulary,
    omegas: torch.Tensor,
    padded_bytes: torch.Tensor,
    byte_lengths: torch.Tensor,
    train_data: MemmapCausalDataset,
    val_data: MemmapCausalDataset,
    geometry_ids: Sequence[int],
    similar_pairs: Sequence[Tuple[int, int]],
    random_pairs: Sequence[Tuple[int, int]],
    order_values: Sequence[bytes],
    device: torch.device,
    expected_hash: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, np.ndarray]]]:
    print(f"\n{'=' * 80}\nARM {arm.upper()}\n{'=' * 80}", flush=True)
    table_dtype = torch.float16 if device.type == "cuda" else torch.float32
    table = build_codec_table(vocabulary, arm, omegas, table_dtype)
    seed_everything(config.seed)
    model = CausalTransformer(
        preset, len(vocabulary), arm, table, padded_bytes, byte_lengths,
        omegas, config,
    ).to(device)
    del table
    initial_hash = trainable_sha256(model)
    if expected_hash is not None and initial_hash != expected_hash:
        raise AssertionError(f"Trainable state mismatch: {initial_hash} != {expected_hash}")
    print(f"parameters={parameter_count(model):,}; initial_sha256={initial_hash}", flush=True)

    initial_geometry, initial_arrays = geometry_audit(
        model, vocabulary, geometry_ids, similar_pairs, random_pairs,
        order_values, device,
    )
    train_loader = make_loader(train_data, config.micro_batch_size, True, config.seed, config.num_workers, device.type == "cuda")
    val_loader = make_loader(val_data, config.micro_batch_size, False, config.seed, config.num_workers, device.type == "cuda")
    iterator = cycle(train_loader)
    optimizer = optimizer_for(model, config, device)
    # A conservative initial scale is preferable for this gradient-audit script:
    # we care more about a finite, interpretable first step than rapid automatic
    # scale discovery over the first few updates.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=1_024.0
    )
    initial_validation = validate(model, val_loader, device, config.val_batches)
    history: List[Dict[str, Any]] = [{
        "step": 0, "train_loss": None, "val_loss": initial_validation["loss"],
        "val_accuracy": initial_validation["accuracy"],
    }]
    first_step_gradients: Optional[Dict[str, Any]] = None
    interval_loss, interval_steps = 0.0, 0
    interval_started = time.perf_counter()
    train_started = interval_started

    for step in range(1, config.max_steps + 1):
        learning_rate = lr_at_step(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        for _ in range(config.grad_accum_steps):
            tokens, targets = next(iterator)
            tokens, targets = tokens.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with amp_context(device):
                _logits, loss = model(tokens, targets)
                assert loss is not None
                scaled_loss = loss / config.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            update_loss += float(loss.detach()) / config.grad_accum_steps
        scaler.unscale_(optimizer)
        if step == 1:
            nonfinite = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if nonfinite:
                raise FloatingPointError(
                    "Non-finite first-step gradients: " + ", ".join(nonfinite)
                )
            first_step_gradients = gradient_audit(model)
            cnn_active = first_step_gradients["cnn_convolution"]["active_parameter_elements"]
            if arm == "cnn" and cnn_active == 0:
                raise AssertionError("CNN arm did not activate convolution gradients")
            if arm != "cnn" and cnn_active != 0:
                raise AssertionError(f"{arm} unexpectedly activated convolution gradients")
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        interval_loss += update_loss
        interval_steps += 1

        if step % config.log_interval == 0 or step == config.max_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            validation = validate(model, val_loader, device, config.val_batches)
            elapsed = time.perf_counter() - interval_started
            tokens_per_second = interval_steps * config.micro_batch_size * config.grad_accum_steps * config.block_size / max(elapsed, 1e-9)
            row = {
                "step": step, "train_loss": interval_loss / interval_steps,
                "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
                "learning_rate": learning_rate, "grad_norm": grad_norm,
                "tokens_per_second": tokens_per_second,
            }
            history.append(row)
            print(
                f"step={step:04d} train={row['train_loss']:.5f} val={row['val_loss']:.5f} "
                f"acc={row['val_accuracy']:.3%} tok/s={tokens_per_second:,.0f}", flush=True,
            )
            interval_loss, interval_steps, interval_started = 0.0, 0, time.perf_counter()

    assert first_step_gradients is not None
    final_geometry, final_arrays = geometry_audit(
        model, vocabulary, geometry_ids, similar_pairs, random_pairs,
        order_values, device,
    )
    result = {
        "arm": arm,
        "trainable_parameters_stored": parameter_count(model),
        "initial_trainable_sha256": initial_hash,
        "initial_validation": initial_validation,
        "final_val_loss": history[-1]["val_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "history": history,
        "geometry": {"initial": initial_geometry, "final": final_geometry},
        "first_step_gradients": first_step_gradients,
        "wall_seconds": time.perf_counter() - train_started,
        "control_note": (
            "CNN parameters stored but inert" if arm in ("v1", "v3")
            else "CNN parameters active; lightweight diagnostic is not effective-parameter matched"
        ),
    }
    del model, optimizer, scaler, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, {"initial": initial_arrays, "final": final_arrays}


# ---------------------------------------------------------------------------
# Plotting and Markdown artifact generation.
# ---------------------------------------------------------------------------

ARM_LABELS = {"v1": "V1 cutoff", "v3": "V3 Fourier", "cnn": "1-layer char CNN"}
ARM_COLORS = {"v1": "#1565C0", "v3": "#D84315", "cnn": "#2E7D32"}


def plot_training(results: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for result in results:
        arm, history = str(result["arm"]), result["history"]
        trained = [row for row in history if row["train_loss"] is not None]
        axes[0].plot([r["step"] for r in trained], [r["train_loss"] for r in trained], marker="o", color=ARM_COLORS[arm], label=ARM_LABELS[arm])
        axes[1].plot([r["step"] for r in history], [r["val_loss"] for r in history], marker="o", color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    for axis, title in zip(axes, ("Training loss", "Validation loss")):
        axis.set_title(title); axis.set_xlabel("Optimizer update"); axis.set_ylabel("Cross-entropy (nats)")
        axis.grid(alpha=0.25); axis.legend()
    figure.suptitle("Phase-1 Polish: short-token representation control")
    figure.tight_layout(); figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


def plot_embedding_geometry(results: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(results)); width = 0.35
    for offset, stage in ((-width / 2, "initial"), (width / 2, "final")):
        axes[0].bar(x + offset, [r["geometry"][stage]["mean_embedding_norm"] for r in results], width, label=stage.title())
        axes[1].bar(x + offset, [r["geometry"][stage]["mean_per_dimension_variance"] for r in results], width, label=stage.title())
        axes[2].bar(x + offset, [r["geometry"][stage]["embedding_effective_rank"] for r in results], width, label=stage.title())
    titles = ("Mean embedding norm", "Mean per-dimension variance", "Embedding effective rank")
    for axis, title in zip(axes, titles):
        axis.set_title(title); axis.set_xticks(x, [ARM_LABELS[str(r["arm"])] for r in results], rotation=12)
        axis.grid(axis="y", alpha=0.25); axis.legend()
    figure.suptitle("Projected token-embedding geometry")
    figure.tight_layout(); figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


def plot_pair_cosines(results: Sequence[Mapping[str, Any]], arrays: Mapping[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True)
    bins = np.linspace(-1.0, 1.0, 45)
    for column, result in enumerate(results):
        arm = str(result["arm"])
        for row, stage in enumerate(("initial", "final")):
            data = arrays[arm][stage]
            axes[row, column].hist(data["random_cosines"], bins=bins, density=True, alpha=0.55, color="#616161", label="Length-matched random")
            axes[row, column].hist(data["similar_cosines"], bins=bins, density=True, alpha=0.55, color=ARM_COLORS[arm], label="Edit-similar words")
            axes[row, column].set_title(f"{ARM_LABELS[arm]} — {stage}")
            axes[row, column].grid(alpha=0.2); axes[row, column].legend(fontsize=8)
    for axis in axes[-1, :]: axis.set_xlabel("Cosine similarity")
    for axis in axes[:, 0]: axis.set_ylabel("Density")
    figure.suptitle("Random versus orthographically similar token pairs")
    figure.tight_layout(); figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


def plot_reversal(results: Sequence[Mapping[str, Any]], arrays: Mapping[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    bins = np.linspace(-1.0, 1.0, 45)
    for axis, stage in zip(axes, ("initial", "final")):
        for result in results:
            arm = str(result["arm"])
            axis.hist(arrays[arm][stage]["reversal_cosines"], bins=bins, density=True, histtype="step", linewidth=2, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
        axis.set_title(stage.title()); axis.set_xlabel("cos(original, byte-reversed)")
        axis.grid(alpha=0.25); axis.legend()
    axes[0].set_ylabel("Density")
    figure.suptitle("Within-token byte-order sensitivity (lower cosine = greater sensitivity)")
    figure.tight_layout(); figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


def plot_gradients(results: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modules = ["codec_projection", "cnn_byte_embedding", "cnn_convolution", "sequence_position", "attention", "mlp", "normalization", "lm_head"]
    x = np.arange(len(modules)); width = 0.24
    figure, axis = plt.subplots(figsize=(15, 6))
    for index, result in enumerate(results):
        arm = str(result["arm"])
        values = [max(1e-12, result["first_step_gradients"][module]["l2_norm"]) for module in modules]
        axis.bar(x + (index - 1) * width, values, width, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    axis.set_yscale("log"); axis.set_ylabel("First-step gradient L2 norm (log scale)")
    axis.set_xticks(x, [name.replace("_", "\n") for name in modules]); axis.grid(axis="y", alpha=0.25); axis.legend()
    axis.set_title("First-step gradient flow by module")
    figure.tight_layout(); figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


def fmt(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def write_markdown(
    output: Path,
    config: Config,
    data_paths: DataPaths,
    search: Mapping[str, Any],
    pair_metadata: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> None:
    hashes = {str(result["initial_trainable_sha256"]) for result in results}
    lines = [
        "# Phase 1 Polish — Representation Geometry Ablation", "",
        "> Local diagnostic on the Week 6 standard-BPE corpus. This experiment audits representation geometry and a lightweight local-convolution control; it is not a substitute for large-scale pretraining.", "",
        "## Experimental controls", "",
        f"- **Data:** {data_paths.source}",
        f"- **Updates per arm:** {config.max_steps:,}",
        f"- **Stored trainable parameters per arm:** {int(results[0]['trainable_parameters_stored']):,}",
        f"- **Identical initial trainable state:** {'verified' if len(hashes) == 1 else 'FAILED'}",
        f"- **Initial SHA-256:** `{next(iter(hashes)) if len(hashes) == 1 else 'mismatch'}`",
        f"- **Word-pair probe:** {pair_metadata['similar_pair_count']} edit-similar and {pair_metadata['random_pair_count']} length-matched random pairs",
        "- **CNN control:** one width-3 convolution over 32 bytes; CNN parameters are stored but inert in V1/V3 and active only in the CNN arm.", "",
        "## Data-aware Fourier conditioning", "",
        "| Window | Effective rank | Ceiling | Condition number | Max off-diagonal cosine |", "|---:|---:|---:|---:|---:|",
    ]
    for metric in search["metrics"]:
        lines.append(
            f"| {int(metric['window'])} | {fmt(metric['effective_rank'], 4)} | {int(metric['rank_ceiling'])} | {fmt(metric['condition_number'], 4)} | {fmt(metric['max_abs_off_diagonal_cosine'], 4)} |"
        )
    lines += ["", "## Short-token language-model result", "", "| Arm | Final validation loss | Final accuracy | Mean tokens/s |", "|---|---:|---:|---:|"]
    for result in results:
        trained = [row for row in result["history"] if row["train_loss"] is not None]
        mean_tps = float(np.mean([row["tokens_per_second"] for row in trained]))
        lines.append(f"| {ARM_LABELS[str(result['arm'])]} | {fmt(result['final_val_loss'])} | {100 * result['final_val_accuracy']:.3f}% | {mean_tps:,.0f} |")

    lines += ["", "## Projected embedding geometry", "", "| Arm | Stage | Mean norm | Mean dimension variance | Effective rank | Similar cosine | Random cosine | Similar − random | Reversal sensitivity |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        for stage in ("initial", "final"):
            geometry = result["geometry"][stage]
            lines.append(
                f"| {ARM_LABELS[str(result['arm'])]} | {stage} | {fmt(geometry['mean_embedding_norm'])} | "
                f"{fmt(geometry['mean_per_dimension_variance'])} | {fmt(geometry['embedding_effective_rank'], 2)} | "
                f"{fmt(geometry['similar_word_cosine']['mean'])} | {fmt(geometry['random_word_cosine']['mean'])} | "
                f"{fmt(geometry['similar_minus_random_mean_cosine'])} | {fmt(geometry['reversal_sensitivity_one_minus_cosine'])} |"
            )

    modules = ["codec_projection", "cnn_byte_embedding", "cnn_convolution", "sequence_position", "attention", "mlp", "normalization", "lm_head"]
    lines += ["", "## First-step gradient norms", "", "| Module | V1 | V3 | Char CNN |", "|---|---:|---:|---:|"]
    by_arm = {str(result["arm"]): result for result in results}
    for module in modules:
        lines.append(
            f"| {module.replace('_', ' ')} | {by_arm['v1']['first_step_gradients'][module]['l2_norm']:.3e} | "
            f"{by_arm['v3']['first_step_gradients'][module]['l2_norm']:.3e} | {by_arm['cnn']['first_step_gradients'][module]['l2_norm']:.3e} |"
        )
    lines += [
        "", "## Figures", "",
        "![Training curves](training_curves.png)", "",
        "![Embedding geometry](embedding_geometry.png)", "",
        "![Pair cosine distributions](pair_cosine_distributions.png)", "",
        "![Byte reversal sensitivity](byte_reversal_sensitivity.png)", "",
        "![First-step gradients](first_step_gradients.png)", "",
        "## Interpretation guardrails", "",
        "- Similar-word cosine is descriptive geometry, not semantic understanding.",
        "- Reversal sensitivity shows that an encoder responds to order; it does not by itself show that the response is useful.",
        "- The CNN is deliberately lightweight and is not effective-parameter matched: most columns of the common 8,192-dimensional projection receive zero input in its forward path.",
        "- V1 and V3 have identical complete trainable initialization. The CNN shares that state too, but activates additional convolution parameters.",
        "- This local ablation addresses representation controls and gradient flow only; it does not establish large-scale scaling behavior.", "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--geometry-tokens", type=int, default=2_048)
    parser.add_argument("--pair-count", type=int, default=512)
    parser.add_argument("--order-probe-count", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7_007)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--train-tokens", type=Path)
    parser.add_argument("--val-tokens", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "phase1_polish_representation_run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        preset="smoke" if args.smoke_test else "mid",
        block_size=32 if args.smoke_test else args.block_size,
        micro_batch_size=1 if args.smoke_test else args.micro_batch_size,
        grad_accum_steps=1 if args.smoke_test else args.grad_accum_steps,
        max_steps=1 if args.smoke_test else args.steps,
        val_batches=1 if args.smoke_test else args.val_batches,
        log_interval=1 if args.smoke_test else args.log_interval,
        warmup_steps=1 if args.smoke_test else max(1, min(25, args.steps // 20)),
        geometry_tokens=min(256, args.geometry_tokens) if args.smoke_test else args.geometry_tokens,
        pair_count=min(64, args.pair_count) if args.smoke_test else args.pair_count,
        order_probe_count=min(64, args.order_probe_count) if args.smoke_test else args.order_probe_count,
        seed=args.seed, num_workers=0 if args.smoke_test else args.num_workers,
        device=args.device,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = discover_data(args)
    vocabulary = load_vocabulary(paths.tokenizer_json)
    if len(vocabulary) != 8_192:
        raise ValueError(f"This controlled preset expects 8,192 tokens, found {len(vocabulary):,}")
    counts, profile = empirical_profile(vocabulary, paths.train_tokens)
    omegas, frequency_search = search_frequencies(profile)
    padded_bytes, byte_lengths = build_padded_byte_table(vocabulary)
    train_data = MemmapCausalDataset(paths.train_tokens, config.block_size)
    val_data = MemmapCausalDataset(paths.val_tokens, config.block_size)
    device = configure_device(config.device)
    preset = PRESETS[config.preset]

    ranked = [int(token_id) for token_id in np.argsort(-counts) if vocabulary.token_bytes[int(token_id)]]
    geometry_ids = ranked[: min(config.geometry_tokens, len(ranked))]
    similar_pairs, random_pairs, pair_metadata = build_pair_sets(vocabulary, counts, config.pair_count, config.seed + 17)
    order_candidates = [
        vocabulary.token_bytes[token_id] for token_id in ranked
        if 4 <= len(vocabulary.token_bytes[token_id]) <= 16
        and vocabulary.token_bytes[token_id] != vocabulary.token_bytes[token_id][::-1]
    ][: config.order_probe_count]
    if not order_candidates:
        raise RuntimeError("No non-palindromic order-probe tokens were found")

    print(json.dumps({
        "config": asdict(config), "preset": asdict(preset), "device": str(device),
        "data": paths.source, "pair_metadata": pair_metadata,
    }, indent=2), flush=True)
    results: List[Dict[str, Any]] = []
    all_arrays: Dict[str, Any] = {}
    expected_hash: Optional[str] = None
    for arm in ("v1", "v3", "cnn"):
        result, arrays = train_arm(
            arm, config, preset, vocabulary, omegas, padded_bytes, byte_lengths,
            train_data, val_data, geometry_ids, similar_pairs, random_pairs,
            order_candidates, device, expected_hash,
        )
        expected_hash = expected_hash or result["initial_trainable_sha256"]
        results.append(result); all_arrays[arm] = arrays
    if len({result["initial_trainable_sha256"] for result in results}) != 1:
        raise AssertionError("Initial trainable hashes differ across arms")

    plot_training(results, output / "training_curves.png")
    plot_embedding_geometry(results, output / "embedding_geometry.png")
    plot_pair_cosines(results, all_arrays, output / "pair_cosine_distributions.png")
    plot_reversal(results, all_arrays, output / "byte_reversal_sensitivity.png")
    plot_gradients(results, output / "first_step_gradients.png")
    report = {
        "title": "Phase-1 Polish representation-geometry ablation",
        "config": asdict(config), "preset": asdict(preset),
        "runtime": {
            "torch": torch.__version__, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "fp16_amp": device.type == "cuda",
        },
        "data": {
            "source": paths.source, "tokenizer": str(paths.tokenizer_json),
            "train_tokens": str(paths.train_tokens), "validation_tokens": str(paths.val_tokens),
        },
        "frequency_search": frequency_search,
        "pair_probe": pair_metadata,
        "order_probe_count": len(order_candidates),
        "controlled_initialization": {"verified": True, "sha256": expected_hash},
        "arms": results,
        "artifacts": {
            "markdown": str(output / "README_PHASE1_POLISH.md"),
            "training_curves": str(output / "training_curves.png"),
            "embedding_geometry": str(output / "embedding_geometry.png"),
            "pair_cosines": str(output / "pair_cosine_distributions.png"),
            "reversal_sensitivity": str(output / "byte_reversal_sensitivity.png"),
            "first_step_gradients": str(output / "first_step_gradients.png"),
        },
    }
    json_dump(report, output / "metrics.json")
    write_markdown(output / "README_PHASE1_POLISH.md", config, paths, frequency_search, pair_metadata, results)
    print(f"\nComplete. README-ready artifact: {output / 'README_PHASE1_POLISH.md'}", flush=True)


if __name__ == "__main__":
    main()
