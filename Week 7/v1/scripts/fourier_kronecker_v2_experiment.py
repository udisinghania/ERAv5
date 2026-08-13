#!/usr/bin/env python3
"""Controlled Kronecker V1 vs. Fourier-Kronecker V2 language-model ablation.

This standalone script trains two otherwise identical causal Transformers on the
same frozen token stream:

    Arm A -- Kronecker V1 with a hard 32-byte, one-hot position basis.
    Arm B -- Fourier-Kronecker V2 with a continuous 32-channel Fourier basis and
             no hard token-length cutoff.

The default paths target the ERA Week 6 V2 artifacts.  The input files are
memory-mapped, so the full corpus is never copied into RAM.  Both arms receive
the same initial trainable weights, data order, optimizer, scheduler, and number
of updates.  Only the fixed token-to-codec table differs.

Examples
--------
Full controlled experiment (1,000 optimizer updates per arm):

    python fourier_kronecker_v2_experiment.py

Fast mathematical diagnostics only:

    python fourier_kronecker_v2_experiment.py --diagnostics-only

One-update end-to-end integration test:

    python fourier_kronecker_v2_experiment.py --smoke-test

Scale the Transformer body later (the codec remains D=256*32=8192):

    python fourier_kronecker_v2_experiment.py --preset gpt2_124m

Important scope statement
-------------------------
V2 removes V1's explicit 32-byte truncation.  It does *not* prove that the
fixed-dimensional Fourier sum is injective for arbitrary strings; collisions
remain possible.  The proof suite therefore claims "no hard cutoff", not
"perfectly reversible unbounded encoding".
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
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


# -----------------------------------------------------------------------------
# Clearly marked scale presets
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPreset:
    """Transformer size, independent of the fixed 8192-D codec dimension."""

    n_layers: int
    n_heads: int
    d_model: int
    mlp_ratio: int


MODEL_PRESETS: Dict[str, ModelPreset] = {
    # The 8x MLP expansion puts the Week-6-vocab model near the requested 25M
    # trainable parameters.  Parameter count is printed rather than assumed.
    "mid": ModelPreset(n_layers=6, n_heads=6, d_model=384, mlp_ratio=8),
    # Scaling switch requested in the assignment.  A 6x MLP expansion offsets
    # the supplied 8K vocabulary (GPT-2 used ~50K) and yields ~126M trainable
    # parameters while retaining the requested 12-layer, d_model=768 geometry.
    "gpt2_124m": ModelPreset(n_layers=12, n_heads=12, d_model=768, mlp_ratio=6),
    # Used only by --smoke-test; it leaves both codec definitions unchanged.
    "smoke": ModelPreset(n_layers=1, n_heads=2, d_model=64, mlp_ratio=2),
}


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_DATA_ROOT = PACKAGE_ROOT / "data" / "week6_v2"


@dataclass
class ExperimentConfig:
    # Model/data
    preset: str = "mid"
    block_size: int = 256
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 1_000
    val_batches: int = 20
    dropout: float = 0.0

    # Optimizer
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    warmup_steps: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Codec -- fixed by the proposed formulation
    d_char: int = 256
    d_pos: int = 32
    fourier_tau: float = 10_000.0
    z_norm_eps: float = 1.0e-5

    # Runtime/reproducibility
    seed: int = 7_007
    log_interval: int = 100
    num_workers: int = 2
    device: str = "auto"
    compile_model: bool = False

    @property
    def codec_dim(self) -> int:
        return self.d_char * self.d_pos


# -----------------------------------------------------------------------------
# Week 6 artifact discovery and token vocabulary
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DataPaths:
    tokenizer_json: Path
    train_tokens: Path
    val_tokens: Path
    source_label: str


def discover_week6_data(
    dataset_root: Optional[Path] = None,
    tokenizer_json: Optional[Path] = None,
    train_tokens: Optional[Path] = None,
    val_tokens: Optional[Path] = None,
) -> DataPaths:
    """Resolve explicit paths first, then use the bundled Phase-1 corpus.

    A valid corpus consists of a tokenizer JSON containing token_id/bytes_hex
    records and little-endian uint16 train/validation streams.  ``dataset_root``
    remains available for a reviewer who wants to point at a separate Week 6
    export, but no author-specific path is embedded in this submission.
    """

    if tokenizer_json or train_tokens or val_tokens:
        if not (tokenizer_json and train_tokens and val_tokens):
            raise ValueError(
                "--tokenizer-json, --train-tokens, and --val-tokens must be "
                "provided together."
            )
        candidate = DataPaths(
            tokenizer_json=tokenizer_json,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            source_label="explicit CLI paths",
        )
        _validate_data_paths(candidate)
        return candidate

    candidates: List[DataPaths] = []
    if dataset_root is None:
        candidates.append(
            DataPaths(
                tokenizer_json=BUNDLED_DATA_ROOT / "tokenizer.json",
                train_tokens=BUNDLED_DATA_ROOT / "train/tokens.uint16.bin",
                val_tokens=BUNDLED_DATA_ROOT / "validation/tokens.uint16.bin",
                source_label="bundled Week 6 V2 standard-BPE corpus",
            )
        )
    else:
        root = dataset_root.resolve()
        # Accept either the compact submission layout or the original Week 6
        # export layout when --dataset-root is supplied explicitly.
        candidates.extend([
            DataPaths(
                tokenizer_json=root / "tokenizer.json",
                train_tokens=root / "train/tokens.uint16.bin",
                val_tokens=root / "validation/tokens.uint16.bin",
                source_label=f"compact corpus ({root})",
            ),
            DataPaths(
                tokenizer_json=root
                / "artifacts/tokenizer_experiments_v2/full/standard_bpe_byte_fallback/tokenizer.json",
                train_tokens=root / "data/tokenized_v2/general/train/tokens.uint16.bin",
                val_tokens=root / "data/tokenized_v2/general/validation/tokens.uint16.bin",
                source_label=f"Week 6 V2 ({root})",
            ),
            DataPaths(
                tokenizer_json=root / "submission_artifacts/manifests/tokenizer.json",
                train_tokens=root / "data/tokenized_v1/general/train/tokens.uint16.bin",
                val_tokens=root / "data/tokenized_v1/general/validation/tokens.uint16.bin",
                source_label=f"Week 6 V1 ({root})",
            ),
        ])

    for candidate in candidates:
        if all(
            path.is_file()
            for path in (
                candidate.tokenizer_json,
                candidate.train_tokens,
                candidate.val_tokens,
            )
        ):
            return candidate

    checked = "\n".join(
        f"  - {c.tokenizer_json}\n    {c.train_tokens}\n    {c.val_tokens}"
        for c in candidates
    )
    raise FileNotFoundError(
        "Could not discover a complete Week 6 dataset. Checked:\n" + checked
    )


def _validate_data_paths(paths: DataPaths) -> None:
    for path in (paths.tokenizer_json, paths.train_tokens, paths.val_tokens):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (paths.train_tokens, paths.val_tokens):
        if path.stat().st_size % np.dtype("<u2").itemsize:
            raise ValueError(f"Token file size is not uint16-aligned: {path}")


class TokenByteVocabulary:
    """Immutable token-id -> exact byte-string mapping from Week 6 JSON."""

    def __init__(self, tokenizer_json: Path) -> None:
        with tokenizer_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        records = payload.get("tokens")
        if not isinstance(records, list) or not records:
            raise ValueError(f"No non-empty 'tokens' list in {tokenizer_json}")

        by_id: Dict[int, bytes] = {}
        displays: Dict[int, str] = {}
        for record in records:
            token_id = int(record["token_id"])
            if token_id in by_id:
                raise ValueError(f"Duplicate token_id {token_id}")
            by_id[token_id] = bytes.fromhex(record["bytes_hex"])
            displays[token_id] = str(record.get("display", ""))

        expected = list(range(len(by_id)))
        if sorted(by_id) != expected:
            raise ValueError("Tokenizer IDs must be contiguous from 0 to V-1")
        if len(by_id) > np.iinfo(np.uint16).max + 1:
            raise ValueError("uint16 token stream cannot represent this vocabulary")

        self.path = tokenizer_json
        self.token_bytes: Tuple[bytes, ...] = tuple(by_id[i] for i in expected)
        self.displays: Tuple[str, ...] = tuple(displays[i] for i in expected)
        self.algorithm = str(payload.get("algorithm", "unknown"))

    def __len__(self) -> int:
        return len(self.token_bytes)

    @property
    def max_token_bytes(self) -> int:
        return max(len(value) for value in self.token_bytes)

    def count_longer_than(self, byte_limit: int) -> int:
        return sum(len(value) > byte_limit for value in self.token_bytes)


# -----------------------------------------------------------------------------
# Flexible memory-mapped Dataset and DataLoader
# -----------------------------------------------------------------------------


class MemmapCausalTokenDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-length causal blocks over a little-endian uint16 token stream.

    The NumPy memmap is opened lazily per process.  That detail matters on
    Windows, where DataLoader workers use process spawning and open file handles
    should not be pickled from the parent process.
    """

    def __init__(self, path: Path, block_size: int, stride: Optional[int] = None) -> None:
        if block_size < 2:
            raise ValueError("block_size must be >= 2")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size % 2:
            raise ValueError(f"Expected uint16-aligned file: {path}")

        self.path = path
        self.block_size = block_size
        self.stride = stride or block_size
        self.n_tokens = path.stat().st_size // 2
        if self.n_tokens <= block_size:
            raise ValueError(
                f"{path} has {self.n_tokens} tokens; need > block_size={block_size}"
            )
        self.n_blocks = 1 + (self.n_tokens - block_size - 1) // self.stride
        self._tokens: Optional[np.memmap] = None

    def _array(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(self.path, mode="r", dtype="<u2")
        return self._tokens

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.n_blocks
        if not 0 <= index < self.n_blocks:
            raise IndexError(index)

        start = index * self.stride
        # Copy detaches the sample from the read-only memmap.  torch.from_numpy
        # then shares memory with this small writable sample, avoiding warnings.
        sample = np.array(
            self._array()[start : start + self.block_size + 1],
            dtype=np.int64,
            copy=True,
        )
        tokens = torch.from_numpy(sample)
        return tokens[:-1], tokens[1:]

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state


def make_dataloader(
    dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def infinite_batches(loader: DataLoader[Any]) -> Iterator[Any]:
    while True:
        yield from loader


# -----------------------------------------------------------------------------
# Exact mathematical codecs
# -----------------------------------------------------------------------------


ArmName = Literal["v1", "v2"]


class KroneckerCodec:
    """Reference codec used for tables, diagnostics, and the README figure."""

    def __init__(
        self,
        arm: ArmName,
        d_char: int = 256,
        d_pos: int = 32,
        tau: float = 10_000.0,
        eps: float = 1.0e-5,
    ) -> None:
        if d_char != 256:
            raise ValueError("UTF-8 byte one-hot requires d_char=256")
        if d_pos <= 0 or d_pos % 2:
            raise ValueError("d_pos must be a positive even number")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.arm = arm
        self.d_char = d_char
        self.d_pos = d_pos
        self.tau = tau
        self.eps = eps
        self.dim = d_char * d_pos

    def fourier_position(self, position: int) -> torch.Tensor:
        """Return F(p) in R^d_pos using the exact requested phase equation.

        Positions are zero-based: the first byte uses p=0, exactly like the
        conventional Transformer sinusoid.  For k=0..15 when d_pos=32:

            denominator[k] = tau ** (2*k / (d_pos/2))
            phase[k]       = p / denominator[k]
            F(p)[2*k]      = cos(phase[k])
            F(p)[2*k+1]    = sin(phase[k])

        The interleaving assignment is explicit so no hidden reshape changes
        the requested even/odd channel order.
        """

        half = self.d_pos // 2
        k = torch.arange(half, dtype=torch.float64)
        denominator = self.tau ** (2.0 * k / float(half))
        phase = float(position) / denominator
        wave = torch.empty(self.d_pos, dtype=torch.float64)
        wave[0::2] = torch.cos(phase)
        wave[1::2] = torch.sin(phase)
        return wave.float()

    def raw_terms(self, token_bytes: bytes) -> torch.Tensor:
        """Return each byte-position Kronecker term before summation.

        Shape is [min(L,32),8192] for V1 and [L,8192] for V2.  This method is
        intentionally diagnostic; training uses precomputed summed vectors.
        """

        if not token_bytes:
            raise ValueError("Cannot encode an empty byte string")
        used = token_bytes[: self.d_pos] if self.arm == "v1" else token_bytes
        terms = torch.zeros((len(used), self.dim), dtype=torch.float32)

        if self.arm == "v1":
            # c_b (256 one-hot) tensor-product p_p (32 one-hot) is itself a
            # single one-hot at flattened coordinate b*d_pos + p.
            byte_values = torch.tensor(list(used), dtype=torch.long)
            positions = torch.arange(len(used), dtype=torch.long)
            flat_indices = byte_values * self.d_pos + positions
            terms[torch.arange(len(used)), flat_indices] = 1.0
        else:
            for p, byte_value in enumerate(used):
                # outer(c_b, F(p)) has 256 rows; only row b is non-zero.
                # Flattening makes that row the contiguous slice
                # [b*d_pos : (b+1)*d_pos].
                start = byte_value * self.d_pos
                terms[p, start : start + self.d_pos] = self.fourier_position(p)
        return terms

    def encode_bytes(
        self, token_bytes: bytes, return_terms: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not token_bytes:
            raise ValueError("Cannot encode an empty byte string")
        used = token_bytes[: self.d_pos] if self.arm == "v1" else token_bytes
        used_length = len(used)

        # Build the summed vector directly for vocabulary-table generation.
        # Allocating [L,D] terms here would create needless transient tensors;
        # raw_terms() remains available when the proof suite explicitly needs
        # those individual contributions.
        raw = torch.zeros(self.dim, dtype=torch.float32)
        if self.arm == "v1":
            byte_values = torch.tensor(list(used), dtype=torch.long)
            positions = torch.arange(used_length, dtype=torch.long)
            raw[byte_values * self.d_pos + positions] = 1.0
        else:
            for p, byte_value in enumerate(used):
                start = byte_value * self.d_pos
                raw[start : start + self.d_pos] += self.fourier_position(p)

        # Variance-preserving aggregation.  V1 scales by the retained length
        # after truncation; V2 scales by the full byte length.
        raw.mul_(1.0 / math.sqrt(float(used_length)))

        # Population z-normalization across D=8192 coordinates, identical to
        # LayerNorm without learned affine parameters.  unbiased=False is the
        # mathematically appropriate population standard deviation here.
        mean = raw.mean()
        variance = (raw - mean).square().mean()
        normalized = (raw - mean) * torch.rsqrt(variance + self.eps)

        metadata: Dict[str, Any] = {
            "input_bytes": len(token_bytes),
            "encoded_bytes": used_length,
            "truncated_bytes": len(token_bytes) - used_length,
            "term_shape": [used_length, self.dim],
            "vector_shape": list(normalized.shape),
        }
        if return_terms:
            metadata["terms"] = self.raw_terms(token_bytes)
        return normalized, metadata

    def encode_text(self, text: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.encode_bytes(text.encode("utf-8"))


def build_codec_table(
    vocabulary: TokenByteVocabulary,
    codec: KroneckerCodec,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    """Precompute fixed, z-normalized codec rows for the entire vocabulary."""

    table = torch.empty((len(vocabulary), codec.dim), dtype=torch.float32)
    for token_id, token_bytes in enumerate(vocabulary.token_bytes):
        vector, _ = codec.encode_bytes(token_bytes)
        table[token_id] = vector
    if not torch.isfinite(table).all():
        raise FloatingPointError("Codec table contains NaN or infinity")
    return table.to(dtype=storage_dtype)


# -----------------------------------------------------------------------------
# Structured input module and SDPA Causal Transformer
# -----------------------------------------------------------------------------


class StructuredTokenEmbedding(nn.Module):
    """Fixed codec lookup followed by the sole learned input projection."""

    def __init__(
        self,
        codec_table: torch.Tensor,
        d_model: int,
        codec_dim: int,
    ) -> None:
        super().__init__()
        if codec_table.ndim != 2 or codec_table.shape[1] != codec_dim:
            raise ValueError(
                f"codec_table must be [V,{codec_dim}], got {tuple(codec_table.shape)}"
            )
        # persistent=False keeps this deterministic/reconstructable table out
        # of checkpoints.  It moves with model.to(device) but is not trainable.
        self.register_buffer("codec_table", codec_table, persistent=False)
        self.projection = nn.Linear(codec_dim, d_model, bias=False)
        nn.init.normal_(self.projection.weight, mean=0.0, std=1.0 / math.sqrt(codec_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        shape = token_ids.shape
        flat_ids = token_ids.reshape(-1)

        # Project each distinct token type only once per microbatch.  This is
        # exactly equivalent to gathering [B,T,D] and applying W_proj to every
        # occurrence, but greatly reduces the expensive 8192->d_model GEMM for
        # repeated tokens.  inverse restores the original [B,T] layout and
        # autograd correctly sums gradients from repeated occurrences.
        unique_ids, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
        codec_rows = self.codec_table.index_select(0, unique_ids)
        unique_embeddings = self.projection(codec_rows)
        embeddings = unique_embeddings.index_select(0, inverse)
        return embeddings.view(*shape, -1)


class CausalSelfAttention(nn.Module):
    """Multi-head attention backed by PyTorch 2.x fused SDPA kernels."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels = x.shape
        qkv = self.qkv(x)
        # [B,T,3C] -> [B,T,3,H,C/H].  Unbinding dimension 2 produces three
        # [B,T,H,head_dim] tensors, then transpose gives SDPA's [B,H,T,D].
        qkv = qkv.view(batch, time_steps, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, time_steps, channels)
        return self.out(attended)


class TransformerBlock(nn.Module):
    def __init__(self, preset: ModelPreset, dropout: float) -> None:
        super().__init__()
        hidden = preset.mlp_ratio * preset.d_model
        self.ln1 = nn.LayerNorm(preset.d_model)
        self.attention = CausalSelfAttention(preset.d_model, preset.n_heads, dropout)
        self.ln2 = nn.LayerNorm(preset.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(preset.d_model, hidden, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, preset.d_model, bias=False),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attention(self.ln1(x)))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(
        self,
        preset: ModelPreset,
        vocab_size: int,
        block_size: int,
        codec_dim: int,
        codec_table: torch.Tensor,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding = StructuredTokenEmbedding(codec_table, preset.d_model, codec_dim)
        # This is sequence-level position.  It is distinct from the within-token
        # byte position basis being ablated in V1 vs V2.
        self.sequence_position = nn.Embedding(block_size, preset.d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(preset, dropout) for _ in range(preset.n_layers)]
        )
        self.final_norm = nn.LayerNorm(preset.d_model)
        # Both arms use the same independent output head.  No weight tying is
        # possible with an 8192-D codec feeding a d_model-dimensional body.
        self.lm_head = nn.Linear(preset.d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        # self.apply touched projection; restore the formulation's exact scale.
        nn.init.normal_(
            self.token_embedding.projection.weight,
            mean=0.0,
            std=1.0 / math.sqrt(codec_dim),
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, time_steps = token_ids.shape
        if time_steps > self.block_size:
            raise ValueError(f"Sequence {time_steps} exceeds block_size {self.block_size}")
        positions = torch.arange(time_steps, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.sequence_position(positions)[None, :, :]
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return logits, loss


# -----------------------------------------------------------------------------
# Controlled training utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(device_request: str) -> torch.device:
    if device_request == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    if device.type == "cuda":
        # RTX 3070 (Ampere) supports fused/memory-efficient SDPA and TF32.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return device


def autocast_context(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    # CUDA AMP with float16 is the exact requested numerical mode.  PyTorch 2.4+
    # moved torch.cuda.amp.autocast to torch.amp.autocast; keep a fallback so the
    # script remains compatible with the requested PyTorch 2.0+ range.
    if device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(device: torch.device) -> Any:
    """Construct the non-finite-safe FP16 gradient scaler across torch versions."""

    enabled = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def trainable_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        contiguous = parameter.detach().cpu().contiguous()
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def make_optimizer(model: nn.Module, config: ExperimentConfig, device: torch.device) -> torch.optim.Optimizer:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=1.0e-8,
    )
    try:
        return torch.optim.AdamW(groups, fused=device.type == "cuda", **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, **kwargs)


def learning_rate_at_step(step: int, config: ExperimentConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * float(step + 1) / float(max(1, config.warmup_steps))
    progress = (step - config.warmup_steps) / float(
        max(1, config.max_steps - config.warmup_steps)
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (
        config.learning_rate - config.min_learning_rate
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    losses: List[float] = []
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device):
            _, loss = model(inputs, targets)
        assert loss is not None
        losses.append(float(loss.detach().cpu()))
    model.train()
    if not losses:
        raise RuntimeError("Validation loader yielded no batches")
    return float(np.mean(losses))


def train_one_arm(
    arm: ArmName,
    config: ExperimentConfig,
    preset: ModelPreset,
    vocabulary: TokenByteVocabulary,
    train_dataset: MemmapCausalTokenDataset,
    val_dataset: MemmapCausalTokenDataset,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    """Train one arm from the identical seed and identical batch ordering."""

    print(f"\n{'=' * 80}\nTraining Arm {arm.upper()}\n{'=' * 80}", flush=True)
    seed_everything(config.seed)

    codec = KroneckerCodec(
        arm=arm,
        d_char=config.d_char,
        d_pos=config.d_pos,
        tau=config.fourier_tau,
        eps=config.z_norm_eps,
    )
    # Fixed codec rows do not require fp32 on GPU; fp16 halves table/gather memory.
    table_dtype = torch.float16 if device.type == "cuda" else torch.float32
    table_start = time.perf_counter()
    codec_table = build_codec_table(vocabulary, codec, table_dtype)
    print(
        f"Codec table: {tuple(codec_table.shape)}, {codec_table.nbytes / 2**20:.1f} MiB, "
        f"built in {time.perf_counter() - table_start:.1f}s",
        flush=True,
    )

    model = CausalTransformer(
        preset=preset,
        vocab_size=len(vocabulary),
        block_size=config.block_size,
        codec_dim=config.codec_dim,
        codec_table=codec_table,
        dropout=config.dropout,
    ).to(device)
    del codec_table

    initial_hash = trainable_state_hash(model)
    parameter_count = count_parameters(model)
    print(
        f"Trainable parameters: {parameter_count:,} ({parameter_count / 1e6:.2f}M)\n"
        f"Initial trainable-state SHA256: {initial_hash}",
        flush=True,
    )

    optimizer = make_optimizer(model, config, device)
    scaler = make_grad_scaler(device)

    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_iterator = infinite_batches(train_loader)

    if config.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires PyTorch 2.0+")
        model = torch.compile(model)  # type: ignore[assignment]

    history: List[Dict[str, float]] = []
    initial_val = evaluate(model, val_loader, device, config.val_batches)
    print(f"step=0000 train_loss=nan val_loss={initial_val:.4f}", flush=True)

    model.train()
    run_start = time.perf_counter()
    interval_start = run_start
    accumulated_for_log = 0.0

    for step in range(1, config.max_steps + 1):
        lr = learning_rate_at_step(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(config.grad_accum_steps):
            inputs, targets = next(train_iterator)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(device):
                _, loss = model(inputs, targets)
                assert loss is not None
                scaled_loss = loss / config.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            step_loss += float(loss.detach().cpu()) / config.grad_accum_steps

        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        accumulated_for_log += step_loss

        if step % config.log_interval == 0 or step == config.max_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            val_loss = evaluate(model, val_loader, device, config.val_batches)
            elapsed = time.perf_counter() - interval_start
            steps_in_interval = config.log_interval if step % config.log_interval == 0 else step % config.log_interval
            mean_train = accumulated_for_log / max(1, steps_in_interval)
            tokens_per_second = (
                steps_in_interval
                * config.micro_batch_size
                * config.grad_accum_steps
                * config.block_size
                / max(elapsed, 1.0e-9)
            )
            record = {
                "step": float(step),
                "train_loss": mean_train,
                "val_loss": val_loss,
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "tokens_per_second": tokens_per_second,
            }
            history.append(record)
            print(
                f"step={step:04d} train_loss={mean_train:.4f} val_loss={val_loss:.4f} "
                f"lr={lr:.3e} grad_norm={grad_norm:.3f} tok/s={tokens_per_second:,.0f}",
                flush=True,
            )
            accumulated_for_log = 0.0
            interval_start = time.perf_counter()

    wall_seconds = time.perf_counter() - run_start
    # torch.compile wraps the original module; state_dict remains usable, but
    # saving the uncompiled module avoids compiler-specific key prefixes.
    save_model = getattr(model, "_orig_mod", model)
    checkpoint_path = output_dir / f"arm_{arm}_final.pt"
    torch.save(
        {
            "arm": arm,
            "model_state_dict": save_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "preset": asdict(preset),
            "vocab_size": len(vocabulary),
            "tokenizer_json": str(vocabulary.path),
            "initial_trainable_sha256": initial_hash,
            "history": history,
        },
        checkpoint_path,
    )

    result = {
        "arm": arm,
        "parameter_count": parameter_count,
        "initial_trainable_sha256": initial_hash,
        "initial_val_loss": initial_val,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "wall_seconds": wall_seconds,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    del model, optimizer, scaler, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# -----------------------------------------------------------------------------
# Quantitative proof suite and interference visualization
# -----------------------------------------------------------------------------


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a[None, :], b[None, :]).item())


def save_interference_plot(
    v1: KroneckerCodec,
    v2: KroneckerCodec,
    output_path: Path,
    word: str = "listen",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to produce fourier_interference.png"
        ) from exc

    v1_vector, _ = v1.encode_text(word)
    v2_vector, _ = v2.encode_text(word)
    x_axis = np.arange(v1.dim)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].plot(x_axis, v1_vector.numpy(), color="#1565C0", linewidth=0.7)
    axes[0].set_title(f'Kronecker V1 pre-projection signal: "{word}"')
    axes[0].set_ylabel("z-normalized amplitude")
    axes[0].grid(alpha=0.2)

    axes[1].plot(x_axis, v2_vector.numpy(), color="#D84315", linewidth=0.7)
    axes[1].set_title(f'Fourier-Kronecker V2 pre-projection signal: "{word}"')
    axes[1].set_xlabel("Flattened byte x position/frequency coordinate (0..8191)")
    axes[1].set_ylabel("z-normalized amplitude")
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        "Discrete position spikes vs. continuous Fourier phase interference",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_proof_suite(config: ExperimentConfig, output_dir: Path) -> Dict[str, Any]:
    v1 = KroneckerCodec(
        "v1", config.d_char, config.d_pos, config.fourier_tau, config.z_norm_eps
    )
    v2 = KroneckerCodec(
        "v2", config.d_char, config.d_pos, config.fourier_tau, config.z_norm_eps
    )

    long_text = "Fourier_Kronecker_preserves_every_byte_beyond_position_thirty_two_0123456789"
    long_bytes = long_text.encode("utf-8")
    v1_long, v1_meta = v1.encode_bytes(long_bytes)
    v2_long, v2_meta = v2.encode_bytes(long_bytes)

    # Both *summed* codec vectors must be 8192-D because W_proj is controlled.
    # The diagnostic term tensors reveal how many byte contributions were used.
    print("\n" + "=" * 80)
    print("UNBOUNDED-LENGTH / NO-HARD-CUTOFF TEST")
    print("=" * 80)
    print(f"Input UTF-8 bytes: {len(long_bytes)}")
    print(
        f"Arm A V1 term tensor={tuple(v1_meta['term_shape'])}, "
        f"summed vector={tuple(v1_long.shape)}, encoded={v1_meta['encoded_bytes']}, "
        f"truncated={v1_meta['truncated_bytes']}"
    )
    print(
        f"Arm B V2 term tensor={tuple(v2_meta['term_shape'])}, "
        f"summed vector={tuple(v2_long.shape)}, encoded={v2_meta['encoded_bytes']}, "
        f"truncated={v2_meta['truncated_bytes']}"
    )

    assert v1_meta["encoded_bytes"] == config.d_pos
    assert v1_meta["truncated_bytes"] == len(long_bytes) - config.d_pos
    assert v2_meta["encoded_bytes"] == len(long_bytes)
    assert v2_meta["truncated_bytes"] == 0
    assert tuple(v1_long.shape) == tuple(v2_long.shape) == (config.codec_dim,)

    # Stronger collision witness: V1 maps these different strings to exactly the
    # same vector because the first 32 bytes match.  V2 observes their suffixes.
    common_prefix = "P" * config.d_pos
    collision_left = (common_prefix + "_LEFT_SUFFIX_is_visible_to_V2").encode("utf-8")
    collision_right = (common_prefix + "_RIGHT_SUFFIX_is_visible_to_V2").encode("utf-8")
    v1_left, _ = v1.encode_bytes(collision_left)
    v1_right, _ = v1.encode_bytes(collision_right)
    v2_left, _ = v2.encode_bytes(collision_left)
    v2_right, _ = v2.encode_bytes(collision_right)
    v1_suffix_distance = float(torch.linalg.vector_norm(v1_left - v1_right))
    v2_suffix_distance = float(torch.linalg.vector_norm(v2_left - v2_right))
    print(
        f"Shared-32-byte-prefix collision: V1 L2 distance={v1_suffix_distance:.8f}, "
        f"V2 L2 distance={v2_suffix_distance:.8f}"
    )
    assert v1_suffix_distance == 0.0
    assert v2_suffix_distance > 0.0

    print("\n" + "=" * 80)
    print("TYPO ROBUSTNESS TEST: separate vs. seperate")
    print("=" * 80)
    typo_vectors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, codec in (("v1", v1), ("v2", v2)):
        correct, _ = codec.encode_text("separate")
        misspelled, _ = codec.encode_text("seperate")
        typo_vectors[name] = (correct, misspelled)
    v1_typo_cosine = cosine(*typo_vectors["v1"])
    v2_typo_cosine = cosine(*typo_vectors["v2"])
    print(f"Arm A V1 cosine similarity: {v1_typo_cosine:.6f}")
    print(f"Arm B V2 cosine similarity: {v2_typo_cosine:.6f}")

    figure_path = output_dir / "fourier_interference.png"
    save_interference_plot(v1, v2, figure_path, word="listen")
    print(f"Saved interference visualization: {figure_path}")

    proof = {
        "claim_scope": (
            "V2 removes the explicit 32-byte cutoff; fixed-dimensional Fourier "
            "summation is not claimed to be injective for arbitrary strings."
        ),
        "long_input": {
            "text": long_text,
            "utf8_bytes": len(long_bytes),
            "v1": v1_meta,
            "v2": v2_meta,
        },
        "shared_prefix_collision": {
            "shared_prefix_bytes": config.d_pos,
            "v1_l2_distance": v1_suffix_distance,
            "v2_l2_distance": v2_suffix_distance,
        },
        "typo_robustness": {
            "pair": ["separate", "seperate"],
            "v1_cosine": v1_typo_cosine,
            "v2_cosine": v2_typo_cosine,
        },
        "interference_figure": str(figure_path),
    }
    with (output_dir / "proof_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(proof, handle, indent=2)
    return proof


def write_experiment_report(
    output_dir: Path,
    config: ExperimentConfig,
    preset: ModelPreset,
    data_paths: DataPaths,
    vocabulary: TokenByteVocabulary,
    proof: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> None:
    by_arm = {str(result["arm"]): result for result in results}
    if by_arm and by_arm["v1"]["initial_trainable_sha256"] != by_arm["v2"]["initial_trainable_sha256"]:
        raise AssertionError("Controlled ablation failed: initial trainable weights differ")

    report: Dict[str, Any] = {
        "experiment": "Fourier-Kronecker V2 controlled two-arm ablation",
        "config": asdict(config),
        "model_preset": asdict(preset),
        "data": {
            "source": data_paths.source_label,
            "tokenizer_json": str(data_paths.tokenizer_json),
            "train_tokens": str(data_paths.train_tokens),
            "val_tokens": str(data_paths.val_tokens),
            "vocab_size": len(vocabulary),
            "tokenizer_algorithm": vocabulary.algorithm,
            "max_vocabulary_token_bytes": vocabulary.max_token_bytes,
            "vocabulary_tokens_longer_than_32_bytes": vocabulary.count_longer_than(32),
            "training_scope_note": (
                "The supplied frozen vocabulary contains no token longer than 32 bytes; "
                "training compares positional geometry but does not empirically exercise "
                "the beyond-byte-32 capability. That capability is tested constructively "
                "in the proof suite."
                if vocabulary.count_longer_than(32) == 0
                else "The supplied vocabulary contains tokens longer than 32 bytes."
            ),
        },
        "proof": proof,
        "arms": list(results),
    }
    if by_arm:
        report["ablation"] = {
            "controlled_initialization_verified": True,
            "v2_minus_v1_final_val_loss": (
                by_arm["v2"]["final_val_loss"] - by_arm["v1"]["final_val_loss"]
            ),
            "v2_relative_val_loss_change_percent": 100.0
            * (
                by_arm["v2"]["final_val_loss"] / by_arm["v1"]["final_val_loss"]
                - 1.0
            ),
        }
    with (output_dir / "experiment_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a controlled Kronecker V1 vs Fourier-Kronecker V2 ablation."
    )
    parser.add_argument("--preset", choices=("mid", "gpt2_124m"), default="mid")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7_007)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--train-tokens", type=Path)
    parser.add_argument("--val-tokens", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fourier_kronecker_run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        preset="smoke" if args.smoke_test else args.preset,
        block_size=32 if args.smoke_test else args.block_size,
        micro_batch_size=1 if args.smoke_test else args.micro_batch_size,
        grad_accum_steps=1 if args.smoke_test else args.grad_accum_steps,
        max_steps=1 if args.smoke_test else args.steps,
        val_batches=1 if args.smoke_test else args.val_batches,
        warmup_steps=1 if args.smoke_test else 50,
        log_interval=1 if args.smoke_test else args.log_interval,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        compile_model=args.compile_model,
    )
    if config.max_steps <= 0:
        raise ValueError("--steps must be positive")
    if config.log_interval <= 0:
        raise ValueError("--log-interval must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preset = MODEL_PRESETS[config.preset]

    print("Fourier-Kronecker V2 controlled experiment")
    print(json.dumps({"config": asdict(config), "preset": asdict(preset)}, indent=2))
    proof = run_proof_suite(config, output_dir)
    if args.diagnostics_only:
        print("Diagnostics completed; training skipped by --diagnostics-only.")
        return

    data_paths = discover_week6_data(
        dataset_root=args.dataset_root,
        tokenizer_json=args.tokenizer_json,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
    )
    _validate_data_paths(data_paths)
    vocabulary = TokenByteVocabulary(data_paths.tokenizer_json)
    train_dataset = MemmapCausalTokenDataset(
        data_paths.train_tokens, block_size=config.block_size
    )
    val_dataset = MemmapCausalTokenDataset(
        data_paths.val_tokens, block_size=config.block_size
    )

    max_seen_id = max(
        int(np.memmap(data_paths.train_tokens, mode="r", dtype="<u2").max()),
        int(np.memmap(data_paths.val_tokens, mode="r", dtype="<u2").max()),
    )
    if max_seen_id >= len(vocabulary):
        raise ValueError(
            f"Token stream contains id {max_seen_id}, but vocab size is {len(vocabulary)}"
        )

    device = configure_runtime(config.device)
    print(
        f"Data source: {data_paths.source_label}\n"
        f"Tokenizer: {vocabulary.algorithm}, vocab={len(vocabulary):,}, "
        f"max token bytes={vocabulary.max_token_bytes}\n"
        f"Train tokens={train_dataset.n_tokens:,}, validation tokens={val_dataset.n_tokens:,}\n"
        f"Device: {device}"
        + (
            f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
        ),
        flush=True,
    )
    if vocabulary.count_longer_than(config.d_pos) == 0:
        print(
            "RESEARCH SCOPE WARNING: this frozen vocabulary has no token longer "
            f"than {config.d_pos} bytes. The training ablation measures V1/V2 "
            "position-basis geometry, while the separate constructive proof suite "
            "establishes removal of the hard cutoff.",
            flush=True,
        )

    results = [
        train_one_arm(
            arm,
            config,
            preset,
            vocabulary,
            train_dataset,
            val_dataset,
            device,
            output_dir,
        )
        for arm in ("v1", "v2")
    ]
    write_experiment_report(
        output_dir, config, preset, data_paths, vocabulary, proof, results
    )

    v1_result, v2_result = results
    delta = v2_result["final_val_loss"] - v1_result["final_val_loss"]
    relative = 100.0 * (
        v2_result["final_val_loss"] / v1_result["final_val_loss"] - 1.0
    )
    print("\n" + "=" * 80)
    print("FINAL CONTROLLED ABLATION")
    print("=" * 80)
    print(f"Arm A V1 final validation loss: {v1_result['final_val_loss']:.6f}")
    print(f"Arm B V2 final validation loss: {v2_result['final_val_loss']:.6f}")
    print(f"V2 - V1: {delta:+.6f} nats ({relative:+.3f}%)")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
