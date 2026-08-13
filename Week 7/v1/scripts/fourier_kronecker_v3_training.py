#!/usr/bin/env python3
"""1,000-step controlled LM ablation: Kronecker V1 vs Fourier-Kronecker V3.

Arm A (baseline)
----------------
    c_byte (256-D one-hot) x p_position (32-D one-hot), hard truncation at
    byte 32, 1/sqrt(L) scaling, per-token z-normalization, learned
    W_proj: 8192 -> d_model.

Arm B (V3 + dynamic amplitude)
--------------------------------
    c_byte x F_V3(p), no hard position limit, with a learned loudness scalar
    a[byte] applied before summation:

        kappa(b; a) = ZNorm(
            1/sqrt(L) * sum_p a[b_p] * (c_{b_p} x F_V3(p))
        )

    F_V3 uses one global Euler phase and fifteen geometrically spaced local
    phases.  Each complex phase is stored as two real channels (cos, sin), so
    d_pos remains 32 and the Kronecker/projection dimension remains 8192.

Strict initialization control
-----------------------------
Dynamic amplitude gives V3 256 learnable values that V1 does not mathematically
need.  To make the complete trainable states byte-for-byte identical at step 0,
both model instances contain the same parameter `byte_amplitude`, initialized to
ones.  V1 deliberately leaves it inert (gradient stays None); V3 uses it.  Thus
both arms have identical parameter names, shapes, values, counts, initialization
SHA-256, optimizer grouping, Transformer body, and output head.

This is a controlled two-arm test of the *combined* V3 basis + loudness system.
It does not isolate whether a result comes from the basis or amplitude.  A later
three-arm study (V1, V3 fixed-amplitude, V3 learned-amplitude) is required for
that mechanistic attribution.

Default execution trains each arm for 1,000 optimizer updates on the frozen Week
6 V2 standard-BPE general train/validation streams.  It targets an RTX 3070 8GB
with FP16 AMP, gradient accumulation, and PyTorch 2.x fused scaled-dot-product
attention.

Run the full experiment:

    python fourier_kronecker_v3_training.py

Fast end-to-end integration check:

    python fourier_kronecker_v3_training.py --smoke-test
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
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


MODEL_PRESETS: Dict[str, ModelPreset] = {
    # 24.09M for V2; the shared 256-value amplitude vector makes V3/V1 here
    # 24.095M.  This is the same requested ~25M scaling regime.
    "mid": ModelPreset(n_layers=6, n_heads=6, d_model=384, mlp_ratio=8),
    "smoke": ModelPreset(n_layers=1, n_heads=2, d_model=64, mlp_ratio=2),
}


@dataclass
class ExperimentConfig:
    preset: str = "mid"
    block_size: int = 256
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 1_000
    val_batches: int = 20
    log_interval: int = 100

    dropout: float = 0.0
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    warmup_steps: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    d_char: int = 256
    d_pos: int = 32
    z_norm_eps: float = 1.0e-5

    # V3 anchored band-limited frequency schedule.
    global_wavelength: float = 256.0
    local_omega_min: float = 0.2
    local_omega_max: float = 0.9 * math.pi

    seed: int = 7_007
    num_workers: int = 2
    device: str = "auto"
    compile_model: bool = False

    @property
    def codec_dim(self) -> int:
        return self.d_char * self.d_pos


@dataclass(frozen=True)
class DataPaths:
    tokenizer_json: Path
    train_tokens: Path
    val_tokens: Path
    source_label: str


def discover_data(
    tokenizer_json: Optional[Path],
    train_tokens: Optional[Path],
    val_tokens: Optional[Path],
) -> DataPaths:
    """Use bundled Week 6 V2 artifacts or a fully explicit alternative."""

    if tokenizer_json or train_tokens or val_tokens:
        if not (tokenizer_json and train_tokens and val_tokens):
            raise ValueError(
                "--tokenizer-json, --train-tokens, and --val-tokens must be supplied together"
            )
        explicit = DataPaths(
            tokenizer_json.resolve(),
            train_tokens.resolve(),
            val_tokens.resolve(),
            "explicit CLI paths",
        )
        validate_data_paths(explicit)
        return explicit

    bundled = DataPaths(
        BUNDLED_DATA_ROOT / "tokenizer.json",
        BUNDLED_DATA_ROOT / "train/tokens.uint16.bin",
        BUNDLED_DATA_ROOT / "validation/tokens.uint16.bin",
        "bundled Week 6 V2 standard-BPE general corpus",
    )
    validate_data_paths(bundled)
    return bundled


def validate_data_paths(paths: DataPaths) -> None:
    for path in (paths.tokenizer_json, paths.train_tokens, paths.val_tokens):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (paths.train_tokens, paths.val_tokens):
        if path.stat().st_size % 2:
            raise ValueError(f"uint16 token file has an odd byte size: {path}")


class TokenByteVocabulary:
    """Exact, immutable token-id to UTF-8-byte mapping from tokenizer JSON."""

    def __init__(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("tokens")
        if not isinstance(records, list) or not records:
            raise ValueError(f"Missing non-empty tokens array in {path}")

        by_id: Dict[int, bytes] = {}
        for record in records:
            token_id = int(record["token_id"])
            if token_id in by_id:
                raise ValueError(f"Duplicate token id: {token_id}")
            by_id[token_id] = bytes.fromhex(record["bytes_hex"])
        expected = list(range(len(by_id)))
        if sorted(by_id) != expected:
            raise ValueError("Tokenizer IDs must be contiguous from 0")

        self.path = path
        self.algorithm = str(payload.get("algorithm", "unknown"))
        self.token_bytes: Tuple[bytes, ...] = tuple(by_id[index] for index in expected)

    def __len__(self) -> int:
        return len(self.token_bytes)

    @property
    def max_token_bytes(self) -> int:
        return max(map(len, self.token_bytes))

    def count_longer_than(self, limit: int) -> int:
        return sum(len(value) > limit for value in self.token_bytes)


class MemmapCausalDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Memory-mapped, fixed-block next-token dataset for uint16 streams."""

    def __init__(self, path: Path, block_size: int, stride: Optional[int] = None) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path
        self.block_size = block_size
        self.stride = stride or block_size
        self.n_tokens = path.stat().st_size // 2
        if self.n_tokens <= block_size:
            raise ValueError("Token stream is shorter than block_size")
        self.n_blocks = 1 + (self.n_tokens - block_size - 1) // self.stride
        self._array: Optional[np.memmap] = None

    def __len__(self) -> int:
        return self.n_blocks

    def _tokens(self) -> np.memmap:
        # Workers open their own read-only mapping, which is safe under Windows spawn.
        if self._array is None:
            self._array = np.memmap(self.path, mode="r", dtype="<u2")
        return self._array

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.n_blocks
        if not 0 <= index < self.n_blocks:
            raise IndexError(index)
        start = index * self.stride
        sample = np.array(
            self._tokens()[start : start + self.block_size + 1],
            dtype=np.int64,
            copy=True,
        )
        tensor = torch.from_numpy(sample)
        return tensor[:-1], tensor[1:]

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_array"] = None
        return state


def make_loader(
    dataset: Dataset[Any],
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        generator=generator,
    )


def cycle_batches(loader: DataLoader[Any]) -> Iterator[Any]:
    while True:
        yield from loader


ArmName = Literal["v1", "v3"]


def geometric_values(start: float, stop: float, count: int) -> torch.Tensor:
    return torch.exp(
        torch.linspace(math.log(start), math.log(stop), count, dtype=torch.float64)
    )


def v3_omegas(config: ExperimentConfig) -> torch.Tensor:
    """One global angular frequency followed by 15 geometric local frequencies."""

    half = config.d_pos // 2
    global_frequency = torch.tensor(
        [2.0 * math.pi / config.global_wavelength], dtype=torch.float64
    )
    local = geometric_values(
        config.local_omega_min, config.local_omega_max, half - 1
    )
    return torch.cat((global_frequency, local))


def phase_waves(length: int, omegas: torch.Tensor, d_pos: int) -> torch.Tensor:
    """Evaluate interleaved real Euler channels for p=0..length-1.

    phase has shape [L,16] through broadcasting.  Interleaving cosine and sine
    creates [L,32] without ever indexing a finite position table.
    """

    positions = torch.arange(length, dtype=torch.float64)[:, None]
    phase = positions * omegas[None, :]
    waves = torch.empty((length, d_pos), dtype=torch.float64)
    waves[:, 0::2] = torch.cos(phase)
    waves[:, 1::2] = torch.sin(phase)
    return waves


def build_raw_codec_table(
    vocabulary: TokenByteVocabulary,
    arm: ArmName,
    config: ExperimentConfig,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    """Build length-scaled but not z-normalized [V,256,32] codec rows.

    Runtime z-normalization is necessary because V3's learned amplitude changes
    the vector mean and variance on every optimizer update.  V1 follows the same
    runtime normalization path to avoid a computational-path confound.
    """

    table = torch.zeros(
        (len(vocabulary), config.d_char, config.d_pos), dtype=torch.float32
    )
    omegas = v3_omegas(config) if arm == "v3" else None

    for token_id, full_value in enumerate(vocabulary.token_bytes):
        value = full_value[: config.d_pos] if arm == "v1" else full_value
        if not value:
            raise ValueError(f"Empty token byte string at id {token_id}")
        length = len(value)
        byte_indices = torch.tensor(list(value), dtype=torch.long)

        if arm == "v1":
            # c_b x p_p is one at [byte, position].  Positions >=32 never enter.
            positions = torch.arange(length, dtype=torch.long)
            table[token_id, byte_indices, positions] = 1.0
        else:
            assert omegas is not None
            waves = phase_waves(length, omegas, config.d_pos).float()
            # index_add_ sums wave interference for repeated bytes.  Loudness is
            # not applied here because it must remain differentiable at runtime.
            table[token_id].index_add_(0, byte_indices, waves)

        table[token_id] *= 1.0 / math.sqrt(float(length))

    if not torch.isfinite(table).all():
        raise FloatingPointError("Codec table contains non-finite values")
    return table.to(storage_dtype)


class ControlledStructuredEmbedding(nn.Module):
    """Same parameters in both arms; V3 alone activates byte loudness."""

    def __init__(
        self,
        raw_codec_table: torch.Tensor,
        d_model: int,
        config: ExperimentConfig,
        use_dynamic_amplitude: bool,
    ) -> None:
        super().__init__()
        expected = (len(raw_codec_table), config.d_char, config.d_pos)
        if tuple(raw_codec_table.shape) != expected:
            raise ValueError(f"Expected codec table {expected}, got {raw_codec_table.shape}")

        self.register_buffer("raw_codec_table", raw_codec_table, persistent=False)
        self.use_dynamic_amplitude = use_dynamic_amplitude
        self.z_norm_eps = config.z_norm_eps

        # Identical trainable parameter in both arms, initialized exactly to 1.0.
        # It is inert in V1 and active in V3.  Direct scalar parameterization is
        # used exactly as requested; values are not forced positive.
        self.byte_amplitude = nn.Parameter(torch.ones(config.d_char))
        self.projection = nn.Linear(config.codec_dim, d_model, bias=False)
        nn.init.normal_(
            self.projection.weight,
            mean=0.0,
            std=1.0 / math.sqrt(config.codec_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        original_shape = token_ids.shape
        flat_ids = token_ids.reshape(-1)

        # Projection is expensive at D=8192.  Project each distinct token once,
        # then use inverse indices to restore all [B,T] occurrences exactly.
        unique_ids, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
        rows = self.raw_codec_table.index_select(0, unique_ids).float()

        if self.use_dynamic_amplitude:
            # [U,256,32] * [1,256,1].  Every occurrence of byte b receives the
            # same learned loudness a[b] before rows are flattened and summed.
            rows = rows * self.byte_amplitude[None, :, None]
        # V1 intentionally does not reference byte_amplitude, leaving grad=None.

        vectors = rows.flatten(start_dim=1)  # [unique tokens, 8192]
        means = vectors.mean(dim=1, keepdim=True)
        centered = vectors - means
        variances = centered.square().mean(dim=1, keepdim=True)
        normalized = centered * torch.rsqrt(variances + self.z_norm_eps)

        unique_embeddings = self.projection(normalized)
        embeddings = unique_embeddings.index_select(0, inverse)
        return embeddings.view(*original_shape, -1)


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
        qkv = self.qkv(x).view(
            batch, steps, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # PyTorch 2.x chooses Flash/memory-efficient SDPA on supported CUDA GPUs.
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.output(attended)


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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.ln1(x)))
        return x + self.mlp(self.ln2(x))


class CausalTransformer(nn.Module):
    def __init__(
        self,
        preset: ModelPreset,
        vocab_size: int,
        raw_codec_table: torch.Tensor,
        config: ExperimentConfig,
        use_dynamic_amplitude: bool,
    ) -> None:
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding = ControlledStructuredEmbedding(
            raw_codec_table, preset.d_model, config, use_dynamic_amplitude
        )
        # Sequence position is separate from within-token byte position.
        self.sequence_position = nn.Embedding(config.block_size, preset.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(preset, config.dropout) for _ in range(preset.n_layers)]
        )
        self.final_norm = nn.LayerNorm(preset.d_model)
        self.lm_head = nn.Linear(preset.d_model, vocab_size, bias=False)

        self.apply(self._initialize)
        # Restore the specified D-dependent projection initialization after apply.
        nn.init.normal_(
            self.token_embedding.projection.weight,
            mean=0.0,
            std=1.0 / math.sqrt(config.codec_dim),
        )
        nn.init.ones_(self.token_embedding.byte_amplitude)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, steps = tokens.shape
        if steps > self.block_size:
            raise ValueError("Input sequence exceeds block_size")
        positions = torch.arange(steps, device=tokens.device)
        x = self.token_embedding(tokens) + self.sequence_position(positions)[None, :, :]
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_device(request: str) -> torch.device:
    device = torch.device(
        "cuda" if request == "auto" and torch.cuda.is_available() else
        "cpu" if request == "auto" else request
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return device


def amp_context(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    if device.type != "cuda":
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def make_scaler(device: torch.device) -> Any:
    enabled = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def trainable_sha256(model: nn.Module) -> str:
    """Hash parameter names, shapes/dtypes through raw contiguous values."""

    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_for(
    model: nn.Module, config: ExperimentConfig, device: torch.device
) -> torch.optim.Optimizer:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for value in model.parameters():
        (decay if value.ndim >= 2 else no_decay).append(value)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {
        "lr": config.learning_rate,
        "betas": (config.beta1, config.beta2),
        "eps": 1.0e-8,
    }
    try:
        return torch.optim.AdamW(groups, fused=device.type == "cuda", **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, **kwargs)


def lr_at_step(step: int, config: ExperimentConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(
        1, config.max_steps - config.warmup_steps
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (
        config.learning_rate - config.min_learning_rate
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    batches: int,
) -> float:
    model.eval()
    losses: List[float] = []
    for index, (tokens, targets) in enumerate(loader):
        if index >= batches:
            break
        tokens = tokens.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with amp_context(device):
            _, loss = model(tokens, targets)
        assert loss is not None
        losses.append(float(loss.detach().cpu()))
    model.train()
    if not losses:
        raise RuntimeError("Validation loader produced no batches")
    return float(np.mean(losses))


def amplitude_statistics(model: nn.Module) -> Dict[str, float]:
    original = getattr(model, "_orig_mod", model)
    values = original.token_embedding.byte_amplitude.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def train_arm(
    arm: ArmName,
    config: ExperimentConfig,
    preset: ModelPreset,
    vocabulary: TokenByteVocabulary,
    train_data: MemmapCausalDataset,
    val_data: MemmapCausalDataset,
    device: torch.device,
    output_dir: Path,
    expected_initial_hash: Optional[str],
) -> Dict[str, Any]:
    print(f"\n{'=' * 80}\nTRAINING ARM {arm.upper()}\n{'=' * 80}", flush=True)

    table_start = time.perf_counter()
    table_dtype = torch.float16 if device.type == "cuda" else torch.float32
    raw_table = build_raw_codec_table(vocabulary, arm, config, table_dtype)
    print(
        f"Raw codec table {tuple(raw_table.shape)}; "
        f"{raw_table.nbytes / 2**20:.1f} MiB; "
        f"built in {time.perf_counter() - table_start:.1f}s",
        flush=True,
    )

    # Reset immediately before every model construction.  Codec construction
    # cannot perturb the controlled trainable initialization.
    seed_everything(config.seed)
    model = CausalTransformer(
        preset,
        len(vocabulary),
        raw_table,
        config,
        use_dynamic_amplitude=arm == "v3",
    ).to(device)
    del raw_table

    initial_hash = trainable_sha256(model)
    if expected_initial_hash is not None and initial_hash != expected_initial_hash:
        raise AssertionError(
            "Controlled initialization failed:\n"
            f"expected {expected_initial_hash}\nobserved {initial_hash}"
        )
    print(
        f"Trainable parameters: {parameter_count(model):,}\n"
        f"Initial trainable-state SHA256: {initial_hash}",
        flush=True,
    )

    optimizer = optimizer_for(model, config, device)
    scaler = make_scaler(device)
    train_loader = make_loader(
        train_data,
        config.micro_batch_size,
        True,
        config.seed,
        config.num_workers,
        device.type == "cuda",
    )
    val_loader = make_loader(
        val_data,
        config.micro_batch_size,
        False,
        config.seed,
        config.num_workers,
        device.type == "cuda",
    )
    train_iterator = cycle_batches(train_loader)

    if config.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires PyTorch 2.0+")
        model = torch.compile(model)  # type: ignore[assignment]

    history: List[Dict[str, Any]] = []
    initial_val = validate(model, val_loader, device, config.val_batches)
    history.append(
        {
            "step": 0,
            "train_loss": None,
            "val_loss": initial_val,
            "learning_rate": 0.0,
            "amplitude": amplitude_statistics(model),
        }
    )
    print(f"step=0000 train_loss=nan val_loss={initial_val:.4f}", flush=True)

    start_time = time.perf_counter()
    interval_start = start_time
    accumulated_loss = 0.0
    steps_since_log = 0

    for step in range(1, config.max_steps + 1):
        learning_rate = lr_at_step(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)

        update_loss = 0.0
        for _ in range(config.grad_accum_steps):
            tokens, targets = next(train_iterator)
            tokens = tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with amp_context(device):
                _, loss = model(tokens, targets)
                assert loss is not None
                micro_loss = loss / config.grad_accum_steps
            scaler.scale(micro_loss).backward()
            update_loss += float(loss.detach().cpu()) / config.grad_accum_steps

        scaler.unscale_(optimizer)
        # First-update structural assertion: V1's shared control parameter must
        # remain outside its graph, while V3 loudness must receive finite signal.
        if step == 1:
            gradient_model = getattr(model, "_orig_mod", model)
            amplitude_grad = gradient_model.token_embedding.byte_amplitude.grad
            if arm == "v1" and amplitude_grad is not None:
                raise AssertionError("V1 inert byte_amplitude unexpectedly received a gradient")
            if arm == "v3":
                if amplitude_grad is None:
                    raise AssertionError("V3 byte_amplitude received no gradient")
                if not torch.isfinite(amplitude_grad).all():
                    raise FloatingPointError("V3 byte_amplitude gradient is non-finite")
            print(
                f"Amplitude gradient control verified for Arm {arm.upper()}: "
                + ("inert (grad=None)" if arm == "v1" else "active and finite"),
                flush=True,
            )
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
        scaler.step(optimizer)
        scaler.update()

        accumulated_loss += update_loss
        steps_since_log += 1
        if step % config.log_interval == 0 or step == config.max_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            val_loss = validate(model, val_loader, device, config.val_batches)
            elapsed = time.perf_counter() - interval_start
            tokens_per_second = (
                steps_since_log
                * config.micro_batch_size
                * config.grad_accum_steps
                * config.block_size
                / max(elapsed, 1.0e-9)
            )
            amplitude = amplitude_statistics(model)
            record = {
                "step": step,
                "train_loss": accumulated_loss / steps_since_log,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "grad_norm": grad_norm,
                "tokens_per_second": tokens_per_second,
                "amplitude": amplitude,
            }
            history.append(record)
            print(
                f"step={step:04d} train_loss={record['train_loss']:.4f} "
                f"val_loss={val_loss:.4f} lr={learning_rate:.3e} "
                f"grad_norm={grad_norm:.3f} tok/s={tokens_per_second:,.0f} "
                f"amp(mean/std/min/max)={amplitude['mean']:.4f}/"
                f"{amplitude['std']:.4f}/{amplitude['minimum']:.4f}/"
                f"{amplitude['maximum']:.4f}",
                flush=True,
            )
            accumulated_loss = 0.0
            steps_since_log = 0
            interval_start = time.perf_counter()

    original_model = getattr(model, "_orig_mod", model)
    final_amplitudes = (
        original_model.token_embedding.byte_amplitude.detach().float().cpu().tolist()
    )
    checkpoint = output_dir / f"arm_{arm}_final.pt"
    torch.save(
        {
            "arm": arm,
            "model_state_dict": original_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "preset": asdict(preset),
            "initial_trainable_sha256": initial_hash,
            "history": history,
            "final_byte_amplitudes": final_amplitudes,
            "codec_table_note": "deterministic raw codec buffer is reconstructed, not checkpointed",
        },
        checkpoint,
    )

    result = {
        "arm": arm,
        "trainable_parameters": parameter_count(original_model),
        "initial_trainable_sha256": initial_hash,
        "initial_val_loss": initial_val,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "wall_seconds": time.perf_counter() - start_time,
        "history": history,
        "final_byte_amplitudes": final_amplitudes,
        "checkpoint": str(checkpoint),
    }

    del model, optimizer, scaler, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_loss_curves(results: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Create publication-ready train/validation loss curves."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for training loss curves") from exc

    colors = {"v1": "#1565C0", "v3": "#D84315"}
    labels = {"v1": "Arm A: Kronecker V1", "v3": "Arm B: Fourier-Kronecker V3 + loudness"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for result in results:
        arm = str(result["arm"])
        history = result["history"]
        train_rows = [row for row in history if row["train_loss"] is not None]
        axes[0].plot(
            [row["step"] for row in train_rows],
            [row["train_loss"] for row in train_rows],
            marker="o",
            linewidth=2,
            color=colors[arm],
            label=labels[arm],
        )
        axes[1].plot(
            [row["step"] for row in history],
            [row["val_loss"] for row in history],
            marker="o",
            linewidth=2,
            color=colors[arm],
            label=labels[arm],
        )

    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xlabel("Optimizer update")
        axis.set_ylabel("Cross-entropy (nats)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Controlled Kronecker V1 vs Fourier-Kronecker V3 ablation")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_amplitude_plot(v3_result: Mapping[str, Any], output: Path) -> None:
    """Visualize the learned loudness assigned to every byte channel."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(v3_result["final_byte_amplitudes"], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(14, 4))
    axis.bar(np.arange(256), values, width=1.0, color="#6A1B9A")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="initial amplitude")
    axis.set_title("Fourier-Kronecker V3 learned byte loudness")
    axis.set_xlabel("Byte value")
    axis.set_ylabel("Learned scalar amplitude")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    output_dir: Path,
    config: ExperimentConfig,
    preset: ModelPreset,
    data_paths: DataPaths,
    vocabulary: TokenByteVocabulary,
    device: torch.device,
    results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_arm = {str(result["arm"]): result for result in results}
    controlled = (
        by_arm["v1"]["initial_trainable_sha256"]
        == by_arm["v3"]["initial_trainable_sha256"]
    )
    if not controlled:
        raise AssertionError("Initial trainable hashes differ")

    v1_loss = float(by_arm["v1"]["final_val_loss"])
    v3_loss = float(by_arm["v3"]["final_val_loss"])
    gap = v3_loss - v1_loss
    report = {
        "experiment": "Kronecker V1 vs Fourier-Kronecker V3 + dynamic amplitude",
        "hypothesis_scope": (
            "Combined V3 basis plus dynamic byte amplitude; this two-arm run does "
            "not isolate the causal contribution of amplitude."
        ),
        "config": asdict(config),
        "preset": asdict(preset),
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
        },
        "data": {
            "source": data_paths.source_label,
            "tokenizer_json": str(data_paths.tokenizer_json),
            "train_tokens": str(data_paths.train_tokens),
            "val_tokens": str(data_paths.val_tokens),
            "vocab_size": len(vocabulary),
            "max_token_bytes": vocabulary.max_token_bytes,
            "tokens_longer_than_32_bytes": vocabulary.count_longer_than(32),
        },
        "controlled_initialization": {
            "verified": controlled,
            "sha256": by_arm["v1"]["initial_trainable_sha256"],
            "shared_inert_v1_amplitude_parameter": True,
        },
        "arms": list(results),
        "ablation": {
            "v1_final_val_loss": v1_loss,
            "v3_final_val_loss": v3_loss,
            "v3_minus_v1_nats": gap,
            "v3_relative_loss_change_percent": 100.0 * (v3_loss / v1_loss - 1.0),
            "v3_perplexity_ratio_vs_v1": math.exp(gap),
            "winner": "v3" if gap < 0 else "v1" if gap > 0 else "tie",
        },
        "artifacts": {
            "loss_curves": str(output_dir / "training_loss_curves.png"),
            "v3_amplitudes": str(output_dir / "v3_byte_amplitudes.png"),
        },
    }
    with (output_dir / "training_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7_007)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--train-tokens", type=Path)
    parser.add_argument("--val-tokens", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fourier_kronecker_v3_run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        preset="smoke" if args.smoke_test else "mid",
        block_size=32 if args.smoke_test else args.block_size,
        micro_batch_size=1 if args.smoke_test else args.micro_batch_size,
        grad_accum_steps=1 if args.smoke_test else args.grad_accum_steps,
        max_steps=1 if args.smoke_test else args.steps,
        val_batches=1 if args.smoke_test else args.val_batches,
        log_interval=1 if args.smoke_test else args.log_interval,
        warmup_steps=1 if args.smoke_test else 50,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        compile_model=args.compile_model,
    )
    if config.max_steps <= 0 or config.log_interval <= 0:
        raise ValueError("steps and log_interval must be positive")
    if config.d_pos != 32 or config.d_char != 256:
        raise ValueError("This controlled experiment requires d_char=256, d_pos=32")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_paths = discover_data(
        args.tokenizer_json, args.train_tokens, args.val_tokens
    )
    validate_data_paths(data_paths)
    vocabulary = TokenByteVocabulary(data_paths.tokenizer_json)
    train_data = MemmapCausalDataset(data_paths.train_tokens, config.block_size)
    val_data = MemmapCausalDataset(data_paths.val_tokens, config.block_size)
    preset = MODEL_PRESETS[config.preset]
    device = configure_device(config.device)

    max_id = max(
        int(np.memmap(data_paths.train_tokens, mode="r", dtype="<u2").max()),
        int(np.memmap(data_paths.val_tokens, mode="r", dtype="<u2").max()),
    )
    if max_id >= len(vocabulary):
        raise ValueError(f"Token id {max_id} exceeds vocab size {len(vocabulary)}")

    print("Fourier-Kronecker V3 controlled language-model experiment")
    print(json.dumps({"config": asdict(config), "preset": asdict(preset)}, indent=2))
    print(
        f"Data: {data_paths.source_label}\n"
        f"Vocabulary: {len(vocabulary):,}; max token bytes={vocabulary.max_token_bytes}\n"
        f"Train tokens: {train_data.n_tokens:,}; validation tokens: {val_data.n_tokens:,}\n"
        f"Device: {device}"
        + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
        flush=True,
    )
    if vocabulary.count_longer_than(config.d_pos) == 0:
        print(
            "RESEARCH SCOPE WARNING: the supplied vocabulary has no token longer "
            "than 32 bytes. This training run tests V3 geometry and loudness, while "
            "the separate diagnostic proves the no-hard-cutoff construction.",
            flush=True,
        )

    v1_result = train_arm(
        "v1",
        config,
        preset,
        vocabulary,
        train_data,
        val_data,
        device,
        output_dir,
        expected_initial_hash=None,
    )
    v3_result = train_arm(
        "v3",
        config,
        preset,
        vocabulary,
        train_data,
        val_data,
        device,
        output_dir,
        expected_initial_hash=v1_result["initial_trainable_sha256"],
    )
    results = [v1_result, v3_result]

    save_loss_curves(results, output_dir / "training_loss_curves.png")
    save_amplitude_plot(v3_result, output_dir / "v3_byte_amplitudes.png")
    report = write_report(
        output_dir,
        config,
        preset,
        data_paths,
        vocabulary,
        device,
        results,
    )

    comparison = report["ablation"]
    print("\n" + "=" * 80)
    print("FINAL CONTROLLED ABLATION")
    print("=" * 80)
    print(f"Arm A V1 validation loss: {comparison['v1_final_val_loss']:.6f}")
    print(f"Arm B V3 validation loss: {comparison['v3_final_val_loss']:.6f}")
    print(
        f"V3 - V1: {comparison['v3_minus_v1_nats']:+.6f} nats "
        f"({comparison['v3_relative_loss_change_percent']:+.3f}%)"
    )
    print(f"Winner: {comparison['winner'].upper()}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
