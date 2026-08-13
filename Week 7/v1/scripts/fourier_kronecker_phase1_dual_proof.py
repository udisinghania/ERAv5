#!/usr/bin/env python3
"""Phase-1 dual proof for Fourier-Kronecker token embeddings.

This standalone program runs two controlled, mid-scale language-model studies:

Experiment 1 -- data fix / long tokens
    A balanced suffix-prediction task is built from 64-byte DNA tokens.  Four
    tokens share every one of their first 32 bytes and differ only in bytes
    32..63.  The next token is a four-way label determined by that suffix.
    V1 therefore receives identical embeddings for all four cases and has the
    information-theoretic floor log(4) nats / 25% accuracy.  V3 sees the full
    64-byte token through analytic Fourier phases.  Prefixes are streamed from
    a Hugging Face genomic dataset through the Dataset Viewer API; --long-source
    auto falls back to deterministic synthetic DNA when networking is absent.

Experiment 2 -- math fix / short Week-6 BPE tokens
    The program measures the empirical BPE token-length distribution, performs
    a deterministic search for 16 non-periodic Euler frequencies conditioned
    jointly at byte windows 8, 16, and 32, then trains:
        A. V1 baseline
        B. data-aware V3, fixed byte amplitude
        C. data-aware V3, learned byte amplitude

Every arm in an experiment has exactly the same trainable parameter names,
shapes, values, optimizer groups, Transformer, and output head at step zero.
The program asserts a byte-for-byte SHA-256 match.  Only deterministic codec
buffers and forward-path mode flags differ.

Full run (five models x 1,000 optimizer updates):
    python fourier_kronecker_phase1_dual_proof.py

Fast integration test:
    python fourier_kronecker_phase1_dual_proof.py --smoke-test

Run one study only:
    python fourier_kronecker_phase1_dual_proof.py --experiment long
    python fourier_kronecker_phase1_dual_proof.py --experiment short
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import time
import urllib.parse
import urllib.request
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
BUNDLED_LONG_PREFIXES = PACKAGE_ROOT / "data" / "long_tokens" / "long_dna_prefixes.json"
HF_VIEWER = "https://datasets-server.huggingface.co"
LONG_VOCAB_SIZE = 8_192
LONG_FAMILIES = (LONG_VOCAB_SIZE - 8) // 4  # 2,046 prefix families x 4 suffixes.


@dataclass(frozen=True)
class ModelPreset:
    n_layers: int
    n_heads: int
    d_model: int
    mlp_ratio: int


PRESETS = {
    # ~25M parameters with an 8,192-way output head and 8,192 -> 384 codec map.
    "mid": ModelPreset(n_layers=6, n_heads=6, d_model=384, mlp_ratio=8),
    "smoke": ModelPreset(n_layers=1, n_heads=2, d_model=64, mlp_ratio=2),
}


@dataclass
class TrainConfig:
    preset: str = "mid"
    block_size: int = 256
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 1_000
    val_batches: int = 20
    log_interval: int = 100
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    warmup_steps: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float = 0.0
    seed: int = 7_007
    num_workers: int = 2
    device: str = "auto"
    compile_model: bool = False
    d_char: int = 256
    d_pos: int = 32
    z_norm_eps: float = 1.0e-5

    @property
    def codec_dim(self) -> int:
        return self.d_char * self.d_pos


@dataclass(frozen=True)
class Week6Paths:
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


def discover_week6(args: argparse.Namespace) -> Week6Paths:
    if args.tokenizer_json or args.train_tokens or args.val_tokens:
        if not (args.tokenizer_json and args.train_tokens and args.val_tokens):
            raise ValueError("The three explicit Week-6 paths must be supplied together")
        answer = Week6Paths(
            args.tokenizer_json.resolve(),
            args.train_tokens.resolve(),
            args.val_tokens.resolve(),
            "explicit CLI paths",
        )
    else:
        answer = Week6Paths(
            BUNDLED_DATA_ROOT / "tokenizer.json",
            BUNDLED_DATA_ROOT / "train/tokens.uint16.bin",
            BUNDLED_DATA_ROOT / "validation/tokens.uint16.bin",
            "bundled Week 6 V2 standard-BPE general corpus",
        )
    for path in (answer.tokenizer_json, answer.train_tokens, answer.val_tokens):
        if not path.is_file():
            raise FileNotFoundError(path)
    if answer.train_tokens.stat().st_size % 2 or answer.val_tokens.stat().st_size % 2:
        raise ValueError("Week-6 token streams must be little-endian uint16 files")
    return answer


def load_week6_vocabulary(path: Path) -> Vocabulary:
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


# ---------------------------------------------------------------------------
# Hugging Face Dataset Viewer data path for natural genomic prefixes.
# ---------------------------------------------------------------------------

def hf_get(endpoint: str, params: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{HF_VIEWER}/{endpoint}?{query}",
        headers={"User-Agent": "Fourier-Kronecker-Phase1/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def recursively_find_dna(value: Any) -> Iterator[str]:
    """Yield DNA-like strings from arbitrary Dataset Viewer row structures."""
    if isinstance(value, str):
        compact = "".join(value.upper().split())
        dna = "".join(ch for ch in compact if ch in "ACGTN")
        # The ratio guard prevents prose containing occasional A/C/G/T letters
        # from being misidentified as genomic sequence.
        if len(dna) >= 64 and len(dna) / max(1, len(compact)) >= 0.90:
            yield dna
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from recursively_find_dna(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from recursively_find_dna(child)


def fetch_hf_prefixes(
    dataset: str,
    config: Optional[str],
    split: Optional[str],
    count: int,
    timeout: float,
    max_pages: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """Read only the small number of rows needed through the Viewer /rows API."""
    split_payload = hf_get("splits", {"dataset": dataset}, timeout)
    choices = split_payload.get("splits", [])
    if not choices:
        raise RuntimeError(f"Hugging Face Dataset Viewer exposes no splits for {dataset}")
    eligible = [
        item for item in choices
        if (config is None or item.get("config") == config)
        and (split is None or item.get("split") == split)
    ]
    if not eligible:
        raise RuntimeError(f"Requested config/split not present for {dataset}")
    chosen = eligible[0]
    config = str(chosen["config"])
    split = str(chosen["split"])

    prefixes: List[str] = []
    seen: set[str] = set()
    pages_read = 0
    for page in range(max_pages):
        payload = hf_get(
            "rows",
            {"dataset": dataset, "config": config, "split": split, "offset": page * 100, "length": 100},
            timeout,
        )
        rows = payload.get("rows", [])
        if not rows:
            break
        pages_read += 1
        for wrapper in rows:
            for sequence in recursively_find_dna(wrapper.get("row", wrapper)):
                # Non-overlapping natural windows provide many distinct prefixes
                # without making the remote data request large.
                for start in range(0, len(sequence) - 31, 32):
                    prefix = sequence[start : start + 32].replace("N", "A")
                    if len(prefix) == 32 and prefix not in seen:
                        seen.add(prefix)
                        prefixes.append(prefix)
                        if len(prefixes) >= count:
                            return prefixes, {
                                "source": "hugging_face_dataset_viewer",
                                "dataset": dataset,
                                "config": config,
                                "split": split,
                                "pages_read": pages_read,
                            }
    raise RuntimeError(f"Only found {len(prefixes):,}/{count:,} distinct DNA prefixes")


def synthetic_prefixes(count: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    alphabet = np.asarray(list("ACGT"))
    answer: List[str] = []
    seen: set[str] = set()
    while len(answer) < count:
        value = "".join(rng.choice(alphabet, size=32).tolist())
        if value not in seen:
            seen.add(value)
            answer.append(value)
    return answer


def obtain_long_prefixes(args: argparse.Namespace, output_dir: Path, seed: int) -> Tuple[List[str], Dict[str, Any]]:
    cache = output_dir / "long_dna_prefixes.json"
    if args.long_source == "auto" and cache.is_file() and not args.refresh_long_data:
        with cache.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        prefixes = [str(value) for value in payload["prefixes"]]
        if len(prefixes) >= LONG_FAMILIES:
            return prefixes[:LONG_FAMILIES], dict(payload["provenance"])

    # The submission bundles the exact 2,046 Hugging Face prefixes used by the
    # reported run.  In auto mode this immutable cache takes precedence over a
    # network request, making the experiment reproducible offline.  Explicit
    # --long-source hf and --long-source synthetic still force those behaviors.
    if args.long_source == "auto" and BUNDLED_LONG_PREFIXES.is_file() and not args.refresh_long_data:
        with BUNDLED_LONG_PREFIXES.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        prefixes = [str(value) for value in payload["prefixes"]]
        if len(prefixes) < LONG_FAMILIES:
            raise ValueError(
                f"Bundled long-token cache contains {len(prefixes):,} prefixes; "
                f"expected at least {LONG_FAMILIES:,}"
            )
        provenance = dict(payload["provenance"])
        json_dump({"provenance": provenance, "prefixes": prefixes[:LONG_FAMILIES]}, cache)
        return prefixes[:LONG_FAMILIES], provenance

    provenance: Dict[str, Any]
    if args.long_source in ("auto", "hf"):
        try:
            prefixes, provenance = fetch_hf_prefixes(
                args.hf_dataset,
                args.hf_config,
                args.hf_split,
                LONG_FAMILIES,
                args.hf_timeout,
                args.hf_max_pages,
            )
        except Exception as exc:
            if args.long_source == "hf":
                raise
            print(f"Hugging Face stream unavailable ({exc}); using deterministic synthetic DNA", flush=True)
            prefixes = synthetic_prefixes(LONG_FAMILIES, seed)
            provenance = {"source": "deterministic_synthetic_fallback", "hf_error": repr(exc)}
    else:
        prefixes = synthetic_prefixes(LONG_FAMILIES, seed)
        provenance = {"source": "deterministic_synthetic", "seed": seed}
    json_dump({"provenance": provenance, "prefixes": prefixes}, cache)
    return prefixes, provenance


def make_long_vocabulary(prefixes: Sequence[str]) -> Tuple[Vocabulary, Dict[str, int]]:
    if len(prefixes) != LONG_FAMILIES:
        raise ValueError(f"Long experiment requires exactly {LONG_FAMILIES:,} prefix families")
    # Cyclic shifts have identical byte histograms (eight A/C/G/T apiece), so V3
    # cannot solve the task from aggregate nucleotide counts.  It must preserve
    # Fourier phase/order in bytes 32..63, all beyond V1's inclusive 0..31 view.
    suffixes = tuple(("ACGT"[shift:] + "ACGT"[:shift]) * 8 for shift in range(4))
    values: List[bytes] = [b"<BOS>", b"<A>", b"<C>", b"<G>", b"<T>", b"<EOS>", b"<R0>", b"<R1>"]
    for prefix in prefixes:
        for suffix in suffixes:
            values.append((prefix + suffix).encode("ascii"))
    if len(values) != LONG_VOCAB_SIZE or any(len(v) != 64 for v in values[8:]):
        raise AssertionError("Long vocabulary construction invariant failed")
    return Vocabulary(tuple(values), "64-byte balanced DNA suffix vocabulary"), {
        "bos": 0, "label_start": 1, "label_stop": 5, "long_start": 8
    }


# ---------------------------------------------------------------------------
# Datasets.  Both produce (input IDs, next-token IDs, boolean loss mask).
# ---------------------------------------------------------------------------

class MemmapCausalDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, path: Path, block_size: int, stride: Optional[int] = None) -> None:
        self.path = path
        self.block_size = block_size
        self.stride = stride or block_size
        self.n_tokens = path.stat().st_size // 2
        self.n_blocks = 1 + (self.n_tokens - block_size - 1) // self.stride
        if self.n_blocks <= 0:
            raise ValueError(f"Token stream is too short: {path}")
        self._array: Optional[np.memmap] = None

    def _tokens(self) -> np.memmap:
        if self._array is None:
            self._array = np.memmap(self.path, mode="r", dtype="<u2")
        return self._array

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = index * self.stride
        sample = np.array(self._tokens()[start : start + self.block_size + 1], dtype=np.int64, copy=True)
        value = torch.from_numpy(sample)
        return value[:-1], value[1:], torch.ones(self.block_size, dtype=torch.bool)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_array"] = None
        return state


class LongSuffixDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Deterministic blocks; only long-token -> class-label targets are scored."""
    def __init__(
        self,
        families: int,
        block_size: int,
        samples: int,
        seed: int,
        ids: Mapping[str, int],
    ) -> None:
        self.families = families
        self.block_size = block_size
        self.samples = samples
        self.seed = seed
        self.ids = dict(ids)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index * 1_000_003)
        # Triples make this a conventional causal-LM stream:
        # <BOS>, <64-byte-DNA-token>, <suffix-class-label>, ...
        needed = self.block_size + 1
        stream: List[int] = []
        while len(stream) < needed:
            family = int(rng.integers(self.families))
            variant = int(rng.integers(4))
            long_id = self.ids["long_start"] + 4 * family + variant
            stream.extend((self.ids["bos"], long_id, self.ids["label_start"] + variant))
        values = torch.tensor(stream[:needed], dtype=torch.long)
        tokens, targets = values[:-1], values[1:]
        mask = (tokens >= self.ids["long_start"]) & (
            (targets >= self.ids["label_start"]) & (targets < self.ids["label_stop"])
        )
        if not mask.any():
            raise AssertionError("A long-token block contains no supervised suffix label")
        return tokens, targets, mask


def make_loader(dataset: Dataset[Any], batch_size: int, shuffle: bool, seed: int, workers: int, pin: bool) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def cycle(loader: DataLoader[Any]) -> Iterator[Any]:
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Fourier schedules and deterministic data-aware conditioning search.
# ---------------------------------------------------------------------------

def phase_waves(length: int, omegas: torch.Tensor, d_pos: int = 32) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float64)[:, None]
    phase = positions * omegas[None, :]
    values = torch.empty((length, d_pos), dtype=torch.float64)
    values[:, 0::2] = torch.cos(phase)
    values[:, 1::2] = torch.sin(phase)
    return values


def original_v3_omegas() -> torch.Tensor:
    """The prior 0..255-calibrated V3 basis used only in Experiment 1."""
    global_frequency = torch.tensor([2.0 * math.pi / 256.0], dtype=torch.float64)
    local = torch.exp(torch.linspace(math.log(0.2), math.log(0.9 * math.pi), 15, dtype=torch.float64))
    return torch.cat((global_frequency, local))


def frequency_metrics(omegas: np.ndarray, window: int) -> Dict[str, float]:
    positions = np.arange(window, dtype=np.float64)[:, None]
    waves = np.empty((window, 32), dtype=np.float64)
    waves[:, 0::2] = np.cos(positions * omegas[None, :])
    waves[:, 1::2] = np.sin(positions * omegas[None, :])
    waves /= math.sqrt(16.0)  # each row is now unit norm
    gram = waves @ waves.T
    off_diagonal = gram - np.eye(window)
    singular = np.linalg.svd(waves, compute_uv=False)
    energy = np.square(singular)
    probability = energy / energy.sum()
    effective_rank = float(np.exp(-np.sum(probability * np.log(probability + 1e-300))))
    positive = singular[singular > 1e-12]
    return {
        "window": float(window),
        "rms_off_diagonal_cosine": float(np.sqrt(np.mean(np.square(off_diagonal)))),
        "max_abs_off_diagonal_cosine": float(np.max(np.abs(off_diagonal))),
        "effective_rank": effective_rank,
        "rank_ceiling": float(min(window, 32)),
        "condition_number": float(positive.max() / positive.min()),
    }


def long_alias_score(omegas: np.ndarray) -> float:
    # Stationarity gives cosine similarity between position p and p+lag as
    # mean_k cos(lag*omega_k).  Penalize recurrence beyond the fitted window.
    lags = np.arange(32, 257, dtype=np.float64)[:, None]
    similarities = np.mean(np.cos(lags * omegas[None, :]), axis=1)
    return float(np.max(similarities))


def empirical_length_profile(vocabulary: Vocabulary, train_path: Path) -> Dict[str, Any]:
    tokens = np.memmap(train_path, mode="r", dtype="<u2")
    if int(tokens.max()) >= len(vocabulary):
        raise ValueError("Week-6 stream contains a token outside the vocabulary")
    counts = np.bincount(tokens, minlength=len(vocabulary)).astype(np.float64)
    occurrence_hist = np.bincount(vocabulary.lengths, weights=counts)
    total = float(occurrence_hist.sum())
    buckets = np.asarray(
        [occurrence_hist[1:9].sum(), occurrence_hist[9:17].sum(), occurrence_hist[17:].sum()],
        dtype=np.float64,
    ) / total
    # A small floor prevents a rare length bucket from being ignored completely;
    # 85% of the objective remains determined by actual corpus occurrence mass.
    search_weights = 0.85 * buckets + 0.15 / 3.0
    return {
        "token_occurrences": int(total),
        "type_histogram": np.bincount(vocabulary.lengths).tolist(),
        "occurrence_histogram": occurrence_hist.astype(np.int64).tolist(),
        "bucket_mass_le8_9to16_17plus": buckets.tolist(),
        "search_weights_8_16_32": search_weights.tolist(),
    }


def search_data_aware_omegas(profile: Mapping[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Grid-search a low-discrepancy perturbation of the 32-point DCT grid.

    Exact DCT midpoints are exceptionally conditioned locally but recur at long
    lags.  A golden-ratio sinusoidal perturbation breaks the common period while
    preserving near-orthogonality.  Strength and phase are selected empirically,
    never hand-picked after looking at LM validation loss.
    """
    weights = np.asarray(profile["search_weights_8_16_32"], dtype=np.float64)
    golden_conjugate = (math.sqrt(5.0) - 1.0) / 2.0
    indices = np.arange(16, dtype=np.float64)
    best: Optional[Tuple[float, float, float, np.ndarray, List[Dict[str, float]], float]] = None
    for strength in np.linspace(0.05, 0.35, 61):
        for phase in np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False):
            offsets = strength * np.sin(2.0 * math.pi * golden_conjugate * (indices + 1.0) + phase)
            omegas = math.pi * (indices + 0.5 + offsets) / 16.0
            metrics = [frequency_metrics(omegas, window) for window in (8, 16, 32)]
            score = 0.0
            for weight, metric in zip(weights, metrics):
                rank_deficit = 1.0 - metric["effective_rank"] / metric["rank_ceiling"]
                score += float(weight) * (
                    metric["rms_off_diagonal_cosine"]
                    + 0.20 * metric["max_abs_off_diagonal_cosine"]
                    + 0.50 * rank_deficit
                )
            alias = long_alias_score(omegas)
            # Local empirical conditioning is primary.  This small term is only
            # enough to reject the exactly recurring DCT grid, not to sacrifice
            # the near-perfect ranks actually used by the short-token corpus.
            score += 0.02 * alias
            candidate = (score, float(strength), float(phase), omegas, metrics, alias)
            if best is None or candidate[0] < best[0]:
                best = candidate
    assert best is not None
    score, strength, phase, omegas, metrics, alias = best
    artifact = {
        "method": "golden-ratio perturbation of 16 DCT midpoint frequencies",
        "objective": "empirical weighted local conditioning + long-lag recurrence penalty",
        "grid": {"strength": [0.05, 0.35, 61], "phase_count": 72},
        "selected_strength": strength,
        "selected_phase_radians": phase,
        "objective_value": score,
        "max_positive_cosine_lag_32_to_256": alias,
        "omegas_radians_per_byte": omegas.tolist(),
        "metrics": metrics,
        "profile": dict(profile),
    }
    return torch.tensor(omegas, dtype=torch.float64), artifact


# ---------------------------------------------------------------------------
# Structured token codec and shared ~25M causal Transformer.
# ---------------------------------------------------------------------------

CodecKind = Literal["v1", "v3"]
AmplitudeMode = Literal["inert", "fixed", "dynamic"]


def build_codec_table(vocabulary: Vocabulary, kind: CodecKind, omegas: Optional[torch.Tensor], storage_dtype: torch.dtype) -> torch.Tensor:
    """Build deterministic [vocab, 256 bytes, 32 position channels] rows."""
    table = torch.zeros((len(vocabulary), 256, 32), dtype=torch.float32)
    for token_id, complete in enumerate(vocabulary.token_bytes):
        value = complete[:32] if kind == "v1" else complete
        if not value:
            raise ValueError(f"Empty token at id {token_id}")
        byte_ids = torch.tensor(list(value), dtype=torch.long)
        if kind == "v1":
            # V1: c_byte x one_hot(position), with a literal hard stop at 32.
            table[token_id, byte_ids, torch.arange(len(value))] = 1.0
        else:
            if omegas is None:
                raise ValueError("V3 requires frequencies")
            waves = phase_waves(len(value), omegas).float()
            table[token_id].index_add_(0, byte_ids, waves)
        table[token_id] *= 1.0 / math.sqrt(float(len(value)))
    if not torch.isfinite(table).all():
        raise FloatingPointError("Codec table contains non-finite values")
    return table.to(storage_dtype)


class StructuredEmbedding(nn.Module):
    def __init__(self, table: torch.Tensor, d_model: int, config: TrainConfig, amplitude_mode: AmplitudeMode) -> None:
        super().__init__()
        self.register_buffer("raw_codec_table", table, persistent=False)
        self.amplitude_mode = amplitude_mode
        self.eps = config.z_norm_eps
        # Present in every arm so complete trainable states remain identical.
        self.byte_amplitude = nn.Parameter(torch.ones(256))
        self.projection = nn.Linear(config.codec_dim, d_model, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        flat = token_ids.reshape(-1)
        unique, inverse = torch.unique(flat, sorted=False, return_inverse=True)
        rows = self.raw_codec_table.index_select(0, unique).float()
        if self.amplitude_mode == "dynamic":
            rows = rows * self.byte_amplitude[None, :, None]
        # "fixed" and "inert" are deliberately the same mathematical value
        # a[b]=1.  The naming distinguishes ablation intent from V1's spare
        # control parameter; both leave its gradient at None.
        vectors = rows.flatten(start_dim=1)
        centered = vectors - vectors.mean(dim=1, keepdim=True)
        normalized = centered * torch.rsqrt(centered.square().mean(dim=1, keepdim=True) + self.eps)
        embedded = self.projection(normalized).index_select(0, inverse)
        return embedded.view(*token_ids.shape, -1)


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
    def __init__(self, preset: ModelPreset, vocab_size: int, table: torch.Tensor, config: TrainConfig, amplitude_mode: AmplitudeMode) -> None:
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding = StructuredEmbedding(table, preset.d_model, config, amplitude_mode)
        self.sequence_position = nn.Embedding(config.block_size, preset.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(preset, config.dropout) for _ in range(preset.n_layers)])
        self.final_norm = nn.LayerNorm(preset.d_model)
        self.lm_head = nn.Linear(preset.d_model, vocab_size, bias=False)
        self.apply(self._initialize)
        nn.init.normal_(self.token_embedding.projection.weight, 0.0, 1.0 / math.sqrt(config.codec_dim))
        nn.init.ones_(self.token_embedding.byte_amplitude)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, steps = tokens.shape
        if steps > self.block_size:
            raise ValueError("Sequence exceeds configured block size")
        positions = torch.arange(steps, device=tokens.device)
        x = self.token_embedding(tokens) + self.sequence_position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = mask.reshape(-1)
    if not selected.any():
        raise ValueError("Loss mask selected no targets")
    return F.cross_entropy(logits.flatten(0, 1)[selected], targets.reshape(-1)[selected])


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
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16)


def make_scaler(device: torch.device) -> Any:
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


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


def optimizer_for(model: nn.Module, config: TrainConfig, device: torch.device) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for value in model.parameters():
        (decay if value.ndim >= 2 else no_decay).append(value)
    groups = [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"lr": config.learning_rate, "betas": (config.beta1, config.beta2), "eps": 1e-8}
    try:
        return torch.optim.AdamW(groups, fused=device.type == "cuda", **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, **kwargs)


def lr_at_step(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader[Any], device: torch.device, batches: int) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for index, (tokens, targets, mask) in enumerate(loader):
        if index >= batches:
            break
        tokens, targets, mask = tokens.to(device), targets.to(device), mask.to(device)
        with amp_context(device):
            logits = model(tokens)
            loss = masked_loss(logits, targets, mask)
        selected = mask.reshape(-1)
        selected_targets = targets.reshape(-1)[selected]
        predictions = logits.flatten(0, 1)[selected].argmax(dim=-1)
        n = int(selected.sum())
        loss_sum += float(loss) * n
        correct += int((predictions == selected_targets).sum())
        count += n
    model.train()
    if not count:
        raise RuntimeError("Validation loader produced no scored targets")
    return {"loss": loss_sum / count, "accuracy": correct / count, "scored_targets": float(count)}


@torch.no_grad()
def collision_diagnostic(vocabulary: Vocabulary, table: torch.Tensor, families: int) -> Dict[str, float]:
    # Raw equality is enough to prove V1 information loss; normalized distances
    # show the V3 alternatives remain separable after the shared z-normalization.
    values = table[8 : 8 + 4 * families].float().flatten(start_dim=1).view(families, 4, -1)
    centered = values - values.mean(dim=-1, keepdim=True)
    values = centered * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + 1e-5)
    distances: List[torch.Tensor] = []
    for left in range(4):
        for right in range(left + 1, 4):
            distances.append(torch.linalg.vector_norm(values[:, left] - values[:, right], dim=-1))
    all_distances = torch.cat(distances)
    return {"minimum_pair_l2": float(all_distances.min()), "maximum_pair_l2": float(all_distances.max())}


def train_arm(
    experiment: str,
    arm: str,
    codec_kind: CodecKind,
    amplitude_mode: AmplitudeMode,
    omegas: Optional[torch.Tensor],
    vocabulary: Vocabulary,
    train_data: Dataset[Any],
    val_data: Dataset[Any],
    config: TrainConfig,
    preset: ModelPreset,
    device: torch.device,
    output_dir: Path,
    expected_hash: Optional[str],
    save_checkpoint: bool,
) -> Dict[str, Any]:
    print(f"\n{'=' * 80}\n{experiment.upper()} / ARM {arm.upper()}\n{'=' * 80}", flush=True)
    storage_dtype = torch.float16 if device.type == "cuda" else torch.float32
    started = time.perf_counter()
    table = build_codec_table(vocabulary, codec_kind, omegas, storage_dtype)
    diagnostic = collision_diagnostic(vocabulary, table, LONG_FAMILIES) if experiment == "long" else None
    print(f"codec={codec_kind}; table={tuple(table.shape)}; {table.nbytes / 2**20:.1f} MiB", flush=True)

    # Seeding immediately before construction makes every trainable tensor equal.
    seed_everything(config.seed)
    model = CausalTransformer(preset, len(vocabulary), table, config, amplitude_mode).to(device)
    del table
    initial_hash = trainable_sha256(model)
    if expected_hash is not None and initial_hash != expected_hash:
        raise AssertionError(f"Controlled initialization failed: {initial_hash} != {expected_hash}")
    print(f"parameters={parameter_count(model):,}; initial_sha256={initial_hash}", flush=True)

    optimizer = optimizer_for(model, config, device)
    scaler = make_scaler(device)
    train_loader = make_loader(train_data, config.micro_batch_size, True, config.seed, config.num_workers, device.type == "cuda")
    val_loader = make_loader(val_data, config.micro_batch_size, False, config.seed, config.num_workers, device.type == "cuda")
    iterator = cycle(train_loader)
    if config.compile_model:
        model = torch.compile(model)  # type: ignore[assignment]

    initial = evaluate(model, val_loader, device, config.val_batches)
    history: List[Dict[str, Any]] = [{"step": 0, "train_loss": None, "val_loss": initial["loss"], "val_accuracy": initial["accuracy"]}]
    print(f"step=0000 val_loss={initial['loss']:.5f} val_accuracy={initial['accuracy']:.3%}", flush=True)
    interval_loss = 0.0
    interval_steps = 0
    interval_started = time.perf_counter()

    for step in range(1, config.max_steps + 1):
        learning_rate = lr_at_step(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        for _ in range(config.grad_accum_steps):
            tokens, targets, mask = next(iterator)
            tokens, targets, mask = tokens.to(device, non_blocking=True), targets.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with amp_context(device):
                logits = model(tokens)
                loss = masked_loss(logits, targets, mask)
                scaled_loss = loss / config.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            update_loss += float(loss.detach()) / config.grad_accum_steps
        scaler.unscale_(optimizer)
        original = getattr(model, "_orig_mod", model)
        if step == 1:
            gradient = original.token_embedding.byte_amplitude.grad
            if amplitude_mode == "dynamic" and (gradient is None or not torch.isfinite(gradient).all()):
                raise AssertionError("Dynamic amplitude must receive a finite gradient")
            if amplitude_mode != "dynamic" and gradient is not None:
                raise AssertionError("Fixed/inert amplitude unexpectedly received a gradient")
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        interval_loss += update_loss
        interval_steps += 1

        if step % config.log_interval == 0 or step == config.max_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            metrics = evaluate(model, val_loader, device, config.val_batches)
            elapsed = time.perf_counter() - interval_started
            tokens_per_second = interval_steps * config.micro_batch_size * config.grad_accum_steps * config.block_size / max(elapsed, 1e-9)
            amplitudes = original.token_embedding.byte_amplitude.detach().float()
            row = {
                "step": step,
                "train_loss": interval_loss / interval_steps,
                "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"],
                "learning_rate": learning_rate,
                "grad_norm": grad_norm,
                "tokens_per_second": tokens_per_second,
                "amplitude_std": float(amplitudes.std(unbiased=False)),
            }
            history.append(row)
            print(
                f"step={step:04d} train={row['train_loss']:.5f} val={row['val_loss']:.5f} "
                f"acc={row['val_accuracy']:.3%} lr={learning_rate:.2e} tok/s={tokens_per_second:,.0f}",
                flush=True,
            )
            interval_loss, interval_steps, interval_started = 0.0, 0, time.perf_counter()

    original = getattr(model, "_orig_mod", model)
    checkpoint: Optional[str] = None
    if save_checkpoint:
        checkpoint_path = output_dir / f"{experiment}_{arm}_final.pt"
        torch.save({"model": original.state_dict(), "config": asdict(config), "history": history}, checkpoint_path)
        checkpoint = str(checkpoint_path)
    answer = {
        "experiment": experiment,
        "arm": arm,
        "codec_kind": codec_kind,
        "amplitude_mode": amplitude_mode,
        "trainable_parameters": parameter_count(original),
        "initial_trainable_sha256": initial_hash,
        "initial_validation": initial,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "history": history,
        "collision_diagnostic": diagnostic,
        "wall_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint,
    }
    del model, optimizer, scaler, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return answer


# ---------------------------------------------------------------------------
# Reporting and plots.
# ---------------------------------------------------------------------------

def save_comparative_curves(results: Sequence[Mapping[str, Any]], title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#1565C0", "#D84315", "#6A1B9A"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for color, result in zip(colors, results):
        history = result["history"]
        trained = [row for row in history if row["train_loss"] is not None]
        label = f"Arm {result['arm']}"
        axes[0].plot([r["step"] for r in trained], [r["train_loss"] for r in trained], marker="o", color=color, label=label)
        axes[1].plot([r["step"] for r in history], [r["val_loss"] for r in history], marker="o", color=color, label=label)
        axes[2].plot([r["step"] for r in history], [r["val_accuracy"] for r in history], marker="o", color=color, label=label)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    axes[2].set_title("Validation accuracy")
    for axis in axes:
        axis.set_xlabel("Optimizer update")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Cross-entropy (nats)")
    axes[1].set_ylabel("Cross-entropy (nats)")
    axes[2].set_ylabel("Accuracy")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_long(args: argparse.Namespace, root: Path, config: TrainConfig, preset: ModelPreset, device: torch.device) -> Dict[str, Any]:
    output = root / "experiment_1_long_tokens"
    output.mkdir(parents=True, exist_ok=True)
    prefixes, provenance = obtain_long_prefixes(args, output, config.seed)
    vocabulary, ids = make_long_vocabulary(prefixes)
    train_data = LongSuffixDataset(LONG_FAMILIES, config.block_size, args.long_train_samples, config.seed + 11, ids)
    val_data = LongSuffixDataset(LONG_FAMILIES, config.block_size, args.long_val_samples, config.seed + 99_991, ids)
    omegas = original_v3_omegas()
    specifications = [("A_V1", "v1", "inert", None), ("B_V3", "v3", "fixed", omegas)]
    results: List[Dict[str, Any]] = []
    expected_hash: Optional[str] = None
    for arm, kind, amplitude, frequencies in specifications:
        result = train_arm("long", arm, kind, amplitude, frequencies, vocabulary, train_data, val_data, config, preset, device, output, expected_hash, args.save_checkpoints)
        expected_hash = expected_hash or result["initial_trainable_sha256"]
        results.append(result)
    if len({r["initial_trainable_sha256"] for r in results}) != 1:
        raise AssertionError("Long-token arm hashes differ")
    save_comparative_curves(results, "Experiment 1: 64-byte DNA suffix identifiability", output / "long_token_loss_curves.png")
    report = {
        "experiment": "Data fix: long-token causal identifiability",
        "data_provenance": provenance,
        "construction": {
            "vocab_size": len(vocabulary),
            "prefix_families": LONG_FAMILIES,
            "variants_per_prefix": 4,
            "long_token_bytes": 64,
            "supervision": "predict balanced suffix class after each long token",
            "v1_information_theoretic_loss_floor_nats": math.log(4.0),
            "v1_information_theoretic_accuracy_ceiling": 0.25,
        },
        "controlled_initialization": {"verified": True, "sha256": expected_hash},
        "arms": results,
        "v3_minus_v1_final_loss_nats": results[1]["final_val_loss"] - results[0]["final_val_loss"],
        "v3_minus_v1_final_accuracy": results[1]["final_val_accuracy"] - results[0]["final_val_accuracy"],
        "loss_curves": str(output / "long_token_loss_curves.png"),
    }
    json_dump(report, output / "report.json")
    return report


def run_short(args: argparse.Namespace, root: Path, config: TrainConfig, preset: ModelPreset, device: torch.device) -> Dict[str, Any]:
    output = root / "experiment_2_short_tokens"
    output.mkdir(parents=True, exist_ok=True)
    paths = discover_week6(args)
    vocabulary = load_week6_vocabulary(paths.tokenizer_json)
    profile = empirical_length_profile(vocabulary, paths.train_tokens)
    omegas, search = search_data_aware_omegas(profile)
    json_dump(search, output / "frequency_search.json")
    train_data = MemmapCausalDataset(paths.train_tokens, config.block_size)
    val_data = MemmapCausalDataset(paths.val_tokens, config.block_size)
    specifications = [
        ("A_V1", "v1", "inert", None),
        ("B_V3_fixed", "v3", "fixed", omegas),
        ("C_V3_dynamic", "v3", "dynamic", omegas),
    ]
    results: List[Dict[str, Any]] = []
    expected_hash: Optional[str] = None
    for arm, kind, amplitude, frequencies in specifications:
        result = train_arm("short", arm, kind, amplitude, frequencies, vocabulary, train_data, val_data, config, preset, device, output, expected_hash, args.save_checkpoints)
        expected_hash = expected_hash or result["initial_trainable_sha256"]
        results.append(result)
    if len({r["initial_trainable_sha256"] for r in results}) != 1:
        raise AssertionError("Short-token arm hashes differ")
    save_comparative_curves(results, "Experiment 2: data-aware Fourier basis on Week-6 BPE", output / "short_token_loss_curves.png")
    baseline = results[0]["final_val_loss"]
    report = {
        "experiment": "Math fix: short-token data-aware conditioning",
        "data": {"source": paths.source, "tokenizer": str(paths.tokenizer_json), "train": str(paths.train_tokens), "validation": str(paths.val_tokens)},
        "frequency_search": search,
        "controlled_initialization": {"verified": True, "sha256": expected_hash},
        "arms": results,
        "loss_deltas_vs_v1_nats": {result["arm"]: result["final_val_loss"] - baseline for result in results[1:]},
        "loss_curves": str(output / "short_token_loss_curves.png"),
    }
    json_dump(report, output / "report.json")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "long", "short"), default="all")
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
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "fourier_kronecker_phase1_dual_proof_run")
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--train-tokens", type=Path)
    parser.add_argument("--val-tokens", type=Path)
    parser.add_argument("--long-source", choices=("auto", "hf", "synthetic"), default="auto")
    parser.add_argument("--hf-dataset", default="shivendrra/EnigmaDataset")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split")
    parser.add_argument("--hf-timeout", type=float, default=30.0)
    parser.add_argument("--hf-max-pages", type=int, default=20)
    parser.add_argument("--refresh-long-data", action="store_true")
    parser.add_argument("--long-train-samples", type=int, default=8_192)
    parser.add_argument("--long-val-samples", type=int, default=1_024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
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
    if min(config.max_steps, config.log_interval, config.block_size, config.micro_batch_size, config.grad_accum_steps) <= 0:
        raise ValueError("Training sizes and intervals must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = configure_device(config.device)
    preset = PRESETS[config.preset]
    print(json.dumps({"config": asdict(config), "preset": asdict(preset), "device": str(device)}, indent=2), flush=True)
    reports: Dict[str, Any] = {}
    if args.experiment in ("all", "long"):
        reports["long"] = run_long(args, output, config, preset, device)
    if args.experiment in ("all", "short"):
        reports["short"] = run_short(args, output, config, preset, device)
    master = {
        "title": "Fourier-Kronecker Phase-1 dual empirical proof",
        "runtime": {
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "fp16_amp": device.type == "cuda",
        },
        "config": asdict(config),
        "reports": reports,
    }
    json_dump(master, output / "master_report.json")
    print(f"\nComplete. Reports and comparative curves: {output}", flush=True)


if __name__ == "__main__":
    main()
