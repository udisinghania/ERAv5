"""Reproducible Adam, warmup, scheduler, and width/LR experiments.

Run from the submission directory with any supported Python environment:

    python experiment.py --device auto

All experimental outputs are written beside this file under ``artifacts/``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
DATASET_SNAPSHOT = DATA_DIR / "synthetic_teacher_v1.npz"
EXPECTED_DATASET_CONTENT_HASH = (
    "sha256:5482356c445d8bb9e29f969dc2da0958cac7740a44c62382d89df71af422bbb5"
)
SEED = 20260911
DTYPE = torch.float32


@dataclass(frozen=True)
class AdamConfig:
    beta1: float = 0.9
    beta2: float = 0.999
    learning_rate: float = 0.001
    epsilon: float = 1e-8
    initial_weight: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    input_dim: int = 32
    classes: int = 10
    train_examples: int = 8192
    validation_examples: int = 2048
    test_examples: int = 4096
    batch_size: int = 128
    planned_steps: int = 300
    stop_step: int = 200
    warmup_steps: int = 20
    min_lr_ratio: float = 0.0
    wsd_decay_start: int = 270
    weight_decay: float = 1e-4


def configure_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def manual_adam_rows(
    gradients: Iterable[float], config: AdamConfig = AdamConfig()
) -> list[dict[str, float | int]]:
    """Compute scalar Adam exactly, with bias correction and no weight decay."""
    m = 0.0
    v = 0.0
    weight = config.initial_weight
    rows: list[dict[str, float | int]] = []
    for step, gradient in enumerate(gradients, start=1):
        weight_before = weight
        m = config.beta1 * m + (1.0 - config.beta1) * gradient
        v = config.beta2 * v + (1.0 - config.beta2) * gradient * gradient
        m_hat = m / (1.0 - config.beta1**step)
        v_hat = v / (1.0 - config.beta2**step)
        update = config.learning_rate * m_hat / (math.sqrt(v_hat) + config.epsilon)
        weight -= update
        rows.append(
            {
                "step": step,
                "gradient": gradient,
                "weight_before": weight_before,
                "m": m,
                "v": v,
                "m_hat": m_hat,
                "v_hat": v_hat,
                "update": update,
                "weight_delta": -update,
                "weight_after": weight,
            }
        )
    return rows


def pytorch_adam_rows(
    gradients: Iterable[float], config: AdamConfig = AdamConfig()
) -> list[dict[str, float | int]]:
    weight = torch.tensor([config.initial_weight], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam(
        [weight],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    rows: list[dict[str, float | int]] = []
    for step, gradient in enumerate(gradients, start=1):
        before = float(weight.detach().item())
        weight.grad = torch.tensor([gradient], dtype=torch.float64)
        optimizer.step()
        state = optimizer.state[weight]
        m = float(state["exp_avg"].item())
        v = float(state["exp_avg_sq"].item())
        m_hat = m / (1.0 - config.beta1**step)
        v_hat = v / (1.0 - config.beta2**step)
        after = float(weight.detach().item())
        rows.append(
            {
                "step": step,
                "gradient": gradient,
                "weight_before": before,
                "m": m,
                "v": v,
                "m_hat": m_hat,
                "v_hat": v_hat,
                "update": before - after,
                "weight_delta": after - before,
                "weight_after": after,
            }
        )
    return rows


def compare_adam_rows(
    manual: list[dict[str, float | int]], pytorch: list[dict[str, float | int]]
) -> tuple[list[dict[str, float | int]], float]:
    columns = ("m", "v", "m_hat", "v_hat", "update", "weight_delta", "weight_after")
    compared: list[dict[str, float | int]] = []
    maximum = 0.0
    for hand, torch_row in zip(manual, pytorch, strict=True):
        row = dict(hand)
        for column in columns:
            torch_value = float(torch_row[column])
            difference = abs(float(hand[column]) - torch_value)
            row[f"torch_{column}"] = torch_value
            row[f"abs_diff_{column}"] = difference
            maximum = max(maximum, difference)
        compared.append(row)
    return compared, maximum


def bias_correction_factor(step: int, beta1: float, beta2: float) -> float:
    return math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)


def bias_threshold_sensitivity(
    config: AdamConfig, tolerances: Iterable[float], maximum_step: int = 100_000
) -> list[dict[str, float | int]]:
    """Find the first step after the last tolerance violation."""
    tolerance_values = list(tolerances)
    last_violation = {tolerance: 0 for tolerance in tolerance_values}
    for step in range(1, maximum_step + 1):
        error = abs(bias_correction_factor(step, config.beta1, config.beta2) - 1.0)
        for tolerance in tolerance_values:
            if error >= tolerance:
                last_violation[tolerance] = step
    return [
        {
            "tolerance_fraction": tolerance,
            "tolerance_percent": 100.0 * tolerance,
            "first_step_permanently_below": last_violation[tolerance] + 1,
        }
        for tolerance in tolerance_values
    ]


def bias_correction_experiment(
    gradients: list[float], config: AdamConfig, plotted_steps: int = 20
) -> tuple[list[dict[str, float | int]], int]:
    repeated = [gradients[i % len(gradients)] for i in range(plotted_steps)]
    corrected = manual_adam_rows(repeated, config)
    m = 0.0
    v = 0.0
    weight = config.initial_weight
    rows: list[dict[str, float | int]] = []
    for step, (gradient, corrected_row) in enumerate(zip(repeated, corrected, strict=True), start=1):
        m = config.beta1 * m + (1.0 - config.beta1) * gradient
        v = config.beta2 * v + (1.0 - config.beta2) * gradient * gradient
        raw_update = config.learning_rate * m / (math.sqrt(v) + config.epsilon)
        weight -= raw_update
        factor = bias_correction_factor(step, config.beta1, config.beta2)
        rows.append(
            {
                "step": step,
                "gradient": gradient,
                "corrected_weight": float(corrected_row["weight_after"]),
                "uncorrected_weight": weight,
                "corrected_update": float(corrected_row["update"]),
                "uncorrected_update": raw_update,
                "correction_factor": factor,
                "relative_factor_error": abs(factor - 1.0),
            }
        )

    # "Stops mattering" is declared when correction changes the instantaneous
    # update scale by less than 1%, and remains below 1% thereafter.
    threshold_step = next(
        step
        for step in range(1, 100_001)
        if all(
            abs(bias_correction_factor(later, config.beta1, config.beta2) - 1.0) < 0.01
            for later in range(step, 100_001)
        )
    )
    return rows, threshold_step


class ResidualMLP(nn.Module):
    """Small width-scalable classifier with named parameter-bearing layers."""

    def __init__(self, input_dim: int, width: int, classes: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.hidden_norm = nn.LayerNorm(width)
        self.hidden = nn.Linear(width, width)
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Standard parameterization: initialization scale follows
                # fan-in. The width sweep deliberately does not use muP.
                nn.init.normal_(
                    module.weight, mean=0.0, std=1.0 / math.sqrt(module.in_features)
                )
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input(inputs))
        hidden = (hidden + F.gelu(self.hidden(self.hidden_norm(hidden)))) / math.sqrt(2.0)
        return self.output(self.output_norm(hidden))


def generate_dataset_cpu(config: TrainConfig) -> dict[str, torch.Tensor]:
    """Generate the canonical dataset on CPU for snapshot creation only."""
    generator = torch.Generator(device="cpu").manual_seed(SEED + 17)
    total = config.train_examples + config.validation_examples + config.test_examples
    inputs = torch.randn(total, config.input_dim, generator=generator)
    teacher_1 = torch.randn(config.input_dim, 96, generator=generator) / math.sqrt(config.input_dim)
    teacher_2 = torch.randn(96, config.classes, generator=generator) / math.sqrt(96)
    teacher_skip = torch.randn(config.input_dim, config.classes, generator=generator) / math.sqrt(
        config.input_dim
    )
    logits = torch.tanh(inputs @ teacher_1) @ teacher_2 + 0.35 * (inputs @ teacher_skip)
    # A small deterministic perturbation prevents excessive teacher ties.
    logits += 0.03 * torch.randn(logits.shape, generator=generator)
    labels = logits.argmax(dim=1)
    train_end = config.train_examples
    validation_end = train_end + config.validation_examples
    return {
        "train_x": inputs[:train_end].to(dtype=DTYPE),
        "train_y": labels[:train_end],
        "validation_x": inputs[train_end:validation_end].to(dtype=DTYPE),
        "validation_y": labels[train_end:validation_end],
        "test_x": inputs[validation_end:].to(dtype=DTYPE),
        "test_y": labels[validation_end:],
    }


def write_dataset_snapshot(path: Path = DATASET_SNAPSHOT) -> str:
    """Write the exact canonical arrays distributed with this submission."""
    data = generate_dataset_cpu(TrainConfig())
    content_hash = hash_named_tensors(data.items())
    if content_hash != EXPECTED_DATASET_CONTENT_HASH:
        raise RuntimeError(
            "Generated dataset differs from the canonical content hash; "
            "use the bundled snapshot instead."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: tensor.numpy() for name, tensor in data.items()})
    return content_hash


def make_dataset(
    config: TrainConfig,
    device: torch.device,
    snapshot_path: Path = DATASET_SNAPSHOT,
) -> dict[str, torch.Tensor]:
    """Load and validate the exact bundled dataset before moving it to a device."""
    if config != TrainConfig():
        raise ValueError("The bundled dataset is defined for the default TrainConfig only")
    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"Required dataset snapshot is missing: {snapshot_path}. "
            "Copy the complete submission folder, including data/."
        )

    expected_shapes = {
        "train_x": (config.train_examples, config.input_dim),
        "train_y": (config.train_examples,),
        "validation_x": (config.validation_examples, config.input_dim),
        "validation_y": (config.validation_examples,),
        "test_x": (config.test_examples, config.input_dim),
        "test_y": (config.test_examples,),
    }
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        if set(snapshot.files) != set(expected_shapes):
            raise ValueError("Dataset snapshot has unexpected or missing arrays")
        cpu_data = {
            name: torch.from_numpy(np.array(snapshot[name], copy=True)) for name in expected_shapes
        }

    for name, expected_shape in expected_shapes.items():
        tensor = cpu_data[name]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected_shape}")
        expected_dtype = torch.float32 if name.endswith("_x") else torch.int64
        if tensor.dtype != expected_dtype:
            raise ValueError(f"{name} has dtype {tensor.dtype}, expected {expected_dtype}")

    content_hash = hash_named_tensors(cpu_data.items())
    if content_hash != EXPECTED_DATASET_CONTENT_HASH:
        raise ValueError(
            f"Dataset content hash mismatch: got {content_hash}, "
            f"expected {EXPECTED_DATASET_CONTENT_HASH}"
        )
    return {name: tensor.to(device=device) for name, tensor in cpu_data.items()}


def hash_named_tensors(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def batch_sequence_hash(
    steps: int, batch_size: int, dataset_size: int, seed: int
) -> str:
    digest = hashlib.sha256()
    for step in range(1, steps + 1):
        generator = torch.Generator(device="cpu").manual_seed(seed + step * 1_000_003)
        indices = torch.randint(0, dataset_size, (batch_size,), generator=generator)
        digest.update(indices.numpy().astype("<i8", copy=False).tobytes())
    return f"sha256:{digest.hexdigest()}"


def batch_indices(step: int, batch_size: int, dataset_size: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + step * 1_000_003)
    return torch.randint(0, dataset_size, (batch_size,), generator=generator).to(device)


def warmup_multiplier(step: int, warmup_steps: int) -> float:
    return min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0


def cosine_multiplier(step: int, config: TrainConfig) -> float:
    if step <= config.warmup_steps:
        return warmup_multiplier(step, config.warmup_steps)
    progress = (step - config.warmup_steps) / (config.planned_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine


def wsd_multiplier(step: int, config: TrainConfig) -> float:
    if step <= config.warmup_steps:
        return warmup_multiplier(step, config.warmup_steps)
    if step <= config.wsd_decay_start:
        return 1.0
    progress = (step - config.wsd_decay_start) / (config.planned_steps - config.wsd_decay_start)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine


def constant_after_warmup_multiplier(step: int, warmup_steps: int) -> float:
    return warmup_multiplier(step, warmup_steps)


def evaluate(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, chunk: int = 512) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], chunk):
            end = min(inputs.shape[0], start + chunk)
            loss = F.cross_entropy(model(inputs[start:end]), labels[start:end], reduction="sum")
            total_loss += float(loss.item())
            total += end - start
    model.train()
    return total_loss / total


def initial_state(width: int, config: TrainConfig, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = ResidualMLP(config.input_dim, width, config.classes)
    return copy.deepcopy(model.state_dict())


def train_run(
    *,
    width: int,
    peak_lr: float,
    steps: int,
    schedule: Callable[[int], float],
    data: dict[str, torch.Tensor],
    config: TrainConfig,
    device: torch.device,
    init_seed: int,
    batch_seed: int,
    log_every: int = 10,
    track_layer_ratios: bool = False,
    return_state: bool = False,
) -> dict[str, object]:
    torch.manual_seed(init_seed)
    model = ResidualMLP(config.input_dim, width, config.classes).to(device)
    initial_state_hash = hash_named_tensors(model.state_dict().items())
    # AdamW decay is applied only to matrix weights. Biases and normalization
    # scales/shifts are explicitly excluded, matching the session guidance.
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim >= 2 and "norm" not in name:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": config.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=peak_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )
    trace: list[dict[str, float | int]] = []
    ratio_rows: list[dict[str, float | int | str | bool]] = []
    validation_before = evaluate(model, data["validation_x"], data["validation_y"])
    for step in range(1, steps + 1):
        multiplier = schedule(step)
        learning_rate = peak_lr * multiplier
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        indices = batch_indices(
            step, config.batch_size, config.train_examples, batch_seed, device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(data["train_x"][indices])
        loss = F.cross_entropy(logits, data["train_y"][indices])
        loss.backward()

        before: dict[str, torch.Tensor] = {}
        gradient_norms: dict[str, float] = {}
        if track_layer_ratios:
            before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            gradient_groups: dict[str, list[torch.Tensor]] = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                layer = name.split(".", maxsplit=1)[0]
                gradient_groups.setdefault(layer, []).append(parameter.grad.detach())
            gradient_norms = {
                layer: math.sqrt(
                    sum(float(gradient.double().pow(2).sum().item()) for gradient in gradients)
                )
                for layer, gradients in gradient_groups.items()
            }
        optimizer.step()

        if track_layer_ratios:
            groups: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
            for name, parameter in model.named_parameters():
                layer = name.split(".", maxsplit=1)[0]
                groups.setdefault(layer, []).append((before[name], parameter.detach()))
            for layer, pairs in groups.items():
                weight_norm_sq = sum(float(old.double().pow(2).sum().item()) for old, _ in pairs)
                update_norm_sq = sum(
                    float((new.double() - old.double()).pow(2).sum().item()) for old, new in pairs
                )
                ratio = math.sqrt(update_norm_sq) / max(math.sqrt(weight_norm_sq), 1e-30)
                ratio_rows.append(
                    {
                        "step": step,
                        "layer": layer,
                        "learning_rate": learning_rate,
                        "lr_multiplier": multiplier,
                        "warmup_active": step < config.warmup_steps,
                        "gradient_norm": gradient_norms[layer],
                        "update_norm": math.sqrt(update_norm_sq),
                        "weight_norm": math.sqrt(weight_norm_sq),
                        "update_to_weight_ratio": ratio,
                        "peak_lr_counterfactual_ratio": ratio / max(multiplier, 1e-30),
                    }
                )

        if step == 1 or step % log_every == 0 or step == steps:
            trace.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach().item()),
                    "validation_loss": evaluate(
                        model, data["validation_x"], data["validation_y"]
                    ),
                    "learning_rate": learning_rate,
                    "lr_multiplier": multiplier,
                }
            )
    result: dict[str, object] = {
        "width": width,
        "peak_lr": peak_lr,
        "steps": steps,
        "validation_loss_before": validation_before,
        "validation_loss": evaluate(model, data["validation_x"], data["validation_y"]),
        "test_loss": evaluate(model, data["test_x"], data["test_y"]),
        "initial_state_hash": initial_state_hash,
        "batch_sequence_hash": batch_sequence_hash(
            steps, config.batch_size, config.train_examples, batch_seed
        ),
        "trace": trace,
        "layer_ratios": ratio_rows,
    }
    if return_state:
        result["model_state"] = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def bootstrap_mean_ci(
    values: Iterable[float], *, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if array.size == 1:
        return float(array[0]), float(array[0])
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def aggregate_traces(runs: list[dict[str, object]]) -> list[dict[str, float | int]]:
    grouped: dict[int, list[float]] = {}
    for run in runs:
        trace = run["trace"]
        assert isinstance(trace, list)
        for row in trace:
            grouped.setdefault(int(row["step"]), []).append(float(row["validation_loss"]))
    return [
        {
            "step": step,
            "mean_validation_loss": float(np.mean(values)),
            "std_validation_loss": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        for step, values in sorted(grouped.items())
    ]


def aggregate_layer_control(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], dict[str, list[float]]] = {}
    for row in rows:
        key = (str(row["condition"]), int(row["step"]), str(row["layer"]))
        bucket = grouped.setdefault(key, {"ratio": [], "gradient": []})
        bucket["ratio"].append(float(row["update_to_weight_ratio"]))
        bucket["gradient"].append(float(row["gradient_norm"]))
    aggregate: list[dict[str, object]] = []
    for (condition, step, layer), values in sorted(grouped.items()):
        ratios = values["ratio"]
        gradients = values["gradient"]
        aggregate.append(
            {
                "condition": condition,
                "step": step,
                "layer": layer,
                "seeds": len(ratios),
                "mean_update_to_weight_ratio": float(np.mean(ratios)),
                "std_update_to_weight_ratio": (
                    float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0
                ),
                "mean_gradient_norm": float(np.mean(gradients)),
                "std_gradient_norm": (
                    float(np.std(gradients, ddof=1)) if len(gradients) > 1 else 0.0
                ),
            }
        )
    return aggregate


def warmup_control_gaps(
    rows: list[dict[str, object]], tolerance: float = 0.05
) -> tuple[list[dict[str, float | int]], dict[str, int | None], int | None]:
    lookup = {
        (str(row["condition"]), int(row["step"]), str(row["layer"])): float(
            row["mean_update_to_weight_ratio"]
        )
        for row in rows
    }
    steps = sorted({int(row["step"]) for row in rows})
    layers = sorted({str(row["layer"]) for row in rows})
    gap_rows: list[dict[str, float | int]] = []
    per_layer: dict[str, list[tuple[int, float]]] = {layer: [] for layer in layers}
    for step in steps:
        layer_gaps: list[float] = []
        for layer in layers:
            warmup = lookup[("warmup", step, layer)]
            control = lookup[("no_warmup", step, layer)]
            gap = abs(warmup - control) / max(abs(control), 1e-30)
            per_layer[layer].append((step, gap))
            layer_gaps.append(gap)
        gap_rows.append({"step": step, "maximum_relative_gap": max(layer_gaps)})

    def first_permanent_below(series: list[tuple[int, float]]) -> int | None:
        violating_steps = [step for step, gap in series if gap >= tolerance]
        if not violating_steps:
            return series[0][0]
        candidate = max(violating_steps) + 1
        return candidate if candidate <= series[-1][0] else None

    per_layer_steps = {
        layer: first_permanent_below(series) for layer, series in per_layer.items()
    }
    overall_series = [
        (int(row["step"]), float(row["maximum_relative_gap"])) for row in gap_rows
    ]
    return gap_rows, per_layer_steps, first_permanent_below(overall_series)


def fit_log_lr_quadratic(rows: list[dict[str, object]]) -> dict[str, float | bool]:
    ordered = sorted(rows, key=lambda row: float(row["learning_rate"]))
    x = np.log(np.asarray([float(row["learning_rate"]) for row in ordered]))
    y = np.asarray([float(row["mean_validation_loss"]) for row in ordered])
    a, b, c = np.polyfit(x, y, deg=2)
    unconstrained_log_lr = -b / (2.0 * a) if a > 0 else math.nan
    if a <= 0:
        best = min(ordered, key=lambda row: float(row["mean_validation_loss"]))
        optimum_log_lr = math.log(float(best["learning_rate"]))
    else:
        optimum_log_lr = float(np.clip(unconstrained_log_lr, x.min(), x.max()))
    return {
        "learning_rate": float(math.exp(optimum_log_lr)),
        "fitted_validation_loss": float(np.polyval([a, b, c], optimum_log_lr)),
        "quadratic_curvature": float(a),
        "boundary_clamped": bool(
            a <= 0 or unconstrained_log_lr <= x.min() or unconstrained_log_lr >= x.max()
        ),
    }


def bootstrap_width_extrapolation(
    raw_rows: list[dict[str, object]], *, seed: int, samples: int = 5_000
) -> dict[str, object]:
    grouped: dict[tuple[int, float], list[float]] = {}
    for row in raw_rows:
        key = (int(row["width"]), float(row["learning_rate"]))
        grouped.setdefault(key, []).append(float(row["validation_loss"]))
    generator = np.random.default_rng(seed)
    widths = np.asarray([256.0, 512.0, 1024.0])
    predictions: list[float] = []
    exponents: list[float] = []
    fitted_by_width: dict[int, list[float]] = {256: [], 512: [], 1024: []}
    for _ in range(samples):
        optima: list[float] = []
        for width in (256, 512, 1024):
            fit_rows: list[dict[str, object]] = []
            rates = sorted(rate for candidate_width, rate in grouped if candidate_width == width)
            for rate in rates:
                losses = np.asarray(grouped[(width, rate)], dtype=np.float64)
                draw = generator.choice(losses, size=losses.size, replace=True)
                fit_rows.append(
                    {"learning_rate": rate, "mean_validation_loss": float(draw.mean())}
                )
            optimum = fit_log_lr_quadratic(fit_rows)["learning_rate"]
            optima.append(optimum)
            fitted_by_width[width].append(optimum)
        exponent, intercept = np.polyfit(np.log(widths), np.log(optima), deg=1)
        exponents.append(float(exponent))
        predictions.append(float(math.exp(intercept + exponent * math.log(4096.0))))

    def interval(values: list[float]) -> list[float]:
        low, high = np.percentile(np.asarray(values), [2.5, 97.5])
        return [float(low), float(high)]

    return {
        "bootstrap_samples": samples,
        "prediction_4096_95_percentile_interval": interval(predictions),
        "exponent_95_percentile_interval": interval(exponents),
        "fitted_minimum_95_percentile_interval": {
            str(width): interval(values) for width, values in fitted_by_width.items()
        },
    }


def plot_bias_correction(rows: list[dict[str, float | int]], path: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    corrected = [float(row["corrected_weight"]) for row in rows]
    uncorrected = [float(row["uncorrected_weight"]) for row in rows]
    factors = [float(row["correction_factor"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(steps, corrected, marker="o", label="Adam (bias corrected)")
    axes[0].plot(steps, uncorrected, marker="s", label="Bias correction disabled")
    axes[0].set(xlabel="Optimizer step", ylabel="Scalar weight", title="First 20 Adam steps")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(steps, factors, marker="o", color="#7a3db8")
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].fill_between(steps, 0.99, 1.01, color="green", alpha=0.12, label="within 1%")
    axes[1].set(
        xlabel="Optimizer step",
        ylabel="Corrected / uncorrected step scale",
        title="Bias-correction scale factor",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_layer_ratios(rows: list[dict[str, object]], warmup_steps: int, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)
    layers = sorted({str(row["layer"]) for row in rows})
    for layer in layers:
        selected = [row for row in rows if row["layer"] == layer]
        axes[0].plot(
            [int(row["step"]) for row in selected],
            [float(row["update_to_weight_ratio"]) for row in selected],
            label=layer,
            linewidth=1.5,
        )
        axes[1].plot(
            [int(row["step"]) for row in selected],
            [float(row["gradient_norm"]) for row in selected],
            label=layer,
            linewidth=1.3,
        )
    for axis in axes:
        axis.axvline(warmup_steps, color="black", linestyle="--", linewidth=1.2, label="warmup ends")
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0].set(
        ylabel=r"$\|\Delta\theta\|_2 / \|\theta\|_2$",
        title="Update-to-weight ratio for every parameter-bearing layer",
    )
    axes[1].set(xlabel="Optimizer step", ylabel="Gradient norm", title="Layer gradient norms")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_warmup_control(
    rows: list[dict[str, object]],
    gap_rows: list[dict[str, float | int]],
    warmup_steps: int,
    convergence_step: int | None,
    path: Path,
) -> None:
    layers = sorted({str(row["layer"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), sharex=True)
    flat_axes = list(axes.flat)
    colors = {"warmup": "#3366cc", "no_warmup": "#d95f02"}
    labels = {"warmup": "20-step warmup", "no_warmup": "no warmup"}
    for axis, layer in zip(flat_axes[:5], layers, strict=True):
        for condition in ("warmup", "no_warmup"):
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["layer"] == layer and row["condition"] == condition
                ],
                key=lambda row: int(row["step"]),
            )
            x = np.asarray([int(row["step"]) for row in selected])
            mean = np.asarray([float(row["mean_update_to_weight_ratio"]) for row in selected])
            std = np.asarray([float(row["std_update_to_weight_ratio"]) for row in selected])
            axis.plot(x, mean, color=colors[condition], label=labels[condition])
            axis.fill_between(
                x,
                np.maximum(mean - std, 1e-12),
                mean + std,
                color=colors[condition],
                alpha=0.14,
            )
        axis.axvline(warmup_steps, color="black", linestyle="--", linewidth=1)
        axis.axhline(1e-3, color="gray", linestyle=":", linewidth=1)
        axis.set_yscale("log")
        axis.set_title(layer)
        axis.grid(alpha=0.22, which="both")
    flat_axes[0].legend(fontsize=8)
    flat_axes[3].set_xlabel("Optimizer step")
    flat_axes[4].set_xlabel("Optimizer step")
    flat_axes[0].set_ylabel("Update / weight")
    flat_axes[3].set_ylabel("Update / weight")

    gap_axis = flat_axes[5]
    x = [int(row["step"]) for row in gap_rows]
    y = [float(row["maximum_relative_gap"]) for row in gap_rows]
    gap_axis.plot(x, y, color="#6a3d9a", label="max layer gap")
    gap_axis.axhline(0.05, color="gray", linestyle=":", linewidth=1, label="5% threshold")
    gap_axis.axvline(warmup_steps, color="black", linestyle="--", linewidth=1)
    if convergence_step is not None:
        gap_axis.axvline(
            convergence_step, color="green", linestyle="-.", linewidth=1.2, label="permanent <5%"
        )
    gap_axis.set_yscale("log")
    gap_axis.set_title("Maximum relative gap across layers")
    gap_axis.set_xlabel("Optimizer step")
    gap_axis.set_ylabel("Relative gap")
    gap_axis.grid(alpha=0.22, which="both")
    gap_axis.legend(fontsize=8)
    fig.suptitle("Matched warmup control (mean ± 1 SD across paired seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_schedule_comparison(
    final_runs: dict[str, dict[str, object]], config: TrainConfig, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {"cosine": "#3366cc", "wsd": "#d95f02"}
    for name, run in final_runs.items():
        trace = run["trace"]
        assert isinstance(trace, list)
        steps = np.asarray([int(row["step"]) for row in trace])
        mean = np.asarray([float(row["mean_validation_loss"]) for row in trace])
        std = np.asarray([float(row["std_validation_loss"]) for row in trace])
        axes[0].plot(
            steps,
            mean,
            marker="o",
            markersize=3,
            color=colors[name],
            label=(
                f"{name.upper()} (LR={float(run['peak_lr']):g}, "
                f"test={float(run['mean_test_loss']):.4f})"
            ),
        )
        axes[0].fill_between(steps, mean - std, mean + std, color=colors[name], alpha=0.14)
        multiplier_fn = cosine_multiplier if name == "cosine" else wsd_multiplier
        planned_steps = list(range(1, config.planned_steps + 1))
        axes[1].plot(
            planned_steps,
            [float(run["peak_lr"]) * multiplier_fn(step, config) for step in planned_steps],
            color=colors[name],
            label=name.upper(),
        )
    for axis in axes:
        axis.axvline(config.stop_step, color="black", linestyle="--", linewidth=1.1, label="stop at 200")
        axis.grid(alpha=0.25)
    axes[0].set(
        xlabel="Optimizer step",
        ylabel="Validation cross-entropy",
        title="Observed through stop step (mean ± 1 SD)",
    )
    axes[1].set(xlabel="Planned optimizer step", ylabel="Learning rate", title="300-step schedules")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_width_sweep(
    rows: list[dict[str, object]],
    minima: dict[str, object],
    fitted_minima: dict[str, object],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    colors = {256: "#1b9e77", 512: "#7570b3", 1024: "#e7298a"}
    for width in (256, 512, 1024):
        selected = sorted(
            [row for row in rows if int(row["width"]) == width],
            key=lambda row: float(row["learning_rate"]),
        )
        x = np.array([float(row["learning_rate"]) for row in selected])
        y = np.array([float(row["mean_validation_loss"]) for row in selected])
        err = np.array([float(row["std_validation_loss"]) for row in selected])
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, color=colors[width], label=f"width {width}")
        coefficients = np.polyfit(np.log(x), y, deg=2)
        smooth_x = np.geomspace(x.min(), x.max(), 200)
        ax.plot(
            smooth_x,
            np.polyval(coefficients, np.log(smooth_x)),
            color=colors[width],
            linestyle="--",
            alpha=0.65,
        )
        minimum = minima[str(width)]
        ax.scatter(
            [float(minimum["learning_rate"])],
            [float(minimum["mean_validation_loss"])],
            s=150,
            marker="*",
            color=colors[width],
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
        )
        fitted = fitted_minima[str(width)]
        ax.scatter(
            [float(fitted["learning_rate"])],
            [float(fitted["fitted_validation_loss"])],
            s=70,
            marker="D",
            facecolor="white",
            edgecolor=colors[width],
            linewidth=1.8,
            zorder=6,
        )
    ax.set_xscale("log", base=2)
    ax.set(
        xlabel="Peak learning rate",
        ylabel="Validation cross-entropy after sweep budget",
        title="Dense local learning-rate sweep (mean ± 1 SD, five seeds)",
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_all(device: torch.device, quick: bool = False) -> dict[str, object]:
    configure_reproducibility()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    adam_config = AdamConfig()
    gradients = [0.50, 0.40, 0.60, 0.45, 0.55]
    manual = manual_adam_rows(gradients, adam_config)
    pytorch = pytorch_adam_rows(gradients, adam_config)
    compared, maximum_adam_error = compare_adam_rows(manual, pytorch)
    write_csv(ARTIFACTS / "adam_hand_vs_pytorch.csv", compared)

    bias_rows, bias_negligible_step = bias_correction_experiment(gradients, adam_config)
    write_csv(ARTIFACTS / "bias_correction_first_20.csv", bias_rows)
    plot_bias_correction(bias_rows, ARTIFACTS / "bias_correction_first_20.png")
    bias_sensitivity = bias_threshold_sensitivity(
        adam_config, tolerances=(0.10, 0.05, 0.01, 0.001)
    )
    write_csv(ARTIFACTS / "bias_correction_threshold_sensitivity.csv", bias_sensitivity)

    train_config = TrainConfig()
    data = make_dataset(train_config, device)

    ratio_steps = 30 if quick else 60
    ratio_run = train_run(
        width=256,
        peak_lr=0.00015,
        steps=ratio_steps,
        schedule=lambda step: constant_after_warmup_multiplier(step, train_config.warmup_steps),
        data=data,
        config=train_config,
        device=device,
        init_seed=SEED + 100,
        batch_seed=SEED + 200,
        log_every=5,
        track_layer_ratios=True,
    )
    ratio_rows = ratio_run["layer_ratios"]
    assert isinstance(ratio_rows, list)
    write_csv(ARTIFACTS / "layer_update_weight_ratios.csv", ratio_rows)
    plot_layer_ratios(ratio_rows, train_config.warmup_steps, ARTIFACTS / "layer_update_weight_ratios.png")

    control_steps = 40 if quick else 200
    control_seeds = [SEED + 211] if quick else [SEED + 211 + index for index in range(5)]
    control_raw_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(control_seeds):
        for condition in ("warmup", "no_warmup"):
            schedule = (
                (lambda step: constant_after_warmup_multiplier(step, train_config.warmup_steps))
                if condition == "warmup"
                else (lambda step: 1.0)
            )
            run = train_run(
                width=256,
                peak_lr=0.00015,
                steps=control_steps,
                schedule=schedule,
                data=data,
                config=train_config,
                device=device,
                init_seed=seed,
                batch_seed=SEED + 250 + seed_index,
                log_every=control_steps,
                track_layer_ratios=True,
            )
            rows = run["layer_ratios"]
            assert isinstance(rows, list)
            for row in rows:
                control_raw_rows.append({"condition": condition, "seed": seed, **row})
    control_summary_rows = aggregate_layer_control(control_raw_rows)
    gap_rows, per_layer_convergence, overall_convergence = warmup_control_gaps(
        control_summary_rows, tolerance=0.05
    )
    write_csv(ARTIFACTS / "warmup_control_raw.csv", control_raw_rows)
    write_csv(ARTIFACTS / "warmup_control_summary.csv", control_summary_rows)
    write_csv(ARTIFACTS / "warmup_control_gap.csv", gap_rows)
    plot_warmup_control(
        control_summary_rows,
        gap_rows,
        train_config.warmup_steps,
        overall_convergence,
        ARTIFACTS / "warmup_matched_control.png",
    )

    schedule_candidates = [0.0001875, 0.000375, 0.00075, 0.0015, 0.003, 0.006, 0.012]
    schedule_steps = 80 if quick else train_config.stop_step
    schedule_tuning_raw_rows: list[dict[str, object]] = []
    schedule_tuning_rows: list[dict[str, object]] = []
    tuned_runs: dict[str, list[dict[str, object]]] = {"cosine": [], "wsd": []}
    schedule_tuning_seeds = [SEED + 301] if quick else [SEED + 301, SEED + 302, SEED + 303]
    for schedule_name in ("cosine", "wsd"):
        fn = cosine_multiplier if schedule_name == "cosine" else wsd_multiplier
        for learning_rate in schedule_candidates:
            losses: list[float] = []
            for seed in schedule_tuning_seeds:
                run = train_run(
                    width=512,
                    peak_lr=learning_rate,
                    steps=schedule_steps,
                    schedule=lambda step, chosen=fn: chosen(step, train_config),
                    data=data,
                    config=train_config,
                    device=device,
                    init_seed=seed,
                    batch_seed=SEED + 400,
                    log_every=max(10, schedule_steps),
                )
                validation_loss = float(run["validation_loss"])
                losses.append(validation_loss)
                schedule_tuning_raw_rows.append(
                    {
                        "schedule": schedule_name,
                        "peak_learning_rate": learning_rate,
                        "seed": seed,
                        "tuning_steps": schedule_steps,
                        "validation_loss": validation_loss,
                    }
                )
            mean_loss = float(np.mean(losses))
            aggregate = {"peak_lr": learning_rate, "validation_loss": mean_loss}
            tuned_runs[schedule_name].append(aggregate)
            schedule_tuning_rows.append(
                {
                    "schedule": schedule_name,
                    "peak_learning_rate": learning_rate,
                    "tuning_steps": schedule_steps,
                    "seeds": len(losses),
                    "mean_validation_loss": mean_loss,
                    "std_validation_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
                }
            )
    write_csv(ARTIFACTS / "schedule_tuning_raw.csv", schedule_tuning_raw_rows)
    write_csv(ARTIFACTS / "schedule_tuning.csv", schedule_tuning_rows)

    selected_lr = {
        name: float(min(runs, key=lambda run: float(run["validation_loss"]))["peak_lr"])
        for name, runs in tuned_runs.items()
    }
    final_seeds = (
        [SEED + 351, SEED + 352]
        if quick
        else [SEED + 351 + index for index in range(5)]
    )
    final_runs_by_schedule: dict[str, list[dict[str, object]]] = {"cosine": [], "wsd": []}
    final_pair_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(final_seeds):
        pair: dict[str, dict[str, object]] = {}
        pair_batch_seed = SEED + 450 + seed_index
        for schedule_name in ("cosine", "wsd"):
            fn = cosine_multiplier if schedule_name == "cosine" else wsd_multiplier
            final_run = train_run(
                width=512,
                peak_lr=selected_lr[schedule_name],
                steps=train_config.stop_step,
                schedule=lambda step, chosen=fn: chosen(step, train_config),
                data=data,
                config=train_config,
                device=device,
                init_seed=seed,
                batch_seed=pair_batch_seed,
                log_every=10,
                return_state=seed_index == 0,
            )
            if seed_index == 0:
                state = final_run.pop("model_state")
                torch.save(
                    {
                        "model_state": state,
                        "width": 512,
                        "schedule": schedule_name,
                        "peak_lr": selected_lr[schedule_name],
                        "stopped_at_step": train_config.stop_step,
                        "planned_steps": train_config.planned_steps,
                        "paired_seed": seed,
                        "initial_state_hash": final_run["initial_state_hash"],
                        "batch_sequence_hash": final_run["batch_sequence_hash"],
                    },
                    ARTIFACTS / f"{schedule_name}_step_200.pt",
                )
            final_runs_by_schedule[schedule_name].append(final_run)
            pair[schedule_name] = final_run
        assert pair["cosine"]["initial_state_hash"] == pair["wsd"]["initial_state_hash"]
        assert pair["cosine"]["batch_sequence_hash"] == pair["wsd"]["batch_sequence_hash"]
        final_pair_rows.append(
            {
                "seed": seed,
                "initial_state_hash": pair["cosine"]["initial_state_hash"],
                "batch_sequence_hash": pair["cosine"]["batch_sequence_hash"],
                "cosine_validation_loss": pair["cosine"]["validation_loss"],
                "wsd_validation_loss": pair["wsd"]["validation_loss"],
                "cosine_test_loss": pair["cosine"]["test_loss"],
                "wsd_test_loss": pair["wsd"]["test_loss"],
                "paired_test_difference_cosine_minus_wsd": (
                    float(pair["cosine"]["test_loss"]) - float(pair["wsd"]["test_loss"])
                ),
            }
        )
    write_csv(ARTIFACTS / "schedule_final_paired.csv", final_pair_rows)

    final_aggregate_rows: list[dict[str, object]] = []
    final_plot_runs: dict[str, dict[str, object]] = {}
    for schedule_index, schedule_name in enumerate(("cosine", "wsd")):
        runs = final_runs_by_schedule[schedule_name]
        validation_losses = [float(run["validation_loss"]) for run in runs]
        test_losses = [float(run["test_loss"]) for run in runs]
        test_ci = bootstrap_mean_ci(test_losses, seed=SEED + 470 + schedule_index)
        final_aggregate_rows.append(
            {
                "schedule": schedule_name,
                "peak_learning_rate": selected_lr[schedule_name],
                "final_seeds": len(runs),
                "mean_validation_loss": float(np.mean(validation_losses)),
                "std_validation_loss": float(np.std(validation_losses, ddof=1)),
                "mean_test_loss": float(np.mean(test_losses)),
                "std_test_loss": float(np.std(test_losses, ddof=1)),
                "test_loss_bootstrap_95_low": test_ci[0],
                "test_loss_bootstrap_95_high": test_ci[1],
            }
        )
        final_plot_runs[schedule_name] = {
            "peak_lr": selected_lr[schedule_name],
            "mean_test_loss": float(np.mean(test_losses)),
            "trace": aggregate_traces(runs),
        }
    write_csv(ARTIFACTS / "schedule_final_summary.csv", final_aggregate_rows)
    paired_differences = [
        float(row["paired_test_difference_cosine_minus_wsd"]) for row in final_pair_rows
    ]
    paired_difference_ci = bootstrap_mean_ci(paired_differences, seed=SEED + 480)
    plot_schedule_comparison(
        final_plot_runs, train_config, ARTIFACTS / "cosine_vs_wsd_stop_200.png"
    )

    prior_centers = {256: 0.0015, 512: 0.00075, 1024: 0.000375}
    local_offsets = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
    sweep_learning_rates = {
        width: [center * (2.0**offset) for offset in local_offsets]
        for width, center in prior_centers.items()
    }
    sweep_steps = 40 if quick else 80
    sweep_seeds = (
        [SEED + 501, SEED + 502]
        if quick
        else [SEED + 501 + index for index in range(5)]
    )
    raw_sweep_rows: list[dict[str, object]] = []
    aggregate_sweep_rows: list[dict[str, object]] = []
    for width in (256, 512, 1024):
        for learning_rate in sweep_learning_rates[width]:
            losses: list[float] = []
            for seed in sweep_seeds:
                run = train_run(
                    width=width,
                    peak_lr=learning_rate,
                    steps=sweep_steps,
                    schedule=lambda step: constant_after_warmup_multiplier(step, 10),
                    data=data,
                    config=train_config,
                    device=device,
                    init_seed=seed,
                    batch_seed=SEED + 600,
                    log_every=sweep_steps,
                )
                validation_loss = float(run["validation_loss"])
                losses.append(validation_loss)
                raw_sweep_rows.append(
                    {
                        "width": width,
                        "learning_rate": learning_rate,
                        "seed": seed,
                        "steps": sweep_steps,
                        "validation_loss": validation_loss,
                    }
                )
            aggregate_sweep_rows.append(
                {
                    "width": width,
                    "learning_rate": learning_rate,
                    "steps": sweep_steps,
                    "seeds": len(losses),
                    "mean_validation_loss": float(np.mean(losses)),
                    "std_validation_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
                }
            )
    write_csv(ARTIFACTS / "width_lr_sweep_raw.csv", raw_sweep_rows)
    write_csv(ARTIFACTS / "width_lr_sweep_summary.csv", aggregate_sweep_rows)
    minima: dict[str, object] = {}
    fitted_minima: dict[str, object] = {}
    for width in (256, 512, 1024):
        width_rows = [row for row in aggregate_sweep_rows if int(row["width"]) == width]
        chosen = min(
            width_rows,
            key=lambda row: float(row["mean_validation_loss"]),
        )
        minima[str(width)] = chosen
        fitted_minima[str(width)] = fit_log_lr_quadratic(width_rows)

    widths = np.array([256.0, 512.0, 1024.0])
    optimum_lrs = np.array(
        [float(fitted_minima[str(int(width))]["learning_rate"]) for width in widths]
    )
    exponent, intercept = np.polyfit(np.log(widths), np.log(optimum_lrs), deg=1)
    predicted_4096 = float(math.exp(intercept + exponent * math.log(4096.0)))
    tested_learning_rates = sorted(
        {rate for rates in sweep_learning_rates.values() for rate in rates}
    )
    recommended_4096 = min(
        tested_learning_rates, key=lambda lr: abs(math.log(lr / predicted_4096))
    )
    width_bootstrap = bootstrap_width_extrapolation(
        raw_sweep_rows, seed=SEED + 700, samples=500 if quick else 5_000
    )
    plot_width_sweep(
        aggregate_sweep_rows,
        minima,
        fitted_minima,
        ARTIFACTS / "width_lr_sweep.png",
    )

    final_summary_by_schedule = {
        str(row["schedule"]): row for row in final_aggregate_rows
    }
    dataset_digest = hash_named_tensors(data.items())
    summary = {
        "runtime": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "elapsed_seconds": time.perf_counter() - started,
            "quick_mode": quick,
        },
        "adam": {
            "config": asdict(adam_config),
            "gradients": gradients,
            "maximum_absolute_error_vs_pytorch": maximum_adam_error,
        },
        "bias_correction": {
            "plotted_steps": 20,
            "negligible_definition": "correction changes instantaneous update scale by <1% now and thereafter",
            "negligible_from_step": bias_negligible_step,
            "threshold_sensitivity": bias_sensitivity,
        },
        "warmup": {
            "warmup_steps": train_config.warmup_steps,
            "first_step_at_full_peak_lr": train_config.warmup_steps,
            "logged_steps": ratio_steps,
            "width": 256,
            "peak_lr": 0.00015,
            "matched_control_steps": control_steps,
            "matched_control_seeds": control_seeds,
            "causal_convergence_definition": (
                "warmup/no-warmup mean ratio gap stays below 5% for every remaining observed step"
            ),
            "per_layer_permanent_5_percent_step": per_layer_convergence,
            "all_layers_permanent_5_percent_step": overall_convergence,
        },
        "schedule_comparison": {
            "planned_steps": train_config.planned_steps,
            "stopped_at_step": train_config.stop_step,
            "width": 512,
            "tuning_candidates": schedule_candidates,
            "tuning_steps": schedule_steps,
            "tuning_seeds": schedule_tuning_seeds,
            "final_seeds": final_seeds,
            "selected_peak_lr": selected_lr,
            "held_out_test_loss_at_stop": {
                name: {
                    "mean": float(row["mean_test_loss"]),
                    "std": float(row["std_test_loss"]),
                    "bootstrap_95_interval": [
                        float(row["test_loss_bootstrap_95_low"]),
                        float(row["test_loss_bootstrap_95_high"]),
                    ],
                }
                for name, row in final_summary_by_schedule.items()
            },
            "paired_difference_cosine_minus_wsd": {
                "mean": float(np.mean(paired_differences)),
                "std": float(np.std(paired_differences, ddof=1)),
                "bootstrap_95_interval": list(paired_difference_ci),
            },
            "keep": min(
                final_summary_by_schedule,
                key=lambda name: float(final_summary_by_schedule[name]["mean_test_loss"]),
            ),
        },
        "width_lr_sweep": {
            "steps": sweep_steps,
            "seeds": sweep_seeds,
            "learning_rates_by_width": sweep_learning_rates,
            "discrete_minima": minima,
            "quadratic_fitted_minima": fitted_minima,
            "log_log_fit_exponent": float(exponent),
            "continuous_extrapolation_at_4096": predicted_4096,
            "recommended_grid_lr_at_4096": recommended_4096,
            "bootstrap_uncertainty": width_bootstrap,
        },
        "reproducibility": {
            "dataset_hash": dataset_digest,
            "dataset_content_hash": dataset_digest,
            "dataset_snapshot": "data/synthetic_teacher_v1.npz",
            "dataset_snapshot_file_sha256": sha256_file(DATASET_SNAPSHOT),
            "final_pair_initial_state_hashes": [
                row["initial_state_hash"] for row in final_pair_rows
            ],
            "final_pair_batch_sequence_hashes": [
                row["batch_sequence_hash"] for row in final_pair_rows
            ],
        },
        "training_optimizer": {
            "name": "AdamW",
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": train_config.weight_decay,
            "decay_exclusions": "biases and normalization parameters",
        },
        "train_config": asdict(train_config),
    }
    save_json(ARTIFACTS / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--quick", action="store_true", help="smoke-test with reduced budgets")
    parser.add_argument(
        "--write-data-snapshot",
        action="store_true",
        help="maintainer-only: recreate the canonical bundled dataset",
    )
    parser.add_argument(
        "--verify-data",
        action="store_true",
        help="validate the bundled dataset and print its content hash",
    )
    args = parser.parse_args()
    if args.write_data_snapshot:
        print(write_dataset_snapshot())
        return
    if args.verify_data:
        data = make_dataset(TrainConfig(), torch.device("cpu"))
        print(hash_named_tensors(data.items()))
        return
    summary = run_all(choose_device(args.device), quick=args.quick)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
