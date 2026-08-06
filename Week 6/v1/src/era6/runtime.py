from __future__ import annotations

import os
from typing import Any

import torch


DEVICE_ENVIRONMENT_VARIABLE = "ERA6_DEVICE"


def choose_device_type(
    requested: str,
    *,
    cuda_available: bool,
    supported_devices: list[str] | tuple[str, ...] = ("cuda", "cpu"),
) -> str:
    """Resolve auto/cuda/cpu without consulting ambient hardware in tests."""
    normalized = requested.strip().lower()
    supported = tuple(str(value).lower() for value in supported_devices)
    if normalized not in {"auto", *supported}:
        raise ValueError(
            f"unsupported device request {requested!r}; expected auto or one of {supported}"
        )
    if normalized == "auto":
        normalized = "cuda" if cuda_available and "cuda" in supported else "cpu"
    if normalized == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if normalized not in supported:
        raise RuntimeError(f"selected device {normalized!r} is not enabled by the runtime contract")
    return normalized


def resolve_execution_device(
    config: dict[str, Any], requested: str | None = None
) -> tuple[torch.device, dict[str, Any]]:
    runtime = config["runtime"]
    requested = requested or os.environ.get(
        DEVICE_ENVIRONMENT_VARIABLE, runtime.get("default_device", "auto")
    )
    supported_versions = tuple(str(value) for value in runtime["supported_torch_major_minor"])
    if not torch.__version__.startswith(supported_versions):
        raise RuntimeError(
            f"unsupported torch version {torch.__version__}; expected prefix {supported_versions}"
        )
    device_type = choose_device_type(
        requested,
        cuda_available=torch.cuda.is_available(),
        supported_devices=runtime["supported_devices"],
    )
    device = torch.device(device_type)
    device_name = torch.cuda.get_device_name(0) if device_type == "cuda" else "CPU"
    descriptor = {
        "request": str(requested).lower(),
        "device_type": device_type,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "fallback_used": str(requested).lower() == "auto" and device_type == "cpu",
    }
    return device, descriptor


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_accelerator_memory(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    cuda_states = state.get("cuda", [])
    if device.type == "cuda":
        if not cuda_states:
            raise RuntimeError("CUDA resume checkpoint does not contain CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_states)

