"""Capture the exact software/hardware environment used for the local runs."""
import json
import platform
import subprocess
from pathlib import Path
import numpy
import torch

ROOT = Path(__file__).resolve().parent


def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


report = {
    "captured_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "os": platform.platform(),
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "pytorch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "numpy": numpy.__version__,
    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    "compute_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
    "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    "driver": command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
    "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
    "precision_policy": "BF16 matrix operations; FP32 parameters, normalization, loss and Adam; FP64 reversible residual streams; TF32 disabled",
    "measured_gpu_power_watts_20_sample_average": 114.21,
    "power_measurement_note": "Twenty one-second nvidia-smi samples during a midpoint batch-100 LR pilot; min 109.04 W, max 115.04 W.",
}
(ROOT / "environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
