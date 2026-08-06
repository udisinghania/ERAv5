from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.runtime import DEVICE_ENVIRONMENT_VARIABLE, resolve_execution_device  # noqa: E402


def main() -> int:
    config = json.loads((ROOT / "configs" / "training_v1.json").read_text(encoding="utf-8"))
    device, descriptor = resolve_execution_device(
        config, os.environ.get(DEVICE_ENVIRONMENT_VARIABLE)
    )
    probe = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    if float((probe * probe).sum().item()) != 5.0:
        raise RuntimeError("selected training device failed the arithmetic preflight")
    print(
        json.dumps(
            {
                "status": "PASS",
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                **descriptor,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
