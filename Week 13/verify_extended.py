"""Verify the original, tuned, and replicated 50M-token experiment artifacts."""
import hashlib
import json
import math
from pathlib import Path

import torch

from model_precise import Decoder


ROOT = Path(__file__).resolve().parent
RUNS = [
    "baseline_precise_b32",
    "midpoint_precise_b32",
    "midpoint_precise_b100",
    "euler_precise_b32",
    "euler_precise_b102",
    "midpoint_precise_b100_lr900_tuned",
    "baseline_precise_b32_seed20260920",
    "midpoint_precise_b32_seed20260920",
    "euler_precise_b32_seed20260920",
]


def model_initial_hash(variant: str, seed: int) -> str:
    torch.manual_seed(seed)
    model = Decoder(variant)
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    torch.set_num_threads(4)
    checks = []
    initialization_groups = {}

    for name in RUNS:
        path = ROOT / name
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        assert result["state"] == "COMPLETE"
        assert result["training_tokens"] == 50_000_000
        assert result["physical_tokens"] == 53_689_856
        assert result["final_validation"]["tokens"] == 5_049_456
        assert result["updates"] == math.ceil(104_863 / result["batch"])

        final_path = path / "final.pt"
        digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
        assert digest == result["final_checkpoint_sha256"]
        state = torch.load(final_path, map_location="cpu", weights_only=True)
        assert state["signature"] == result["signature"]
        assert state["consumed_tokens"] == 50_000_000

        model = Decoder(state["config"]["variant"])
        model.load_state_dict(state["model"])
        assert sum(p.numel() for p in model.parameters()) == 20_166_912
        assert all(torch.isfinite(p).all().item() for p in model.parameters())

        last = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        assert last["cursor"] == 104_863
        assert last["consumed"] == 50_000_000
        assert all(torch.equal(tensor, last["model"][key]) for key, tensor in state["model"].items())

        if result["variant"] != "baseline":
            audit = json.loads((path / "gradient_audit.json").read_text(encoding="utf-8"))
            assert audit["status"] == "PASS"
            assert audit["gradient_relative_l2"] < 1e-4
            assert audit["reconstruction_relative_l2"] < 1e-4

        seed = int(state["config"]["seed"])
        initialization_groups.setdefault(seed, []).append(model_initial_hash(result["variant"], seed))
        checks.append({
            "run": name,
            "status": "PASS",
            "seed": seed,
            "sha256": digest,
        })
        del model, last, state

    initial_hashes = {}
    for seed, hashes in initialization_groups.items():
        assert len(set(hashes)) == 1
        initial_hashes[str(seed)] = hashes[0]

    report = {
        "status": "PASS",
        "checks": checks,
        "identical_initial_parameters_within_seed": initial_hashes,
        "notes": [
            "Nine complete 50M-token checkpoints verified.",
            "Initialization equality is asserted among model variants within each seed.",
            "Euler maximum-batch LR tuning selected the original 6e-4 run, so it is not duplicated.",
        ],
    }
    (ROOT / "verification_extended.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
