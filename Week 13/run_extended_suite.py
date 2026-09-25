"""Reproducible extension: LR tuning, tuned max-batch runs, and seed replication."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "experiment_precise.py"
SEED2 = 20260920
PILOT_TOKENS = 5_000_000
CANDIDATES = (4.5e-4, 6e-4, 7.5e-4, 9e-4, 1.2e-3, 1.5e-3)
MAXIMUM = {"midpoint": 100, "euler": 102}


def tag(value):
    return str(int(round(value * 1_000_000)))


def call(args):
    print("+", sys.executable, "-B", RUNNER.name, *map(str, args), flush=True)
    subprocess.run([sys.executable, "-B", str(RUNNER), *map(str, args)], cwd=ROOT, check=True)


def train(name, variant, batch, tokens, seed, peak_lr):
    folder = ROOT / name
    if (folder / "result.json").exists():
        return
    args = ["train", "--variant", variant, "--batch", batch, "--tokens", tokens,
            "--name", name, "--seed", seed, "--peak-lr", peak_lr, "--min-lr", peak_lr / 10]
    if (folder / "last.pt").exists():
        args.append("--resume")
    call(args)


def audit(name, seed):
    if (ROOT / name / "gradient_audit.json").exists():
        return
    call(["audit", "--name", name, "--seed", seed])


def main():
    tuning = {"criterion": "Lowest fixed-probe loss after 5M supervised targets; same maximum batch and seed.",
              "pilot_tokens": PILOT_TOKENS, "variants": {}}
    for variant, batch in MAXIMUM.items():
        rows = []
        for peak in CANDIDATES:
            name = f"lrpilot_{variant}_b{batch}_lr{tag(peak)}"
            train(name, variant, batch, PILOT_TOKENS, 20260919, peak)
            result = json.loads((ROOT / name / "result.json").read_text())
            rows.append({"name": name, "peak_lr": peak, "min_lr": peak / 10,
                         "loss": result["final_validation"]["loss"],
                         "perplexity": result["final_validation"]["perplexity"]})
        best = min(rows, key=lambda row: row["loss"])
        tuning["variants"][variant] = {"batch": batch, "candidates": rows, "selected": best}
    (ROOT / "lr_tuning.json").write_text(json.dumps(tuning, indent=2) + "\n")

    for variant, details in tuning["variants"].items():
        batch = details["batch"]
        peak = details["selected"]["peak_lr"]
        if peak == 6e-4:
            name = f"{variant}_precise_b{batch}"
        else:
            name = f"{variant}_precise_b{batch}_lr{tag(peak)}_tuned"
            train(name, variant, batch, 50_000_000, 20260919, peak)
            audit(name, 20260919)
        details["full_run"] = name
    (ROOT / "lr_tuning.json").write_text(json.dumps(tuning, indent=2) + "\n")

    for variant in ("baseline", "midpoint", "euler"):
        name = f"{variant}_precise_b32_seed{SEED2}"
        train(name, variant, 32, 50_000_000, SEED2, 6e-4)
        if variant != "baseline":
            audit(name, SEED2)

    (ROOT / "extended_suite_status.json").write_text(json.dumps({
        "state": "COMPLETE", "additional_seed": SEED2,
        "lr_tuning": "lr_tuning.json",
        "runs": [
            "baseline_precise_b32_seed20260920", "midpoint_precise_b32_seed20260920",
            "euler_precise_b32_seed20260920",
            *[d["full_run"] for d in tuning["variants"].values()],
        ],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
