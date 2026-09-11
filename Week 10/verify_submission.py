"""Fast, read-only checks for the saved submission artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
results = json.loads((ROOT / "artifacts" / "results.json").read_text(encoding="utf-8"))
ledger = json.loads((ROOT / "artifacts" / "shape_ledger.json").read_text(encoding="utf-8"))
notebook = json.loads((ROOT / "small_model_truth.ipynb").read_text(encoding="utf-8"))
with (ROOT / "artifacts" / "training_log.csv").open(encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

assert notebook["nbformat"] == 4 and len(notebook["cells"]) >= 15
assert len(rows) == results["artifacts"]["training_steps"] == 40
assert len(ledger) == results["artifacts"]["shape_ledger_entries"] >= 240
assert all({"shape", "dtype", "dims"} <= item.keys() for item in ledger.values())
assert any(name.startswith("microbatch_0_short.") for name in ledger)
assert any(name.startswith("microbatch_1_long.") for name in ledger)
assert results["gradient_check"]["absolute_error"] < 1e-8
assert results["accumulation"]["final_absolute_gap"] > 0
assert results["grad_norm_leads_loss"]["grad_norm_relative_change"] > 0.5
assert results["grad_norm_leads_loss"]["loss_relative_change"] < 0.003
assert results["mfu"]["available"] and 0 < results["mfu"]["mfu_percent"] < 40
assert results["mfu"]["standard_6n_flops_per_token"] == 6 * results["model"]["parameter_count"]
assert results["float_representations"]["fp32"]["hex"] == "0x3DCCCCCD"
assert results["float_representations"]["bf16"]["hex"] == "0x3DCD"
assert results["float_representations"]["fp8_e4m3"]["hex"] == "0x1D"
for filename in ("accumulation_curves.svg", "grad_norms.svg"):
    ET.parse(ROOT / "artifacts" / filename)

print("PASS: notebook, 40 log rows, shape ledger, plots, gradient check, MFU, and bit patterns")
