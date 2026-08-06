from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import canonical_json_bytes, read_jsonl_gz, sha256_bytes, sha256_file  # noqa: E402


def main() -> int:
    curriculum_root = ROOT / "data" / "curriculum_v1"
    schedule_path = ROOT / "artifacts" / "schedule_v1" / "schedule.json"
    curriculum = json.loads((curriculum_root / "curriculum_report.json").read_text(encoding="utf-8"))
    schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    claimed_hash = schedule_payload.pop("schedule_hash")
    actual_hash = f"sha256:{sha256_bytes(canonical_json_bytes(schedule_payload))}"
    if claimed_hash != actual_hash:
        raise AssertionError("schedule hash mismatch")
    schedule_payload["schedule_hash"] = claimed_hash
    overlay_path = ROOT / curriculum["overlay"]["path"]
    if sha256_file(overlay_path) != curriculum["overlay"]["compressed_sha256"]:
        raise AssertionError("curriculum overlay hash mismatch")
    rows = list(read_jsonl_gz(overlay_path))
    if len(rows) != curriculum["records"] or len({row["record_id"] for row in rows}) != len(rows):
        raise AssertionError("curriculum record count or uniqueness failure")
    bands = {"B0", "B1", "B2", "B3", "B4", "B5"}
    if {row["difficulty_band"] for row in rows} != bands:
        raise AssertionError("not all difficulty bands are represented")
    invalid_tiers = [
        row["record_id"]
        for row in rows
        if (row["lane"] == "indic") != (row["indic_tier"] is not None)
    ]
    if invalid_tiers:
        raise AssertionError("Indic tier mapping is incomplete or leaked to another lane")

    schedule = schedule_payload["schedule"]
    if sum(stage["target_loss_tokens"] for stage in schedule["stages"]) != schedule["total_loss_token_budget"]:
        raise AssertionError("stage targets do not sum to total budget")
    for stage in schedule["stages"]:
        if sum(stage["lane_targets"].values()) != stage["target_loss_tokens"]:
            raise AssertionError(f"lane targets do not sum in stage {stage['name']}")
        if stage["indic_tier_targets"] and sum(stage["indic_tier_targets"].values()) != stage["lane_targets"]["indic"]:
            raise AssertionError(f"Indic tier targets do not sum in stage {stage['name']}")
        for lane, target in stage["lane_targets"].items():
            if target > stage["lane_eligible_supply"][lane]:
                raise AssertionError(f"infeasible lane target {stage['name']} {lane}")
        for tier, target in stage["indic_tier_targets"].items():
            if target > stage["indic_tier_eligible_supply"][tier]:
                raise AssertionError(f"infeasible Indic target {stage['name']} {tier}")
    if not all(result["passed"] for result in schedule["protected_floors"].values()):
        raise AssertionError("protected floor failure")
    if sum(schedule["pre_anneal_indic_tier_targets"].values()) != schedule["pre_anneal_lane_targets"]["indic"]:
        raise AssertionError("pre-anneal Indic tier totals do not match the Indic lane")
    if any(stage["permission"] == "anneal" for stage in schedule["stages"][:-1]):
        raise AssertionError("anneal permission appears before final stage")
    if schedule["stages"][-1]["permission"] != "anneal":
        raise AssertionError("final stage is not isolated anneal")
    print(
        json.dumps(
            {
                "status": "PASS",
                "schedule_hash": claimed_hash,
                "curriculum_records": len(rows),
                "total_loss_token_budget": schedule["total_loss_token_budget"],
                "stages": len(schedule["stages"]),
                "protected_floors": schedule["protected_floors"],
                "anneal_isolated": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
