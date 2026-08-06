from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.batching import select_opus_batches  # noqa: E402
from era6.canonical import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    read_jsonl_gz,
    sha256_bytes,
    sha256_file,
    write_jsonl_gz,
)


OUTPUT_ROOT = ROOT / "data" / "batches_v1"


def main() -> int:
    config_path = ROOT / "configs" / "batching_v1.json"
    packing_report_path = ROOT / "data" / "packed_v1" / "packing_report.json"
    schedule_path = ROOT / "artifacts" / "schedule_v1" / "schedule.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    packing_report = json.loads(packing_report_path.read_text(encoding="utf-8"))
    schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule = schedule_payload["schedule"]
    if config["candidate_microbatches"] != schedule["opus"]["candidate_microbatches"]:
        raise RuntimeError("batch config disagrees with scheduled OPUS candidate count")
    if config["proxy_version"] != schedule["opus"]["proxy_version"]:
        raise RuntimeError("batch config disagrees with scheduled OPUS proxy version")
    sequences = list(read_jsonl_gz(ROOT / packing_report["paths"]["sequences"]))
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sequences:
        by_stage[row["stage"]].append(row)

    batches: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    stage_summaries = []
    global_batch_index = 0
    global_decision_index = 0
    for stage in schedule["stages"]:
        stage_batches, stage_decisions = select_opus_batches(
            by_stage[stage["name"]], stage_schedule=stage, config=config
        )
        for batch in stage_batches:
            batch["batch_index"] = global_batch_index
            batch["global_opus_decision_index"] = global_decision_index
            global_batch_index += 1
            global_decision_index += 1
        for decision, batch in zip(stage_decisions, stage_batches, strict=True):
            decision["global_decision_index"] = batch["global_opus_decision_index"]
            decision["selected_batch_index"] = batch["batch_index"]
        batches.extend(stage_batches)
        decisions.extend(stage_decisions)
        stage_summaries.append(
            {
                "stage": stage["name"],
                "sequence_length": stage["sequence_length"],
                "sequences": len(by_stage[stage["name"]]),
                "microbatch_capacity_sequences": config["microbatch_physical_token_budget"]
                // stage["sequence_length"],
                "microbatches": len(stage_batches),
                "loss_bearing_tokens": sum(row["loss_bearing_tokens"] for row in stage_batches),
                "zero_loss_sequences": sum(row["zero_loss_sequences"] for row in stage_batches),
            }
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    batches_path = OUTPUT_ROOT / "batches.jsonl.gz"
    decisions_path = OUTPUT_ROOT / "opus_decisions.jsonl.gz"
    batch_stats = write_jsonl_gz(batches_path, batches)
    decision_stats = write_jsonl_gz(decisions_path, decisions)
    component_hashes = {
        "batches": sha256_file(batches_path),
        "opus_decisions": sha256_file(decisions_path),
    }
    batch_plan_hash = f"sha256:{sha256_bytes(canonical_json_bytes(component_hashes))}"
    lane_loss: Counter[tuple[str, str]] = Counter()
    tier_loss: Counter[tuple[str, str]] = Counter()
    selected_candidate_ids: Counter[int] = Counter()
    candidate_counts: Counter[int] = Counter()
    candidate_outcomes: Counter[str] = Counter()
    decision_outcomes: Counter[str] = Counter()
    for batch in batches:
        for lane, value in batch["lane_loss_tokens"].items():
            lane_loss[(batch["stage"], lane)] += value
        for tier, value in batch["indic_tier_loss_tokens"].items():
            tier_loss[(batch["stage"], tier)] += value
    for decision in decisions:
        selected_candidate_ids[int(decision["selected_candidate_id"])] += 1
        candidate_counts[int(decision["candidate_count"])] += 1
        decision_outcomes[str(decision["decision_outcome"])] += 1
        candidate_outcomes.update(str(row["outcome"]) for row in decision["candidates"])
    report = {
        "schema_version": 1,
        "status": "FROZEN",
        "batch_plan_hash": batch_plan_hash,
        "packing_hash": packing_report["packing_hash"],
        "schedule_hash": schedule_payload["schedule_hash"],
        "batching_config_sha256": sha256_file(config_path),
        "packing_report_sha256": sha256_file(packing_report_path),
        "schedule_file_sha256": sha256_file(schedule_path),
        "proxy_version": config["proxy_version"],
        "candidate_microbatches_maximum": config["candidate_microbatches"],
        "microbatch_physical_token_budget": config["microbatch_physical_token_budget"],
        "component_hashes": component_hashes,
        "paths": {
            "batches": batches_path.relative_to(ROOT).as_posix(),
            "opus_decisions": decisions_path.relative_to(ROOT).as_posix(),
        },
        "sequences": sum(row["sequence_count"] for row in batches),
        "unique_sequences": len(
            {index for row in batches for index in row["sequence_indices"]}
        ),
        "microbatches": len(batches),
        "opus_decisions": len(decisions),
        "physical_tokens": sum(row["physical_tokens"] for row in batches),
        "nonpadding_tokens": sum(row["nonpadding_tokens"] for row in batches),
        "loss_bearing_tokens": sum(row["loss_bearing_tokens"] for row in batches),
        "zero_loss_sequences": sum(row["zero_loss_sequences"] for row in batches),
        "zero_loss_microbatches": sum(row["loss_bearing_tokens"] == 0 for row in batches),
        "opus_summary": {
            "selected_candidate_id_distribution": {
                str(key): value for key, value in sorted(selected_candidate_ids.items())
            },
            "candidate_count_distribution": {
                str(key): value for key, value in sorted(candidate_counts.items())
            },
            "decisions_not_candidate_zero": sum(
                decision["selected_candidate_id"] != 0 for decision in decisions
            ),
            "candidate_outcome_distribution": dict(sorted(candidate_outcomes.items())),
            "decision_outcome_distribution": dict(sorted(decision_outcomes.items())),
            "protected_floor_overrides": sum(
                bool(decision["protected_floor_override"]) for decision in decisions
            ),
            "sparse_fallbacks": sum(bool(decision["sparse_fallback"]) for decision in decisions),
        },
        "loss_by_stage_lane": {
            f"{stage}|{lane}": value for (stage, lane), value in sorted(lane_loss.items())
        },
        "loss_by_stage_indic_tier": {
            f"{stage}|{tier}": value for (stage, tier), value in sorted(tier_loss.items())
        },
        "stage_summaries": stage_summaries,
        "batch_index": batch_stats,
        "decision_index": decision_stats,
    }
    atomic_write_json(OUTPUT_ROOT / "batch_report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "batch_plan_hash": report["batch_plan_hash"],
                "microbatches": report["microbatches"],
                "sequences": report["sequences"],
                "loss_bearing_tokens": report["loss_bearing_tokens"],
                "zero_loss_microbatches": report["zero_loss_microbatches"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
