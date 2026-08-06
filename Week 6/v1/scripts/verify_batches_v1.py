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
    canonical_json_bytes,
    read_jsonl_gz,
    sha256_bytes,
    sha256_file,
)


def main() -> int:
    report_path = ROOT / "data" / "batches_v1" / "batch_report.json"
    config_path = ROOT / "configs" / "batching_v1.json"
    packing_report_path = ROOT / "data" / "packed_v1" / "packing_report.json"
    schedule_path = ROOT / "artifacts" / "schedule_v1" / "schedule.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    packing_report = json.loads(packing_report_path.read_text(encoding="utf-8"))
    schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule = schedule_payload["schedule"]

    if report["packing_hash"] != packing_report["packing_hash"]:
        raise AssertionError("batch plan refers to the wrong packing artifact")
    if report["schedule_hash"] != schedule_payload["schedule_hash"]:
        raise AssertionError("batch plan refers to the wrong schedule")
    if report["batching_config_sha256"] != sha256_file(config_path):
        raise AssertionError("batching config hash mismatch")
    if report["proxy_version"] != config["proxy_version"]:
        raise AssertionError("OPUS proxy version mismatch")
    for name, expected in report["component_hashes"].items():
        if sha256_file(ROOT / report["paths"][name]) != expected:
            raise AssertionError(f"batch component hash mismatch: {name}")
    expected_plan_hash = f"sha256:{sha256_bytes(canonical_json_bytes(report['component_hashes']))}"
    if report["batch_plan_hash"] != expected_plan_hash:
        raise AssertionError("batch plan hash mismatch")

    sequences = list(read_jsonl_gz(ROOT / packing_report["paths"]["sequences"]))
    batches = list(read_jsonl_gz(ROOT / report["paths"]["batches"]))
    decisions = list(read_jsonl_gz(ROOT / report["paths"]["opus_decisions"]))
    sequence_by_index = {int(row["sequence_index"]): row for row in sequences}
    consumed = [index for batch in batches for index in batch["sequence_indices"]]
    if len(consumed) != len(set(consumed)):
        raise AssertionError("a packed sequence was consumed more than once")
    if set(consumed) != set(sequence_by_index):
        raise AssertionError("batch plan does not cover every packed sequence")
    if len(batches) != len(decisions) or len(batches) != report["microbatches"]:
        raise AssertionError("batch/decision counts differ")

    stage_order = {stage["name"]: index for index, stage in enumerate(schedule["stages"])}
    previous_stage = -1
    loss_by_stage: Counter[str] = Counter()
    loss_by_stage_lane: Counter[tuple[str, str]] = Counter()
    loss_by_stage_tier: Counter[tuple[str, str]] = Counter()
    zero_loss_sequences = 0
    candidate_outcomes: Counter[str] = Counter()
    decision_outcomes: Counter[str] = Counter()
    protected_floor_overrides = 0
    sparse_fallbacks = 0
    minimum_density = int(config["acceptance_policy"]["minimum_loss_density_ppm"])
    for expected_index, (batch, decision) in enumerate(zip(batches, decisions, strict=True)):
        if batch["batch_index"] != expected_index:
            raise AssertionError("batch indices are not contiguous")
        current_stage = stage_order[batch["stage"]]
        if current_stage < previous_stage:
            raise AssertionError("curriculum stage order moved backwards")
        previous_stage = current_stage
        rows = [sequence_by_index[int(index)] for index in batch["sequence_indices"]]
        if any(row["stage"] != batch["stage"] for row in rows):
            raise AssertionError("a microbatch mixes curriculum stages")
        if any(row["sequence_length"] != batch["sequence_length"] for row in rows):
            raise AssertionError("a microbatch mixes sequence lengths")
        capacity = config["microbatch_physical_token_budget"] // batch["sequence_length"]
        if not 1 <= len(rows) <= capacity:
            raise AssertionError("microbatch sequence capacity violated")
        loss = sum(int(row["loss_bearing_tokens"]) for row in rows)
        if loss != batch["loss_bearing_tokens"]:
            raise AssertionError("microbatch loss accounting mismatch")
        if loss <= 0:
            raise AssertionError("standalone zero-loss microbatch found")
        if sum(int(row["sequence_length"]) for row in rows) != batch["physical_tokens"]:
            raise AssertionError("microbatch physical-token accounting mismatch")
        if sum(int(row["nonpadding_tokens"]) for row in rows) != batch["nonpadding_tokens"]:
            raise AssertionError("microbatch nonpadding accounting mismatch")
        zero_loss_sequences += sum(int(row["loss_bearing_tokens"]) == 0 for row in rows)
        accepted = [row for row in decision["candidates"] if row["outcome"] == "accepted"]
        rejected = [row for row in decision["candidates"] if row["outcome"] == "rejected"]
        deferred = [row for row in decision["candidates"] if row["outcome"] == "deferred"]
        if len(accepted) != 1 or accepted[0]["candidate_id"] != batch["selected_candidate_id"]:
            raise AssertionError("OPUS decision does not identify exactly one accepted candidate")
        if decision["accepted_candidate_id"] != batch["selected_candidate_id"]:
            raise AssertionError("accepted candidate and emitted batch differ")
        if sorted(decision["rejected_candidate_ids"]) != sorted(row["candidate_id"] for row in rejected):
            raise AssertionError("rejected candidate index mismatch")
        if sorted(decision["deferred_candidate_ids"]) != sorted(row["candidate_id"] for row in deferred):
            raise AssertionError("deferred candidate index mismatch")
        for candidate in rejected:
            if candidate["gate_result"] != "reject":
                raise AssertionError("rejected candidate passed its gate")
            if not candidate["policy_reasons"]:
                raise AssertionError("rejected candidate has no reason")
            if (
                "below_minimum_loss_density" in candidate["policy_reasons"]
                and int(candidate["metrics"]["loss_density_ppm"]) >= minimum_density
            ):
                raise AssertionError("loss-density rejection threshold mismatch")
        for candidate in deferred:
            if candidate["gate_result"] != "pass":
                raise AssertionError("deferred candidate did not pass the gate")
        if decision["protected_floor_override"]:
            protected_floor_overrides += 1
            normal = next(
                row
                for row in decision["candidates"]
                if row["candidate_id"] == decision["normal_proxy_winner_candidate_id"]
            )
            if int(accepted[0]["metrics"]["protected_floor_deficit_tokens"]) >= int(
                normal["metrics"]["protected_floor_deficit_tokens"]
            ):
                raise AssertionError("protected-floor override did not reduce projected deficit")
        if decision["sparse_fallback"]:
            sparse_fallbacks += 1
            if accepted[0]["gate_result"] != "fallback":
                raise AssertionError("sparse fallback candidate is not marked as fallback")
        if batch["opus_decision_outcome"] != decision["decision_outcome"]:
            raise AssertionError("batch and OPUS decision outcome differ")
        candidate_outcomes.update(row["outcome"] for row in decision["candidates"])
        decision_outcomes[decision["decision_outcome"]] += 1
        loss_by_stage[batch["stage"]] += loss
        for row in rows:
            row_loss = int(row["loss_bearing_tokens"])
            loss_by_stage_lane[(row["stage"], row["lane"])] += row_loss
            if row.get("indic_tier"):
                loss_by_stage_tier[(row["stage"], row["indic_tier"])] += row_loss

    if not {"accepted", "rejected", "deferred"}.issubset(candidate_outcomes):
        raise AssertionError("OPUS did not demonstrate accept, reject, and defer outcomes")
    if protected_floor_overrides <= 0:
        raise AssertionError("OPUS did not demonstrate a protected-floor override")
    expected_summary = report["opus_summary"]
    if expected_summary["candidate_outcome_distribution"] != dict(sorted(candidate_outcomes.items())):
        raise AssertionError("OPUS candidate outcome summary mismatch")
    if expected_summary["decision_outcome_distribution"] != dict(sorted(decision_outcomes.items())):
        raise AssertionError("OPUS decision outcome summary mismatch")
    if expected_summary["protected_floor_overrides"] != protected_floor_overrides:
        raise AssertionError("OPUS protected-floor override summary mismatch")
    if expected_summary["sparse_fallbacks"] != sparse_fallbacks:
        raise AssertionError("OPUS sparse-fallback summary mismatch")

    for stage in schedule["stages"]:
        name = stage["name"]
        if loss_by_stage[name] != stage["target_loss_tokens"]:
            raise AssertionError(f"stage target mismatch: {name}")
        for lane, target in stage["lane_targets"].items():
            if loss_by_stage_lane[(name, lane)] != target:
                raise AssertionError(f"stage/lane target mismatch: {name} {lane}")
        for tier, target in stage["indic_tier_targets"].items():
            if loss_by_stage_tier[(name, tier)] != target:
                raise AssertionError(f"stage/tier target mismatch: {name} {tier}")

    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sequences:
        by_stage[row["stage"]].append(row)
    rebuilt_batches: list[dict[str, Any]] = []
    rebuilt_decisions: list[dict[str, Any]] = []
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
        rebuilt_batches.extend(stage_batches)
        rebuilt_decisions.extend(stage_decisions)
    if rebuilt_batches != batches:
        raise AssertionError("deterministic batch reconstruction differs from frozen plan")
    if rebuilt_decisions != decisions:
        raise AssertionError("deterministic OPUS reconstruction differs from decision ledger")

    print(
        json.dumps(
            {
                "status": "PASS",
                "batch_plan_hash": report["batch_plan_hash"],
                "microbatches": len(batches),
                "opus_decisions": len(decisions),
                "sequences_consumed_once": len(consumed),
                "loss_bearing_tokens": sum(loss_by_stage.values()),
                "zero_loss_sequences_paired": zero_loss_sequences,
                "zero_loss_microbatches": 0,
                "curriculum_stage_order": "verified",
                "schedule_accounting": "exact",
                "deterministic_reconstruction": "exact",
                "candidate_outcomes": dict(sorted(candidate_outcomes.items())),
                "decision_outcomes": dict(sorted(decision_outcomes.items())),
                "protected_floor_overrides": protected_floor_overrides,
                "sparse_fallbacks": sparse_fallbacks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
