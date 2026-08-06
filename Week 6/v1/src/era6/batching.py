from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any


def _hash_rank(
    sequence: dict[str, Any], *, seed: str, stage: str, decision_index: int, candidate_id: int
) -> tuple[bytes, int]:
    payload = (
        f"{seed}|{stage}|{decision_index}|{candidate_id}|{sequence['sequence_index']}"
    ).encode("utf-8")
    return hashlib.sha256(payload).digest(), int(sequence["sequence_index"])


def _take_ranked(
    rows: list[dict[str, Any]],
    count: int,
    *,
    seed: str,
    stage: str,
    decision_index: int,
    candidate_id: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    ranked = sorted(
        rows,
        key=lambda row: _hash_rank(
            row,
            seed=seed,
            stage=stage,
            decision_index=decision_index,
            candidate_id=candidate_id,
        ),
    )
    return ranked[:count]


def zero_loss_slots_for_next_batch(
    *, zero_remaining: int, nonzero_remaining: int, batch_size: int
) -> int:
    """Spread zero-loss context sequences so no standalone zero-loss batch is created."""
    total = zero_remaining + nonzero_remaining
    if total == 0 or zero_remaining == 0:
        return 0
    batches_remaining = (total + batch_size - 1) // batch_size
    slots = (zero_remaining + batches_remaining - 1) // batches_remaining
    if nonzero_remaining:
        slots = min(slots, batch_size - 1)
    return min(slots, zero_remaining)


def build_candidate(
    remaining: list[dict[str, Any]],
    *,
    batch_size: int,
    seed: str,
    stage: str,
    decision_index: int,
    candidate_id: int,
) -> list[dict[str, Any]]:
    target_size = min(batch_size, len(remaining))
    zero_rows = [row for row in remaining if int(row["loss_bearing_tokens"]) == 0]
    nonzero_rows = [row for row in remaining if int(row["loss_bearing_tokens"]) > 0]
    zero_slots = zero_loss_slots_for_next_batch(
        zero_remaining=len(zero_rows),
        nonzero_remaining=len(nonzero_rows),
        batch_size=batch_size,
    )
    zero_slots = min(zero_slots, target_size)
    nonzero_slots = min(len(nonzero_rows), target_size - zero_slots)
    if zero_slots + nonzero_slots < target_size:
        zero_slots = target_size - nonzero_slots
    chosen = _take_ranked(
        zero_rows,
        zero_slots,
        seed=seed,
        stage=stage,
        decision_index=decision_index,
        candidate_id=candidate_id,
    )
    chosen.extend(
        _take_ranked(
            nonzero_rows,
            nonzero_slots,
            seed=seed,
            stage=stage,
            decision_index=decision_index,
            candidate_id=candidate_id,
        )
    )
    return sorted(
        chosen,
        key=lambda row: _hash_rank(
            row,
            seed=seed,
            stage=stage,
            decision_index=decision_index,
            candidate_id=candidate_id,
        ),
    )


def proportional_error_ppm(
    observed: Counter[str], targets: dict[str, int], observed_total: int, target_total: int
) -> int:
    if observed_total <= 0 or target_total <= 0:
        return 0
    numerator = sum(
        abs(int(observed.get(name, 0)) * target_total - observed_total * int(target))
        for name, target in targets.items()
    )
    return numerator * 1_000_000 // (observed_total * target_total)


def candidate_metrics(
    candidate: list[dict[str, Any]],
    *,
    cumulative_lane_loss: Counter[str],
    cumulative_tier_loss: Counter[str],
    lane_targets: dict[str, int],
    tier_targets: dict[str, int],
    stage_target_loss: int,
    protected_lanes: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    projected_lane = cumulative_lane_loss.copy()
    projected_tier = cumulative_tier_loss.copy()
    candidate_loss = 0
    candidate_nonpadding = 0
    physical_tokens = 0
    sources: set[str] = set()
    for row in candidate:
        loss = int(row["loss_bearing_tokens"])
        candidate_loss += loss
        candidate_nonpadding += int(row["nonpadding_tokens"])
        physical_tokens += int(row["sequence_length"])
        projected_lane[row["lane"]] += loss
        if row.get("indic_tier"):
            projected_tier[row["indic_tier"]] += loss
        sources.update(fragment["source_id"] for fragment in row["fragments"])
    projected_total = sum(projected_lane.values())
    projected_indic = int(projected_lane.get("indic", 0))
    lane_error = proportional_error_ppm(
        projected_lane, lane_targets, projected_total, stage_target_loss
    )
    tier_error = proportional_error_ppm(
        projected_tier,
        tier_targets,
        projected_indic,
        int(lane_targets.get("indic", 0)),
    )
    protected_floor_projection: dict[str, dict[str, int]] = {}
    protected_floor_deficit = 0
    for lane in protected_lanes:
        lane_target = int(lane_targets.get(lane, 0))
        if lane_target <= 0:
            continue
        required = (
            projected_total * lane_target + max(1, stage_target_loss) - 1
        ) // max(1, stage_target_loss)
        observed = int(projected_lane.get(lane, 0))
        deficit = max(0, required - observed)
        protected_floor_projection[lane] = {
            "required_at_projected_pace": required,
            "observed_after_candidate": observed,
            "deficit_tokens": deficit,
        }
        protected_floor_deficit += deficit
    return {
        "candidate_loss_tokens": candidate_loss,
        "candidate_nonpadding_tokens": candidate_nonpadding,
        "candidate_physical_tokens": physical_tokens,
        "zero_loss_sequences": sum(
            int(row["loss_bearing_tokens"]) == 0 for row in candidate
        ),
        "source_diversity": len(sources),
        "lane_error_ppm": lane_error,
        "indic_tier_error_ppm": tier_error,
        "loss_density_ppm": candidate_loss * 1_000_000 // max(1, physical_tokens),
        "packing_utilization_ppm": candidate_nonpadding * 1_000_000 // max(1, physical_tokens),
        "protected_floor_deficit_tokens": protected_floor_deficit,
        "protected_floor_projection": protected_floor_projection,
    }


def proxy_score(metrics: dict[str, int], weights: dict[str, int]) -> int:
    return sum(int(weights[name]) * int(metrics[name]) for name in weights)


def apply_opus_policy(
    candidates: list[dict[str, Any]], *, config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Apply auditable OPUS gate, deferral, and protected-floor override rules.

    Rejection applies to a candidate microbatch proposal, not to its underlying
    sequences.  Sequences from rejected or deferred proposals stay in the
    remaining pool unless they also occur in the accepted proposal.
    """
    if not candidates:
        raise ValueError("OPUS requires at least one candidate")
    acceptance = config.get("acceptance_policy", {})
    minimum_density = int(acceptance.get("minimum_loss_density_ppm", 0))
    floor_policy = config.get("protected_floor_policy", {})
    floor_override_enabled = bool(floor_policy.get("override_enabled", False))

    annotated: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for source in candidates:
        candidate = dict(source)
        reasons: list[str] = []
        metrics = candidate["metrics"]
        if int(metrics["candidate_loss_tokens"]) <= 0:
            reasons.append("zero_loss_only_candidate")
        if int(metrics["loss_density_ppm"]) < minimum_density:
            reasons.append("below_minimum_loss_density")
        candidate["gate_result"] = "reject" if reasons else "pass"
        candidate["policy_reasons"] = reasons
        annotated.append(candidate)
        if not reasons:
            eligible.append(candidate)

    fallback_used = False
    if not eligible:
        # The final partial microbatch can naturally be sparse.  Accepting the
        # best deterministic proposal is safer than stranding scheduled data.
        eligible = annotated
        fallback_used = True

    normal_winner = max(
        eligible, key=lambda item: (int(item["proxy_score"]), -int(item["candidate_id"]))
    )
    selected = normal_winner
    protected_override = False
    if floor_override_enabled and not fallback_used:
        minimum_deficit = min(
            int(item["metrics"].get("protected_floor_deficit_tokens", 0))
            for item in eligible
        )
        if int(normal_winner["metrics"].get("protected_floor_deficit_tokens", 0)) > minimum_deficit:
            selected = max(
                (
                    item
                    for item in eligible
                    if int(item["metrics"].get("protected_floor_deficit_tokens", 0))
                    == minimum_deficit
                ),
                key=lambda item: (int(item["proxy_score"]), -int(item["candidate_id"])),
            )
            protected_override = True

    selected_id = int(selected["candidate_id"])
    rejected_ids: list[int] = []
    deferred_ids: list[int] = []
    for candidate in annotated:
        candidate_id = int(candidate["candidate_id"])
        if candidate_id == selected_id:
            candidate["outcome"] = "accepted"
            if fallback_used:
                candidate["gate_result"] = "fallback"
                candidate["policy_reasons"].append("final_sparse_batch_fallback")
            elif protected_override:
                candidate["policy_reasons"].append("protected_floor_override_selected")
        elif candidate["gate_result"] == "reject":
            candidate["outcome"] = "rejected"
            rejected_ids.append(candidate_id)
        else:
            candidate["outcome"] = "deferred"
            candidate["policy_reasons"].append("valid_candidate_remains_available")
            deferred_ids.append(candidate_id)

    selected = next(item for item in annotated if int(item["candidate_id"]) == selected_id)
    policy = {
        "accepted_candidate_id": selected_id,
        "normal_proxy_winner_candidate_id": int(normal_winner["candidate_id"]),
        "rejected_candidate_ids": rejected_ids,
        "deferred_candidate_ids": deferred_ids,
        "protected_floor_override": protected_override,
        "sparse_fallback": fallback_used,
        "decision_outcome": (
            "protected_floor_override"
            if protected_override
            else "sparse_fallback"
            if fallback_used
            else "proxy_accept"
        ),
    }
    return selected, annotated, policy


def select_opus_batches(
    sequences: list[dict[str, Any]],
    *,
    stage_schedule: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequence_length = int(stage_schedule["sequence_length"])
    token_budget = int(config["microbatch_physical_token_budget"])
    if token_budget % sequence_length:
        raise ValueError("microbatch token budget must divide the stage sequence length")
    batch_size = token_budget // sequence_length
    if batch_size < 1:
        raise ValueError("microbatch must contain at least one sequence")
    if any(int(row["sequence_length"]) != sequence_length for row in sequences):
        raise ValueError("stage contains a mismatched sequence length")
    zero_count = sum(int(row["loss_bearing_tokens"]) == 0 for row in sequences)
    batch_count = (len(sequences) + batch_size - 1) // batch_size
    if zero_count and zero_count > (batch_size - 1) * batch_count:
        raise ValueError("zero-loss sequences cannot all be paired with loss-bearing sequences")

    remaining = list(sequences)
    cumulative_lane: Counter[str] = Counter()
    cumulative_tier: Counter[str] = Counter()
    batches: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    decision_index = 0
    while remaining:
        candidates: list[dict[str, Any]] = []
        seen_sets: set[tuple[int, ...]] = set()
        for candidate_id in range(int(config["candidate_microbatches"])):
            rows = build_candidate(
                remaining,
                batch_size=batch_size,
                seed=config["candidate_seed"],
                stage=stage_schedule["name"],
                decision_index=decision_index,
                candidate_id=candidate_id,
            )
            identity = tuple(sorted(int(row["sequence_index"]) for row in rows))
            if identity in seen_sets:
                continue
            seen_sets.add(identity)
            metrics = candidate_metrics(
                rows,
                cumulative_lane_loss=cumulative_lane,
                cumulative_tier_loss=cumulative_tier,
                lane_targets=stage_schedule["lane_targets"],
                tier_targets=stage_schedule["indic_tier_targets"],
                stage_target_loss=int(stage_schedule["target_loss_tokens"]),
                protected_lanes=tuple(
                    config.get("protected_floor_policy", {}).get("lanes", [])
                ),
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "sequence_indices": [int(row["sequence_index"]) for row in rows],
                    "metrics": metrics,
                    "proxy_score": proxy_score(metrics, config["score_weights"]),
                    "rows": rows,
                }
            )
        selected, candidates, policy = apply_opus_policy(candidates, config=config)
        selected_rows = selected["rows"]
        selected_indices = set(selected["sequence_indices"])
        for row in selected_rows:
            loss = int(row["loss_bearing_tokens"])
            cumulative_lane[row["lane"]] += loss
            if row.get("indic_tier"):
                cumulative_tier[row["indic_tier"]] += loss
        batch_lane = Counter()
        batch_tier = Counter()
        for row in selected_rows:
            batch_lane[row["lane"]] += int(row["loss_bearing_tokens"])
            if row.get("indic_tier"):
                batch_tier[row["indic_tier"]] += int(row["loss_bearing_tokens"])
        batches.append(
            {
                "stage": stage_schedule["name"],
                "stage_batch_index": decision_index,
                "sequence_length": sequence_length,
                "sequence_count": len(selected_rows),
                "sequence_indices": selected["sequence_indices"],
                "physical_tokens": selected["metrics"]["candidate_physical_tokens"],
                "nonpadding_tokens": selected["metrics"]["candidate_nonpadding_tokens"],
                "loss_bearing_tokens": selected["metrics"]["candidate_loss_tokens"],
                "zero_loss_sequences": selected["metrics"]["zero_loss_sequences"],
                "lane_loss_tokens": dict(sorted(batch_lane.items())),
                "indic_tier_loss_tokens": dict(sorted(batch_tier.items())),
                "opus_decision_index": decision_index,
                "selected_candidate_id": selected["candidate_id"],
                "opus_decision_outcome": policy["decision_outcome"],
            }
        )
        decisions.append(
            {
                "stage": stage_schedule["name"],
                "decision_index": decision_index,
                "remaining_sequences_before": len(remaining),
                "candidate_count": len(candidates),
                "selected_candidate_id": selected["candidate_id"],
                "accepted_candidate_id": policy["accepted_candidate_id"],
                "normal_proxy_winner_candidate_id": policy[
                    "normal_proxy_winner_candidate_id"
                ],
                "rejected_candidate_ids": policy["rejected_candidate_ids"],
                "deferred_candidate_ids": policy["deferred_candidate_ids"],
                "protected_floor_override": policy["protected_floor_override"],
                "sparse_fallback": policy["sparse_fallback"],
                "decision_outcome": policy["decision_outcome"],
                "candidates": [
                    {key: value for key, value in candidate.items() if key != "rows"}
                    for candidate in candidates
                ],
                "cumulative_lane_loss_after": dict(sorted(cumulative_lane.items())),
                "cumulative_indic_tier_loss_after": dict(sorted(cumulative_tier.items())),
            }
        )
        remaining = [
            row for row in remaining if int(row["sequence_index"]) not in selected_indices
        ]
        decision_index += 1
    return batches, decisions
