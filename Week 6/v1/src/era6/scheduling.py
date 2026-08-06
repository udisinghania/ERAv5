from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, ROUND_CEILING
from typing import Any, Iterable


def allocate_integer(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation with deterministic key tie-breaking."""
    if total < 0 or not weights:
        raise ValueError("allocation requires a non-negative total and weights")
    decimal_weights = {key: Decimal(str(value)) for key, value in weights.items()}
    weight_sum = sum(decimal_weights.values())
    if weight_sum <= 0:
        raise ValueError("allocation weights must sum to a positive value")
    exact = {key: Decimal(total) * value / weight_sum for key, value in decimal_weights.items()}
    result = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(result.values())
    order = sorted(result, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return dict(sorted(result.items()))


def enforce_integer_floors(
    allocation: dict[str, int], *, total: int, floors: dict[str, float]
) -> dict[str, int]:
    """Move rounding tokens so every protected share is a true minimum."""
    result = dict(allocation)
    required = {
        lane: int((Decimal(total) * Decimal(str(floor))).to_integral_value(rounding=ROUND_CEILING))
        for lane, floor in floors.items()
    }
    for lane in sorted(required):
        while result[lane] < required[lane]:
            donors = [
                candidate
                for candidate in result
                if candidate != lane and result[candidate] > required.get(candidate, 0)
            ]
            if not donors:
                raise ValueError(f"cannot enforce protected floor for {lane}")
            donor = max(donors, key=lambda candidate: (result[candidate], candidate))
            result[donor] -= 1
            result[lane] += 1
    if sum(result.values()) != total:
        raise AssertionError("floor enforcement changed allocation total")
    return dict(sorted(result.items()))


def assign_difficulty_bands(
    rows: Iterable[dict[str, Any]], bands: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Assign within-source complexity percentiles without conflating quality and difficulty."""
    previous = Decimal("0")
    cutoffs: list[tuple[str, Decimal]] = []
    for band in bands:
        maximum = Decimal(str(band["maximum_fraction"]))
        if not previous < maximum <= 1:
            raise ValueError("difficulty cutoffs must be strictly increasing and end at one")
        cutoffs.append((str(band["name"]), maximum))
        previous = maximum
    if cutoffs[-1][1] != 1:
        raise ValueError("final difficulty cutoff must equal one")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["lane"], row["source_id"])].append(row)
    assigned = []
    for (_lane, _source), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: (row["token_count"], row["record_id"]))
        size = len(ordered)
        for rank, row in enumerate(ordered):
            percentile = (Decimal(rank) + Decimal("0.5")) / Decimal(size)
            band_name = next(name for name, maximum in cutoffs if percentile <= maximum)
            assigned.append(
                {
                    **row,
                    "difficulty_proxy": "within_source_token_count_rank",
                    "difficulty_rank": rank,
                    "difficulty_population": size,
                    "difficulty_percentile": float(percentile),
                    "difficulty_band": band_name,
                }
            )
    assigned.sort(key=lambda row: row["record_id"])
    return assigned


def build_executable_schedule(
    *,
    rows: list[dict[str, Any]],
    mixture: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    budget = int(execution["total_loss_token_budget"])
    stages = mixture["stages"]
    stage_weights = {stage["name"]: float(stage["fraction_of_total"]) for stage in stages}
    stage_targets = allocate_integer(budget, stage_weights)
    lane_weights = {key: float(value) for key, value in mixture["pre_anneal_mixture"].items()}
    indic_weights = {key: float(value) for key, value in mixture["indic_tiers"].items()}

    supply: dict[tuple[str, str, str, str | None], int] = defaultdict(int)
    for row in rows:
        if row["loss_bearing_token_count"] <= 0:
            continue
        supply[(row["permission"], row["difficulty_band"], row["lane"], row["indic_tier"])] += int(
            row["loss_bearing_token_count"]
        )

    schedule_stages = []
    cumulative_lane_targets: dict[str, int] = defaultdict(int)
    cumulative_pre_anneal = 0
    for stage in stages:
        stage_name = stage["name"]
        target = stage_targets[stage_name]
        permission = "anneal" if stage.get("reserve_only") else "train"
        eligible_bands = tuple(stage["difficulty_bands"])
        lane_targets = enforce_integer_floors(
            allocate_integer(target, lane_weights),
            total=target,
            floors={key: float(value) for key, value in mixture["protected_floors"].items()},
        )
        lane_supply = {}
        tier_targets: dict[str, int] = {}
        tier_supply: dict[str, int] = {}
        for lane, lane_target in lane_targets.items():
            available = sum(
                value
                for (row_permission, band, row_lane, _tier), value in supply.items()
                if row_permission == permission and row_lane == lane and band in eligible_bands
            )
            lane_supply[lane] = available
            if lane_target > available:
                raise ValueError(
                    f"stage {stage_name} lane {lane} needs {lane_target} loss tokens but has {available}"
                )
            cumulative_lane_targets[lane] += lane_target
        if not stage.get("reserve_only"):
            cumulative_pre_anneal += target

        indic_target = lane_targets.get("indic", 0)
        if indic_target:
            for tier in indic_weights:
                tier_supply[tier] = sum(
                    value
                    for (row_permission, band, row_lane, row_tier), value in supply.items()
                    if row_permission == permission
                    and row_lane == "indic"
                    and row_tier == tier
                    and band in eligible_bands
                )
            stage_indic_weights = indic_weights
            if stage.get("reserve_only"):
                stage_indic_weights = {
                    tier: weight for tier, weight in indic_weights.items() if tier_supply[tier] > 0
                }
                if not stage_indic_weights:
                    raise ValueError("anneal stage has no eligible Indic tier supply")
            tier_targets = allocate_integer(indic_target, stage_indic_weights)
            for tier, tier_target in tier_targets.items():
                available = tier_supply[tier]
                if tier_target > available:
                    raise ValueError(
                        f"stage {stage_name} Indic tier {tier} needs {tier_target} but has {available}"
                    )
        schedule_stages.append(
            {
                "name": stage_name,
                "permission": permission,
                "sequence_length": int(stage["sequence_length"]),
                "eligible_difficulty_bands": list(eligible_bands),
                "target_loss_tokens": target,
                "lane_targets": lane_targets,
                "lane_eligible_supply": dict(sorted(lane_supply.items())),
                "indic_tier_targets": tier_targets,
                "indic_tier_eligible_supply": tier_supply,
                "indic_tier_policy": (
                    execution["anneal_indic_tier_policy"]
                    if stage.get("reserve_only")
                    else "configured_four_tier_mixture"
                ),
            }
        )

    pre_anneal_lane_targets = {
        lane: sum(
            stage["lane_targets"][lane]
            for stage in schedule_stages
            if stage["permission"] == "train"
        )
        for lane in lane_weights
    }
    protected_floor_results = {}
    for lane, floor in mixture["protected_floors"].items():
        required = int(
            (Decimal(str(floor)) * cumulative_pre_anneal).to_integral_value(rounding=ROUND_CEILING)
        )
        actual = pre_anneal_lane_targets[lane]
        if actual < required:
            raise ValueError(f"protected floor failed for {lane}: {actual} < {required}")
        protected_floor_results[lane] = {
            "required_minimum_loss_tokens": required,
            "scheduled_loss_tokens": actual,
            "scheduled_fraction": actual / max(1, cumulative_pre_anneal),
            "passed": True,
        }
    pre_anneal_indic_tier_targets = {
        tier: sum(
            stage["indic_tier_targets"].get(tier, 0)
            for stage in schedule_stages
            if stage["permission"] == "train"
        )
        for tier in indic_weights
    }
    return {
        "total_loss_token_budget": budget,
        "accounting_unit": mixture["accounting_unit"],
        "sampling_policy": execution["sampling_policy"],
        "sampling_seed": execution["sampling_seed"],
        "transition_warmup_fraction": mixture["transition_warmup_fraction"],
        "stages": schedule_stages,
        "pre_anneal_loss_tokens": cumulative_pre_anneal,
        "anneal_loss_tokens": sum(
            stage["target_loss_tokens"] for stage in schedule_stages if stage["permission"] == "anneal"
        ),
        "pre_anneal_lane_targets": pre_anneal_lane_targets,
        "pre_anneal_indic_tier_targets": pre_anneal_indic_tier_targets,
        "protected_floors": protected_floor_results,
        "opus": mixture["opus"],
    }
