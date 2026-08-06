from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Callable


def weighted_priority(record: dict[str, Any], *, seed: str) -> tuple[int, str]:
    """Deterministic integer approximation to weighted random priority."""
    digest = hashlib.sha256(f"{seed}|{record['record_id']}".encode("utf-8")).digest()
    random_integer = int.from_bytes(digest[:16], "big")
    weight_micros = max(1, int(round(float(record.get("quality_weight", 1.0)) * 1_000_000)))
    return (random_integer * 1_000_000 // weight_micros, record["record_id"])


def select_for_schedule(
    rows: list[dict[str, Any]], schedule: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_eligibility: dict[tuple[str, str, str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["loss_bearing_token_count"] <= 0:
            continue
        by_eligibility[(row["permission"], row["lane"], row["indic_tier"], row["difficulty_band"])].append(row)

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    group_summaries = []
    seed = schedule["sampling_seed"]
    for stage_order, stage in enumerate(schedule["stages"]):
        groups: list[tuple[str, str | None, int]] = []
        for lane, target in stage["lane_targets"].items():
            if lane == "indic":
                groups.extend((lane, tier, value) for tier, value in stage["indic_tier_targets"].items())
            else:
                groups.append((lane, None, target))
        for lane, tier, target in sorted(groups, key=lambda item: (item[0], item[1] or "")):
            candidates = []
            for band in stage["eligible_difficulty_bands"]:
                candidates.extend(by_eligibility.get((stage["permission"], lane, tier, band), []))
            candidates = [row for row in candidates if row["record_id"] not in used]
            candidates.sort(
                key=lambda row: weighted_priority(
                    row, seed=f"{seed}|{stage['name']}|{lane}|{tier or '-'}"
                )
            )
            remaining = target
            selected_count = 0
            available_loss = sum(row["loss_bearing_token_count"] for row in candidates)
            for row in candidates:
                if remaining <= 0:
                    break
                take = min(remaining, int(row["loss_bearing_token_count"]))
                selected.append(
                    {
                        **row,
                        "stage": stage["name"],
                        "stage_order": stage_order,
                        "sequence_length": stage["sequence_length"],
                        "selected_loss_tokens": take,
                        "record_loss_tokens": int(row["loss_bearing_token_count"]),
                        "loss_clipped": take < int(row["loss_bearing_token_count"]),
                        "selection_order": selected_count,
                    }
                )
                used.add(row["record_id"])
                selected_count += 1
                remaining -= take
            if remaining:
                raise ValueError(
                    f"selection exhausted for {stage['name']} {lane} {tier}: {remaining} loss tokens short"
                )
            group_summaries.append(
                {
                    "stage": stage["name"],
                    "lane": lane,
                    "indic_tier": tier,
                    "target_loss_tokens": target,
                    "unused_candidate_loss_supply_before_selection": available_loss,
                    "selected_records": selected_count,
                }
            )
    selected.sort(
        key=lambda row: (
            row["stage_order"],
            row["lane"],
            row["indic_tier"] or "",
            row["selection_order"],
        )
    )
    return selected, {
        "selected_records": len(selected),
        "unique_selected_records": len(used),
        "selected_loss_tokens": sum(row["selected_loss_tokens"] for row in selected),
        "groups": group_summaries,
    }


def clip_record_to_loss(
    token_ids: list[int], loss_mask: list[int], selected_loss_tokens: int
) -> tuple[list[int], list[int]]:
    if len(token_ids) != len(loss_mask) or selected_loss_tokens < 1:
        raise ValueError("invalid token/loss input or selected loss target")
    total_loss = sum(loss_mask)
    if selected_loss_tokens > total_loss:
        raise ValueError("selected loss exceeds record loss")
    if selected_loss_tokens == total_loss:
        return token_ids, loss_mask
    running = 0
    end = 0
    for end, value in enumerate(loss_mask, 1):
        running += value
        if running == selected_loss_tokens:
            break
    return token_ids[:end], loss_mask[:end]


def pack_group(
    records: list[dict[str, Any]],
    *,
    sequence_length: int,
    pad_token_id: int,
    load_payload: Callable[[dict[str, Any]], tuple[list[int], list[int]]],
) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    current = {"tokens": [], "loss": [], "segments": [], "positions": [], "fragments": []}
    segment_id = 0

    def flush() -> None:
        nonlocal current, segment_id
        if not current["tokens"]:
            return
        nonpadding = len(current["tokens"])
        padding = sequence_length - nonpadding
        current["tokens"].extend([pad_token_id] * padding)
        current["loss"].extend([0] * padding)
        current["segments"].extend([-1] * padding)
        current["positions"].extend([0] * padding)
        sequences.append(
            {
                **current,
                "nonpadding_tokens": nonpadding,
                "loss_tokens": sum(current["loss"]),
            }
        )
        current = {"tokens": [], "loss": [], "segments": [], "positions": [], "fragments": []}
        segment_id = 0

    for record in records:
        token_ids, loss_mask = load_payload(record)
        token_ids, loss_mask = clip_record_to_loss(
            token_ids, loss_mask, int(record["selected_loss_tokens"])
        )
        cursor = 0
        previous_token: int | None = None
        while cursor < len(token_ids):
            if len(current["tokens"]) == sequence_length:
                flush()
            continuation_overlap = previous_token is not None and not current["tokens"]
            start_in_sequence = len(current["tokens"])
            if continuation_overlap:
                current["tokens"].append(previous_token)
                current["loss"].append(0)
                current["segments"].append(segment_id)
                current["positions"].append(0)
            capacity = sequence_length - len(current["tokens"])
            take = min(capacity, len(token_ids) - cursor)
            position_start = 1 if continuation_overlap else 0
            current["tokens"].extend(token_ids[cursor : cursor + take])
            current["loss"].extend(loss_mask[cursor : cursor + take])
            current["segments"].extend([segment_id] * take)
            current["positions"].extend(range(position_start, position_start + take))
            current["fragments"].append(
                {
                    "record_id": record["record_id"],
                    "source_id": record["source_id"],
                    "segment_id": segment_id,
                    "record_token_start": cursor,
                    "record_token_end": cursor + take,
                    "sequence_token_start": start_in_sequence,
                    "sequence_token_end": len(current["tokens"]),
                    "continuation_overlap": 1 if continuation_overlap else 0,
                }
            )
            cursor += take
            if cursor < len(token_ids):
                previous_token = token_ids[cursor - 1]
                flush()
            else:
                previous_token = None
                segment_id += 1
    flush()
    return sequences
