from __future__ import annotations

from decimal import Decimal
import hashlib


PARTITIONS = ("train", "validation", "anneal")


def fraction_threshold(fraction: float) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    return int(Decimal(str(fraction)) * (1 << 64))


def deterministic_partition(
    group_id: str,
    *,
    seed: str,
    validation_fraction: float = 0.10,
    anneal_fraction: float = 0.02,
) -> str:
    if validation_fraction + anneal_fraction >= 1.0:
        raise ValueError("validation and anneal fractions must leave training supply")
    value = int.from_bytes(hashlib.sha256(f"{seed}|{group_id}".encode("utf-8")).digest()[:8], "big")
    validation_limit = fraction_threshold(validation_fraction)
    anneal_limit = validation_limit + fraction_threshold(anneal_fraction)
    if value < validation_limit:
        return "validation"
    if value < anneal_limit:
        return "anneal"
    return "train"

