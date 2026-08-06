from __future__ import annotations


def reconstructable_throughput(
    *,
    elapsed_nanoseconds: int,
    physical_tokens: int,
    nonpadding_tokens: int,
    loss_bearing_tokens: int,
) -> dict[str, float | int]:
    """Return rates whose token numerators and elapsed denominator are retained."""
    if elapsed_nanoseconds <= 0:
        raise ValueError("elapsed_nanoseconds must be positive")
    counts = (physical_tokens, nonpadding_tokens, loss_bearing_tokens)
    if any(value < 0 for value in counts):
        raise ValueError("token counts cannot be negative")
    scale = 1_000_000_000 / elapsed_nanoseconds
    return {
        "elapsed_nanoseconds": elapsed_nanoseconds,
        "elapsed_seconds": elapsed_nanoseconds / 1_000_000_000,
        "physical_tokens": physical_tokens,
        "nonpadding_tokens": nonpadding_tokens,
        "loss_bearing_tokens": loss_bearing_tokens,
        "physical_tokens_per_second": physical_tokens * scale,
        "nonpadding_tokens_per_second": nonpadding_tokens * scale,
        "useful_loss_bearing_tokens_per_second": loss_bearing_tokens * scale,
        "packing_utilization": nonpadding_tokens / max(1, physical_tokens),
        "useful_loss_fraction": loss_bearing_tokens / max(1, physical_tokens),
    }
