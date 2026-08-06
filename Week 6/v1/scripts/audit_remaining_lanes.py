from __future__ import annotations

import gzip
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.acquisition import candidate_text_and_metadata, load_source_lock  # noqa: E402
from era6.canonical import atomic_write_json, read_jsonl_gz  # noqa: E402
from era6.cleaning import BasicQualityFilter, PIIScrubber, TextNormalizer  # noqa: E402
from era6.quality import extract_quality_signals  # noqa: E402


OUTPUT_PATH = ROOT / "analysis" / "data_quality" / "remaining_lanes_baseline_audit.json"
REPORT_PATH = ROOT / "docs" / "quality" / "REMAINING_LANES_BASELINE_AUDIT.md"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"minimum": 0.0, "p05": 0.0, "median": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "minimum": min(values),
        "p05": percentile(values, 0.05),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
    }


def load_cached_rows(source_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "data" / "acquisition_cache" / source_id).glob("offset-*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(item["row"] for item in json.load(stream).get("rows", []))
    return rows


def compact_preview(text: str, limit: int = 300) -> dict[str, str]:
    value = " ".join(text.split())
    if len(value) <= limit:
        return {"start": value, "end": value}
    return {"start": value[: limit - 1].rstrip() + "…", "end": "…" + value[-(limit - 1) :].lstrip()}


def deterministic_examples(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    def take(values: list[dict[str, Any]], key: Any, reverse: bool = False) -> list[dict[str, Any]]:
        ordered = sorted(values, key=lambda row: (key(row), row["upstream_id"]), reverse=reverse)
        return ordered[:3]

    categories = {
        "shortest_admitted": take(rows, lambda row: row["signals"]["characters"]),
        "lowest_alpha": take(rows, lambda row: row["signals"]["alpha_fraction"]),
        "highest_repetition": take(
            rows, lambda row: row["signals"]["repeated_trigram_fraction"], reverse=True
        ),
        "largest_truncations": take(rows, lambda row: row["characters_truncated"], reverse=True),
        "pii_redactions": take(
            [row for row in rows if row["pii_redactions"]],
            lambda row: row["pii_redactions"],
            reverse=True,
        ),
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for category, members in categories.items():
        result[category] = [
            {
                "upstream_id": row["upstream_id"],
                "raw_characters": row["raw_characters"],
                "cleaned_characters": row["signals"]["characters"],
                "characters_truncated": row["characters_truncated"],
                "pii_redactions": row["pii_redactions"],
                "signals": row["signals"],
                "preview": compact_preview(row["text"]),
            }
            for row in members
        ]
    return result


def audit_cached_target(target: dict[str, Any]) -> dict[str, Any]:
    raw_rows = load_cached_rows(target["source_id"])
    normalizer, scrubber = TextNormalizer(), PIIScrubber()
    rejection_counts: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    raw_characters = 0
    normalized_characters = 0
    total_pii_redactions = 0

    allowed_licenses = {value.casefold() for value in target.get("allowed_row_licenses", [])}
    for raw in raw_rows:
        if allowed_licenses and str(raw.get("license", "")).casefold() not in allowed_licenses:
            rejection_counts["row_license_not_allowed"] += 1
            continue
        text, upstream_id, _metadata = candidate_text_and_metadata(target, raw)
        raw_characters += len(text)
        normalized = normalizer.normalize(text)
        normalized_characters += len(normalized)
        scrubbed = scrubber.scrub(normalized)
        total_pii_redactions += scrubbed.num_redactions
        before_slice = scrubbed.text
        if len(before_slice) < int(target["min_chars"]):
            rejection_counts["below_minimum_characters"] += 1
            continue
        cleaned = before_slice[: int(target["max_chars"])]
        decision = BasicQualityFilter(
            min_characters=int(target["min_chars"]), max_characters=int(target["max_chars"])
        ).evaluate(cleaned)
        if not decision.admitted:
            rejection_counts.update(decision.reasons)
            continue
        accepted.append(
            {
                "upstream_id": str(upstream_id),
                "raw_characters": len(text),
                "normalized_characters": len(normalized),
                "characters_truncated": max(0, len(before_slice) - len(cleaned)),
                "pii_redactions": scrubbed.num_redactions,
                "text": cleaned,
                "signals": extract_quality_signals(cleaned),
            }
        )

    snapshot_path = ROOT / "data" / "source_snapshots" / f"{target['source_id']}.jsonl.gz"
    snapshot_records = list(read_jsonl_gz(snapshot_path))
    signal_names = (
        "characters",
        "alpha_fraction",
        "digit_fraction",
        "non_ascii_fraction",
        "list_line_fraction",
        "duplicate_line_fraction",
        "repeated_trigram_fraction",
    )
    return {
        "source_id": target["source_id"],
        "lane": target["lane"],
        "provenance_tier": target["provenance_tier"],
        "transform": target["transform"],
        "audit_status": "raw_cache_compared_to_cleaning_pipeline",
        "counts": {
            "cached_raw_candidates": len(raw_rows),
            "accepted_candidates_before_quota": len(accepted),
            "selected_snapshot_records": len(snapshot_records),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "truncated_candidates": sum(bool(row["characters_truncated"]) for row in accepted),
            "pii_redacted_candidates": sum(bool(row["pii_redactions"]) for row in accepted),
            "pii_redactions": total_pii_redactions,
        },
        "character_flow": {
            "raw_characters_after_license_gate": raw_characters,
            "normalized_characters": normalized_characters,
            "normalization_delta": normalized_characters - raw_characters,
            "accepted_cleaned_characters": sum(row["signals"]["characters"] for row in accepted),
            "truncated_characters": sum(row["characters_truncated"] for row in accepted),
        },
        "accepted_signal_distributions": {
            name: distribution([float(row["signals"][name]) for row in accepted])
            for name in signal_names
        },
        "review_samples": deterministic_examples(accepted),
    }


def audit_snapshot_only_target(target: dict[str, Any]) -> dict[str, Any]:
    snapshot_path = ROOT / "data" / "source_snapshots" / f"{target['source_id']}.jsonl.gz"
    records = list(read_jsonl_gz(snapshot_path))
    return {
        "source_id": target["source_id"],
        "lane": target["lane"],
        "provenance_tier": target["provenance_tier"],
        "transform": target["transform"],
        "audit_status": "snapshot_only_parent_manifest_available",
        "counts": {"selected_snapshot_records": len(records)},
        "character_flow": {"accepted_cleaned_characters": sum(len(row["text"]) for row in records)},
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Remaining lanes: baseline cleaning audit",
        "",
        "This is a diagnostic pass. It measures each lane with the existing source transform and cleaner; it does not copy Wikipedia admission thresholds or modify any source snapshot.",
        "",
        "## Inventory",
        "",
        "| Lane | Source | Raw candidates | Accepted before quota | Snapshot | Truncated | PII-redacted |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for source in report["sources"]:
        counts = source["counts"]
        lines.append(
            f"| {source['lane']} | {source['source_id']} | "
            f"{counts.get('cached_raw_candidates', 'n/a')} | "
            f"{counts.get('accepted_candidates_before_quota', 'n/a')} | "
            f"{counts['selected_snapshot_records']} | "
            f"{counts.get('truncated_candidates', 'n/a')} | "
            f"{counts.get('pii_redacted_candidates', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## How to interpret this",
            "",
            "- Raw candidates are cached rows examined, not the full upstream dataset.",
            "- Accepted-before-quota shows what passed the current deterministic transform and basic gate.",
            "- Snapshot count is the pinned quota actually preserved for downstream curation.",
            "- Truncation and generic text signals are diagnostic. Their meaning differs by lane: punctuation and low alpha can be healthy in code or mathematics, while role markers are required in agentic data.",
            "",
            "## Measured priority order",
            "",
            "1. **Source-aware PII policy:** the generic phone pattern masks valid numeric constants, mathematical answers, years, identifiers, and JSON values. This is label corruption, not merely a sampling preference.",
            "2. **Boundary-aware retention instead of slicing:** science/math, code, and long-context sources lose millions of characters at hard maximum-length slices. Each requires boundaries appropriate to its structure.",
            "3. **Lane validators:** code needs syntax/file-boundary checks; reasoning needs question/derivation/final-answer integrity; agentic data needs role, tool-call, and JSON consistency; Indic data needs script/language and provenance-tier checks.",
            "4. **Only then freeze the tokenizer input:** tokenization must consume repaired, versioned snapshots rather than the current diagnostic baseline.",
            "",
            "Read the per-source start/end samples in the JSON artifact before defining any thresholds. The order above is chosen from measured corruption risk, truncation, and structural extremes rather than corpus size alone.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    config = load_source_lock(ROOT / "configs" / "sources.lock.json")
    targets = [
        target
        for target in config["targets"]
        if target["permission"] == "train" and target["source_id"] != "wikipedia_general_en"
    ]
    sources = []
    for target in targets:
        cache = ROOT / "data" / "acquisition_cache" / target["source_id"]
        sources.append(audit_cached_target(target) if cache.exists() else audit_snapshot_only_target(target))
    report = {
        "schema_version": 1,
        "policy_status": "diagnostic_only_no_snapshot_changes",
        "source_count": len(sources),
        "lane_count": len({source["lane"] for source in sources}),
        "sources": sources,
    }
    atomic_write_json(OUTPUT_PATH, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "status": "READY_FOR_LANE_REVIEW",
                "sources": report["source_count"],
                "lanes": report["lane_count"],
                "report": REPORT_PATH.relative_to(ROOT).as_posix(),
                "details": OUTPUT_PATH.relative_to(ROOT).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
