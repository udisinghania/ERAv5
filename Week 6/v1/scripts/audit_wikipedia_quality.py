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

from era6.acquisition import load_source_lock  # noqa: E402
from era6.canonical import atomic_write_json, read_jsonl_gz  # noqa: E402
from era6.cleaning import BasicQualityFilter, PIIScrubber, TextNormalizer  # noqa: E402
from era6.quality import extract_quality_signals, provisional_quality_flags  # noqa: E402


SOURCE_ID = "wikipedia_general_en"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def signal_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row["signals"][key]) for row in rows]
    return {
        "min": min(values),
        "p05": percentile(values, 0.05),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def load_raw_rows() -> list[dict[str, Any]]:
    rows = []
    cache = ROOT / "data" / "acquisition_cache" / SOURCE_ID
    for path in sorted(cache.glob("offset-*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        rows.extend(item["row"] for item in payload.get("rows", []))
    return rows


def preview(text: str, limit: int = 500) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[:limit].rstrip() + "…"


def audit_row(raw: dict[str, Any], selected_ids: set[str], target: dict[str, Any]) -> dict[str, Any]:
    raw_combined = f"{str(raw.get('title', '')).strip()}\n\n{str(raw.get('text', ''))}"
    normalized = TextNormalizer().normalize(raw_combined)
    scrubbed = PIIScrubber().scrub(normalized)
    before_truncation = scrubbed.text
    truncated = len(before_truncation) > int(target["max_chars"])
    cleaned = before_truncation[: int(target["max_chars"])]
    decision = BasicQualityFilter(
        min_characters=int(target["min_chars"]), max_characters=int(target["max_chars"])
    ).evaluate(cleaned)
    signals = extract_quality_signals(cleaned)
    flags = provisional_quality_flags(signals, truncated=truncated, pii_redactions=scrubbed.num_redactions)
    return {
        "upstream_id": str(raw["id"]),
        "title": raw.get("title"),
        "url": raw.get("url"),
        "selected_in_snapshot": str(raw["id"]) in selected_ids,
        "passes_current_cleaner": decision.admitted,
        "current_rejection_reasons": list(decision.reasons),
        "raw_characters": len(raw_combined),
        "normalized_characters": len(normalized),
        "cleaned_characters": len(cleaned),
        "characters_removed_by_normalization": len(raw_combined) - len(normalized),
        "characters_removed_by_truncation": max(0, len(before_truncation) - len(cleaned)),
        "pii_redactions": scrubbed.num_redactions,
        "truncated": truncated,
        "signals": signals,
        "provisional_flags": list(flags),
        "before_preview": preview(raw_combined),
        "after_preview": preview(cleaned),
    }


def sample_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    admitted = [row for row in rows if row["passes_current_cleaner"]]
    rejected = [row for row in rows if not row["passes_current_cleaner"]]
    pii = [row for row in admitted if row["pii_redactions"]]
    truncated = [row for row in admitted if row["truncated"]]
    disambiguation = [row for row in admitted if "disambiguation_page" in row["provisional_flags"]]
    repetitive = sorted(admitted, key=lambda row: row["signals"]["repeated_trigram_fraction"], reverse=True)
    return {
        "rejected_shortest_boundary_cases": sorted(rejected, key=lambda row: row["raw_characters"], reverse=True)[:12],
        "pii_redaction_cases": pii[:12],
        "truncation_cases": sorted(truncated, key=lambda row: row["characters_removed_by_truncation"], reverse=True)[:12],
        "disambiguation_cases": disambiguation[:12],
        "most_repetitive_admitted": repetitive[:12],
    }


def write_markdown(report: dict[str, Any], samples: dict[str, list[dict[str, Any]]]) -> None:
    counts = report["counts"]
    lines = [
        "# Wikipedia general lane: corpus-v0 quality audit",
        "",
        "This is a diagnostic report, not a final admission policy. Provisional flags identify examples for review; they do not silently delete records.",
        "",
        "## Current transformation",
        "",
        "`title + text → NFC/HTML normalization → control and ghost-tag removal → whitespace normalization → PII-pattern masking → 8,000-character slice → basic length/content gate`",
        "",
        "## Inventory",
        "",
        f"- Cached raw candidates: {counts['raw_candidates']:,}",
        f"- Pass the current cleaner: {counts['pass_current_cleaner']:,}",
        f"- Fail the current cleaner: {counts['fail_current_cleaner']:,}",
        f"- Selected before the 4,000-row quota stopped acquisition: {counts['selected_snapshot']:,}",
        f"- Character-truncated candidates: {counts['truncated']:,}",
        f"- Truncated away from a detected text boundary: {counts['truncated_mid_boundary']:,}",
        f"- Candidates with PII-pattern redactions: {counts['pii_redacted']:,}",
        "",
        "## Provisional flag counts",
        "",
        "| Flag | Records |",
        "|---|---:|",
    ]
    lines.extend(f"| {flag} | {count:,} |" for flag, count in report["flag_counts"].items())
    lines.extend(["", "## Signal distributions for currently admitted candidates", "", "| Signal | Min | P05 | Median | P95 | Max |", "|---|---:|---:|---:|---:|---:|"])
    for key, values in report["admitted_signal_summary"].items():
        lines.append(
            f"| {key} | {values['min']:.4g} | {values['p05']:.4g} | {values['median']:.4g} | {values['p95']:.4g} | {values['max']:.4g} |"
        )
    lines.extend(["", "## Review samples", ""])
    for category, rows in samples.items():
        lines.extend([f"### {category.replace('_', ' ').title()}", ""])
        if not rows:
            lines.extend(["No examples in this category.", ""])
            continue
        for row in rows[:5]:
            lines.extend(
                [
                    f"- **{row['title']}** (`{row['upstream_id']}`): flags={row['provisional_flags']}; raw={row['raw_characters']}; cleaned={row['cleaned_characters']}",
                    f"  - Before: {row['before_preview']}",
                    f"  - After: {row['after_preview']}",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Questions for the admission-policy review",
            "",
            "1. Should short factual stubs remain excluded, or become a low-difficulty band?",
            "2. Should disambiguation and list-dominant pages be rejected or downweighted?",
            "3. Should long articles be split at paragraph boundaries instead of sliced at 8,000 characters?",
            "4. Which PII patterns are genuine personal data versus false positives such as dates or identifiers?",
            "5. Which signal combinations define B0–B5 without using an opaque quality model?",
            "",
        ]
    )
    destination = ROOT / "docs" / "quality" / "WIKIPEDIA_GENERAL_V0_AUDIT.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    config = load_source_lock(ROOT / "configs" / "sources.lock.json")
    target = next(item for item in config["targets"] if item["source_id"] == SOURCE_ID)
    snapshot_path = ROOT / "data" / "source_snapshots" / f"{SOURCE_ID}.jsonl.gz"
    selected_ids = {record["upstream_id"] for record in read_jsonl_gz(snapshot_path)}
    rows = [audit_row(raw, selected_ids, target) for raw in load_raw_rows()]
    if not rows:
        raise RuntimeError("No cached raw rows found; acquisition cache is required for before/after auditing")
    admitted = [row for row in rows if row["passes_current_cleaner"]]
    flags = Counter(flag for row in rows for flag in row["provisional_flags"])
    signal_keys = [
        "characters",
        "words",
        "paragraphs",
        "alpha_fraction",
        "list_line_fraction",
        "duplicate_line_fraction",
        "repeated_trigram_fraction",
        "compression_ratio",
        "character_entropy_bits",
    ]
    report = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "baseline_tag": "corpus-v0",
        "policy_status": "diagnostic_only",
        "counts": {
            "raw_candidates": len(rows),
            "pass_current_cleaner": len(admitted),
            "fail_current_cleaner": len(rows) - len(admitted),
            "selected_snapshot": sum(row["selected_in_snapshot"] for row in rows),
            "truncated": sum(row["truncated"] for row in rows),
            "truncated_mid_boundary": sum(
                row["truncated"] and not row["signals"]["ends_at_boundary"] for row in rows
            ),
            "pii_redacted": sum(bool(row["pii_redactions"]) for row in rows),
        },
        "flag_counts": dict(sorted(flags.items())),
        "admitted_signal_summary": {key: signal_summary(admitted, key) for key in signal_keys},
    }
    samples = sample_rows(rows)
    output = ROOT / "analysis" / "data_quality"
    atomic_write_json(output / "wikipedia_general_v0_metrics.json", report)
    atomic_write_json(output / "wikipedia_general_v0_samples.json", samples)
    write_markdown(report, samples)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
