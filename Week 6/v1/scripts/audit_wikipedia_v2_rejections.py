from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.acquisition import load_source_lock  # noqa: E402
from era6.canonical import atomic_write_json  # noqa: E402
from era6.chunking import boundary_aware_chunks_v2  # noqa: E402
from era6.cleaning import SourceAwarePIIScrubber, TextNormalizer  # noqa: E402
from era6.quality import extract_quality_signals_v2, provisional_quality_flags_v2  # noqa: E402
from build_wikipedia_v2 import load_raw_rows  # noqa: E402


OUTPUT_PATH = ROOT / "analysis" / "data_quality" / "wikipedia_v2_rejection_audit.json"
REPORT_PATH = ROOT / "docs" / "quality" / "WIKIPEDIA_V2_REJECTION_AUDIT.md"
HARD_FLAGS = {"raw_wikitable_markup", "orphaned_table_footnotes"}


def preview(text: str, limit: int = 400) -> dict[str, str]:
    value = " ".join(text.split())
    if len(value) <= limit:
        return {"start": value, "end": value}
    return {"start": value[: limit - 1].rstrip() + "…", "end": "…" + value[-(limit - 1) :].lstrip()}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round(fraction * (len(ordered) - 1))
    return ordered[index]


def main() -> int:
    policy = json.loads((ROOT / "configs" / "quality_policy_v2.json").read_text(encoding="utf-8"))
    sources = load_source_lock(ROOT / "configs" / "sources.lock.json")
    target = next(item for item in sources["targets"] if item["source_id"] == "wikipedia_general_en")
    normalizer, scrubber = TextNormalizer(), SourceAwarePIIScrubber()
    rejected: list[dict[str, Any]] = []

    for raw in load_raw_rows():
        title = normalizer.normalize(str(raw.get("title", "")))
        body = normalizer.normalize(str(raw.get("text", "")))
        combined = f"{title}\n\n{body}".strip()
        if len(combined) < int(policy["minimum_parent_characters"]):
            continue
        clean_title = scrubber.scrub(title, source_class=policy["source_class"])
        clean_body = scrubber.scrub(body, source_class=policy["source_class"])
        pii_redactions = clean_title.num_redactions + clean_body.num_redactions
        chunks = boundary_aware_chunks_v2(
            clean_title.text,
            clean_body.text,
            maximum_characters=int(policy["chunking"]["maximum_characters"]),
            minimum_continuation_characters=int(policy["chunking"]["minimum_continuation_characters"]),
        )
        for chunk in chunks:
            signals = extract_quality_signals_v2(chunk.text)
            flags = provisional_quality_flags_v2(
                signals,
                short_continuation=len(combined) >= 400 and len(chunk.text) < 400,
                pii_redactions=pii_redactions,
                rules=policy["reviewed_signal_rules"],
            )
            reasons = sorted(HARD_FLAGS & set(flags))
            if not reasons:
                continue
            rejected.append(
                {
                    "parent_upstream_id": str(raw["id"]),
                    "title": clean_title.text,
                    "source_chunk_index": chunk.index,
                    "source_chunk_count": chunk.count,
                    "reasons": reasons,
                    "signals": {
                        "characters": signals["characters"],
                        "wikitable_markup_lines": signals["wikitable_markup_lines"],
                        "wikitable_markup_line_fraction": signals["wikitable_markup_line_fraction"],
                        "list_line_fraction": signals["list_line_fraction"],
                        "duplicate_line_fraction": signals["duplicate_line_fraction"],
                        "repeated_trigram_fraction": signals["repeated_trigram_fraction"],
                    },
                    "preview": preview(chunk.text),
                }
            )

    rejected.sort(key=lambda item: (item["title"], item["source_chunk_index"]))
    raw_table = [item for item in rejected if "raw_wikitable_markup" in item["reasons"]]
    orphaned = [item for item in rejected if "orphaned_table_footnotes" in item["reasons"]]
    marker_counts = [float(item["signals"]["wikitable_markup_lines"]) for item in raw_table]
    marker_fractions = [float(item["signals"]["wikitable_markup_line_fraction"]) for item in raw_table]
    reviewed_titles = {
        "Coach Trip (series 8)",
        "1951 Ohio State Buckeyes baseball team",
        "FC Luch Vladivostok",
        "Results of the 1994 Sri Lankan general election by electoral district",
    }
    report = {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "rejected_chunks": len(rejected),
        "reason_counts": dict(sorted(Counter(reason for item in rejected for reason in item["reasons"]).items())),
        "raw_wikitable_evidence": {
            "minimum_marker_lines": min(marker_counts),
            "median_marker_lines": percentile(marker_counts, 0.50),
            "maximum_marker_lines": max(marker_counts),
            "minimum_marker_fraction": min(marker_fractions),
            "median_marker_fraction": percentile(marker_fractions, 0.50),
            "maximum_marker_fraction": max(marker_fractions),
        },
        "reviewed_titles_found": sorted(reviewed_titles & {item["title"] for item in rejected}),
        "reviewed_titles_missing": sorted(reviewed_titles - {item["title"] for item in rejected}),
        "lowest_marker_rejections": sorted(
            raw_table,
            key=lambda item: (
                item["signals"]["wikitable_markup_lines"],
                item["signals"]["wikitable_markup_line_fraction"],
                item["title"],
            ),
        )[:8],
        "orphaned_footnote_rejections": orphaned,
        "all_rejections": rejected,
    }
    if report["reviewed_titles_missing"]:
        raise RuntimeError(f"Reviewed rejection regression: {report['reviewed_titles_missing']}")
    atomic_write_json(OUTPUT_PATH, report)

    evidence = report["raw_wikitable_evidence"]
    lines = [
        "# Wikipedia corpus-v2 targeted rejection audit",
        "",
        "This audit reconstructs every v2 hard-rejected chunk from the cached raw candidates. It validates rejection evidence independently of the ordinary retained-record review sample.",
        "",
        "## Result",
        "",
        f"- Rejected chunks: {report['rejected_chunks']}",
        f"- Raw-wikitable chunks: {report['reason_counts'].get('raw_wikitable_markup', 0)}",
        f"- Orphaned-footnote chunks: {report['reason_counts'].get('orphaned_table_footnotes', 0)}",
        f"- Wikitable marker lines per rejected chunk: min {evidence['minimum_marker_lines']:.0f}, median {evidence['median_marker_lines']:.0f}, max {evidence['maximum_marker_lines']:.0f}",
        f"- Wikitable marker fraction: min {evidence['minimum_marker_fraction']:.1%}, median {evidence['median_marker_fraction']:.1%}, max {evidence['maximum_marker_fraction']:.1%}",
        "- All four human-reviewed rejection titles were reconstructed and rejected.",
        "",
        "## Lowest-marker spot checks",
        "",
        "These are the closest cases to the rejection threshold and therefore the most useful false-positive audit sample.",
        "",
    ]
    for item in report["lowest_marker_rejections"]:
        lines.extend(
            [
                f"### {item['title']} (chunk {item['source_chunk_index'] + 1}/{item['source_chunk_count']})",
                "",
                f"Marker lines: {item['signals']['wikitable_markup_lines']}; marker fraction: {item['signals']['wikitable_markup_line_fraction']:.1%}",
                "",
                f"> Beginning: {item['preview']['start']}",
                "",
                f"> Ending: {item['preview']['end']}",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "rejected_chunks": len(rejected), "report": REPORT_PATH.relative_to(ROOT).as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
