from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from era6.acquisition import load_source_lock  # noqa: E402
from era6.canonical import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    read_jsonl_gz,
    sha256_bytes,
    sha256_file,
    sha256_text,
    write_jsonl_gz,
)
from era6.chunking import boundary_aware_chunks_v2  # noqa: E402
from era6.cleaning import SourceAwarePIIScrubber, TextNormalizer, strip_wikitable_markup  # noqa: E402
from era6.quality import extract_quality_signals_v2, provisional_quality_flags_v2  # noqa: E402
from build_wikipedia_v2 import build_record, load_raw_rows  # noqa: E402


SOURCE_ID = "wikipedia_general_en"
HARD_CHUNK_FLAGS = {"raw_wikitable_markup", "orphaned_table_footnotes"}


def has_language_content(text: str) -> bool:
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in text)


def write_markdown(report: dict[str, Any]) -> None:
    before, after = report["v2"], report["v3"]
    lines = [
        "# Wikipedia general: corpus-v2 versus corpus-v3",
        "",
        "Corpus-v3 is the final Wikipedia policy candidate from the v2 human review. Corpus-v2 remains the reviewed structural baseline.",
        "",
        "## Before and after",
        "",
        "| Measure | corpus-v2 | corpus-v3 |",
        "|---|---:|---:|",
        f"| Parents retained | {before['parents']:,} | {after['admitted_parent_documents']:,} |",
        f"| Records retained | {before['records']:,} | {after['records']:,} |",
        f"| Text characters | {before['text_characters']:,} | {after['text_characters']:,} |",
        f"| Table-affected chunks salvaged | 0 | {after['flag_counts'].get('table_markup_removed', 0):,} |",
        f"| Stat-heavy compact lists capped | 0 | {after['flag_counts'].get('stat_heavy_list', 0):,} |",
        f"| Human sensitive-context caps | 0 | {after['flag_counts'].get('human_sensitive_context_reviewed', 0):,} |",
        "",
        "## Decision logic",
        "",
        "- Raw wikitable blocks are removed rather than rendered. Any salvaged surrounding prose is conservatively placed in capped B0.",
        "- Pelopas-style compact numeric/honours lists use a separate stat-heavy-list rule and capped B0.",
        "- The Alachua County case is a hashed human-review override. A naive automated sensitive-name detector was rejected because it flagged 891 mostly ordinary public-reference chunks.",
        "- Orphaned table footnotes remain hard rejections because the substantive table content is absent.",
        "",
        "## v3 bands",
        "",
        "| Band | Records |",
        "|---|---:|",
    ]
    lines.extend(f"| {band} | {count:,} |" for band, count in after["quality_band_counts"].items())
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "The v3 verifier checks the reviewed Pelopas and Alachua decisions, table salvage, hard-rejection removal, lineage, hashes, and deterministic record counts.",
            "",
        ]
    )
    path = ROOT / "docs" / "quality" / "WIKIPEDIA_V2_V3_COMPARISON.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    policy_path = ROOT / "configs" / "quality_policy_v3.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_hash = f"sha256:{sha256_bytes(canonical_json_bytes(policy))}"
    sources = load_source_lock(ROOT / "configs" / "sources.lock.json")
    target = next(item for item in sources["targets"] if item["source_id"] == SOURCE_ID)
    normalizer, scrubber = TextNormalizer(), SourceAwarePIIScrubber()
    review_overrides = {
        (item["parent_upstream_id"], item["content_sha256"]): item
        for item in policy.get("manual_review_overrides", [])
    }

    records: list[dict[str, Any]] = []
    parent_rejections: Counter[str] = Counter()
    chunk_rejections: Counter[str] = Counter()
    rejected_titles: Counter[str] = Counter()
    cleanup_counts: Counter[str] = Counter()
    admitted_parents = 0

    for raw in load_raw_rows():
        title = normalizer.normalize(str(raw.get("title", "")))
        original_body = normalizer.normalize(str(raw.get("text", "")))
        original_combined = f"{title}\n\n{original_body}".strip()
        if len(original_combined) < int(policy["minimum_parent_characters"]):
            parent_rejections["parent_below_300_characters"] += 1
            continue
        cleanup = strip_wikitable_markup(original_body)
        body = cleanup.text
        combined = f"{title}\n\n{body}".strip()
        if len(combined) < int(policy["minimum_parent_characters"]):
            parent_rejections["below_300_characters_after_wikitable_cleanup"] += 1
            continue
        if not has_language_content(combined):
            parent_rejections["no_language_content"] += 1
            continue
        if combined.count("�") / max(1, len(combined)) > 0.001:
            parent_rejections["excessive_replacement_characters"] += 1
            continue

        if cleanup.removed_lines:
            cleanup_counts["parents_with_table_cleanup"] += 1
            cleanup_counts["removed_table_lines"] += cleanup.removed_lines
            cleanup_counts["removed_table_characters"] += cleanup.removed_characters
            cleanup_counts["removed_table_blocks"] += cleanup.removed_blocks

        clean_title = scrubber.scrub(title, source_class=policy["source_class"])
        clean_body = scrubber.scrub(body, source_class=policy["source_class"])
        pii_counts = {
            key: clean_title.counts.get(key, 0) + clean_body.counts.get(key, 0)
            for key in ("email", "phone", "ipv4", "secret")
        }
        chunks = boundary_aware_chunks_v2(
            clean_title.text,
            clean_body.text,
            maximum_characters=int(policy["chunking"]["maximum_characters"]),
            minimum_continuation_characters=int(policy["chunking"]["minimum_continuation_characters"]),
        )
        candidates: list[tuple[Any, dict[str, Any], tuple[str, ...]]] = []
        for chunk in chunks:
            signals = extract_quality_signals_v2(chunk.text)
            flags = provisional_quality_flags_v2(
                signals,
                short_continuation=len(combined) >= 400 and len(chunk.text) < 400,
                pii_redactions=sum(pii_counts.values()),
                rules=policy["reviewed_signal_rules"],
                table_markup_removed=bool(cleanup.removed_lines),
            )
            override = review_overrides.get((str(raw["id"]), sha256_text(chunk.text)))
            if override:
                flags = tuple(sorted(set(flags) | {override["flag"]}))
            hard_rejections = HARD_CHUNK_FLAGS & set(flags)
            if hard_rejections:
                chunk_rejections.update(hard_rejections)
                rejected_titles[title] += 1
                continue
            candidates.append((chunk, signals, flags))

        if not candidates:
            parent_rejections["all_chunks_rejected"] += 1
            continue
        admitted_parents += 1
        for output_index, (chunk, signals, flags) in enumerate(candidates):
            record = build_record(
                target=target,
                policy=policy,
                policy_hash=policy_hash,
                raw=raw,
                title=clean_title.text,
                chunk=chunk,
                output_index=output_index,
                output_count=len(candidates),
                parent_characters=len(combined),
                pii_counts=pii_counts,
                signals=signals,
                flags=flags,
            )
            record["metadata"]["wikitable_cleanup"] = {
                "removed_lines": cleanup.removed_lines,
                "removed_characters": cleanup.removed_characters,
                "removed_blocks": cleanup.removed_blocks,
            }
            records.append(record)

    records.sort(key=lambda record: record["record_id"])
    duplicate_hashes = Counter(record["content_sha256"] for record in records)
    if any(count > 1 for count in duplicate_hashes.values()):
        raise RuntimeError("corpus-v3 produced exact duplicate chunks")

    output_root = ROOT / "data" / "experiments" / "corpus_v3"
    snapshot_path = output_root / "wikipedia_general_en.jsonl.gz"
    stats = write_jsonl_gz(snapshot_path, records)
    v2_path = ROOT / "data" / "experiments" / "corpus_v2" / "wikipedia_general_en.jsonl.gz"
    v2_records = list(read_jsonl_gz(v2_path))
    band_counts = Counter(record["quality"]["band"] for record in records)
    flag_counts = Counter(flag for record in records for flag in record["quality"]["flags"])
    cap_counts = Counter(group for record in records for group in record["quality"]["sampling_cap_groups"])
    boundary_counts = Counter(record["metadata"]["chunk_end_boundary"] for record in records)
    report = {
        "schema_version": 3,
        "policy": policy,
        "policy_hash": policy_hash,
        "source_revision": target["revision"],
        "v2": {
            "artifact_sha256": sha256_file(v2_path),
            "parents": len({record["parent_upstream_id"] for record in v2_records}),
            "records": len(v2_records),
            "text_characters": sum(len(record["text"]) for record in v2_records),
        },
        "v3": {
            "admitted_parent_documents": admitted_parents,
            "rejected_parent_documents": sum(parent_rejections.values()),
            "parent_rejection_counts": dict(sorted(parent_rejections.items())),
            "chunk_rejection_counts": dict(sorted(chunk_rejections.items())),
            "rejected_title_counts": dict(sorted(rejected_titles.items())),
            "cleanup_counts": dict(sorted(cleanup_counts.items())),
            "records": len(records),
            "text_characters": sum(len(record["text"]) for record in records),
            "quality_band_counts": dict(sorted(band_counts.items())),
            "flag_counts": dict(sorted(flag_counts.items())),
            "sampling_cap_group_counts": dict(sorted(cap_counts.items())),
            "chunk_end_boundary_counts": dict(sorted(boundary_counts.items())),
        },
        "artifact": {"path": snapshot_path.relative_to(ROOT).as_posix(), **stats},
        "inputs": {
            "policy_sha256": sha256_file(policy_path),
            "v2_snapshot_sha256": sha256_file(v2_path),
        },
    }
    atomic_write_json(output_root / "comparison_report.json", report)
    write_markdown(report)
    print(json.dumps(report["v3"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
