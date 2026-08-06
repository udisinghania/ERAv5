from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import atomic_write_json, read_jsonl_gz  # noqa: E402


EXPERIMENT_ROOT = ROOT / "data" / "experiments" / "remaining_lanes_v1"
OUTPUT_PATH = ROOT / "analysis" / "data_quality" / "remaining_lanes_v1_review_packet.json"
REPORT_PATH = ROOT / "docs" / "quality" / "REMAINING_LANES_V1_REVIEW_PACKET.md"


def preview(text: str, limit: int = 700) -> dict[str, str]:
    value = " ".join(text.split())
    if len(value) <= limit:
        return {"start": value, "end": value}
    return {"start": value[: limit - 1].rstrip() + "…", "end": "…" + value[-(limit - 1) :].lstrip()}


def marker_context(text: str, marker: str = "[PHONE]", radius: int = 180) -> list[str]:
    contexts = []
    start = 0
    while len(contexts) < 3:
        index = text.find(marker, start)
        if index < 0:
            break
        left = max(0, index - radius)
        right = min(len(text), index + len(marker) + radius)
        contexts.append(" ".join(text[left:right].split()))
        start = index + len(marker)
    return contexts


def source_packet(source_report: dict[str, Any]) -> dict[str, Any]:
    source_id = source_report["source_id"]
    baseline = list(
        read_jsonl_gz(ROOT / "data" / "source_snapshots" / f"{source_id}.jsonl.gz")
    )
    experiment = list(read_jsonl_gz(ROOT / source_report["artifact"]["path"]))
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in experiment:
        parent_id = str(record.get("parent_upstream_id", record["upstream_id"]))
        by_parent[parent_id].append(record)
    for records in by_parent.values():
        records.sort(key=lambda row: row.get("metadata", {}).get("chunk_index", 0))

    phone_rows = sorted(
        [row for row in baseline if "[PHONE]" in row["text"]],
        key=lambda row: (-row["text"].count("[PHONE]"), row["upstream_id"]),
    )[:3]
    repairs = []
    for row in phone_rows:
        chunks = by_parent.get(str(row["upstream_id"]), [])
        rebuilt = "\n".join(chunk["text"] for chunk in chunks)
        repairs.append(
            {
                "parent_upstream_id": row["upstream_id"],
                "baseline_phone_markers": row["text"].count("[PHONE]"),
                "baseline_marker_contexts": marker_context(row["text"]),
                "baseline_preview": preview(row["text"]),
                "experiment_preview": preview(rebuilt),
                "experiment_chunks": len(chunks),
            }
        )

    split_parents = sorted(
        [(parent_id, rows) for parent_id, rows in by_parent.items() if len(rows) > 1],
        key=lambda item: (-len(item[1]), item[0]),
    )[:3]
    boundaries = []
    for parent_id, chunks in split_parents:
        joins = []
        for left, right in zip(chunks, chunks[1:]):
            joins.append(
                {
                    "left_boundary": left["metadata"]["chunk_end_boundary"],
                    "left_end": preview(left["text"], 300)["end"],
                    "right_start": preview(right["text"], 300)["start"],
                }
            )
        boundaries.append(
            {"parent_upstream_id": parent_id, "chunks": len(chunks), "joins": joins[:3]}
        )
    return {
        "source_id": source_id,
        "lane": source_report["lane"],
        "summary": source_report["summary"],
        "numeric_repair_samples": repairs,
        "boundary_samples": boundaries,
    }


def write_markdown(packet: dict[str, Any]) -> None:
    lines = [
        "# Remaining lanes v1 review packet",
        "",
        "Review two claims separately: whether former `[PHONE]` replacements were valid numeric content, and whether new chunks end/start at acceptable structural boundaries. The experiment does not replace baseline snapshots until this gate closes.",
        "",
    ]
    for source in packet["sources"]:
        summary = source["summary"]
        lines.extend(
            [
                f"## {source['lane']} — {source['source_id']}",
                "",
                f"Parents: {summary['parents']:,}; records: {summary['records']:,}; recovered markers: {summary['recovered_phone_markers']:,}; character gain: {summary['experiment_characters'] - summary['baseline_characters']:,}",
                "",
                "### Numeric-repair samples",
                "",
            ]
        )
        if not source["numeric_repair_samples"]:
            lines.extend(["No baseline `[PHONE]` marker in this source.", ""])
        for sample in source["numeric_repair_samples"]:
            lines.extend(
                [
                    f"#### Parent `{sample['parent_upstream_id']}` ({sample['baseline_phone_markers']} former markers)",
                    "",
                ]
            )
            for context in sample["baseline_marker_contexts"]:
                lines.extend([f"> Baseline: {context}", ""])
            lines.extend(
                [
                    f"> Rebuilt beginning: {sample['experiment_preview']['start']}",
                    "",
                    f"> Rebuilt ending: {sample['experiment_preview']['end']}",
                    "",
                    "Repair judgment: valid numeric recovery [ ]  actual phone exposed [ ]  unclear [ ]",
                    "",
                ]
            )
        lines.extend(["### Boundary samples", ""])
        if not source["boundary_samples"]:
            lines.extend(["No selected parent required splitting.", ""])
        for sample in source["boundary_samples"]:
            lines.extend(
                [
                    f"#### Parent `{sample['parent_upstream_id']}` ({sample['chunks']} chunks)",
                    "",
                ]
            )
            for join in sample["joins"]:
                lines.extend(
                    [
                        f"> Left end ({join['left_boundary']}): {join['left_end']}",
                        "",
                        f"> Right start: {join['right_start']}",
                        "",
                    ]
                )
            lines.extend(["Boundary judgment: clean [ ]  broken [ ]  unclear [ ]", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    comparison = json.loads((EXPERIMENT_ROOT / "comparison_report.json").read_text(encoding="utf-8"))
    sources = [source_packet(source) for source in comparison["sources"]]
    packet = {
        "schema_version": 1,
        "policy_hash": comparison["policy_hash"],
        "source_count": len(sources),
        "sources": sources,
    }
    atomic_write_json(OUTPUT_PATH, packet)
    write_markdown(packet)
    print(
        json.dumps(
            {
                "status": "READY_FOR_REVIEW",
                "sources": len(sources),
                "report": REPORT_PATH.relative_to(ROOT).as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
