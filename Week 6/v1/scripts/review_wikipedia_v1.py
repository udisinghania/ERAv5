from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import atomic_write_json, read_jsonl_gz  # noqa: E402


def paths_for_version(version: str) -> tuple[Path, Path, Path, Path]:
    return (
        ROOT / "data" / "experiments" / f"corpus_{version}" / "wikipedia_general_en.jsonl.gz",
        ROOT / "configs" / f"quality_policy_{version}.json",
        ROOT / "analysis" / "data_quality" / f"wikipedia_{version}_review_packet.json",
        ROOT / "docs" / "quality" / f"WIKIPEDIA_{version.upper()}_REVIEW_PACKET.md",
    )


def evenly_spaced_sample(
    rows: list[dict[str, Any]],
    count: int,
    *,
    order_key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    """Choose deterministic examples across the ordered range, including both ends."""
    ordered = sorted(rows, key=order_key)
    if len(ordered) <= count:
        return ordered
    indexes = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indexes]


def review_item(record: dict[str, Any], stratum: str) -> dict[str, Any]:
    quality = record["quality"]
    signals = quality["signals"]
    preview = " ".join(record["text"].split())
    start_preview = preview if len(preview) <= 700 else preview[:699].rstrip() + "…"
    end_preview = preview if len(preview) <= 700 else "…" + preview[-699:].lstrip()
    return {
        "sample_id": f"{stratum}:{record['record_id']}",
        "stratum": stratum,
        "record_id": record["record_id"],
        "parent_upstream_id": record["parent_upstream_id"],
        "title": record["metadata"]["title"],
        "band": quality["band"],
        "sampling_weight": quality["sampling_weight"],
        "sampling_cap_groups": quality["sampling_cap_groups"],
        "flags": quality["flags"],
        "chunk_index": record["metadata"]["chunk_index"],
        "chunk_count": record["metadata"]["chunk_count"],
        "chunk_end_boundary": record["metadata"]["chunk_end_boundary"],
        "pii_counts": record["metadata"]["pii_counts"],
        "signals": {
            key: signals[key]
            for key in (
                "characters",
                "words",
                "paragraphs",
                "alpha_fraction",
                "list_line_fraction",
                "duplicate_line_fraction",
                "repeated_trigram_fraction",
                "character_entropy_bits",
            )
        },
        "text_preview": start_preview,
        "text_start_preview": start_preview,
        "text_end_preview": end_preview,
        "human_review": {
            "meaningful_language": None,
            "coherent_and_complete": None,
            "band_agrees": None,
            "cap_group_agrees": None,
            "pii_safe": None,
            "boundary_clean": None,
            "decision": None,
            "notes": "",
        },
    }


def weighted_share(rows: list[dict[str, Any]], total_weight: float) -> float:
    return sum(float(row["quality"]["sampling_weight"]) for row in rows) / total_weight


def build_packet(
    records: list[dict[str, Any]], policy: dict[str, Any], source_artifact: Path
) -> dict[str, Any]:
    total_weight = sum(float(record["quality"]["sampling_weight"]) for record in records)
    by_band = {
        band: [record for record in records if record["quality"]["band"] == band]
        for band in policy["quality_bands"]
    }
    by_cap = {
        group: [
            record
            for record in records
            if group in record["quality"]["sampling_cap_groups"]
        ]
        for group in policy["sampling_caps"]
        if group != "all_B0_combined"
    }
    b0_records = by_band["B0"]

    cap_analysis: dict[str, dict[str, Any]] = {}
    for group, cap in policy["sampling_caps"].items():
        members = b0_records if group == "all_B0_combined" else by_cap[group]
        share = weighted_share(members, total_weight)
        cap_analysis[group] = {
            "records": len(members),
            "weighted_records": sum(float(row["quality"]["sampling_weight"]) for row in members),
            "unconstrained_weighted_share": share,
            "configured_cap": float(cap),
            "cap_would_bind": share > float(cap),
        }

    strata: dict[str, list[dict[str, Any]]] = {}
    length_order = lambda row: (row["quality"]["signals"]["characters"], row["record_id"])
    for band, members in by_band.items():
        strata[f"band_{band}"] = evenly_spaced_sample(members, 5, order_key=length_order)
    for group, members in by_cap.items():
        strata[f"cap_{group}"] = evenly_spaced_sample(members, 5, order_key=length_order)

    unusual_boundaries = [
        record
        for record in records
        if record["metadata"]["chunk_end_boundary"] not in {"paragraph"}
    ]
    strata["boundary_non_paragraph"] = sorted(
        unusual_boundaries,
        key=lambda row: (row["metadata"]["chunk_end_boundary"], row["record_id"]),
    )

    samples = [
        review_item(record, stratum)
        for stratum, members in strata.items()
        for record in members
    ]
    return {
        "schema_version": 1,
        "purpose": "Human validation gate before adapting quality rules to other lanes",
        "source_artifact": source_artifact.relative_to(ROOT).as_posix(),
        "policy_id": policy["policy_id"],
        "population": {
            "records": len(records),
            "weighted_records": total_weight,
            "band_counts": dict(sorted(Counter(row["quality"]["band"] for row in records).items())),
            "boundary_counts": dict(
                sorted(Counter(row["metadata"]["chunk_end_boundary"] for row in records).items())
            ),
        },
        "cap_analysis": cap_analysis,
        "review_rubric": {
            "boolean_fields": [
                "meaningful_language",
                "coherent_and_complete",
                "band_agrees",
                "cap_group_agrees",
                "pii_safe",
                "boundary_clean",
            ],
            "decision_values": ["keep", "downweight", "reject"],
            "instruction": "Review text, not just numeric signals. Record disagreements and revise rules only after patterns recur.",
        },
        "sample_count": len(samples),
        "samples": samples,
    }


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_markdown(packet: dict[str, Any], report_path: Path, version: str) -> None:
    population = packet["population"]
    lines = [
        f"# Wikipedia corpus-{version} review packet",
        "",
        "This is the human validation gate between building the Wikipedia-specific quality policy and adapting quality logic to the other six data lanes.",
        "",
        "## What to learn from this stage",
        "",
        "The signals are measurements, the band is a policy decision derived from those measurements, the weight controls ordinary sampling, and a cap is a final safety ceiling. A cap may be configured without activating.",
        "",
        "## Population and weighted supply",
        "",
        f"- Physical records: {population['records']:,}",
        f"- Weighted records before caps: {population['weighted_records']:,.2f}",
        f"- Review examples: {packet['sample_count']:,}",
        "",
        "| Band | Physical records |",
        "|---|---:|",
    ]
    lines.extend(f"| {band} | {count:,} |" for band, count in population["band_counts"].items())
    lines.extend(
        [
            "",
            "## Do the caps currently activate?",
            "",
            "| Group | Records | Share after weights | Cap | Activates? |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for group, values in packet["cap_analysis"].items():
        lines.append(
            f"| {group} | {values['records']:,} | "
            f"{percentage(values['unconstrained_weighted_share'])} | "
            f"{percentage(values['configured_cap'])} | "
            f"{'yes' if values['cap_would_bind'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Only an activating cap changes the distribution beyond the sampling weights. Non-activating caps remain useful as guards if the corpus grows later.",
            "",
            "## Review rubric",
            "",
            "For each example, inspect whether it contains meaningful language, is coherent and complete, belongs in its assigned band/cap group, is PII-safe, and ends cleanly. Then choose keep, downweight, or reject. Do not change a threshold after one unusual example; look for a repeated error pattern.",
            "",
            "## Deterministic sample",
            "",
            f"Five examples are drawn across the length range of every band and cap group. All {sum(population['boundary_counts'].get(key, 0) for key in population['boundary_counts'] if key != 'paragraph')} non-paragraph boundary chunks are included. Some records intentionally appear in more than one stratum because they test different claims.",
            "",
        ]
    )
    current_stratum = None
    for sample in packet["samples"]:
        if sample["stratum"] != current_stratum:
            current_stratum = sample["stratum"]
            lines.extend([f"### {current_stratum}", ""])
        signals = sample["signals"]
        lines.extend(
            [
                f"#### {sample['title']} (`{sample['record_id']}`)",
                "",
                f"Band/weight: {sample['band']} / {sample['sampling_weight']}; "
                f"caps: {sample['sampling_cap_groups'] or 'none'}; flags: {sample['flags'] or 'none'}",
                "",
                f"Chunk: {sample['chunk_index'] + 1}/{sample['chunk_count']}; "
                f"end boundary: {sample['chunk_end_boundary']}; characters: {signals['characters']:,}; "
                f"paragraphs: {signals['paragraphs']}; alpha: {signals['alpha_fraction']:.3f}; "
                f"repeated trigrams: {signals['repeated_trigram_fraction']:.3f}",
                "",
                "Beginning:",
                "",
                f"> {sample['text_start_preview']}",
                "",
                "Ending:",
                "",
                f"> {sample['text_end_preview']}",
                "",
                "Review: meaningful [ ]  coherent [ ]  band [ ]  cap [ ]  PII [ ]  boundary [ ]",
                "",
                "Decision: keep [ ]  downweight [ ]  reject [ ]",
                "",
                "Notes:",
                "",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic Wikipedia quality review packet")
    parser.add_argument("--version", choices=("v1", "v2", "v3"), default="v1")
    args = parser.parse_args()
    corpus_path, policy_path, packet_path, report_path = paths_for_version(args.version)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    records = list(read_jsonl_gz(corpus_path))
    if not records:
        raise RuntimeError(f"Wikipedia corpus-{args.version} is empty")
    packet = build_packet(records, policy, corpus_path)
    atomic_write_json(packet_path, packet)
    write_markdown(packet, report_path, args.version)
    summary = {
        "status": "READY_FOR_HUMAN_REVIEW",
        "records": packet["population"]["records"],
        "weighted_records": packet["population"]["weighted_records"],
        "review_samples": packet["sample_count"],
        "activating_caps": [
            group for group, values in packet["cap_analysis"].items() if values["cap_would_bind"]
        ],
        "packet": packet_path.relative_to(ROOT).as_posix(),
        "report": report_path.relative_to(ROOT).as_posix(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
