from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.acquisition import candidate_text_and_metadata, load_source_lock  # noqa: E402
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
from era6.cleaning import SourceAwarePIIScrubber, TextNormalizer  # noqa: E402


OUTPUT_ROOT = ROOT / "data" / "experiments" / "remaining_lanes_v1"


def load_cached_rows(source_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "data" / "acquisition_cache" / source_id).glob("offset-*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(item["row"] for item in json.load(stream).get("rows", []))
    return rows


def raw_index(target: dict[str, Any]) -> dict[str, tuple[dict[str, Any], str, dict[str, Any]]]:
    result: dict[str, tuple[dict[str, Any], str, dict[str, Any]]] = {}
    allowed = {value.casefold() for value in target.get("allowed_row_licenses", [])}
    for raw in load_cached_rows(target["source_id"]):
        if allowed and str(raw.get("license", "")).casefold() not in allowed:
            continue
        text, upstream_id, metadata = candidate_text_and_metadata(target, raw)
        if upstream_id:
            result[str(upstream_id)] = (raw, text, metadata)
    return result


def build_source(
    target: dict[str, Any], source_policy: dict[str, Any], policy_hash: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_path = ROOT / "data" / "source_snapshots" / f"{target['source_id']}.jsonl.gz"
    baseline = list(read_jsonl_gz(baseline_path))
    baseline_phone_markers = sum(record["text"].count("[PHONE]") for record in baseline)
    baseline_characters = sum(len(record["text"]) for record in baseline)
    if source_policy["pii_mode"] == "baseline_locked":
        replacements = {
            item["upstream_id"]: item
            for item in source_policy.get("adjudicated_replacements", [])
        }
        repaired: list[dict[str, Any]] = []
        applied = 0
        for baseline_record in baseline:
            replacement = replacements.get(str(baseline_record["upstream_id"]))
            if replacement is None:
                repaired.append(baseline_record)
                continue
            expected = int(replacement.get("expected_occurrences", 1))
            if baseline_record["text"].count(replacement["old"]) != expected:
                raise RuntimeError(
                    f"{target['source_id']} adjudicated replacement no longer matches "
                    f"{baseline_record['upstream_id']}"
                )
            text = baseline_record["text"].replace(replacement["old"], replacement["new"])
            content_hash = sha256_text(text)
            record_key = (
                f"{target['source_id']}:{baseline_record['upstream_id']}:"
                f"human-adjudicated:{content_hash}:{policy_hash}"
            )
            repaired.append(
                {
                    **baseline_record,
                    "record_id": f"rec_{sha256_text(record_key)[:24]}",
                    "content_sha256": content_hash,
                    "text": text,
                    "metadata": {
                        **baseline_record.get("metadata", {}),
                        "baseline_record_id": baseline_record["record_id"],
                        "manual_adjudication": replacement["reason"],
                        "policy_hash": policy_hash,
                    },
                }
            )
            applied += 1
        repaired.sort(key=lambda record: record["record_id"])
        experiment_characters = sum(len(record["text"]) for record in repaired)
        experiment_phone_markers = sum(record["text"].count("[PHONE]") for record in repaired)
        return repaired, {
            "status": (
                "baseline_locked_with_human_adjudicated_repair"
                if applied
                else "baseline_locked_raw_parent_not_available"
            ),
            "parents": len(baseline),
            "records": len(repaired),
            "baseline_characters": baseline_characters,
            "experiment_characters": experiment_characters,
            "baseline_phone_markers": baseline_phone_markers,
            "experiment_phone_markers": experiment_phone_markers,
            "recovered_phone_markers": baseline_phone_markers - experiment_phone_markers,
            "adjudicated_replacements": applied,
            "chunk_boundary_counts": {"baseline": len(repaired)},
        }

    index = raw_index(target)
    normalizer, scrubber = TextNormalizer(), SourceAwarePIIScrubber()
    records: list[dict[str, Any]] = []
    missing = 0
    excluded = 0
    excluded_parent_ids = set(source_policy.get("excluded_parent_upstream_ids", []))
    boundary_counts: Counter[str] = Counter()
    pii_counts: Counter[str] = Counter()
    parents_split = 0
    for baseline_record in baseline:
        upstream_id = str(baseline_record["upstream_id"])
        if upstream_id in excluded_parent_ids:
            excluded += 1
            continue
        item = index.get(upstream_id)
        if item is None:
            missing += 1
            continue
        _raw, raw_text, raw_metadata = item
        normalized = normalizer.normalize(raw_text)
        scrubbed = scrubber.scrub(normalized, source_class=source_policy["pii_mode"])
        pii_counts.update(scrubbed.counts)
        if len(scrubbed.text) < int(target["min_chars"]):
            continue
        chunks = boundary_aware_chunks_v2(
            "",
            scrubbed.text,
            maximum_characters=int(target["max_chars"]),
            minimum_continuation_characters=min(300, int(target["min_chars"])),
        )
        if len(chunks) > 1:
            parents_split += 1
        for chunk in chunks:
            content_hash = sha256_text(chunk.text)
            record_key = f"{target['source_id']}:{upstream_id}:{chunk.index}:{content_hash}:{policy_hash}"
            row_license = raw_metadata.get("row_license") or target["license_id"]
            records.append(
                {
                    "record_id": f"rec_{sha256_text(record_key)[:24]}",
                    "group_id": baseline_record["group_id"],
                    "source_id": target["source_id"],
                    "source_revision": target["revision"],
                    "upstream_id": f"{upstream_id}:chunk-{chunk.index:04d}",
                    "parent_upstream_id": upstream_id,
                    "content_sha256": content_hash,
                    "capability_lane": target["lane"],
                    "provenance_tier": target["provenance_tier"],
                    "permission": target["permission"],
                    "language": target["language"],
                    "license_id": row_license,
                    "pii_redactions": sum(scrubbed.counts.values()),
                    "text": chunk.text,
                    "metadata": {
                        **{key: value for key, value in raw_metadata.items() if value is not None},
                        "baseline_record_id": baseline_record["record_id"],
                        "chunk_index": chunk.index,
                        "chunk_count": chunk.count,
                        "chunk_end_boundary": chunk.end_boundary,
                        "pii_mode": source_policy["pii_mode"],
                        "policy_hash": policy_hash,
                    },
                }
            )
            boundary_counts[chunk.end_boundary] += 1
    records.sort(key=lambda record: record["record_id"])
    if missing:
        raise RuntimeError(f"{target['source_id']} could not reconstruct {missing} baseline parents")
    return records, {
        "status": "rebuilt_from_cached_raw_parent",
        "parents": len(baseline) - excluded,
        "records": len(records),
        "excluded_parents": excluded,
        "parents_split": parents_split,
        "baseline_characters": baseline_characters,
        "experiment_characters": sum(len(record["text"]) for record in records),
        "baseline_phone_markers": baseline_phone_markers,
        "experiment_phone_markers": sum(record["text"].count("[PHONE]") for record in records),
        "recovered_phone_markers": baseline_phone_markers
        - sum(record["text"].count("[PHONE]") for record in records),
        "pii_counts": dict(sorted(pii_counts.items())),
        "chunk_boundary_counts": dict(sorted(boundary_counts.items())),
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Remaining lanes: cleaning experiment v1",
        "",
        "This versioned experiment rebuilds the same selected source parents with numeric-safe PII policies and boundary-aware chunks. Baseline snapshots remain unchanged.",
        "",
        "| Lane | Source | Parents | Records | Split parents | Recovered `[PHONE]` | Character gain |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["sources"]:
        values = item["summary"]
        lines.append(
            f"| {item['lane']} | {item['source_id']} | {values['parents']:,} | {values['records']:,} | "
            f"{values.get('parents_split', 0):,} | {values['recovered_phone_markers']:,} | "
            f"{values['experiment_characters'] - values['baseline_characters']:,} |"
        )
    lines.extend(
        [
            "",
            "Recovered `[PHONE]` markers are candidate false-positive repairs; review packets must confirm them before this experiment replaces any baseline supply. Character gain comes from retaining boundary chunks instead of slicing oversized parents.",
            "",
        ]
    )
    path = ROOT / "docs" / "quality" / "REMAINING_LANES_V0_V1_COMPARISON.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    policy_path = ROOT / "configs" / "remaining_lanes_policy_v1.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_hash = f"sha256:{sha256_bytes(canonical_json_bytes(policy))}"
    source_policies = {item["source_id"]: item for item in policy["sources"]}
    config = load_source_lock(ROOT / "configs" / "sources.lock.json")
    targets = [
        target
        for target in config["targets"]
        if target["permission"] == "train" and target["source_id"] != "wikipedia_general_en"
    ]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_reports = []
    for target in targets:
        records, summary = build_source(target, source_policies[target["source_id"]], policy_hash)
        artifact_path = OUTPUT_ROOT / f"{target['source_id']}.jsonl.gz"
        stats = write_jsonl_gz(artifact_path, records)
        source_reports.append(
            {
                "source_id": target["source_id"],
                "lane": target["lane"],
                "summary": summary,
                "artifact": {"path": artifact_path.relative_to(ROOT).as_posix(), **stats},
            }
        )
    report = {
        "schema_version": 1,
        "policy": policy,
        "policy_hash": policy_hash,
        "sources": source_reports,
        "inputs": {"policy_sha256": sha256_file(policy_path)},
    }
    atomic_write_json(OUTPUT_ROOT / "comparison_report.json", report)
    write_markdown(report)
    print(json.dumps({"status": "READY_FOR_LANE_REVIEW", "sources": len(source_reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
