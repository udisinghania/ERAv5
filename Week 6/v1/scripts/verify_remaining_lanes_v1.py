from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.acquisition import load_source_lock  # noqa: E402
from era6.canonical import (  # noqa: E402
    canonical_json_bytes,
    read_jsonl_gz,
    sha256_bytes,
    sha256_file,
    sha256_text,
)


EXPERIMENT_ROOT = ROOT / "data" / "experiments" / "remaining_lanes_v1"
VERIFICATION_STATUS = "PASS_WITH_LANE_BOUNDARY_REVIEW_PENDING"


def fail(message: str) -> None:
    raise AssertionError(message)


def uncompressed_hash(path: Path) -> tuple[str, int, int]:
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    return sha256_bytes(payload), len(payload), payload.count(b"\n")


def main() -> int:
    report_path = EXPERIMENT_ROOT / "comparison_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    policy = report["policy"]
    expected_policy_hash = f"sha256:{sha256_bytes(canonical_json_bytes(policy))}"
    if report["policy_hash"] != expected_policy_hash:
        fail("policy hash does not match the embedded policy")

    source_lock = load_source_lock(ROOT / "configs" / "sources.lock.json")
    targets = {item["source_id"]: item for item in source_lock["targets"]}
    policy_by_source = {item["source_id"]: item for item in policy["sources"]}
    global_record_ids: set[str] = set()
    verification: list[dict[str, Any]] = []

    for source_report in report["sources"]:
        source_id = source_report["source_id"]
        target = targets[source_id]
        source_policy = policy_by_source[source_id]
        artifact = source_report["artifact"]
        artifact_path = ROOT / artifact["path"]
        records = list(read_jsonl_gz(artifact_path))
        summary = source_report["summary"]

        if len(records) != artifact["records"] or len(records) != summary["records"]:
            fail(f"{source_id}: record count mismatch")
        if sha256_file(artifact_path) != artifact["compressed_sha256"]:
            fail(f"{source_id}: compressed hash mismatch")
        raw_hash, raw_bytes, raw_records = uncompressed_hash(artifact_path)
        if raw_hash != artifact["canonical_uncompressed_sha256"]:
            fail(f"{source_id}: uncompressed hash mismatch")
        if raw_bytes != artifact["uncompressed_bytes"] or raw_records != len(records):
            fail(f"{source_id}: uncompressed size mismatch")

        record_ids = [row["record_id"] for row in records]
        if len(record_ids) != len(set(record_ids)):
            fail(f"{source_id}: duplicate record IDs")
        overlap = global_record_ids.intersection(record_ids)
        if overlap:
            fail(f"{source_id}: cross-source duplicate record ID")
        global_record_ids.update(record_ids)

        parents: dict[str, list[dict[str, Any]]] = defaultdict(list)
        boundary_counts: Counter[str] = Counter()
        marker_count = 0
        for row in records:
            if row["source_id"] != source_id or row["capability_lane"] != source_report["lane"]:
                fail(f"{source_id}: source or lane metadata mismatch")
            if row["content_sha256"] != sha256_text(row["text"]):
                fail(f"{source_id}: content hash mismatch for {row['record_id']}")
            if len(row["text"]) > int(target["max_chars"]):
                fail(f"{source_id}: record exceeds max_chars")
            marker_count += row["text"].count("[PHONE]")
            metadata = row.get("metadata", {})
            parent_id = str(row.get("parent_upstream_id", row["upstream_id"]))
            parents[parent_id].append(row)
            boundary_counts[str(metadata.get("chunk_end_boundary", "baseline"))] += 1

        if len(parents) != summary["parents"]:
            fail(f"{source_id}: parent count mismatch")
        if marker_count != summary["experiment_phone_markers"]:
            fail(f"{source_id}: [PHONE] marker count mismatch")
        if dict(sorted(boundary_counts.items())) != summary["chunk_boundary_counts"]:
            fail(f"{source_id}: chunk-boundary count mismatch")

        if source_policy["pii_mode"] != "baseline_locked":
            for parent_id, chunks in parents.items():
                ordered = sorted(chunks, key=lambda row: row["metadata"]["chunk_index"])
                expected_count = len(ordered)
                indexes = [row["metadata"]["chunk_index"] for row in ordered]
                counts = {row["metadata"]["chunk_count"] for row in ordered}
                if indexes != list(range(expected_count)) or counts != {expected_count}:
                    fail(f"{source_id}: broken chunk lineage for parent {parent_id}")
                if any(row["metadata"]["policy_hash"] != report["policy_hash"] for row in ordered):
                    fail(f"{source_id}: policy lineage mismatch")
            if source_policy["pii_mode"] == "structured_numeric" and marker_count:
                fail(f"{source_id}: structured-numeric source still contains [PHONE]")

        verification.append(
            {
                "source_id": source_id,
                "records": len(records),
                "parents": len(parents),
                "artifact_sha256": artifact["compressed_sha256"],
                "phone_markers": marker_count,
                "word_boundaries": boundary_counts.get("word", 0),
            }
        )

    result = {
        "status": VERIFICATION_STATUS,
        "policy_hash": report["policy_hash"],
        "sources": verification,
        "records": sum(item["records"] for item in verification),
        "unique_record_ids": len(global_record_ids),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
