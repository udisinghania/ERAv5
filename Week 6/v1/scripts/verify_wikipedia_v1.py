from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import read_jsonl_gz, sha256_file  # noqa: E402


def main() -> int:
    report_path = ROOT / "data" / "experiments" / "corpus_v1" / "comparison_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = report["artifact"]
    artifact_path = ROOT / artifact["path"]
    if sha256_file(artifact_path) != artifact["compressed_sha256"]:
        raise RuntimeError("corpus-v1 artifact hash mismatch")
    records = list(read_jsonl_gz(artifact_path))
    if len(records) != artifact["records"]:
        raise RuntimeError("corpus-v1 record count mismatch")

    by_parent = defaultdict(list)
    content_hashes = set()
    for record in records:
        if len(record["text"]) > report["policy"]["chunking"]["maximum_characters"]:
            raise RuntimeError(f"Oversized chunk: {record['record_id']}")
        if record["quality"]["policy_hash"] != report["policy_hash"]:
            raise RuntimeError(f"Policy hash mismatch: {record['record_id']}")
        if record["content_sha256"] in content_hashes:
            raise RuntimeError(f"Duplicate content: {record['record_id']}")
        content_hashes.add(record["content_sha256"])
        by_parent[record["parent_upstream_id"]].append(record)

    for parent_id, chunks in by_parent.items():
        groups = {record["group_id"] for record in chunks}
        declared_counts = {record["metadata"]["chunk_count"] for record in chunks}
        indexes = {record["metadata"]["chunk_index"] for record in chunks}
        if len(groups) != 1 or declared_counts != {len(chunks)} or indexes != set(range(len(chunks))):
            raise RuntimeError(f"Broken parent chunk lineage: {parent_id}")

    identifier_expectations = {
        "6864279": "44090600",
        "6864512": "836100",
        "6864677": "2825108",
    }
    for parent_id, identifier in identifier_expectations.items():
        combined = "\n".join(record["text"] for record in by_parent[parent_id])
        if identifier not in combined or "[PHONE]" in combined:
            raise RuntimeError(f"Public-reference identifier masking regression: {parent_id}")

    if report["v1"]["mid_boundary_truncations"] != 0:
        raise RuntimeError("Character truncation remains in corpus-v1")
    summary = {
        "status": "PASS",
        "parents": len(by_parent),
        "records": len(records),
        "unique_content_hashes": len(content_hashes),
        "maximum_record_characters": max(len(record["text"]) for record in records),
        "policy_hash": report["policy_hash"],
        "artifact_sha256": artifact["compressed_sha256"],
        "mid_boundary_truncations": 0,
        "identifier_regressions": 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
