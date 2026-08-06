from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import read_jsonl_gz, sha256_file  # noqa: E402


def main() -> int:
    report_path = ROOT / "data" / "experiments" / "corpus_v2" / "comparison_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = report["artifact"]
    artifact_path = ROOT / artifact["path"]
    if sha256_file(artifact_path) != artifact["compressed_sha256"]:
        raise RuntimeError("corpus-v2 artifact hash mismatch")
    records = list(read_jsonl_gz(artifact_path))
    if len(records) != artifact["records"]:
        raise RuntimeError("corpus-v2 record count mismatch")

    by_parent: dict[str, list[dict]] = defaultdict(list)
    by_title: dict[str, list[dict]] = defaultdict(list)
    content_hashes: set[str] = set()
    forbidden_flags = {"raw_wikitable_markup", "orphaned_table_footnotes"}
    suspicious_line_end = re.compile(r"(?:&|\bvar\.|\bsubsp\.|\bI\.)$")
    for record in records:
        if len(record["text"]) > report["policy"]["chunking"]["maximum_characters"]:
            raise RuntimeError(f"Oversized chunk: {record['record_id']}")
        if record["quality"]["policy_hash"] != report["policy_hash"]:
            raise RuntimeError(f"Policy hash mismatch: {record['record_id']}")
        if forbidden_flags & set(record["quality"]["flags"]):
            raise RuntimeError(f"Hard-rejected flag survived: {record['record_id']}")
        if record["content_sha256"] in content_hashes:
            raise RuntimeError(f"Duplicate content: {record['record_id']}")
        if record["metadata"]["chunk_end_boundary"] == "line":
            final_line = record["text"].splitlines()[-1].strip()
            if suspicious_line_end.search(final_line):
                raise RuntimeError(f"Suspicious mid-entry line boundary: {record['record_id']}")
        content_hashes.add(record["content_sha256"])
        by_parent[record["parent_upstream_id"]].append(record)
        by_title[record["metadata"]["title"]].append(record)

    for parent_id, chunks in by_parent.items():
        declared_counts = {record["metadata"]["chunk_count"] for record in chunks}
        indexes = {record["metadata"]["chunk_index"] for record in chunks}
        if declared_counts != {len(chunks)} or indexes != set(range(len(chunks))):
            raise RuntimeError(f"Broken retained-chunk lineage: {parent_id}")

    expected_rejected = {
        "Results of the 1994 Sri Lankan general election by electoral district",
        "Coach Trip (series 8)",
        "1951 Ohio State Buckeyes baseball team",
        "FC Luch Vladivostok",
    }
    rejected_titles = set(report["v2"]["rejected_title_counts"])
    missing_rejections = expected_rejected - rejected_titles
    if missing_rejections:
        raise RuntimeError(f"Reviewed rejects survived: {sorted(missing_rejections)}")

    category_tail_titles = {"Caning", "Saint Seiya Omega", "Gérard Condé", "Kenneth Leighton"}
    for title in category_tail_titles:
        matches = [
            record
            for record in by_title[title]
            if "category_tail" in record["quality"]["flags"]
        ]
        if not matches or any(record["quality"]["band"] != "B0" for record in matches):
            raise RuntimeError(f"Category-tail regression: {title}")

    linewise_titles = {
        "Anagyrus",
        "List of Saxifragales of South Africa",
        "List of Solanales of South Africa",
        "List of moths of Australia (Cosmopterigidae)",
    }
    for title in linewise_titles:
        matches = [
            record
            for record in by_title[title]
            if "linewise_list" in record["quality"]["flags"]
        ]
        if not matches or any(record["quality"]["band"] != "B0" for record in matches):
            raise RuntimeError(f"Linewise-list regression: {title}")

    boundaries = set(report["v2"]["chunk_end_boundary_counts"])
    if not boundaries <= {"paragraph", "line"}:
        raise RuntimeError(f"Unexpected v2 boundary types: {sorted(boundaries)}")

    summary = {
        "status": "PASS",
        "parents": len(by_parent),
        "records": len(records),
        "unique_content_hashes": len(content_hashes),
        "maximum_record_characters": max(len(record["text"]) for record in records),
        "policy_hash": report["policy_hash"],
        "artifact_sha256": artifact["compressed_sha256"],
        "raw_wikitable_chunks_rejected": report["v2"]["chunk_rejection_counts"]["raw_wikitable_markup"],
        "orphaned_footnote_chunks_rejected": report["v2"]["chunk_rejection_counts"]["orphaned_table_footnotes"],
        "category_tails_capped": report["v2"]["flag_counts"]["category_tail"],
        "linewise_lists_capped": report["v2"]["flag_counts"]["linewise_list"],
        "reviewed_regressions": 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
