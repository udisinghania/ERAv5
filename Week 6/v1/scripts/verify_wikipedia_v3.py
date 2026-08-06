from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import read_jsonl_gz, sha256_file  # noqa: E402


def main() -> int:
    report = json.loads(
        (ROOT / "data" / "experiments" / "corpus_v3" / "comparison_report.json").read_text(
            encoding="utf-8"
        )
    )
    artifact = report["artifact"]
    artifact_path = ROOT / artifact["path"]
    if sha256_file(artifact_path) != artifact["compressed_sha256"]:
        raise RuntimeError("corpus-v3 artifact hash mismatch")
    records = list(read_jsonl_gz(artifact_path))
    if len(records) != artifact["records"]:
        raise RuntimeError("corpus-v3 record count mismatch")

    by_parent: dict[str, list[dict]] = defaultdict(list)
    by_title: dict[str, list[dict]] = defaultdict(list)
    hashes: set[str] = set()
    forbidden = {"raw_wikitable_markup", "orphaned_table_footnotes"}
    for record in records:
        if record["quality"]["policy_hash"] != report["policy_hash"]:
            raise RuntimeError(f"Policy mismatch: {record['record_id']}")
        if forbidden & set(record["quality"]["flags"]):
            raise RuntimeError(f"Hard-rejected flag survived: {record['record_id']}")
        if len(record["text"]) > report["policy"]["chunking"]["maximum_characters"]:
            raise RuntimeError(f"Oversized record: {record['record_id']}")
        if record["content_sha256"] in hashes:
            raise RuntimeError(f"Duplicate record: {record['record_id']}")
        hashes.add(record["content_sha256"])
        by_parent[record["parent_upstream_id"]].append(record)
        by_title[record["metadata"]["title"]].append(record)

    for parent_id, chunks in by_parent.items():
        counts = {record["metadata"]["chunk_count"] for record in chunks}
        indexes = {record["metadata"]["chunk_index"] for record in chunks}
        if counts != {len(chunks)} or indexes != set(range(len(chunks))):
            raise RuntimeError(f"Broken lineage: {parent_id}")

    pelopas = by_title["Pelopas Kiato F.C."]
    if not pelopas or not any(
        record["quality"]["band"] == "B0"
        and "stat_heavy_list" in record["quality"]["flags"]
        and "general_stat_heavy_list" in record["quality"]["sampling_cap_groups"]
        for record in pelopas
    ):
        raise RuntimeError("Pelopas compact-list review regression")

    alachua = by_title["Alachua County, Florida"]
    if not any(
        "human_sensitive_context_reviewed" in record["quality"]["flags"]
        and "general_sensitive_context_review" in record["quality"]["sampling_cap_groups"]
        for record in alachua
    ):
        raise RuntimeError("Alachua sensitive-context review regression")

    for title in {
        "Coach Trip (series 8)",
        "1951 Ohio State Buckeyes baseball team",
        "FC Luch Vladivostok",
    }:
        salvaged = [
            record for record in by_title[title] if "table_markup_removed" in record["quality"]["flags"]
        ]
        if not salvaged or any(record["quality"]["band"] != "B0" for record in salvaged):
            raise RuntimeError(f"Reviewed table-salvage regression: {title}")

    rejected_titles = set(report["v3"]["rejected_title_counts"])
    if "Results of the 1994 Sri Lankan general election by electoral district" not in rejected_titles:
        raise RuntimeError("Orphaned-footnote review regression")

    summary = {
        "status": "PASS",
        "parents": len(by_parent),
        "records": len(records),
        "unique_content_hashes": len(hashes),
        "artifact_sha256": artifact["compressed_sha256"],
        "policy_hash": report["policy_hash"],
        "table_salvage_chunks": report["v3"]["flag_counts"]["table_markup_removed"],
        "stat_heavy_lists_capped": report["v3"]["flag_counts"]["stat_heavy_list"],
        "sensitive_context_overrides": report["v3"]["flag_counts"][
            "human_sensitive_context_reviewed"
        ],
        "reviewed_regressions": 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
