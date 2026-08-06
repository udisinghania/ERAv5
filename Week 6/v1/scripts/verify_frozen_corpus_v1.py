from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import canonical_json_bytes, read_jsonl_gz, sha256_bytes, sha256_file  # noqa: E402
from era6.firewall import NGramDecontaminator  # noqa: E402


def main() -> int:
    output_root = ROOT / "data" / "frozen_corpus_v1"
    report = json.loads((output_root / "freeze_report.json").read_text(encoding="utf-8"))
    if report["status"] != "FROZEN":
        raise AssertionError("corpus is not marked frozen")
    if report["corpus_hash"] != f"sha256:{sha256_bytes(canonical_json_bytes(report['artifacts']))}":
        raise AssertionError("corpus hash mismatch")

    record_ids: set[str] = set()
    content_hashes: set[str] = set()
    group_permissions: dict[str, set[str]] = defaultdict(set)
    training_records = []
    evaluation_records = []
    for artifact in report["artifacts"]:
        path = ROOT / artifact["path"]
        if sha256_file(path) != artifact["compressed_sha256"]:
            raise AssertionError(f"artifact hash mismatch: {artifact['path']}")
        rows = list(read_jsonl_gz(path))
        if len(rows) != artifact["records"]:
            raise AssertionError(f"artifact count mismatch: {artifact['path']}")
        if artifact["permission"] == "never_train":
            evaluation_records.extend(rows)
            continue
        for row in rows:
            if row["record_id"] in record_ids:
                raise AssertionError(f"duplicate admitted record ID: {row['record_id']}")
            if row["content_sha256"] in content_hashes:
                raise AssertionError(f"duplicate admitted content: {row['content_sha256']}")
            record_ids.add(row["record_id"])
            content_hashes.add(row["content_sha256"])
            group_permissions[row["group_id"]].add(row["permission"])
            if row["permission"] != artifact["permission"]:
                raise AssertionError("row/artifact permission mismatch")
            if row["capability_lane"] != artifact["lane"]:
                raise AssertionError("row/artifact lane mismatch")
            if row["cleaning_pipeline_hash"] != report["cleaning_pipeline_hash"]:
                raise AssertionError("cleaning pipeline lineage mismatch")
            training_records.append(row)
    leaking_groups = [group for group, permissions in group_permissions.items() if len(permissions) > 1]
    if leaking_groups:
        raise AssertionError(f"group partition leakage: {len(leaking_groups)} groups")

    eval_hashes = {row["content_sha256"] for row in evaluation_records}
    if content_hashes.intersection(eval_hashes):
        raise AssertionError("exact evaluation overlap")
    firewall = NGramDecontaminator((row["text"] for row in evaluation_records), ngram_size=13)
    contaminated = [row["record_id"] for row in training_records if firewall.is_contaminated(row["text"])]
    if contaminated:
        raise AssertionError(f"evaluation n-gram overlap: {len(contaminated)} records")

    expected_lanes = {"general", "science_math", "code", "reasoning", "long_context", "indic", "agentic"}
    observed_lanes = {artifact["lane"] for artifact in report["artifacts"] if artifact["permission"] != "never_train"}
    if observed_lanes != expected_lanes:
        raise AssertionError("not all seven training lanes are present")
    result = {
        "status": "PASS",
        "corpus_hash": report["corpus_hash"],
        "training_records": len(training_records),
        "evaluation_records": len(evaluation_records),
        "unique_content_hashes": len(content_hashes),
        "group_leaks": 0,
        "evaluation_overlaps": 0,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
