from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_remaining_lanes_v1 import build_source  # noqa: E402
from era6.acquisition import load_source_lock  # noqa: E402
from era6.canonical import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_jsonl_gz,
)


OUTPUT_ROOT = ROOT / "data" / "experiments" / "remaining_lanes_v2"


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Remaining lanes: cleaning experiment v2",
        "",
        "This candidate applies the completed v1 human review without modifying either baseline snapshots or the reviewed v1 experiment.",
        "",
        "| Lane | Source | Parents | Records | Excluded | Phone masks | Bank-account masks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["sources"]:
        summary = item["summary"]
        pii = summary.get("pii_counts", {})
        lines.append(
            f"| {item['lane']} | {item['source_id']} | {summary['parents']:,} | "
            f"{summary['records']:,} | {summary.get('excluded_parents', 0):,} | "
            f"{pii.get('phone', 0):,} | {pii.get('financial_account', 0):,} |"
        )
    lines.extend(
        [
            "",
            "Human-review actions: the WhatsApp number remains masked, both exposed bank-account numbers are masked, the garbled synthetic-Hindi parent is excluded, the anomalous Sanskrit/OCR parent is held pending source verification, and the translated false marker is restored to 250,000 from its paired English state.",
            "",
        ]
    )
    path = ROOT / "docs" / "quality" / "REMAINING_LANES_V1_V2_COMPARISON.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    policy_path = ROOT / "configs" / "remaining_lanes_policy_v2.json"
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
        "schema_version": 2,
        "policy": policy,
        "policy_hash": policy_hash,
        "sources": source_reports,
        "inputs": {"policy_sha256": sha256_file(policy_path)},
    }
    atomic_write_json(OUTPUT_ROOT / "comparison_report.json", report)
    atomic_write_json(
        OUTPUT_ROOT / "human_review_registry.json",
        {
            "schema_version": 1,
            "policy_hash": policy_hash,
            "actions": policy["human_review_actions"],
        },
    )
    write_markdown(report)
    print(json.dumps({"status": "V2_CANDIDATE_BUILT", "sources": len(source_reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
