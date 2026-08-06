from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.acquisition import (  # noqa: E402
    load_source_lock,
    snapshot_target,
    summarize_results,
    verify_snapshot,
)
from era6.canonical import atomic_write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire bounded, pinned Session 6 corpus snapshots")
    parser.add_argument("--source", action="append", help="Acquire only this source_id; repeatable")
    parser.add_argument("--offline-verify", action="store_true", help="Verify existing snapshots without network")
    parser.add_argument("--refresh", action="store_true", help="Replace selected snapshots from pinned sources")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = load_source_lock(ROOT / "configs" / "sources.lock.json")
    targets = [target for target in lock["targets"] if target["transform"] != "assignment4_parquet"]
    if args.source:
        selected = set(args.source)
        targets = [target for target in targets if target["source_id"] in selected]
        missing = selected - {target["source_id"] for target in targets}
        if missing:
            raise SystemExit(f"Unknown or local-only source ids: {sorted(missing)}")

    output_dir = ROOT / "data" / "source_snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "snapshot_index.json"
    existing = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"snapshots": []}
    by_id = {item["source_id"]: item for item in existing.get("snapshots", [])}

    if args.offline_verify:
        for target in targets:
            item = by_id.get(target["source_id"])
            if item is None:
                raise SystemExit(f"Missing snapshot index entry: {target['source_id']}")
            verify_snapshot(ROOT / item["path"], item["compressed_sha256"], item["records"])
            print(f"[verified] {target['source_id']}: {item['records']} records")
        return 0

    results = []
    for target in targets:
        current = by_id.get(target["source_id"])
        snapshot_path = output_dir / f"{target['source_id']}.jsonl.gz"
        if current and snapshot_path.exists() and not args.refresh:
            verify_snapshot(snapshot_path, current["compressed_sha256"], current["records"])
            print(f"[reuse] {target['source_id']}: verified existing snapshot")
            continue
        print(f"[acquire] {target['source_id']} ({target['quota']} records)")
        results.append(snapshot_target(target, output_dir, ROOT))
        print(f"[complete] {target['source_id']}: {results[-1].compressed_bytes:,} bytes")

    merged = list(existing.get("snapshots", []))
    replacement_ids = {item.source_id for item in results}
    merged = [item for item in merged if item["source_id"] not in replacement_ids]
    merged.extend(summarize_results(results)["snapshots"])
    merged.sort(key=lambda item: item["source_id"])
    summary = {
        "schema_version": 1,
        "training_records": sum(x["records"] for x in merged if x["permission"] != "never_train"),
        "never_train_records": sum(x["records"] for x in merged if x["permission"] == "never_train"),
        "compressed_bytes": sum(x["compressed_bytes"] for x in merged),
        "snapshots": merged,
    }
    atomic_write_json(index_path, summary)
    print(f"[index] {index_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
