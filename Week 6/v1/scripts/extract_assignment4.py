from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.acquisition import load_source_lock, snapshot_assignment4_target  # noqa: E402
from era6.canonical import atomic_write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract the translated Indic tier from Assignment 4")
    parser.add_argument("assignment4_root", type=Path)
    args = parser.parse_args()
    lock = load_source_lock(ROOT / "configs" / "sources.lock.json")
    target = next(item for item in lock["targets"] if item["transform"] == "assignment4_parquet")
    output_dir = ROOT / "data" / "source_snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = snapshot_assignment4_target(target, args.assignment4_root, output_dir, ROOT)

    index_path = output_dir / "snapshot_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"snapshots": []}
    snapshots = [item for item in index.get("snapshots", []) if item["source_id"] != result.source_id]
    snapshots.append(result.__dict__)
    snapshots.sort(key=lambda item: item["source_id"])
    atomic_write_json(
        index_path,
        {
            "schema_version": 1,
            "training_records": sum(x["records"] for x in snapshots if x["permission"] != "never_train"),
            "never_train_records": sum(x["records"] for x in snapshots if x["permission"] == "never_train"),
            "compressed_bytes": sum(x["compressed_bytes"] for x in snapshots),
            "snapshots": snapshots,
        },
    )
    print(f"[complete] {result.source_id}: {result.records} records, {result.compressed_bytes:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
