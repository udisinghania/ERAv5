from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.acquisition import load_source_lock, verify_snapshot  # noqa: E402
from era6.canonical import read_jsonl_gz  # noqa: E402
from era6.curation import verify_curated_corpus  # noqa: E402


def main() -> int:
    config = load_source_lock(ROOT / "configs" / "sources.lock.json")
    index = json.loads((ROOT / "data" / "source_snapshots" / "snapshot_index.json").read_text(encoding="utf-8"))
    target_by_id = {target["source_id"]: target for target in config["targets"]}
    snapshot_by_id = {item["source_id"]: item for item in index["snapshots"]}
    if set(target_by_id) != set(snapshot_by_id):
        raise RuntimeError("Snapshot index does not cover exactly the locked sources")

    for source_id, target in sorted(target_by_id.items()):
        item = snapshot_by_id[source_id]
        if item["records"] != target["quota"]:
            raise RuntimeError(f"Snapshot quota mismatch: {source_id}")
        if item["lane"] != target["lane"] or item["permission"] != target["permission"]:
            raise RuntimeError(f"Snapshot lane/permission mismatch: {source_id}")
        verify_snapshot(ROOT / item["path"], item["compressed_sha256"], item["records"])

    report = verify_curated_corpus(ROOT)
    train_hashes: set[str] = set()
    never_train_hashes: set[str] = set()
    for artifact in report["artifacts"]:
        destination = never_train_hashes if artifact["permission"] == "never_train" else train_hashes
        for record in read_jsonl_gz(ROOT / artifact["path"]):
            if record["permission"] != artifact["permission"]:
                raise RuntimeError(f"Record permission/path mismatch: {artifact['path']}")
            destination.add(record["content_sha256"])
    if train_hashes & never_train_hashes:
        raise RuntimeError("Exact evaluation content appears in a training-capable artifact")

    summary = {
        "source_snapshots": len(snapshot_by_id),
        "snapshot_training_records": index["training_records"],
        "snapshot_never_train_records": index["never_train_records"],
        "curated_training_records": report["admitted_training_records"],
        "curated_never_train_records": len(never_train_hashes),
        "exact_train_eval_overlap": 0,
        "lanes": sorted(report["lane_permission_counts"]),
        "status": "PASS",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
