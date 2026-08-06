from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .acquisition import load_source_lock
from .canonical import atomic_write_json, read_jsonl_gz, sha256_file, write_jsonl_gz
from .firewall import EvaluationRegistry, NGramDecontaminator
from .manifests import SourceLock
from .splitting import deterministic_partition


PARTITION_SEED = "era6-partition-v1"
LANE_PRIORITY = {
    "reasoning": 0,
    "code": 1,
    "agentic": 2,
    "science_math": 3,
    "long_context": 4,
    "indic": 5,
    "general": 6,
}


def build_source_locks(source_config: dict[str, Any], parent_manifest_id: str) -> list[dict[str, Any]]:
    locks = []
    for target in source_config["targets"]:
        parent_ids = (parent_manifest_id,) if target["transform"] == "assignment4_parquet" else ()
        lock = SourceLock(
            source_id=target["source_id"],
            source_url=target["source_url"],
            revision=target["revision"],
            license_id=target["license_id"],
            capability_lane=target["lane"],
            provenance_tier=target["provenance_tier"],
            config=target.get("config"),
            split=target.get("split"),
            parent_manifest_ids=parent_ids,
        )
        locks.append({"lock": asdict(lock), "lock_hash": lock.lock_hash})
    return locks


def curate_corpus(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    source_config_path = root / "configs" / "sources.lock.json"
    snapshot_index_path = root / "data" / "source_snapshots" / "snapshot_index.json"
    parent_manifest_path = root / "source_artifacts" / "assignment4_samanantar_parent_manifest.json"
    source_config = load_source_lock(source_config_path)
    snapshot_index = json.loads(snapshot_index_path.read_text(encoding="utf-8"))
    snapshots = {item["source_id"]: item for item in snapshot_index["snapshots"]}
    target_ids = {item["source_id"] for item in source_config["targets"]}
    if set(snapshots) != target_ids:
        missing = sorted(target_ids - set(snapshots))
        extra = sorted(set(snapshots) - target_ids)
        raise RuntimeError(f"Snapshot/source lock mismatch; missing={missing}, extra={extra}")

    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    source_locks = build_source_locks(source_config, parent_manifest["shard_id"])
    lock_hash_by_source = {item["lock"]["source_id"]: item["lock_hash"] for item in source_locks}

    eval_targets = [item for item in source_config["targets"] if item["permission"] == "never_train"]
    evaluation_records: list[dict[str, Any]] = []
    registry = EvaluationRegistry()
    for target in eval_targets:
        snapshot = snapshots[target["source_id"]]
        for record in read_jsonl_gz(root / snapshot["path"]):
            evaluation_records.append(record)
            registry.register(
                evaluation_id=record["record_id"],
                benchmark_id=f"{target['dataset']}:{target['config']}:{target['split']}",
                version=target["revision"],
                content={"record_id": record["record_id"], "text": record["text"]},
            )
    evaluation_records.sort(key=lambda record: record["record_id"])
    firewall = NGramDecontaminator((record["text"] for record in evaluation_records), ngram_size=13)
    eval_content_hashes = {record["content_sha256"] for record in evaluation_records}

    train_targets = sorted(
        (item for item in source_config["targets"] if item["permission"] == "train"),
        key=lambda item: (LANE_PRIORITY[item["lane"]], item["source_id"]),
    )
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_content: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    source_admitted: Counter[str] = Counter()
    for target in train_targets:
        snapshot = snapshots[target["source_id"]]
        records = sorted(read_jsonl_gz(root / snapshot["path"]), key=lambda record: record["record_id"])
        for record in records:
            content_hash = record["content_sha256"]
            if content_hash in seen_content:
                rejection_counts["cross_source_exact_duplicate"] += 1
                continue
            if content_hash in eval_content_hashes:
                rejection_counts["evaluation_exact_overlap"] += 1
                continue
            if firewall.is_contaminated(record["text"]):
                rejection_counts["evaluation_13gram_overlap"] += 1
                continue
            seen_content.add(content_hash)
            partition = deterministic_partition(record["group_id"], seed=PARTITION_SEED)
            admitted = dict(record)
            admitted["source_lock_hash"] = lock_hash_by_source[record["source_id"]]
            admitted["original_permission"] = admitted["permission"]
            admitted["permission"] = partition
            admitted["partition_seed"] = PARTITION_SEED
            buckets[(admitted["capability_lane"], partition)].append(admitted)
            source_admitted[record["source_id"]] += 1

    output_root = root / "data" / "curated"
    artifacts = []
    for (lane, permission), records in sorted(buckets.items()):
        records.sort(key=lambda record: record["record_id"])
        path = output_root / lane / f"{permission}.jsonl.gz"
        stats = write_jsonl_gz(path, records)
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "lane": lane,
                "permission": permission,
                **stats,
            }
        )

    eval_path = output_root / "reasoning" / "never_train.jsonl.gz"
    eval_stats = write_jsonl_gz(eval_path, evaluation_records)
    artifacts.append(
        {
            "path": eval_path.relative_to(root).as_posix(),
            "lane": "reasoning",
            "permission": "never_train",
            **eval_stats,
        }
    )
    registry_payload = {
        "schema_version": 1,
        "registry_hash": registry.registry_hash,
        "entries": [asdict(entry) for _, entry in sorted(registry.entries.items())],
    }
    atomic_write_json(output_root / "evaluation_registry.json", registry_payload)
    atomic_write_json(output_root / "source_locks.json", {"schema_version": 1, "sources": source_locks})

    lane_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for artifact in artifacts:
        lane_counts[artifact["lane"]][artifact["permission"]] = artifact["records"]
    report = {
        "schema_version": 1,
        "partition_seed": PARTITION_SEED,
        "source_config_sha256": sha256_file(source_config_path),
        "snapshot_index_sha256": sha256_file(snapshot_index_path),
        "parent_manifest_copy_sha256": sha256_file(parent_manifest_path),
        "evaluation_registry_hash": registry.registry_hash,
        "input_training_records": snapshot_index["training_records"],
        "input_never_train_records": snapshot_index["never_train_records"],
        "admitted_training_records": sum(source_admitted.values()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "source_admitted_counts": dict(sorted(source_admitted.items())),
        "lane_permission_counts": {lane: dict(sorted(counts.items())) for lane, counts in sorted(lane_counts.items())},
        "artifacts": artifacts,
    }
    atomic_write_json(output_root / "curation_report.json", report)
    return report


def verify_curated_corpus(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    report_path = root / "data" / "curated" / "curation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for artifact in report["artifacts"]:
        path = root / artifact["path"]
        if sha256_file(path) != artifact["compressed_sha256"]:
            raise RuntimeError(f"Curated artifact hash mismatch: {path}")
        count = sum(1 for _ in read_jsonl_gz(path))
        if count != artifact["records"]:
            raise RuntimeError(f"Curated artifact row count mismatch: {path}")
    return report
