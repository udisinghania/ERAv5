from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.acquisition import load_source_lock  # noqa: E402
from era6.canonical import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    read_jsonl_gz,
    sha256_bytes,
    sha256_file,
    write_jsonl_gz,
)
from era6.curation import LANE_PRIORITY, build_source_locks  # noqa: E402
from era6.firewall import EvaluationRegistry, NGramDecontaminator  # noqa: E402
from era6.manifests import pipeline_hash  # noqa: E402
from era6.splitting import deterministic_partition  # noqa: E402


OUTPUT_ROOT = ROOT / "data" / "frozen_corpus_v1"


def main() -> int:
    freeze_path = ROOT / "configs" / "corpus_freeze_v1.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    source_config_path = ROOT / "configs" / "sources.lock.json"
    source_config = load_source_lock(source_config_path)
    targets = {target["source_id"]: target for target in source_config["targets"]}

    configured_ids = [item["source_id"] for item in freeze["training_sources"]]
    expected_ids = sorted(
        target["source_id"] for target in source_config["targets"] if target["permission"] == "train"
    )
    if sorted(configured_ids) != expected_ids or len(configured_ids) != len(set(configured_ids)):
        raise RuntimeError("freeze config must name every training source exactly once")

    parent_manifest = json.loads(
        (ROOT / "source_artifacts" / "assignment4_samanantar_parent_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_locks = build_source_locks(source_config, parent_manifest["shard_id"])
    lock_hash_by_source = {item["lock"]["source_id"]: item["lock_hash"] for item in source_locks}

    eval_source_id = freeze["evaluation_firewall"]["source_id"]
    eval_path = ROOT / "data" / "source_snapshots" / f"{eval_source_id}.jsonl.gz"
    evaluation_records = sorted(read_jsonl_gz(eval_path), key=lambda row: row["record_id"])
    registry = EvaluationRegistry()
    eval_target = targets[eval_source_id]
    for record in evaluation_records:
        registry.register(
            evaluation_id=record["record_id"],
            benchmark_id=f"{eval_target['dataset']}:{eval_target['config']}:{eval_target['split']}",
            version=eval_target["revision"],
            content={"record_id": record["record_id"], "text": record["text"]},
        )
    firewall = NGramDecontaminator(
        (record["text"] for record in evaluation_records),
        ngram_size=int(freeze["evaluation_firewall"]["ngram_size"]),
    )
    eval_content_hashes = {record["content_sha256"] for record in evaluation_records}

    pipeline_paths = [ROOT / item for item in freeze["quality_inputs"]]
    cleaning_hash = pipeline_hash(*pipeline_paths, config=freeze)
    partition = freeze["partition"]
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_content: set[str] = set()
    seen_record_ids: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    source_input: Counter[str] = Counter()
    source_admitted: Counter[str] = Counter()

    ordered_sources = sorted(
        freeze["training_sources"],
        key=lambda item: (LANE_PRIORITY[targets[item["source_id"]]["lane"]], item["source_id"]),
    )
    input_artifacts = []
    for source in ordered_sources:
        source_id = source["source_id"]
        path = ROOT / source["path"]
        input_artifacts.append(
            {
                "source_id": source_id,
                "path": source["path"],
                "compressed_sha256": sha256_file(path),
            }
        )
        for record in sorted(read_jsonl_gz(path), key=lambda row: row["record_id"]):
            source_input[source_id] += 1
            if record["record_id"] in seen_record_ids:
                raise RuntimeError(f"duplicate record ID across frozen inputs: {record['record_id']}")
            seen_record_ids.add(record["record_id"])
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
            permission = deterministic_partition(
                record["group_id"],
                seed=partition["seed"],
                validation_fraction=float(partition["validation_fraction"]),
                anneal_fraction=float(partition["anneal_fraction"]),
            )
            admitted = {
                **record,
                "source_lock_hash": lock_hash_by_source[source_id],
                "original_permission": record["permission"],
                "permission": permission,
                "partition_seed": partition["seed"],
                "cleaning_pipeline_hash": cleaning_hash,
            }
            buckets[(admitted["capability_lane"], permission)].append(admitted)
            source_admitted[source_id] += 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for (lane, permission), records in sorted(buckets.items()):
        records.sort(key=lambda row: row["record_id"])
        path = OUTPUT_ROOT / lane / f"{permission}.jsonl.gz"
        stats = write_jsonl_gz(path, records)
        artifacts.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "lane": lane,
                "permission": permission,
                **stats,
            }
        )

    frozen_eval_path = OUTPUT_ROOT / "reasoning" / "never_train.jsonl.gz"
    eval_stats = write_jsonl_gz(frozen_eval_path, evaluation_records)
    artifacts.append(
        {
            "path": frozen_eval_path.relative_to(ROOT).as_posix(),
            "lane": "reasoning",
            "permission": "never_train",
            **eval_stats,
        }
    )
    artifacts.sort(key=lambda item: item["path"])
    registry_payload = {
        "schema_version": 1,
        "registry_hash": registry.registry_hash,
        "entries": [asdict(entry) for _, entry in sorted(registry.entries.items())],
    }
    atomic_write_json(OUTPUT_ROOT / "evaluation_registry.json", registry_payload)
    atomic_write_json(OUTPUT_ROOT / "source_locks.json", {"schema_version": 1, "sources": source_locks})

    lane_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for artifact in artifacts:
        lane_counts[artifact["lane"]][artifact["permission"]] = artifact["records"]
    corpus_hash = f"sha256:{sha256_bytes(canonical_json_bytes(artifacts))}"
    report = {
        "schema_version": 1,
        "corpus_id": freeze["corpus_id"],
        "status": "FROZEN",
        "corpus_hash": corpus_hash,
        "cleaning_pipeline_hash": cleaning_hash,
        "partition": partition,
        "evaluation_registry_hash": registry.registry_hash,
        "freeze_config_sha256": sha256_file(freeze_path),
        "source_config_sha256": sha256_file(source_config_path),
        "input_artifacts": input_artifacts,
        "input_training_records": sum(source_input.values()),
        "admitted_training_records": sum(source_admitted.values()),
        "input_never_train_records": len(evaluation_records),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "source_input_counts": dict(sorted(source_input.items())),
        "source_admitted_counts": dict(sorted(source_admitted.items())),
        "lane_permission_counts": {
            lane: dict(sorted(counts.items())) for lane, counts in sorted(lane_counts.items())
        },
        "artifacts": artifacts,
    }
    atomic_write_json(OUTPUT_ROOT / "freeze_report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "corpus_hash": corpus_hash,
                "input_training_records": report["input_training_records"],
                "admitted_training_records": report["admitted_training_records"],
                "rejection_counts": report["rejection_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
