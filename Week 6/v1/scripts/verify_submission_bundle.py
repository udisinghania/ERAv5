from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "submission_artifacts"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load(relative: str) -> dict[str, Any]:
    return json.loads((BUNDLE / relative).read_text(encoding="utf-8"))


def main() -> int:
    required_files = ["run.log", "evidence.json", "evidence.md", "performance.json"]
    required_directories = ["manifests", "ledgers", "checkpoints"]
    for relative in required_files:
        if not (BUNDLE / relative).is_file():
            raise AssertionError(f"required submission file is missing: {relative}")
    for relative in required_directories:
        if not (BUNDLE / relative).is_dir():
            raise AssertionError(f"required submission directory is missing: {relative}")

    evidence = load("evidence.json")
    evidence_body = {key: value for key, value in evidence.items() if key != "evidence_hash"}
    if evidence["evidence_hash"] != sha256_bytes(canonical_bytes(evidence_body)):
        raise AssertionError("evidence hash mismatch")
    if evidence["status"] != "PASS" or not evidence["requirements"]:
        raise AssertionError("evidence status is not PASS")
    if any(row["result"] != "PASS" for row in evidence["requirements"]):
        raise AssertionError("evidence contains a failed requirement")

    artifact_manifest = load("manifests/submission_artifact_manifest.json")
    manifest_body = {
        key: value for key, value in artifact_manifest.items() if key != "manifest_hash"
    }
    if artifact_manifest["manifest_hash"] != sha256_bytes(canonical_bytes(manifest_body)):
        raise AssertionError("submission artifact manifest hash mismatch")
    if evidence["artifact_manifest_hash"] != artifact_manifest["manifest_hash"]:
        raise AssertionError("evidence and artifact manifest lineage mismatch")
    for row in artifact_manifest["artifacts"]:
        path = BUNDLE / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise AssertionError(f"artifact size mismatch: {row['path']}")
        if sha256_file(path) != row["sha256"]:
            raise AssertionError(f"artifact hash mismatch: {row['path']}")

    performance = load("performance.json")
    runtime_config = load("manifests/training_config.json")["runtime"]
    if runtime_config["interpreter_policy"] != "launching_python_executable":
        raise AssertionError("submission runner is not interpreter-portable")
    if set(runtime_config["supported_devices"]) != {"cuda", "cpu"}:
        raise AssertionError("submission runtime does not declare CUDA and CPU support")
    if performance["device_type"] not in runtime_config["supported_devices"]:
        raise AssertionError("performance device violates the runtime contract")
    elapsed = int(performance["elapsed_nanoseconds"])
    if elapsed <= 0 or performance["status"] != "PASS":
        raise AssertionError("invalid performance timing")
    expected_rate = int(performance["loss_bearing_tokens"]) * 1_000_000_000 / elapsed
    if not math.isclose(
        float(performance["useful_loss_bearing_tokens_per_second"]),
        expected_rate,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("useful loss-bearing throughput is not reconstructable")

    batch_report = load("manifests/batch_report.json")
    opus = batch_report["opus_summary"]
    if not all(
        int(opus["candidate_outcome_distribution"].get(name, 0)) > 0
        for name in ("accepted", "rejected", "deferred")
    ):
        raise AssertionError("OPUS outcome evidence is incomplete")
    if int(opus["protected_floor_overrides"]) <= 0:
        raise AssertionError("OPUS protected-floor override was not demonstrated")

    audit = load("manifests/recovery_audit.json")
    if audit["status"] != "PASS" or not all(audit["comparisons"].values()):
        raise AssertionError("recovery audit contains a failed comparison")
    resume = load("manifests/resume_event.json")
    if resume["status"] != "PASS" or not all(
        resume[name]
        for name in (
            "batch_id_matched",
            "sequence_ids_matched",
            "token_spans_matched",
            "payload_hashes_matched",
            "proof_hash_matched",
        )
    ):
        raise AssertionError("resume-next-batch proof failed")
    interval = load("manifests/replay_interval_report.json")
    if interval["status"] != "PASS" or not interval["entries"]:
        raise AssertionError("historical replay interval is missing or failed")
    if not all(all(row["comparisons"].values()) for row in interval["entries"]):
        raise AssertionError("historical replay interval comparison failed")

    log = (BUNDLE / "run.log").read_text(encoding="utf-8")
    required_log_events = [
        "[PASS] tokenizer_hash_verified",
        "[PASS] eval_shard_blocked",
        "[PASS] checkpoint_saved",
        "[PASS] resume_next_batch_matched",
        "[PASS] replay_hash_matched",
        "[PASS] shards_created",
        "[PASS] manifests_validated",
        "[PASS] mixture_compiled",
        "[PASS] batches_packed",
        "[PASS] OPUS_decisions_recorded",
        "[PASS] crash_simulated",
        "[PASS] run_resumed",
        "[PASS] historical_stream_replayed",
        "[PASS] branch_forked",
        "[PASS] audit_completed",
        "[PASS] performance_measured",
        "[PASS] portable_runtime_verified",
    ]
    missing = [event for event in required_log_events if event not in log]
    if missing:
        raise AssertionError(f"run.log is missing events: {missing}")

    markdown = (BUNDLE / "evidence.md").read_text(encoding="utf-8")
    if "| Requirement | Result | Evidence |" not in markdown:
        raise AssertionError("human-readable evidence table is missing")
    if any(row["requirement"] not in markdown for row in evidence["requirements"]):
        raise AssertionError("evidence.md does not cover every machine-readable requirement")

    result = {
        "status": "PASS",
        "requirements": len(evidence["requirements"]),
        "supporting_artifacts": len(artifact_manifest["artifacts"]),
        "shard_manifests": len(list((BUNDLE / "manifests/shards").rglob("manifest.json"))),
        "ledgers": len(list((BUNDLE / "ledgers").glob("*.jsonl.gz"))),
        "checkpoints": len(list((BUNDLE / "checkpoints").glob("*.pt"))),
        "useful_loss_bearing_tokens_per_second": performance[
            "useful_loss_bearing_tokens_per_second"
        ],
        "resume_next_batch_index": resume["checkpoint_next_batch_index"],
        "replay_interval_batches": interval["batches_replayed"],
        "evidence_hash": evidence["evidence_hash"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
