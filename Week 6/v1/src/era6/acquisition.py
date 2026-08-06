from __future__ import annotations

import hashlib
import gzip
import http.client
import heapq
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import atomic_write_bytes, canonical_json_bytes, read_jsonl_gz, sha256_file, sha256_text, write_jsonl_gz
from .cleaning import BasicQualityFilter, PIIScrubber, TextNormalizer


BLOCK_SIZE = 100
USER_AGENT = "era-v5-session6-toy-execution-system/1.0"
ROLE_MAP = {"human": "user", "gpt": "assistant", "tool": "tool", "system": "system"}
MIN_REQUEST_INTERVAL_SECONDS = 3.0
_last_request_at = 0.0


@dataclass(frozen=True)
class AcquisitionResult:
    source_id: str
    path: str
    records: int
    compressed_bytes: int
    compressed_sha256: str
    canonical_uncompressed_sha256: str
    permission: str
    lane: str


def load_source_lock(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported sources.lock.json schema")
    targets = value.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Source lock must contain targets")
    ids = [target.get("source_id") for target in targets]
    if any(not source_id for source_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Source ids must be present and unique")
    return value


def deterministic_block_offsets(source_id: str, total_rows: int, max_blocks: int) -> list[int]:
    if total_rows < BLOCK_SIZE:
        return [0]
    slots = total_rows // BLOCK_SIZE
    count = min(max_blocks, slots)
    offsets: list[int] = []
    used: set[int] = set()
    counter = 0
    while len(offsets) < count:
        payload = f"era6-block-v1:{source_id}:{counter}".encode("utf-8")
        slot = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % slots
        counter += 1
        if slot in used:
            continue
        used.add(slot)
        offsets.append(slot * BLOCK_SIZE)
    return offsets


def _request_json(url: str, retries: int = 10) -> dict[str, Any]:
    global _last_request_at
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        elapsed = time.monotonic() - _last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        try:
            _last_request_at = time.monotonic()
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if attempt + 1 == retries:
                raise
            if error.code == 429:
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(15 + 5 * attempt, 60)
            elif 500 <= error.code < 600:
                delay = min(2**attempt, 30)
            else:
                raise
            time.sleep(delay)
        except (
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
            http.client.RemoteDisconnected,
            ConnectionResetError,
        ):
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def verify_hub_revision(target: dict[str, Any]) -> None:
    dataset = target["dataset"]
    revision = target["revision"]
    if dataset.startswith("local/"):
        return
    url = f"https://huggingface.co/api/datasets/{dataset}/revision/{revision}"
    result = _request_json(url)
    if result.get("sha") != revision:
        raise RuntimeError(f"Pinned revision cannot be verified for {dataset}: {revision}")


def fetch_viewer_rows(
    target: dict[str, Any], offset: int, cache_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"offset-{offset:012d}.json.gz"
        if cache_path.exists():
            with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
                cached = json.load(stream)
            return [item["row"] for item in cached.get("rows", [])]
    query = urllib.parse.urlencode(
        {
            "dataset": target["dataset"],
            "config": target["config"],
            "split": target["split"],
            "offset": offset,
            "length": BLOCK_SIZE,
        }
    )
    result = _request_json(f"https://datasets-server.huggingface.co/rows?{query}")
    total = result.get("num_rows_total")
    if total is not None and int(total) != int(target["viewer_total_rows"]):
        raise RuntimeError(
            f"Viewer row count drift for {target['source_id']}: "
            f"expected {target['viewer_total_rows']}, received {total}"
        )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = gzip.compress(canonical_json_bytes(result), compresslevel=9, mtime=0)
        atomic_write_bytes(cache_path, payload)
    return [item["row"] for item in result.get("rows", [])]


def _text_and_metadata(transform: str, row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if transform == "wikipedia":
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", ""))
        return f"{title}\n\n{text}", str(row.get("id", "")), {"title": title, "url": row.get("url")}
    if transform == "openwebmath":
        return str(row.get("text", "")), str(row.get("url", "")), {
            "url": row.get("url"),
            "crawl_date": row.get("date"),
        }
    if transform == "codeparrot":
        upstream_id = f"{row.get('repo_name', '')}:{row.get('path', '')}:{row.get('hash', '')}"
        return str(row.get("content", "")), upstream_id, {
            "repo_name": row.get("repo_name"),
            "path": row.get("path"),
            "row_license": row.get("license"),
        }
    if transform == "gsm8k":
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        return f"<question>\n{question}\n<reasoning>\n{answer}", sha256_text(question), {}
    if transform == "sangraha":
        return str(row.get("text", "")), str(row.get("doc_id", "")), {"source_type": row.get("type")}
    if transform == "hermes":
        turns = []
        for turn in row.get("conversations") or []:
            original_role = str(turn.get("from", "unknown")).lower()
            role = ROLE_MAP.get(original_role, original_role)
            turns.append(f"<{role}>\n{str(turn.get('value', '')).strip()}")
        return "\n".join(turns), str(row.get("id", "")), {
            "category": row.get("category"),
            "subcategory": row.get("subcategory"),
            "task": row.get("task"),
        }
    if transform == "assignment4_parquet":
        return str(row.get("twm", "")), str(row.get("twm_hash", "")), {
            "hub_hash": row.get("hub_hash"),
            "target_languages": row.get("target_languages") or [],
        }
    raise ValueError(f"Unknown transform: {transform}")


def candidate_text_and_metadata(
    target: dict[str, Any], row: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    """Expose the pre-cleaning candidate representation for reproducible audits."""
    return _text_and_metadata(target["transform"], row)


def transform_record(target: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    allowed = target.get("allowed_row_licenses")
    if allowed and str(row.get("license", "")).lower() not in set(allowed):
        return None
    text, upstream_id, metadata = candidate_text_and_metadata(target, row)
    text = TextNormalizer().normalize(text)
    scrubbed = PIIScrubber().scrub(text)
    text = scrubbed.text
    if len(text) < int(target["min_chars"]):
        return None
    text = text[: int(target["max_chars"])]
    decision = BasicQualityFilter(
        min_characters=int(target["min_chars"]), max_characters=int(target["max_chars"])
    ).evaluate(text)
    if not decision.admitted:
        return None
    upstream_id = upstream_id or sha256_text(text)
    content_hash = sha256_text(text)
    record_key = f"{target['source_id']}:{upstream_id}:{content_hash}"
    row_license = metadata.get("row_license") or target["license_id"]
    return {
        "record_id": f"rec_{sha256_text(record_key)[:24]}",
        "group_id": f"grp_{sha256_text(str(upstream_id))[:20]}",
        "source_id": target["source_id"],
        "source_revision": target["revision"],
        "upstream_id": str(upstream_id),
        "content_sha256": content_hash,
        "capability_lane": target["lane"],
        "provenance_tier": target["provenance_tier"],
        "permission": target["permission"],
        "language": target["language"],
        "license_id": row_license,
        "pii_redactions": scrubbed.num_redactions,
        "text": text,
        "metadata": {key: value for key, value in metadata.items() if value is not None},
    }


def collect_viewer_target(
    target: dict[str, Any], cache_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    verify_hub_revision(target)
    quota = int(target["quota"])
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    offsets = deterministic_block_offsets(
        target["source_id"], int(target["viewer_total_rows"]), int(target["max_blocks"])
    )
    for offset in offsets:
        for row in fetch_viewer_rows(target, offset, cache_dir):
            transformed = transform_record(target, row)
            if transformed is None or transformed["record_id"] in seen:
                continue
            seen.add(transformed["record_id"])
            records.append(transformed)
            if len(records) == quota:
                return records
    raise RuntimeError(
        f"{target['source_id']} produced {len(records)} accepted rows; quota is {quota}. "
        "Increase max_blocks only after reviewing the acceptance report."
    )


def deterministic_bucket_ids(source_id: str, count: int, total_buckets: int = 256) -> list[str]:
    if not 0 < count <= total_buckets:
        raise ValueError("Bucket count must be between one and total_buckets")
    selected: list[int] = []
    used: set[int] = set()
    counter = 0
    while len(selected) < count:
        digest = hashlib.sha256(f"era6-bucket-v1:{source_id}:{counter}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % total_buckets
        counter += 1
        if bucket in used:
            continue
        used.add(bucket)
        selected.append(bucket)
    return [f"{bucket:02x}" for bucket in selected]


def collect_assignment4_target(
    target: dict[str, Any], assignment4_root: str | Path, project_root: str | Path
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Assignment 4 extraction requires pyarrow; install the pinned project extra") from error

    source_root = Path(assignment4_root)
    parent_path = source_root / "shard.manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if f"sha256:{parent.get('sha256')}" != target["revision"]:
        raise RuntimeError("Assignment 4 parent manifest revision does not match sources.lock.json")
    if int(parent.get("record_count", -1)) != int(target["viewer_total_rows"]):
        raise RuntimeError("Assignment 4 parent record count does not match sources.lock.json")

    quota = int(target["quota"])
    heap: list[tuple[int, int, dict[str, Any]]] = []
    serial = 0
    for bucket in deterministic_bucket_ids(target["source_id"], int(target["max_blocks"])):
        files = sorted((source_root / "twm_final" / f"hub_bucket={bucket}").glob("part-*.parquet"))
        if not files:
            raise RuntimeError(f"Missing Assignment 4 parquet bucket: {bucket}")
        for parquet_path in files:
            table = parquet.read_table(
                parquet_path,
                columns=["hub_hash", "twm_hash", "twm", "target_languages"],
                partitioning=None,
            )
            for row in table.to_pylist():
                upstream_id = str(row.get("twm_hash", ""))
                score = int.from_bytes(
                    hashlib.sha256(
                        f"era6-local-row-v1:{target['source_id']}:{upstream_id}".encode("utf-8")
                    ).digest()[:8],
                    "big",
                )
                item = (-score, serial, row)
                serial += 1
                if len(heap) < quota * 3:
                    heapq.heappush(heap, item)
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, item)

    candidates = [(-negative_score, row) for negative_score, _, row in heap]
    candidates.sort(key=lambda item: (item[0], str(item[1].get("twm_hash", ""))))
    records: list[dict[str, Any]] = []
    parent_copy = Path(project_root) / target["parent_manifest"]
    for _, row in candidates:
        transformed = transform_record(target, row)
        if transformed is None:
            continue
        transformed["metadata"]["parent_manifest_id"] = parent["shard_id"]
        transformed["metadata"]["parent_manifest_sha256"] = parent["sha256"]
        transformed["metadata"]["parent_manifest_copy_sha256"] = sha256_file(parent_copy)
        records.append(transformed)
        if len(records) == quota:
            return records
    raise RuntimeError(f"Assignment 4 extraction admitted {len(records)} rows; quota is {quota}")


def snapshot_target(target: dict[str, Any], output_dir: str | Path, root: str | Path) -> AcquisitionResult:
    cache_dir = Path(root) / "data" / "acquisition_cache" / target["source_id"]
    records = collect_viewer_target(target, cache_dir)
    output = Path(output_dir) / f"{target['source_id']}.jsonl.gz"
    stats = write_jsonl_gz(output, records)
    return AcquisitionResult(
        source_id=target["source_id"],
        path=output.relative_to(Path(root)).as_posix(),
        records=stats["records"],
        compressed_bytes=stats["compressed_bytes"],
        compressed_sha256=stats["compressed_sha256"],
        canonical_uncompressed_sha256=stats["canonical_uncompressed_sha256"],
        permission=target["permission"],
        lane=target["lane"],
    )


def snapshot_assignment4_target(
    target: dict[str, Any], assignment4_root: str | Path, output_dir: str | Path, root: str | Path
) -> AcquisitionResult:
    records = collect_assignment4_target(target, assignment4_root, root)
    output = Path(output_dir) / f"{target['source_id']}.jsonl.gz"
    stats = write_jsonl_gz(output, records)
    return AcquisitionResult(
        source_id=target["source_id"],
        path=output.relative_to(Path(root)).as_posix(),
        records=stats["records"],
        compressed_bytes=stats["compressed_bytes"],
        compressed_sha256=stats["compressed_sha256"],
        canonical_uncompressed_sha256=stats["canonical_uncompressed_sha256"],
        permission=target["permission"],
        lane=target["lane"],
    )


def verify_snapshot(path: str | Path, expected_sha256: str, expected_records: int) -> None:
    snapshot = Path(path)
    if sha256_file(snapshot) != expected_sha256:
        raise RuntimeError(f"Snapshot hash mismatch: {snapshot}")
    count = sum(1 for _ in read_jsonl_gz(snapshot))
    if count != expected_records:
        raise RuntimeError(f"Snapshot record count mismatch: {snapshot}")


def summarize_results(results: Iterable[AcquisitionResult]) -> dict[str, Any]:
    values = list(results)
    return {
        "schema_version": 1,
        "training_records": sum(item.records for item in values if item.permission != "never_train"),
        "never_train_records": sum(item.records for item in values if item.permission == "never_train"),
        "compressed_bytes": sum(item.compressed_bytes for item in values),
        "snapshots": [asdict(item) for item in values],
    }
