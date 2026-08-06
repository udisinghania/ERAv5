from __future__ import annotations

import gzip
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def atomic_write_json(path: str | Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def write_jsonl_gz(path: str | Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    lines = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    compressed = gzip.compress(lines, compresslevel=9, mtime=0)
    atomic_write_bytes(path, compressed)
    return {
        "compressed_sha256": sha256_bytes(compressed),
        "canonical_uncompressed_sha256": sha256_bytes(lines),
        "compressed_bytes": len(compressed),
        "uncompressed_bytes": len(lines),
        "records": lines.count(b"\n"),
    }


def read_jsonl_gz(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error

