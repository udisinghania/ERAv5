from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import atomic_write_json, read_jsonl_gz, sha256_file  # noqa: E402
from era6.tokenizer import MultilaneTokenizer, train_tokenizer  # noqa: E402


TOKENIZER_ROOT = ROOT / "artifacts" / "tokenizer_v1"


def training_texts(report: dict[str, object]) -> Iterator[str]:
    for artifact in report["artifacts"]:  # type: ignore[index]
        if artifact["permission"] != "train":
            continue
        for record in read_jsonl_gz(ROOT / artifact["path"]):
            yield record["text"]


def main() -> int:
    freeze_path = ROOT / "data" / "frozen_corpus_v1" / "freeze_report.json"
    report = json.loads(freeze_path.read_text(encoding="utf-8"))
    if report["status"] != "FROZEN":
        raise RuntimeError("tokenizer input corpus is not frozen")
    config_path = ROOT / "configs" / "tokenizer_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
    tokenizer_path = TOKENIZER_ROOT / "tokenizer.json"
    payload = train_tokenizer(
        training_texts(report),
        config=config,
        corpus_hash=report["corpus_hash"],
        output_path=tokenizer_path,
    )
    tokenizer = MultilaneTokenizer.load(tokenizer_path)

    lane_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"records": 0, "characters": 0, "words": 0, "tokens": 0}
    )
    roundtrip_failures = 0
    for artifact in report["artifacts"]:
        if artifact["permission"] != "train":
            continue
        lane = artifact["lane"]
        for record in read_jsonl_gz(ROOT / artifact["path"]):
            text = record["text"]
            token_ids = tokenizer.encode(text)
            if tokenizer.decode(token_ids) != text:
                roundtrip_failures += 1
            stats = lane_stats[lane]
            stats["records"] += 1
            stats["characters"] += len(text)
            stats["words"] += len(text.split())
            stats["tokens"] += len(token_ids)
    if roundtrip_failures:
        raise RuntimeError(f"tokenizer round-trip failures: {roundtrip_failures}")

    measured = {}
    for lane, values in sorted(lane_stats.items()):
        measured[lane] = {
            **values,
            "tokens_per_character": values["tokens"] / max(1, values["characters"]),
            "tokens_per_word": values["tokens"] / max(1, values["words"]),
        }
    audit = {
        "schema_version": 1,
        "status": "FROZEN",
        "tokenizer_hash": payload["tokenizer_hash"],
        "tokenizer_file_sha256": sha256_file(tokenizer_path),
        "corpus_hash": report["corpus_hash"],
        "config_sha256": sha256_file(config_path),
        "vocab_size": payload["vocab_size"],
        "training_documents": payload["training_documents"],
        "training_characters": payload["training_characters"],
        "unknown_tokens": 0,
        "roundtrip_failures": 0,
        "lane_fertility": measured,
    }
    atomic_write_json(TOKENIZER_ROOT / "audit.json", audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "tokenizer_hash": audit["tokenizer_hash"],
                "vocab_size": audit["vocab_size"],
                "training_documents": audit["training_documents"],
                "roundtrip_failures": 0,
                "lane_tokens_per_word": {
                    lane: round(values["tokens_per_word"], 3) for lane, values in measured.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
