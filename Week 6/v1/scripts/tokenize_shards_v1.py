from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import atomic_write_json, read_jsonl_gz  # noqa: E402
from era6.tokenization import tokenize_artifact  # noqa: E402
from era6.tokenizer import MultilaneTokenizer  # noqa: E402


def main() -> int:
    corpus_report = json.loads(
        (ROOT / "data" / "frozen_corpus_v1" / "freeze_report.json").read_text(encoding="utf-8")
    )
    tokenizer = MultilaneTokenizer.load(ROOT / "artifacts" / "tokenizer_v1" / "tokenizer.json")
    if tokenizer.payload["corpus_hash"] != corpus_report["corpus_hash"]:
        raise RuntimeError("tokenizer was not trained from this frozen corpus")
    output_root = ROOT / "data" / "tokenized_v1"
    shards = []
    for artifact in corpus_report["artifacts"]:
        records = list(read_jsonl_gz(ROOT / artifact["path"]))
        shards.append(
            tokenize_artifact(
                root=ROOT,
                input_artifact=artifact,
                records=records,
                tokenizer=tokenizer,
                corpus_hash=corpus_report["corpus_hash"],
                cleaning_pipeline_hash=corpus_report["cleaning_pipeline_hash"],
                output_root=output_root,
            )
        )
    report = {
        "schema_version": 1,
        "status": "FROZEN",
        "corpus_hash": corpus_report["corpus_hash"],
        "tokenizer_hash": tokenizer.tokenizer_hash,
        "shards": shards,
        "total_records": sum(shard["record_count"] for shard in shards),
        "total_tokens": sum(shard["token_count"] for shard in shards),
        "total_loss_bearing_tokens": sum(shard["loss_bearing_token_count"] for shard in shards),
    }
    atomic_write_json(output_root / "tokenized_report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "shards": len(shards),
                "records": report["total_records"],
                "tokens": report["total_tokens"],
                "loss_bearing_tokens": report["total_loss_bearing_tokens"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
