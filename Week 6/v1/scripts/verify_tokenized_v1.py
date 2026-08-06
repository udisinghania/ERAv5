from __future__ import annotations

import json
import sys
from array import array
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import canonical_json_bytes, read_jsonl_gz, sha256_bytes, sha256_file  # noqa: E402
from era6.manifests import ShardManifest  # noqa: E402
from era6.tokenizer import MultilaneTokenizer  # noqa: E402


def main() -> int:
    report = json.loads(
        (ROOT / "data" / "tokenized_v1" / "tokenized_report.json").read_text(encoding="utf-8")
    )
    tokenizer = MultilaneTokenizer.load(ROOT / "artifacts" / "tokenizer_v1" / "tokenizer.json")
    corpus = json.loads(
        (ROOT / "data" / "frozen_corpus_v1" / "freeze_report.json").read_text(encoding="utf-8")
    )
    if report["tokenizer_hash"] != tokenizer.tokenizer_hash or report["corpus_hash"] != corpus["corpus_hash"]:
        raise AssertionError("tokenizer/corpus lineage mismatch")

    total_records = total_tokens = total_loss = 0
    for shard in report["shards"]:
        manifest_payload = json.loads((ROOT / shard["manifest_path"]).read_text(encoding="utf-8"))
        manifest = ShardManifest(**manifest_payload["manifest"])
        manifest.validate()
        if manifest.manifest_hash != manifest_payload["manifest_hash"]:
            raise AssertionError(f"manifest hash mismatch: {shard['shard_id']}")
        extra = manifest.extra
        tokens_path, loss_path, index_path = (
            ROOT / extra["tokens_path"],
            ROOT / extra["loss_path"],
            ROOT / extra["index_path"],
        )
        if sha256_file(tokens_path) != extra["tokens_sha256"]:
            raise AssertionError("token payload hash mismatch")
        if sha256_file(loss_path) != manifest.loss_mask_hash.removeprefix("sha256:"):
            raise AssertionError("loss-mask hash mismatch")
        if sha256_file(index_path) != extra["index_sha256"]:
            raise AssertionError("index hash mismatch")
        token_values = array("H")
        token_values.frombytes(tokens_path.read_bytes())
        loss_values = loss_path.read_bytes()
        index_rows = list(read_jsonl_gz(index_path))
        if len(token_values) != manifest.token_count or len(loss_values) != manifest.token_count:
            raise AssertionError("binary token/loss length mismatch")
        if len(index_rows) != manifest.record_count or sum(loss_values) != manifest.loss_bearing_token_count:
            raise AssertionError("manifest totals mismatch")
        expected_offset = 0
        for row in index_rows:
            if row["token_offset"] != expected_offset:
                raise AssertionError("non-contiguous record offsets")
            expected_offset += row["token_count"]
        if expected_offset != manifest.token_count:
            raise AssertionError("index does not cover token payload")
        if manifest.permission == "never_train" and any(loss_values):
            raise AssertionError("never-train shard has loss-bearing tokens")
        component_hash = f"sha256:{sha256_bytes(canonical_json_bytes({'tokens': extra['tokens_sha256'], 'index': extra['index_sha256']}))}"
        if component_hash != manifest.content_hash:
            raise AssertionError("manifest content hash mismatch")
        total_records += manifest.record_count
        total_tokens += manifest.token_count
        total_loss += manifest.loss_bearing_token_count
    if (total_records, total_tokens, total_loss) != (
        report["total_records"], report["total_tokens"], report["total_loss_bearing_tokens"]
    ):
        raise AssertionError("global tokenized totals mismatch")
    print(
        json.dumps(
            {
                "status": "PASS",
                "shards": len(report["shards"]),
                "records": total_records,
                "tokens": total_tokens,
                "loss_bearing_tokens": total_loss,
                "tokenizer_hash": tokenizer.tokenizer_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
