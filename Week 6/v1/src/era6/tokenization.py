from __future__ import annotations

import re
import sys
from array import array
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .canonical import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_jsonl_gz,
)
from .manifests import ShardManifest
from .tokenizer import MultilaneTokenizer


_AGENTIC_ROLE = re.compile(r"(?m)^<(system|user|assistant|tool)>\n")
_TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", flags=re.DOTALL)


def _byte_spans(text: str, character_spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    prefix_bytes = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        prefix_bytes.append(total)
    return [(prefix_bytes[start], prefix_bytes[end]) for start, end in character_spans]


def supervised_byte_spans(record: dict[str, Any]) -> list[tuple[int, int]] | None:
    """Return loss-bearing byte spans; None means ordinary full-text pretraining."""
    text = record["text"]
    if record["source_id"] == "gsm8k_reasoning_train":
        marker = "<reasoning>"
        start = text.find(marker)
        if start < 0:
            raise ValueError("GSM8K record has no reasoning marker")
        start += len(marker)
        if start < len(text) and text[start] == "\n":
            start += 1
        return _byte_spans(text, [(start, len(text))])

    if record["source_id"] == "assignment4_samanantar_translated":
        opening, closing = "<observation>", "</observation>"
        start, end = text.find(opening), text.rfind(closing)
        if start < 0 or end < 0 or end <= start:
            raise ValueError("translated record has no observation span")
        start += len(opening)
        while start < end and text[start].isspace():
            start += 1
        return _byte_spans(text, [(start, end)])

    if record["capability_lane"] == "agentic":
        matches = list(_AGENTIC_ROLE.finditer(text))
        spans = []
        for index, match in enumerate(matches):
            if match.group(1) != "assistant":
                continue
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            spans.append((start, end))
        for match in _TOOL_CALL.finditer(text):
            if not any(start <= match.start() and match.end() <= end for start, end in spans):
                spans.append((match.start(), match.end()))
        spans.sort()
        return _byte_spans(text, spans)
    return None


def record_tokens_and_mask(
    record: dict[str, Any], tokenizer: MultilaneTokenizer, *, permission: str
) -> tuple[list[int], list[int], str]:
    token_ids, offsets = tokenizer.encode_with_offsets(record["text"], add_bos=True, add_eos=True)
    if permission == "never_train":
        return token_ids, [0] * len(token_ids), "never_train_zero_loss"

    spans = supervised_byte_spans(record)
    if spans is None:
        mask = [0] + [1] * (len(token_ids) - 1)
        return token_ids, mask, "full_text_causal"

    mask = []
    text_bytes = len(record["text"].encode("utf-8"))
    for index, (start, end) in enumerate(offsets):
        if index == 0:
            mask.append(0)
        elif start == end == text_bytes:
            mask.append(1 if any(span_end == text_bytes for _span_start, span_end in spans) else 0)
        else:
            mask.append(1 if any(start >= span_start and end <= span_end for span_start, span_end in spans) else 0)
    policy = (
        "reasoning_answer_only"
        if record["source_id"] == "gsm8k_reasoning_train"
        else "translated_observation_only"
        if record["source_id"] == "assignment4_samanantar_translated"
        else "assistant_origin_only"
        if spans
        else "agentic_context_only"
    )
    return token_ids, mask, policy


def tokenize_artifact(
    *,
    root: Path,
    input_artifact: dict[str, Any],
    records: list[dict[str, Any]],
    tokenizer: MultilaneTokenizer,
    corpus_hash: str,
    cleaning_pipeline_hash: str,
    output_root: Path,
) -> dict[str, Any]:
    lane = input_artifact["lane"]
    permission = input_artifact["permission"]
    shard_id = f"{lane}-{permission}-v1"
    token_values = array("H")
    loss_values = bytearray()
    index_rows = []
    languages: Counter[str] = Counter()
    source_locks: set[str] = set()
    parent_manifests: set[str] = set()
    loss_policies: Counter[str] = Counter()
    for record in records:
        token_ids, loss_mask, loss_policy = record_tokens_and_mask(
            record, tokenizer, permission=permission
        )
        if max(token_ids, default=0) >= 65536:
            raise RuntimeError("uint16 token storage requires vocabulary below 65536")
        offset = len(token_values)
        token_values.extend(token_ids)
        loss_values.extend(loss_mask)
        languages[record["language"]] += 1
        if record.get("source_lock_hash"):
            source_locks.add(record["source_lock_hash"])
        parent_id = record.get("metadata", {}).get("parent_manifest_id")
        if parent_id:
            parent_manifests.add(parent_id)
        loss_policies[loss_policy] += 1
        index_rows.append(
            {
                "record_id": record["record_id"],
                "group_id": record["group_id"],
                "source_id": record["source_id"],
                "token_offset": offset,
                "token_count": len(token_ids),
                "loss_bearing_token_count": sum(loss_mask),
                "loss_policy": loss_policy,
                "quality_band": record.get("metadata", {}).get("quality_band"),
                "quality_weight": record.get("metadata", {}).get("quality_weight", 1.0),
            }
        )

    if sys.byteorder != "little":
        token_values.byteswap()
    shard_root = output_root / lane / permission
    tokens_path = shard_root / "tokens.uint16.bin"
    loss_path = shard_root / "loss.uint8.bin"
    index_path = shard_root / "index.jsonl.gz"
    atomic_write_bytes(tokens_path, token_values.tobytes())
    atomic_write_bytes(loss_path, bytes(loss_values))
    index_stats = write_jsonl_gz(index_path, index_rows)
    token_hash = sha256_file(tokens_path)
    loss_hash = sha256_file(loss_path)
    index_hash = sha256_file(index_path)
    content_hash = f"sha256:{sha256_bytes(canonical_json_bytes({'tokens': token_hash, 'index': index_hash}))}"

    manifest = ShardManifest(
        shard_id=shard_id,
        content_hash=content_hash,
        tokenizer_hash=tokenizer.tokenizer_hash,
        cleaning_pipeline_hash=cleaning_pipeline_hash,
        capability_lane=lane,
        permission=permission,
        record_count=len(records),
        token_count=len(token_values),
        loss_bearing_token_count=sum(loss_values),
        source_lock_hashes=tuple(sorted(source_locks)),
        language_distribution=dict(sorted(languages.items())),
        dedup_status="passed",
        pii_screen_status="passed_human_reviewed",
        eval_overlap_status="registered" if permission == "never_train" else "clear",
        position_policy="document_reset_causal",
        loss_mask_hash=f"sha256:{loss_hash}",
        parent_manifest_ids=tuple(sorted(parent_manifests)),
        extra={
            "corpus_hash": corpus_hash,
            "input_artifact": input_artifact["path"],
            "input_compressed_sha256": input_artifact["compressed_sha256"],
            "tokens_path": tokens_path.relative_to(root).as_posix(),
            "tokens_sha256": token_hash,
            "loss_path": loss_path.relative_to(root).as_posix(),
            "index_path": index_path.relative_to(root).as_posix(),
            "index_sha256": index_hash,
            "index_canonical_uncompressed_sha256": index_stats["canonical_uncompressed_sha256"],
            "loss_policy_records": dict(sorted(loss_policies.items())),
        },
    )
    manifest.validate()
    payload = {"schema_version": 1, "manifest": asdict(manifest), "manifest_hash": manifest.manifest_hash}
    manifest_path = shard_root / "manifest.json"
    atomic_write_json(manifest_path, payload)
    return {
        "shard_id": shard_id,
        "lane": lane,
        "permission": permission,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_hash": manifest.manifest_hash,
        "record_count": len(records),
        "token_count": len(token_values),
        "loss_bearing_token_count": sum(loss_values),
        "loss_policy_records": dict(sorted(loss_policies.items())),
    }
