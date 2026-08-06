from __future__ import annotations

import json
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    read_jsonl_gz,
    sha256_bytes,
    sha256_file,
    write_jsonl_gz,
)
from era6.packing import pack_group, select_for_schedule  # noqa: E402
from era6.tokenizer import MultilaneTokenizer  # noqa: E402


OUTPUT_ROOT = ROOT / "data" / "packed_v1"


class ShardPayloads:
    def __init__(self, tokenized: dict[str, Any]) -> None:
        self.resources: dict[tuple[str, str], dict[str, Any]] = {}
        for shard in tokenized["shards"]:
            manifest = json.loads((ROOT / shard["manifest_path"]).read_text(encoding="utf-8"))["manifest"]
            self.resources[(shard["lane"], shard["permission"])] = manifest["extra"]
        self.loaded_key: tuple[str, str] | None = None
        self.tokens = array("H")
        self.loss = b""

    def load(self, row: dict[str, Any]) -> tuple[list[int], list[int]]:
        key = (row["lane"], row["permission"])
        if self.loaded_key != key:
            extra = self.resources[key]
            self.tokens = array("H")
            self.tokens.frombytes((ROOT / extra["tokens_path"]).read_bytes())
            self.loss = (ROOT / extra["loss_path"]).read_bytes()
            self.loaded_key = key
        start = int(row["token_offset"])
        end = start + int(row["token_count"])
        return list(self.tokens[start:end]), list(self.loss[start:end])


def main() -> int:
    schedule_path = ROOT / "artifacts" / "schedule_v1" / "schedule.json"
    curriculum_report = json.loads(
        (ROOT / "data" / "curriculum_v1" / "curriculum_report.json").read_text(encoding="utf-8")
    )
    tokenized_path = ROOT / "data" / "tokenized_v1" / "tokenized_report.json"
    packing_config_path = ROOT / "configs" / "packing_v1.json"
    schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    curriculum_rows = list(read_jsonl_gz(ROOT / curriculum_report["overlay"]["path"]))
    tokenized = json.loads(tokenized_path.read_text(encoding="utf-8"))
    packing_config = json.loads(packing_config_path.read_text(encoding="utf-8"))
    tokenizer = MultilaneTokenizer.load(ROOT / "artifacts" / "tokenizer_v1" / "tokenizer.json")
    schedule = schedule_payload["schedule"]
    selected, selection_summary = select_for_schedule(curriculum_rows, schedule)
    if selection_summary["selected_loss_tokens"] != schedule["total_loss_token_budget"]:
        raise RuntimeError("selection does not match scheduled loss-token budget")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selection_path = OUTPUT_ROOT / "selection.jsonl.gz"
    selection_stats = write_jsonl_gz(selection_path, selected)

    groups: dict[tuple[int, str, str, str | None, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[
            (
                row["stage_order"],
                row["stage"],
                row["lane"],
                row["indic_tier"],
                row["sequence_length"],
            )
        ].append(row)
    payloads = ShardPayloads(tokenized)
    packed_tokens = array("H")
    packed_loss = bytearray()
    packed_segments = array("h")
    packed_positions = array("H")
    sequence_rows = []
    policy_sequence_counts: Counter[str] = Counter()
    loss_by_stage: Counter[str] = Counter()
    loss_by_stage_lane: Counter[tuple[str, str]] = Counter()
    loss_by_stage_tier: Counter[tuple[str, str]] = Counter()
    nonpadding_total = 0
    overlap_total = 0
    sequence_global_index = 0
    for key in sorted(groups, key=lambda value: (value[0], value[2], value[3] or "")):
        _stage_order, stage, lane, tier, sequence_length = key
        policy = packing_config["data_type_policies"].get(lane)
        if policy is None:
            raise RuntimeError(f"no packing policy declared for lane {lane}")
        rows = sorted(groups[key], key=lambda row: row["selection_order"])
        sequences = pack_group(
            rows,
            sequence_length=sequence_length,
            pad_token_id=tokenizer.special_token_ids["<pad>"],
            load_payload=payloads.load,
        )
        for local_index, sequence in enumerate(sequences):
            offset = len(packed_tokens)
            packed_tokens.extend(sequence["tokens"])
            packed_loss.extend(sequence["loss"])
            packed_segments.extend(sequence["segments"])
            packed_positions.extend(sequence["positions"])
            sequence_loss = int(sequence["loss_tokens"])
            nonpadding_total += int(sequence["nonpadding_tokens"])
            overlaps = sum(fragment["continuation_overlap"] for fragment in sequence["fragments"])
            overlap_total += overlaps
            loss_by_stage[stage] += sequence_loss
            loss_by_stage_lane[(stage, lane)] += sequence_loss
            if tier:
                loss_by_stage_tier[(stage, tier)] += sequence_loss
            sequence_rows.append(
                {
                    "sequence_index": sequence_global_index,
                    "group_sequence_index": local_index,
                    "stage": stage,
                    "lane": lane,
                    "packing_policy_id": policy["policy_id"],
                    "packing_boundary_unit": policy["boundary_unit"],
                    "loss_origin_policy": policy["loss_origin"],
                    "indic_tier": tier,
                    "sequence_length": sequence_length,
                    "global_token_offset": offset,
                    "nonpadding_tokens": sequence["nonpadding_tokens"],
                    "padding_tokens": sequence_length - sequence["nonpadding_tokens"],
                    "loss_bearing_tokens": sequence_loss,
                    "continuation_overlap_tokens": overlaps,
                    "attention_policy": packing_config["attention_policy"],
                    "fragments": sequence["fragments"],
                }
            )
            policy_sequence_counts[policy["policy_id"]] += 1
            sequence_global_index += 1

    tokens_path = OUTPUT_ROOT / "input_ids.uint16.bin"
    loss_path = OUTPUT_ROOT / "loss_mask.uint8.bin"
    segments_path = OUTPUT_ROOT / "segment_ids.int16.bin"
    positions_path = OUTPUT_ROOT / "position_ids.uint16.bin"
    index_path = OUTPUT_ROOT / "sequences.jsonl.gz"
    atomic_write_bytes(tokens_path, packed_tokens.tobytes())
    atomic_write_bytes(loss_path, bytes(packed_loss))
    atomic_write_bytes(segments_path, packed_segments.tobytes())
    atomic_write_bytes(positions_path, packed_positions.tobytes())
    index_stats = write_jsonl_gz(index_path, sequence_rows)
    component_hashes = {
        "input_ids": sha256_file(tokens_path),
        "loss_mask": sha256_file(loss_path),
        "segment_ids": sha256_file(segments_path),
        "position_ids": sha256_file(positions_path),
        "sequences": sha256_file(index_path),
        "selection": sha256_file(selection_path),
    }
    packing_hash = f"sha256:{sha256_bytes(canonical_json_bytes(component_hashes))}"
    report = {
        "schema_version": 1,
        "status": "FROZEN",
        "packing_hash": packing_hash,
        "schedule_hash": schedule_payload["schedule_hash"],
        "tokenizer_hash": tokenized["tokenizer_hash"],
        "corpus_hash": tokenized["corpus_hash"],
        "packing_config_sha256": sha256_file(packing_config_path),
        "schedule_file_sha256": sha256_file(schedule_path),
        "curriculum_overlay_sha256": curriculum_report["overlay"]["compressed_sha256"],
        "component_hashes": component_hashes,
        "paths": {
            "input_ids": tokens_path.relative_to(ROOT).as_posix(),
            "loss_mask": loss_path.relative_to(ROOT).as_posix(),
            "segment_ids": segments_path.relative_to(ROOT).as_posix(),
            "position_ids": positions_path.relative_to(ROOT).as_posix(),
            "sequences": index_path.relative_to(ROOT).as_posix(),
            "selection": selection_path.relative_to(ROOT).as_posix(),
        },
        "selected_records": selection_summary["selected_records"],
        "sequences": len(sequence_rows),
        "physical_tokens": len(packed_tokens),
        "nonpadding_tokens": nonpadding_total,
        "padding_tokens": len(packed_tokens) - nonpadding_total,
        "continuation_overlap_tokens": overlap_total,
        "loss_bearing_tokens": sum(packed_loss),
        "packing_utilization": nonpadding_total / max(1, len(packed_tokens)),
        "loss_density": sum(packed_loss) / max(1, len(packed_tokens)),
        "data_type_policies": packing_config["data_type_policies"],
        "policy_sequence_counts": dict(sorted(policy_sequence_counts.items())),
        "loss_by_stage": dict(sorted(loss_by_stage.items())),
        "loss_by_stage_lane": {
            f"{stage}|{lane}": value for (stage, lane), value in sorted(loss_by_stage_lane.items())
        },
        "loss_by_stage_indic_tier": {
            f"{stage}|{tier}": value for (stage, tier), value in sorted(loss_by_stage_tier.items())
        },
        "selection_summary": selection_summary,
        "index": index_stats,
    }
    atomic_write_json(OUTPUT_ROOT / "packing_report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "packing_hash": packing_hash,
                "selected_records": report["selected_records"],
                "sequences": report["sequences"],
                "physical_tokens": report["physical_tokens"],
                "loss_bearing_tokens": report["loss_bearing_tokens"],
                "packing_utilization": report["packing_utilization"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
