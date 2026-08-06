from __future__ import annotations

import json
import sys
from array import array
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import canonical_json_bytes, read_jsonl_gz, sha256_bytes, sha256_file  # noqa: E402
from era6.tokenizer import MultilaneTokenizer  # noqa: E402


def read_array(typecode: str, path: Path) -> array:
    result = array(typecode)
    result.frombytes(path.read_bytes())
    return result


def main() -> int:
    report_path = ROOT / "data" / "packed_v1" / "packing_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    schedule_payload = json.loads(
        (ROOT / "artifacts" / "schedule_v1" / "schedule.json").read_text(encoding="utf-8")
    )
    packing_config = json.loads(
        (ROOT / "configs" / "packing_v1.json").read_text(encoding="utf-8")
    )
    tokenizer = MultilaneTokenizer.load(ROOT / "artifacts" / "tokenizer_v1" / "tokenizer.json")
    for name, expected in report["component_hashes"].items():
        if sha256_file(ROOT / report["paths"][name]) != expected:
            raise AssertionError(f"packed component hash mismatch: {name}")
    if report["packing_hash"] != f"sha256:{sha256_bytes(canonical_json_bytes(report['component_hashes']))}":
        raise AssertionError("packing hash mismatch")
    tokens = read_array("H", ROOT / report["paths"]["input_ids"])
    loss = (ROOT / report["paths"]["loss_mask"]).read_bytes()
    segments = read_array("h", ROOT / report["paths"]["segment_ids"])
    positions = read_array("H", ROOT / report["paths"]["position_ids"])
    sequences = list(read_jsonl_gz(ROOT / report["paths"]["sequences"]))
    selected = list(read_jsonl_gz(ROOT / report["paths"]["selection"]))
    if not (len(tokens) == len(loss) == len(segments) == len(positions) == report["physical_tokens"]):
        raise AssertionError("packed binary lengths differ")
    if len(sequences) != report["sequences"] or len(selected) != report["selected_records"]:
        raise AssertionError("packed index/selection counts differ")
    if len({row["record_id"] for row in selected}) != len(selected):
        raise AssertionError("record was selected more than once")

    pad_id = tokenizer.special_token_ids["<pad>"]
    expected_offset = 0
    stage_loss: Counter[str] = Counter()
    stage_lane_loss: Counter[tuple[str, str]] = Counter()
    stage_tier_loss: Counter[tuple[str, str]] = Counter()
    policy_counts: Counter[str] = Counter()
    observed_lanes: set[str] = set()
    for sequence in sequences:
        start = sequence["global_token_offset"]
        length = sequence["sequence_length"]
        end = start + length
        if start != expected_offset:
            raise AssertionError("sequence offsets are not contiguous")
        expected_offset = end
        ids, mask, seg, pos = tokens[start:end], loss[start:end], segments[start:end], positions[start:end]
        nonpadding = sequence["nonpadding_tokens"]
        if any(value != pad_id for value in ids[nonpadding:]):
            raise AssertionError("padding region contains a non-pad token")
        if any(mask[nonpadding:]) or any(value != -1 for value in seg[nonpadding:]):
            raise AssertionError("padding carries loss or a segment")
        if any(value != 0 for value in pos[nonpadding:]):
            raise AssertionError("padding positions are nonzero")
        if sum(mask) != sequence["loss_bearing_tokens"]:
            raise AssertionError("sequence loss count mismatch")
        lane = sequence["lane"]
        policy = packing_config["data_type_policies"].get(lane)
        if policy is None:
            raise AssertionError(f"missing data-type packing policy: {lane}")
        if sequence.get("packing_policy_id") != policy["policy_id"]:
            raise AssertionError(f"packing policy ID mismatch: {lane}")
        if sequence.get("packing_boundary_unit") != policy["boundary_unit"]:
            raise AssertionError(f"packing boundary policy mismatch: {lane}")
        if sequence.get("loss_origin_policy") != policy["loss_origin"]:
            raise AssertionError(f"loss-origin packing policy mismatch: {lane}")
        policy_counts[policy["policy_id"]] += 1
        observed_lanes.add(lane)
        runs = []
        previous = None
        for index in range(nonpadding):
            if seg[index] != previous:
                if seg[index] in runs:
                    raise AssertionError("segment ID reused non-contiguously")
                runs.append(seg[index])
                previous = seg[index]
                if pos[index] != 0 or mask[index] != 0:
                    raise AssertionError("segment does not start at zero-loss position zero")
            elif pos[index] != pos[index - 1] + 1:
                raise AssertionError("positions are not sequential within a segment")
        stage_loss[sequence["stage"]] += sum(mask)
        stage_lane_loss[(sequence["stage"], sequence["lane"])] += sum(mask)
        if sequence["indic_tier"]:
            stage_tier_loss[(sequence["stage"], sequence["indic_tier"])] += sum(mask)
    if expected_offset != len(tokens):
        raise AssertionError("sequence index does not cover packed payload")
    if observed_lanes != set(packing_config["data_type_policies"]):
        raise AssertionError("not all data-type packing policies were exercised")
    if report.get("data_type_policies") != packing_config["data_type_policies"]:
        raise AssertionError("packing report policy contract mismatch")
    if report.get("policy_sequence_counts") != dict(sorted(policy_counts.items())):
        raise AssertionError("packing policy sequence counts mismatch")

    schedule = schedule_payload["schedule"]
    if sum(loss) != schedule["total_loss_token_budget"]:
        raise AssertionError("packed loss does not match schedule budget")
    for stage in schedule["stages"]:
        if stage_loss[stage["name"]] != stage["target_loss_tokens"]:
            raise AssertionError(f"stage loss mismatch: {stage['name']}")
        for lane, target in stage["lane_targets"].items():
            if stage_lane_loss[(stage["name"], lane)] != target:
                raise AssertionError(f"stage/lane loss mismatch: {stage['name']} {lane}")
        for tier, target in stage["indic_tier_targets"].items():
            if stage_tier_loss[(stage["name"], tier)] != target:
                raise AssertionError(f"stage/tier loss mismatch: {stage['name']} {tier}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "packing_hash": report["packing_hash"],
                "selected_records": len(selected),
                "sequences": len(sequences),
                "physical_tokens": len(tokens),
                "loss_bearing_tokens": sum(loss),
                "packing_utilization": report["packing_utilization"],
                "attention_isolation": "segment_ids_verified",
                "position_resets": "verified",
                "schedule_accounting": "exact",
                "data_type_policies": len(observed_lanes),
                "policy_sequence_counts": dict(sorted(policy_counts.items())),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
