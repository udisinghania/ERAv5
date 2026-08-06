from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    read_jsonl_gz,
    sha256_bytes,
    sha256_file,
    write_jsonl_gz,
)
from era6.scheduling import assign_difficulty_bands, build_executable_schedule  # noqa: E402


CURRICULUM_ROOT = ROOT / "data" / "curriculum_v1"
SCHEDULE_ROOT = ROOT / "artifacts" / "schedule_v1"


def load_token_rows(tokenized: dict[str, Any], execution: dict[str, Any]) -> list[dict[str, Any]]:
    tier_map = execution["indic_tier_by_source"]
    rows = []
    for shard in tokenized["shards"]:
        manifest = json.loads((ROOT / shard["manifest_path"]).read_text(encoding="utf-8"))["manifest"]
        for row in read_jsonl_gz(ROOT / manifest["extra"]["index_path"]):
            rows.append(
                {
                    **row,
                    "lane": shard["lane"],
                    "permission": shard["permission"],
                    "indic_tier": tier_map.get(row["source_id"]),
                }
            )
    return rows


def write_report(
    *, curriculum_report: dict[str, Any], schedule: dict[str, Any], schedule_hash: str
) -> None:
    lines = [
        "# Curriculum and executable mixture schedule v1",
        "",
        "Difficulty is a scheduling proxy, not a quality judgment. Records are ranked by token length within each source, while the independent human-reviewed quality weight remains available for record selection.",
        "",
        f"- Curriculum records: {curriculum_report['records']:,}",
        f"- Demonstration loss-token budget: {schedule['total_loss_token_budget']:,}",
        f"- Pre-anneal loss tokens: {schedule['pre_anneal_loss_tokens']:,}",
        f"- Isolated anneal loss tokens: {schedule['anneal_loss_tokens']:,}",
        f"- Schedule hash: `{schedule_hash}`",
        "",
        "## Stage quotas",
        "",
        "| Stage | Permission | Sequence | Loss tokens | Bands |",
        "|---|---|---:|---:|---|",
    ]
    for stage in schedule["stages"]:
        lines.append(
            f"| {stage['name']} | {stage['permission']} | {stage['sequence_length']} | "
            f"{stage['target_loss_tokens']:,} | {', '.join(stage['eligible_difficulty_bands'])} |"
        )
    lines.extend(["", "## Pre-anneal lane targets", "", "| Lane | Loss tokens |", "|---|---:|"])
    for lane, value in schedule["pre_anneal_lane_targets"].items():
        lines.append(f"| {lane} | {value:,} |")
    lines.extend(["", "## Protected floors", ""])
    for lane, result in schedule["protected_floors"].items():
        lines.append(
            f"- {lane}: {result['scheduled_loss_tokens']:,} scheduled; "
            f"{result['scheduled_fraction']:.3%} of pre-anneal loss tokens; passed."
        )
    lines.extend(["", "## Pre-anneal Indic tier targets", "", "| Tier | Loss tokens |", "|---|---:|"])
    for tier, value in schedule["pre_anneal_indic_tier_targets"].items():
        lines.append(f"| {tier} | {value:,} |")
    lines.extend(
        [
            "",
            "Every stage/lane/tier target is below its eligible no-replacement supply. Validation and never-train shards contribute no scheduling supply. Anneal records are accessible only to the final reserve stage. The ordinary four-tier Indic ratio applies before annealing; the final high-difficulty reserve renormalizes those weights over tiers with eligible B4/B5 supply, so it does not fabricate a synthetic tier that is absent from the reserve.",
            "",
        ]
    )
    (ROOT / "docs" / "CURRICULUM_AND_SCHEDULE_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def main() -> int:
    tokenized_path = ROOT / "data" / "tokenized_v1" / "tokenized_report.json"
    mixture_path = ROOT / "configs" / "mixture.json"
    execution_path = ROOT / "configs" / "execution_v1.json"
    tokenized = json.loads(tokenized_path.read_text(encoding="utf-8"))
    mixture = json.loads(mixture_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    raw_rows = load_token_rows(tokenized, execution)
    curriculum_rows = assign_difficulty_bands(raw_rows, execution["curriculum"]["bands"])
    CURRICULUM_ROOT.mkdir(parents=True, exist_ok=True)
    overlay_path = CURRICULUM_ROOT / "curriculum.jsonl.gz"
    overlay_stats = write_jsonl_gz(overlay_path, curriculum_rows)

    band_counts: dict[str, Counter[str]] = defaultdict(Counter)
    band_loss: dict[str, Counter[str]] = defaultdict(Counter)
    for row in curriculum_rows:
        band_counts[row["lane"]][row["difficulty_band"]] += 1
        band_loss[row["lane"]][row["difficulty_band"]] += row["loss_bearing_token_count"]
    curriculum_report = {
        "schema_version": 1,
        "status": "FROZEN",
        "tokenized_report_sha256": sha256_file(tokenized_path),
        "tokenizer_hash": tokenized["tokenizer_hash"],
        "corpus_hash": tokenized["corpus_hash"],
        "execution_config_sha256": sha256_file(execution_path),
        "overlay": {"path": overlay_path.relative_to(ROOT).as_posix(), **overlay_stats},
        "records": len(curriculum_rows),
        "band_record_counts": {lane: dict(sorted(values.items())) for lane, values in sorted(band_counts.items())},
        "band_loss_token_supply": {lane: dict(sorted(values.items())) for lane, values in sorted(band_loss.items())},
    }
    atomic_write_json(CURRICULUM_ROOT / "curriculum_report.json", curriculum_report)

    schedule = build_executable_schedule(rows=curriculum_rows, mixture=mixture, execution=execution)
    schedule_payload = {
        "schema_version": 1,
        "status": "FROZEN",
        "corpus_hash": tokenized["corpus_hash"],
        "tokenizer_hash": tokenized["tokenizer_hash"],
        "curriculum_overlay_sha256": overlay_stats["compressed_sha256"],
        "mixture_config_sha256": sha256_file(mixture_path),
        "execution_config_sha256": sha256_file(execution_path),
        "schedule": schedule,
    }
    schedule_hash = f"sha256:{sha256_bytes(canonical_json_bytes(schedule_payload))}"
    schedule_payload["schedule_hash"] = schedule_hash
    SCHEDULE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(SCHEDULE_ROOT / "schedule.json", schedule_payload)
    write_report(curriculum_report=curriculum_report, schedule=schedule, schedule_hash=schedule_hash)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "schedule_hash": schedule_hash,
                "curriculum_records": len(curriculum_rows),
                "total_loss_token_budget": schedule["total_loss_token_budget"],
                "protected_floors": schedule["protected_floors"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
