from __future__ import annotations

import gzip
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.acquisition import load_source_lock  # noqa: E402
from era6.canonical import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    read_jsonl_gz,
    sha256_bytes,
    sha256_file,
    sha256_text,
    write_jsonl_gz,
)
from era6.chunking import boundary_aware_chunks  # noqa: E402
from era6.cleaning import SourceAwarePIIScrubber, TextNormalizer  # noqa: E402
from era6.quality import extract_quality_signals, provisional_quality_flags, quality_band_and_weight  # noqa: E402


SOURCE_ID = "wikipedia_general_en"


def load_raw_rows() -> list[dict[str, Any]]:
    rows = []
    cache = ROOT / "data" / "acquisition_cache" / SOURCE_ID
    for path in sorted(cache.glob("offset-*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(item["row"] for item in json.load(stream).get("rows", []))
    return rows


def has_language_content(text: str) -> bool:
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in text)


def build_record(
    *, target: dict[str, Any], policy: dict[str, Any], policy_hash: str,
    raw: dict[str, Any], title: str, chunk: Any, parent_characters: int,
    pii_counts: dict[str, int],
) -> dict[str, Any]:
    signals = extract_quality_signals(chunk.text)
    provisional = list(provisional_quality_flags(signals, truncated=False, pii_redactions=sum(pii_counts.values())))
    if parent_characters >= 400 and signals["characters"] < 400:
        provisional = [flag for flag in provisional if flag != "short_document"]
        provisional.append("short_continuation_chunk")
    flags = tuple(sorted(set(provisional)))
    band, sampling_weight, cap_groups = quality_band_and_weight(signals, flags)
    content_hash = sha256_text(chunk.text)
    parent_id = str(raw["id"])
    record_key = f"{SOURCE_ID}:{parent_id}:{chunk.index}:{content_hash}:{policy_hash}"
    return {
        "record_id": f"rec_{sha256_text(record_key)[:24]}",
        "group_id": f"grp_{sha256_text(parent_id)[:20]}",
        "source_id": SOURCE_ID,
        "source_revision": target["revision"],
        "upstream_id": f"{parent_id}:chunk-{chunk.index:04d}",
        "parent_upstream_id": parent_id,
        "content_sha256": content_hash,
        "capability_lane": "general",
        "provenance_tier": target["provenance_tier"],
        "permission": "train",
        "language": "en",
        "license_id": target["license_id"],
        "pii_redactions": sum(pii_counts.values()),
        "text": chunk.text,
        "quality": {
            "policy_id": policy["policy_id"], "policy_hash": policy_hash,
            "band": band, "sampling_weight": sampling_weight,
            "sampling_cap_groups": list(cap_groups), "flags": list(flags),
            "signals": signals, "hard_rejection_reasons": [],
        },
        "metadata": {
            "title": title, "url": raw.get("url"), "parent_characters": parent_characters,
            "chunk_index": chunk.index, "chunk_count": chunk.count,
            "chunk_end_boundary": chunk.end_boundary, "pii_counts": pii_counts,
            "source_class": policy["source_class"],
        },
    }


def write_report(report: dict[str, Any]) -> None:
    v0, v1 = report["v0"], report["v1"]
    lines = [
        "# Wikipedia general: corpus-v0 versus corpus-v1", "",
        "Corpus-v1 applies the six approved recommendations to the same cached raw candidates. Corpus-v0 remains unchanged through the `corpus-v0` Git tag.", "",
        "## Before and after", "", "| Measure | corpus-v0 | corpus-v1 |", "|---|---:|---:|",
        f"| Raw candidates examined | {v0['raw_candidates']:,} | {v1['raw_candidates']:,} |",
        f"| Parent documents admitted | {v0['selected_parent_documents']:,} | {v1['admitted_parent_documents']:,} |",
        f"| Output records/chunks | {v0['records']:,} | {v1['records']:,} |",
        f"| Output text characters | {v0['text_characters']:,} | {v1['text_characters']:,} |",
        f"| Mid-boundary character truncations | {v0['mid_boundary_truncations']:,} | {v1['mid_boundary_truncations']:,} |",
        f"| Records with PII redactions | {v0['records_with_pii_redactions']:,} | {v1['records_with_pii_redactions']:,} |",
        "", "## corpus-v1 quality bands", "", "| Band | Records | Meaning |", "|---|---:|---|",
    ]
    for band, count in v1["quality_band_counts"].items():
        lines.append(f"| {band} | {count:,} | {report['policy']['quality_bands'][band]['description']} |")
    lines.extend(["", "## Sampling cap groups", "", "| Cap group | Records | Maximum scheduled share |", "|---|---:|---:|"])
    for group, count in v1["sampling_cap_group_counts"].items():
        lines.append(f"| {group} | {count:,} | {report['policy']['sampling_caps'][group]:.1%} |")
    lines.extend([
        "", "## What changed", "",
        "1. Articles split at paragraph, sentence, or—only as a last resort—word boundaries.",
        "2. Public-reference masking preserves bare identifiers and technical IP addresses.",
        "3. Emails, secrets, explicitly labelled phones, and formatted international phones remain masked.",
        "4. Useful 300–399 character parents are retained in B0.",
        "5. Short, disambiguation, list-like, and repetitive material uses lower weights and explicit caps.",
        "6. Every chunk records signals, flags, band, weight, caps, policy hash, boundary, parent, and PII counts.",
        "", "## Remaining gate", "",
        "This policy is Wikipedia-specific. Representative v1 samples and cap sizes must be reviewed before analogous lane-specific policies are applied elsewhere.", "",
    ])
    (ROOT / "docs" / "quality" / "WIKIPEDIA_V0_V1_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    policy_path = ROOT / "configs" / "quality_policy_v1.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_hash = f"sha256:{sha256_bytes(canonical_json_bytes(policy))}"
    sources = load_source_lock(ROOT / "configs" / "sources.lock.json")
    target = next(item for item in sources["targets"] if item["source_id"] == SOURCE_ID)
    normalizer, scrubber = TextNormalizer(), SourceAwarePIIScrubber()
    records: list[dict[str, Any]] = []
    parent_rejections, parent_chunks, pii_parent_counts = Counter(), Counter(), Counter()
    admitted_parents = 0
    raw_rows = load_raw_rows()
    for raw in raw_rows:
        title = normalizer.normalize(str(raw.get("title", "")))
        body = normalizer.normalize(str(raw.get("text", "")))
        combined = f"{title}\n\n{body}".strip()
        if len(combined) < int(policy["minimum_parent_characters"]):
            parent_rejections["parent_below_300_characters"] += 1
            continue
        if not has_language_content(combined):
            parent_rejections["no_language_content"] += 1
            continue
        if combined.count("�") / max(1, len(combined)) > 0.001:
            parent_rejections["excessive_replacement_characters"] += 1
            continue
        clean_title = scrubber.scrub(title, source_class=policy["source_class"])
        clean_body = scrubber.scrub(body, source_class=policy["source_class"])
        pii_counts = {key: clean_title.counts.get(key, 0) + clean_body.counts.get(key, 0) for key in ("email", "phone", "ipv4", "secret")}
        if sum(pii_counts.values()):
            pii_parent_counts["parents_with_redactions"] += 1
            pii_parent_counts.update(pii_counts)
        chunks = boundary_aware_chunks(
            clean_title.text, clean_body.text,
            maximum_characters=int(policy["chunking"]["maximum_characters"]),
            minimum_continuation_characters=int(policy["chunking"]["minimum_continuation_characters"]),
        )
        if not chunks:
            parent_rejections["empty_after_cleaning"] += 1
            continue
        admitted_parents += 1
        parent_chunks[len(chunks)] += 1
        for chunk in chunks:
            records.append(build_record(target=target, policy=policy, policy_hash=policy_hash, raw=raw, title=clean_title.text, chunk=chunk, parent_characters=len(combined), pii_counts=pii_counts))
    records.sort(key=lambda record: record["record_id"])
    duplicate_hashes = Counter(record["content_sha256"] for record in records)
    if any(count > 1 for count in duplicate_hashes.values()):
        raise RuntimeError("Boundary-aware transformation produced exact duplicate chunks")

    output_root = ROOT / "data" / "experiments" / "corpus_v1"
    snapshot_path = output_root / "wikipedia_general_en.jsonl.gz"
    stats = write_jsonl_gz(snapshot_path, records)
    v0_records = list(read_jsonl_gz(ROOT / "data" / "source_snapshots" / "wikipedia_general_en.jsonl.gz"))
    band_counts = Counter(record["quality"]["band"] for record in records)
    cap_counts = Counter(group for record in records for group in record["quality"]["sampling_cap_groups"])
    boundary_counts = Counter(record["metadata"]["chunk_end_boundary"] for record in records)
    report = {
        "schema_version": 1, "policy": policy, "policy_hash": policy_hash,
        "source_revision": target["revision"],
        "source_cache_files": len(list((ROOT / "data" / "acquisition_cache" / SOURCE_ID).glob("*.json.gz"))),
        "v0": {
            "raw_candidates": len(raw_rows), "selected_parent_documents": len({record["upstream_id"] for record in v0_records}),
            "records": len(v0_records), "text_characters": sum(len(record["text"]) for record in v0_records),
            "mid_boundary_truncations": 434, "records_with_pii_redactions": sum(bool(record["pii_redactions"]) for record in v0_records),
        },
        "v1": {
            "raw_candidates": len(raw_rows), "admitted_parent_documents": admitted_parents,
            "rejected_parent_documents": sum(parent_rejections.values()), "parent_rejection_counts": dict(sorted(parent_rejections.items())),
            "records": len(records), "text_characters": sum(len(record["text"]) for record in records),
            "mid_boundary_truncations": boundary_counts.get("character", 0), "chunk_end_boundary_counts": dict(sorted(boundary_counts.items())),
            "chunks_per_parent_counts": {str(key): value for key, value in sorted(parent_chunks.items())},
            "records_with_pii_redactions": sum(bool(record["pii_redactions"]) for record in records),
            "pii_parent_counts": dict(sorted(pii_parent_counts.items())),
            "quality_band_counts": dict(sorted(band_counts.items())), "sampling_cap_group_counts": dict(sorted(cap_counts.items())),
        },
        "artifact": {"path": snapshot_path.relative_to(ROOT).as_posix(), **stats},
        "inputs": {"policy_sha256": sha256_file(policy_path), "v0_snapshot_sha256": sha256_file(ROOT / "data" / "source_snapshots" / "wikipedia_general_en.jsonl.gz")},
    }
    atomic_write_json(output_root / "comparison_report.json", report)
    write_report(report)
    print(json.dumps(report["v1"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
