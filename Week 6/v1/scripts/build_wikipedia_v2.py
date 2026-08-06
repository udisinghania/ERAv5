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
from era6.chunking import boundary_aware_chunks_v2  # noqa: E402
from era6.cleaning import SourceAwarePIIScrubber, TextNormalizer  # noqa: E402
from era6.quality import (  # noqa: E402
    extract_quality_signals_v2,
    provisional_quality_flags_v2,
    quality_band_and_weight_v2,
)


SOURCE_ID = "wikipedia_general_en"


def load_raw_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cache = ROOT / "data" / "acquisition_cache" / SOURCE_ID
    for path in sorted(cache.glob("offset-*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(item["row"] for item in json.load(stream).get("rows", []))
    return rows


def has_language_content(text: str) -> bool:
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in text)


def build_record(
    *,
    target: dict[str, Any],
    policy: dict[str, Any],
    policy_hash: str,
    raw: dict[str, Any],
    title: str,
    chunk: Any,
    output_index: int,
    output_count: int,
    parent_characters: int,
    pii_counts: dict[str, int],
    signals: dict[str, Any],
    flags: tuple[str, ...],
) -> dict[str, Any]:
    band, sampling_weight, cap_groups = quality_band_and_weight_v2(signals, flags)
    content_hash = sha256_text(chunk.text)
    parent_id = str(raw["id"])
    record_key = f"{SOURCE_ID}:{parent_id}:{output_index}:{content_hash}:{policy_hash}"
    return {
        "record_id": f"rec_{sha256_text(record_key)[:24]}",
        "group_id": f"grp_{sha256_text(parent_id)[:20]}",
        "source_id": SOURCE_ID,
        "source_revision": target["revision"],
        "upstream_id": f"{parent_id}:chunk-{output_index:04d}",
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
            "policy_id": policy["policy_id"],
            "policy_hash": policy_hash,
            "band": band,
            "sampling_weight": sampling_weight,
            "sampling_cap_groups": list(cap_groups),
            "flags": list(flags),
            "signals": signals,
            "hard_rejection_reasons": [],
        },
        "metadata": {
            "title": title,
            "url": raw.get("url"),
            "parent_characters": parent_characters,
            "chunk_index": output_index,
            "chunk_count": output_count,
            "source_chunk_index": chunk.index,
            "source_chunk_count": chunk.count,
            "chunk_end_boundary": chunk.end_boundary,
            "pii_counts": pii_counts,
            "source_class": policy["source_class"],
        },
    }


def write_markdown(report: dict[str, Any]) -> None:
    v1, v2 = report["v1"], report["v2"]
    lines = [
        "# Wikipedia general: corpus-v1 versus corpus-v2",
        "",
        "Corpus-v2 is the versioned response to the corpus-v1 human review. Corpus-v1 remains immutable and reproducible.",
        "",
        "## Before and after",
        "",
        "| Measure | corpus-v1 | corpus-v2 |",
        "|---|---:|---:|",
        f"| Parent documents retained | {v1['parents']:,} | {v2['admitted_parent_documents']:,} |",
        f"| Output records | {v1['records']:,} | {v2['records']:,} |",
        f"| Output text characters | {v1['text_characters']:,} | {v2['text_characters']:,} |",
        f"| Raw-wikitable chunks retained | not measured | 0 |",
        f"| Raw-wikitable chunks rejected | 0 | {v2['chunk_rejection_counts'].get('raw_wikitable_markup', 0):,} |",
        "",
        "## corpus-v2 quality bands",
        "",
        "| Band | Records | Meaning |",
        "|---|---:|---|",
    ]
    for band, count in v2["quality_band_counts"].items():
        lines.append(f"| {band} | {count:,} | {report['policy']['quality_bands'][band]['description']} |")
    lines.extend(
        [
            "",
            "## New reviewed signals",
            "",
            "| Flag | Records | Treatment |",
            "|---|---:|---|",
            f"| category_tail | {v2['flag_counts'].get('category_tail', 0):,} | retain in capped B0 |",
            f"| linewise_list | {v2['flag_counts'].get('linewise_list', 0):,} | retain in capped B0 |",
            f"| raw_wikitable_markup | {v2['chunk_rejection_counts'].get('raw_wikitable_markup', 0):,} | reject chunk |",
            f"| orphaned_table_footnotes | {v2['chunk_rejection_counts'].get('orphaned_table_footnotes', 0):,} | reject chunk |",
            "",
            "## Boundary behavior",
            "",
            "| Boundary | Chunks |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {key} | {value:,} |" for key, value in v2["chunk_end_boundary_counts"].items())
    lines.extend(
        [
            "",
            "The new `line` boundary preserves extracted list entries before sentence logic. Sentence splitting also refuses common abbreviations and single-letter initials.",
            "",
            "## Decision gate",
            "",
            "Review the v2 packet, including both beginning and ending previews. If the targeted errors are gone without unacceptable false positives, freeze this Wikipedia policy and begin lane-specific audits for the remaining data lanes.",
            "",
        ]
    )
    path = ROOT / "docs" / "quality" / "WIKIPEDIA_V1_V2_COMPARISON.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    policy_path = ROOT / "configs" / "quality_policy_v2.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_hash = f"sha256:{sha256_bytes(canonical_json_bytes(policy))}"
    sources = load_source_lock(ROOT / "configs" / "sources.lock.json")
    target = next(item for item in sources["targets"] if item["source_id"] == SOURCE_ID)
    normalizer, scrubber = TextNormalizer(), SourceAwarePIIScrubber()

    records: list[dict[str, Any]] = []
    parent_rejections: Counter[str] = Counter()
    chunk_rejections: Counter[str] = Counter()
    rejected_titles: Counter[str] = Counter()
    parent_chunks: Counter[int] = Counter()
    pii_parent_counts: Counter[str] = Counter()
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
        pii_counts = {
            key: clean_title.counts.get(key, 0) + clean_body.counts.get(key, 0)
            for key in ("email", "phone", "ipv4", "secret")
        }
        if sum(pii_counts.values()):
            pii_parent_counts["parents_with_redactions"] += 1
            pii_parent_counts.update(pii_counts)

        chunks = boundary_aware_chunks_v2(
            clean_title.text,
            clean_body.text,
            maximum_characters=int(policy["chunking"]["maximum_characters"]),
            minimum_continuation_characters=int(policy["chunking"]["minimum_continuation_characters"]),
        )
        candidates: list[tuple[Any, dict[str, Any], tuple[str, ...]]] = []
        for chunk in chunks:
            signals = extract_quality_signals_v2(chunk.text)
            short_continuation = len(combined) >= 400 and len(chunk.text) < 400
            flags = provisional_quality_flags_v2(
                signals,
                short_continuation=short_continuation,
                pii_redactions=sum(pii_counts.values()),
                rules=policy["reviewed_signal_rules"],
            )
            hard_chunk_rejections = set(flags) & {
                "raw_wikitable_markup",
                "orphaned_table_footnotes",
            }
            if hard_chunk_rejections:
                chunk_rejections.update(hard_chunk_rejections)
                rejected_titles[title] += 1
                continue
            candidates.append((chunk, signals, flags))

        if not candidates:
            parent_rejections["all_chunks_rejected"] += 1
            continue
        admitted_parents += 1
        parent_chunks[len(candidates)] += 1
        for output_index, (chunk, signals, flags) in enumerate(candidates):
            records.append(
                build_record(
                    target=target,
                    policy=policy,
                    policy_hash=policy_hash,
                    raw=raw,
                    title=clean_title.text,
                    chunk=chunk,
                    output_index=output_index,
                    output_count=len(candidates),
                    parent_characters=len(combined),
                    pii_counts=pii_counts,
                    signals=signals,
                    flags=flags,
                )
            )

    records.sort(key=lambda record: record["record_id"])
    duplicate_hashes = Counter(record["content_sha256"] for record in records)
    if any(count > 1 for count in duplicate_hashes.values()):
        raise RuntimeError("corpus-v2 produced exact duplicate chunks")

    output_root = ROOT / "data" / "experiments" / "corpus_v2"
    snapshot_path = output_root / "wikipedia_general_en.jsonl.gz"
    stats = write_jsonl_gz(snapshot_path, records)
    v1_path = ROOT / "data" / "experiments" / "corpus_v1" / "wikipedia_general_en.jsonl.gz"
    v1_records = list(read_jsonl_gz(v1_path))
    band_counts = Counter(record["quality"]["band"] for record in records)
    flag_counts = Counter(flag for record in records for flag in record["quality"]["flags"])
    cap_counts = Counter(
        group for record in records for group in record["quality"]["sampling_cap_groups"]
    )
    boundary_counts = Counter(record["metadata"]["chunk_end_boundary"] for record in records)
    report = {
        "schema_version": 2,
        "policy": policy,
        "policy_hash": policy_hash,
        "source_revision": target["revision"],
        "v1": {
            "artifact_sha256": sha256_file(v1_path),
            "parents": len({record["parent_upstream_id"] for record in v1_records}),
            "records": len(v1_records),
            "text_characters": sum(len(record["text"]) for record in v1_records),
        },
        "v2": {
            "raw_candidates": len(raw_rows),
            "admitted_parent_documents": admitted_parents,
            "rejected_parent_documents": sum(parent_rejections.values()),
            "parent_rejection_counts": dict(sorted(parent_rejections.items())),
            "records": len(records),
            "text_characters": sum(len(record["text"]) for record in records),
            "chunk_rejection_counts": dict(sorted(chunk_rejections.items())),
            "rejected_title_counts": dict(sorted(rejected_titles.items())),
            "chunks_per_parent_counts": {str(key): value for key, value in sorted(parent_chunks.items())},
            "chunk_end_boundary_counts": dict(sorted(boundary_counts.items())),
            "quality_band_counts": dict(sorted(band_counts.items())),
            "flag_counts": dict(sorted(flag_counts.items())),
            "sampling_cap_group_counts": dict(sorted(cap_counts.items())),
            "pii_parent_counts": dict(sorted(pii_parent_counts.items())),
        },
        "artifact": {"path": snapshot_path.relative_to(ROOT).as_posix(), **stats},
        "inputs": {
            "policy_sha256": sha256_file(policy_path),
            "v1_snapshot_sha256": sha256_file(v1_path),
        },
    }
    atomic_write_json(output_root / "comparison_report.json", report)
    write_markdown(report)
    print(json.dumps(report["v2"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
