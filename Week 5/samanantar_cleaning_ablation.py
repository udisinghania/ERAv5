#!/usr/bin/env python3
"""Samanantar A/B/C cleaning-policy micro-proxy.

This experiment tests a Week 4 cleaning decision with a compact translation
model. It is not a formal 1B/3B mixture proxy. Preparation streams raw
AI4Bharat Samanantar records, assigns source-near-duplicate clusters to
mutually disjoint arms/splits, applies the three policies, and emits equal-size
train JSONL files plus one held-out JSONL. Training fine-tunes the same compact
checkpoint for the same number of steps in every arm.

Required packages for prepare:
  datasets fasttext datasketch

Required packages for train:
  torch transformers sentencepiece datasets sacrebleu
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
from importlib import metadata as package_metadata
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata


LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}

SCRIPT_RANGES = {
    "as": ((0x0980, 0x09FF),),
    "bn": ((0x0980, 0x09FF),),
    "gu": ((0x0A80, 0x0AFF),),
    "hi": ((0x0900, 0x097F),),
    "kn": ((0x0C80, 0x0CFF),),
    "ml": ((0x0D00, 0x0D7F),),
    "mr": ((0x0900, 0x097F),),
    "or": ((0x0B00, 0x0B7F),),
    "pa": ((0x0A00, 0x0A7F),),
    "ta": ((0x0B80, 0x0BFF),),
    "te": ((0x0C00, 0x0C7F),),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def obviously_mutable_revision(revision: str) -> bool:
    normalized = revision.strip().casefold()
    return normalized in {"main", "master"} or normalized.startswith("refs/heads/")


def immutable_hub_commit(revision: object) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{40,64}", str(revision or "")))


def valid_token_matching_execution(world_size: int, visible_cuda_devices: int) -> bool:
    return world_size == 1 and visible_cuda_devices <= 1


def normalized_tokens(text: str) -> list[str]:
    tokens = []
    for raw in text.casefold().split():
        token = "".join(
            char
            for char in raw
            if unicodedata.category(char)[0] in {"L", "M", "N"}
            or char in {"\u200c", "\u200d"}
        )
        if token:
            tokens.append(token)
    return tokens


def script_fraction(text: str, language: str) -> float:
    expected = SCRIPT_RANGES[language]
    letters = [char for char in text if unicodedata.category(char).startswith("L")]
    if not letters:
        return 0.0
    matches = sum(
        any(start <= ord(char) <= end for start, end in expected)
        for char in letters
    )
    return matches / len(letters)


def repeated_output(text: str) -> bool:
    tokens = normalized_tokens(text)
    if len(tokens) < 9:
        return False
    trigrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    return 1.0 - len(set(trigrams)) / len(trigrams) > 0.30


def read_proxy_questions(path: Path) -> list[str]:
    questions = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("question") or row.get("text") or row.get("prompt")
            if value:
                questions.append(str(value))
    if not questions:
        raise ValueError(f"No proxy question text found in {path}")
    return questions


@dataclass
class RawRecord:
    ordinal: int
    parent_pair_id: str
    raw_src: str
    raw_tgt: str
    normalized_src: str
    normalized_tgt: str
    cluster_key: str = ""
    cluster_id: str = ""


class SourceClusterer:
    """In-memory source-side MinHash clustering for bounded micro-proxy pools."""

    def __init__(self, threshold: float, num_perm: int, seed: int):
        try:
            from datasketch import MinHash, MinHashLSH
        except ModuleNotFoundError as exc:
            raise RuntimeError("prepare requires datasketch") from exc
        self.MinHash = MinHash
        self.index = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.seed = seed
        self.parent: dict[str, str] = {}
        self.parent_pair_id: dict[str, str] = {}

    def signature(self, text: str):
        tokens = normalized_tokens(text)
        shingles = (
            {" ".join(tokens)}
            if len(tokens) < 3
            else {" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)}
        )
        signature = self.MinHash(num_perm=self.num_perm, seed=self.seed)
        for shingle in sorted(shingles):
            signature.update(shingle.encode("utf-8"))
        return signature

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            next_key = self.parent[key]
            self.parent[key] = root
            key = next_key
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root

    def add(self, parent_id: str, source: str, ordinal: int) -> str:
        key = f"{parent_id}:{ordinal}"
        signature = self.signature(source)
        hits = sorted(self.index.query(signature))
        self.parent[key] = key
        self.parent_pair_id[key] = parent_id
        for hit in hits:
            self.union(key, hit)
        self.index.insert(key, signature)
        return key

    def canonical_map(self) -> dict[str, str]:
        members: dict[str, list[str]] = defaultdict(list)
        for key in self.parent:
            members[self.find(key)].append(key)
        root_to_cluster = {
            root: min(self.parent_pair_id[key] for key in keys)
            for root, keys in members.items()
        }
        return {key: root_to_cluster[self.find(key)] for key in self.parent}


def arm_and_split(cluster_id: str, seed: int, heldout_fraction: float) -> tuple[str, str]:
    value = deterministic_u64("split", seed, cluster_id) / 2**64
    if value < heldout_fraction:
        return "heldout", "heldout"
    arm_index = deterministic_u64("arm", seed, cluster_id) % 3
    return "train", ("A", "B", "C")[arm_index]


def stable_row_key(row: dict) -> tuple[str, str]:
    return str(row.get("parent_pair_id", "")), str(row.get("cluster_id", ""))


def select_token_balanced_subsets(
    candidates: dict[str, list[dict]],
    selected_count: int,
    tolerance: float,
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int] | float]]:
    """Choose deterministic fixed-size windows with matched combined token totals."""

    if selected_count < 1:
        raise ValueError("selected_count must be positive")
    if not 0 < tolerance <= 0.01:
        raise ValueError("token-exposure tolerance must be in (0, 0.01]")

    ordered: dict[str, list[dict]] = {}
    for arm in ("A", "B", "C"):
        rows = candidates[arm]
        if len(rows) < selected_count:
            raise RuntimeError(
                f"Arm {arm} has {len(rows)} tokenized rows; "
                f"{selected_count} are required for the fixed training schedule."
            )
        ordered[arm] = sorted(
            rows,
            key=lambda row: (
                int(row["_combined_tokens"]),
                stable_row_key(row),
            ),
        )

    lower = max(
        sum(int(row["_combined_tokens"]) for row in rows[:selected_count])
        for rows in ordered.values()
    )
    upper = min(
        sum(int(row["_combined_tokens"]) for row in rows[-selected_count:])
        for rows in ordered.values()
    )
    if lower > upper:
        raise RuntimeError(
            "No common combined-token target is feasible at the required example count."
        )
    target = (lower + upper) // 2

    selected: dict[str, list[dict]] = {}
    exposure: dict[str, dict[str, int] | float] = {}
    for arm, rows in ordered.items():
        window_total = sum(
            int(row["_combined_tokens"]) for row in rows[:selected_count]
        )
        best_start, best_total = 0, window_total
        for start in range(1, len(rows) - selected_count + 1):
            window_total -= int(rows[start - 1]["_combined_tokens"])
            window_total += int(rows[start + selected_count - 1]["_combined_tokens"])
            candidate = (abs(window_total - target), start)
            incumbent = (abs(best_total - target), best_start)
            if candidate < incumbent:
                best_start, best_total = start, window_total
        chosen = rows[best_start : best_start + selected_count]
        selected[arm] = sorted(chosen, key=stable_row_key)
        exposure[arm] = {
            "selected_examples": len(chosen),
            "source_nonpadding_tokens": sum(
                int(row["_source_tokens"]) for row in chosen
            ),
            "target_nonpadding_tokens": sum(
                int(row["_target_tokens"]) for row in chosen
            ),
            "supervised_tokens": sum(
                int(row["_supervised_tokens"]) for row in chosen
            ),
            "combined_nonpadding_tokens": sum(
                int(row["_combined_tokens"]) for row in chosen
            ),
        }

    totals = [
        int(exposure[arm]["combined_nonpadding_tokens"]) for arm in ("A", "B", "C")
    ]
    spread = (max(totals) - min(totals)) / min(totals)
    exposure["summary"] = {
        "target_combined_tokens": target,
        "relative_spread": spread,
        "tolerance": tolerance,
    }
    if spread > tolerance:
        raise RuntimeError(
            f"Combined token exposure differs by {spread:.6%}; "
            f"maximum allowed is {tolerance:.6%}."
        )
    return selected, exposure


def run_self_tests(_args) -> int:
    script_tests = {
        "valid_target_script": script_fraction("मराठी भाषा", "mr") == 1.0,
        "all_english": script_fraction("all English output", "mr") == 0.0,
        "mixed_english_indic_fails": (
            script_fraction("mostly English words with मराठी", "mr") < 0.50
        ),
        "empty": script_fraction("", "mr") == 0.0,
        "punctuation_only": script_fraction("...!?—", "mr") == 0.0,
    }

    token_candidates: dict[str, list[dict]] = {}
    values = {
        "A": [90, 100, 100, 100, 100, 110],
        "B": [80, 100, 100, 100, 100, 120],
        "C": [70, 100, 100, 100, 100, 130],
    }
    for arm, totals in values.items():
        token_candidates[arm] = [
            {
                "parent_pair_id": f"{arm}-{index}",
                "cluster_id": f"{arm}-{index}",
                "_source_tokens": total // 2,
                "_target_tokens": total - total // 2,
                "_supervised_tokens": total - total // 2,
                "_combined_tokens": total,
            }
            for index, total in enumerate(totals)
        ]
    _, exposure = select_token_balanced_subsets(
        token_candidates, selected_count=4, tolerance=0.01
    )
    token_test = float(exposure["summary"]["relative_spread"]) <= 0.01
    revision_tests = {
        "rejects_main": obviously_mutable_revision("main"),
        "rejects_master": obviously_mutable_revision("master"),
        "rejects_refs_heads": obviously_mutable_revision("refs/heads/release"),
        "accepts_immutable_commit": immutable_hub_commit("a" * 40),
        "rejects_noncommit_tag": not immutable_hub_commit("v1.0"),
    }
    execution_mode_tests = {
        "accepts_single_process_one_gpu": valid_token_matching_execution(1, 1),
        "accepts_single_process_cpu": valid_token_matching_execution(1, 0),
        "rejects_distributed_world_size": not valid_token_matching_execution(2, 1),
        "rejects_multiple_visible_gpus": not valid_token_matching_execution(1, 2),
    }
    passed = (
        all(script_tests.values())
        and token_test
        and all(revision_tests.values())
        and all(execution_mode_tests.values())
    )
    payload = {
        "status": "passed" if passed else "failed",
        "wrong_script_behavior": script_tests,
        "revision_handling": revision_tests,
        "execution_mode": execution_mode_tests,
        "token_exposure_balance": {
            "passed": token_test,
            "exposure": exposure,
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 1


def prepare(args) -> int:
    if obviously_mutable_revision(args.revision):
        raise SystemExit(
            "--revision must not be a mutable branch such as main, master, "
            "or refs/heads/*."
        )
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "prepare requires datasets and huggingface_hub"
        ) from exc

    dataset_id = "ai4bharat/samanantar"
    try:
        dataset_info = HfApi().dataset_info(dataset_id, revision=args.revision)
    except Exception as exc:
        raise SystemExit(
            f"Could not resolve dataset revision {args.revision!r} for {dataset_id}."
        ) from exc
    resolved_dataset_revision = getattr(dataset_info, "sha", None)
    if not immutable_hub_commit(resolved_dataset_revision):
        raise SystemExit(
            "The dataset hub did not resolve --revision to an immutable commit."
        )

    week4_root = args.week4_root.resolve()
    sys.path.insert(0, str(week4_root))
    try:
        from lang_filter import (
            FastTextLanguageID,
            HeuristicQualityFilter,
            PairAlignmentFilter,
            QualityConfig,
        )
        from normalizer import TextNormalizer
        from pii_masker import NGramDecontaminator, PIIScrubber
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Week 4 modules or their fastText dependency are unavailable. "
            "Activate the Week 4 environment; do not modify Week 4."
        ) from exc

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer = TextNormalizer()
    alignment = PairAlignmentFilter(0.30, 3.00)
    quality = HeuristicQualityFilter(
        QualityConfig(min_stopword_count=0, require_terminal_punctuation=False)
    )
    lid = FastTextLanguageID(
        model_path=week4_root / "models" / "lid.176.ftz",
        expected_language=args.language,
        min_confidence=0.70,
        short_text_confidence=0.40,
    )
    scrubber = PIIScrubber()
    firewall = NGramDecontaminator(
        [
            scrubber.scrub(normalizer.normalize(question)).text
            for question in read_proxy_questions(
                week4_root / "golden_proxies_benchmark.jsonl"
            )
        ],
        ngram_size=13,
    )

    stream = load_dataset(
        dataset_id,
        args.language,
        split="train",
        streaming=True,
        revision=resolved_dataset_revision,
    )
    clusterer = SourceClusterer(args.cluster_threshold, args.num_perm, args.seed)
    counters = Counter()
    raw_records: list[RawRecord] = []

    # Pass 1: collect the bounded raw pool and finish all near-duplicate unions
    # before assigning any cluster to a split or arm.
    for ordinal, example in enumerate(stream):
        if ordinal >= args.raw_limit:
            break
        counters["raw_seen"] += 1
        raw_src = "" if example.get("src") is None else str(example["src"])
        raw_tgt = "" if example.get("tgt") is None else str(example["tgt"])
        parent_id = sha256_text(f"{args.language}\x1f{raw_src}\x1f{raw_tgt}")
        if not raw_src.strip() or not raw_tgt.strip():
            counters["a_null_or_empty"] += 1
            continue
        normalized_src = normalizer.normalize(raw_src)
        normalized_tgt = normalizer.normalize(raw_tgt)
        cluster_key = clusterer.add(parent_id, normalized_src, ordinal)
        raw_records.append(
            RawRecord(
                ordinal=ordinal,
                parent_pair_id=parent_id,
                raw_src=raw_src.strip(),
                raw_tgt=raw_tgt.strip(),
                normalized_src=normalized_src,
                normalized_tgt=normalized_tgt,
                cluster_key=cluster_key,
            )
        )

    canonical_clusters = clusterer.canonical_map()
    for record in raw_records:
        record.cluster_id = canonical_clusters[record.cluster_key]

    # Pass 2: evaluate each policy on the same bounded raw pool. Arm C retains
    # one deterministic clean representative per completed source cluster.
    processed_b: dict[str, tuple[str, str]] = {}
    processed_c: dict[str, tuple[str, str]] = {}
    selected_c_clusters: set[str] = set()
    for record in raw_records:
        key = record.cluster_key
        normalized_src = record.normalized_src
        normalized_tgt = record.normalized_tgt
        if normalized_src and normalized_tgt:
            processed_b[key] = (normalized_src, normalized_tgt)
        else:
            counters["policy_b_empty_after_normalization"] += 1
            counters["policy_c_empty_after_normalization"] += 1
            continue
        if record.cluster_id in selected_c_clusters:
            counters["policy_c_dedup"] += 1
            continue
        if not alignment.keep(normalized_src, normalized_tgt):
            counters["policy_c_alignment"] += 1
            continue
        quality_report = quality.evaluate(normalized_tgt)
        if not quality_report.accepted:
            counters["policy_c_quality"] += 1
            for reason in quality_report.reasons:
                counters[f"policy_c_quality_{reason}"] += 1
            continue
        if not lid.keep(normalized_tgt):
            counters["policy_c_lid"] += 1
            continue
        if script_fraction(normalized_tgt, args.language) < args.min_script_fraction:
            counters["policy_c_script"] += 1
            continue
        source = scrubber.scrub(normalized_src).text
        target = scrubber.scrub(normalized_tgt).text
        if firewall.is_contaminated_pair(source, target):
            counters["policy_c_decontamination"] += 1
            continue
        selected_c_clusters.add(record.cluster_id)
        processed_c[key] = (source, target)

    # Pass 3: assign complete clusters, then select the appropriate policy view.
    pools: dict[str, list[dict]] = {"A": [], "B": [], "C": [], "heldout": []}
    for record in raw_records:
        parent_id = record.parent_pair_id
        split, arm = arm_and_split(record.cluster_id, args.seed, args.heldout_fraction)

        if split == "heldout":
            if record.cluster_key not in processed_c:
                continue
            src, tgt = processed_c[record.cluster_key]
            pools["heldout"].append(
                {
                    "parent_pair_id": parent_id,
                    "cluster_id": record.cluster_id,
                    "language": args.language,
                    "source": src,
                    "target": tgt,
                    "arm": "heldout",
                    "derived_direction": False,
                }
            )
            continue

        if arm == "A":
            source, target = record.raw_src, record.raw_tgt
        elif arm == "B":
            if record.cluster_key not in processed_b:
                continue
            source, target = processed_b[record.cluster_key]
        else:
            if record.cluster_key not in processed_c:
                continue
            source, target = processed_c[record.cluster_key]

        pools[arm].append(
            {
                "parent_pair_id": parent_id,
                "cluster_id": record.cluster_id,
                "language": args.language,
                "source": source,
                "target": target,
                "arm": arm,
                "derived_direction": False,
                "selection_priority": deterministic_u64("select", args.seed, parent_id),
            }
        )

    policy_retained = {
        "A": len(raw_records),
        "B": len(processed_b),
        "C": len(processed_c),
    }
    eligible = {arm: len(pools[arm]) for arm in ("A", "B", "C")}
    equal_size = min(args.train_per_arm, *eligible.values())
    if equal_size < 1:
        raise RuntimeError(f"No equal-size arm can be created: {eligible}")
    heldout_size = min(args.heldout_size, len(pools["heldout"]))
    if heldout_size < 1:
        raise RuntimeError("No held-out records survived.")

    selected: dict[str, list[dict]] = {}
    for arm in ("A", "B", "C"):
        selected[arm] = sorted(pools[arm], key=lambda row: row["selection_priority"])[:equal_size]
        for row in selected[arm]:
            row.pop("selection_priority", None)
    selected["heldout"] = sorted(
        pools["heldout"],
        key=lambda row: deterministic_u64("heldout", args.seed, row["parent_pair_id"]),
    )[:heldout_size]

    cluster_sets = {
        name: {row["cluster_id"] for row in rows}
        for name, rows in selected.items()
    }
    intersections = {
        f"{left}_{right}": len(cluster_sets[left] & cluster_sets[right])
        for index, left in enumerate(cluster_sets)
        for right in list(cluster_sets)[index + 1 :]
    }
    if any(intersections.values()):
        raise RuntimeError(f"Cluster leakage detected: {intersections}")

    for name, rows in selected.items():
        path = output_dir / (f"train_{name}.jsonl" if name != "heldout" else "heldout.jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "classification": "Samanantar cleaning-policy micro-proxy preparation; not 1B/3B",
        "dataset_id": dataset_id,
        "language_pair": f"en->{args.language}",
        "requested_dataset_revision": args.revision,
        "resolved_dataset_revision": resolved_dataset_revision,
        "seed": args.seed,
        "raw_limit": args.raw_limit,
        "arm_treatments": {
            "A": "schema/non-null/non-empty only",
            "B": "A plus Week 4 normalization",
            "C": "B plus language/script/alignment/quality/PII/decontamination/dedup",
        },
        "policy_retained_before_cluster_assignment": policy_retained,
        "eligible_before_equalization": eligible,
        "selected_train_per_arm": equal_size,
        "selected_heldout": heldout_size,
        "cluster_intersections": intersections,
        "counters": dict(sorted(counters.items())),
        "arm_raw_retention_percent": {
            arm: 100.0 * policy_retained[arm] / max(counters["raw_seen"], 1)
            for arm in ("A", "B", "C")
        },
        "week4_root": str(week4_root),
        "parent_pair_policy": (
            "SHA-256(language, raw source, raw target); derived reverse direction "
            "must reuse the parent ID and split/arm"
        ),
    }
    (output_dir / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train(args) -> int:
    try:
        import sacrebleu
        import torch
        from datasets import Dataset
        from transformers import (
            AutoConfig,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            set_seed,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "train requires torch, transformers, sentencepiece, datasets, and sacrebleu"
        ) from exc

    try:
        environment_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise SystemExit("WORLD_SIZE must be an integer.") from exc
    world_size = (
        int(torch.distributed.get_world_size())
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else environment_world_size
    )
    visible_cuda_devices = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if world_size != 1:
        raise SystemExit(
            "Token-exposure matching requires single-process execution (WORLD_SIZE=1)."
        )
    if not valid_token_matching_execution(world_size, visible_cuda_devices):
        raise SystemExit(
            "Token-exposure matching requires at most one visible GPU; set "
            "CUDA_VISIBLE_DEVICES to one device."
        )

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((data_dir / "preparation_manifest.json").read_text(encoding="utf-8"))
    resolved_dataset_revision = manifest.get("resolved_dataset_revision")
    if not immutable_hub_commit(resolved_dataset_revision):
        raise RuntimeError(
            "preparation_manifest.json lacks an immutable resolved dataset revision."
        )
    language = manifest["language_pair"].split("->", 1)[1]
    language_name = LANGUAGE_NAMES[language]
    validation_rows = load_jsonl(data_dir / "heldout.jsonl")
    model_config = AutoConfig.from_pretrained(
        args.model_name, revision=args.model_revision
    )
    resolved_model_revision = getattr(model_config, "_commit_hash", None)
    if not immutable_hub_commit(resolved_model_revision):
        raise RuntimeError(
            "The model hub did not resolve --model-revision to an immutable commit."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=resolved_model_revision
    )
    resolved_tokenizer_revision = (
        getattr(tokenizer, "_commit_hash", None)
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or resolved_model_revision
    )
    if not immutable_hub_commit(resolved_tokenizer_revision):
        raise RuntimeError(
            "The tokenizer did not resolve to an immutable model commit."
        )

    def encode_rows(rows: list[dict]):
        sources = [f"translate English to {language_name}: {row['source']}" for row in rows]
        targets = [row["target"] for row in rows]
        encoded = tokenizer(
            sources,
            max_length=args.max_source_length,
            truncation=True,
        )
        encoded["labels"] = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
        )["input_ids"]
        return encoded

    def tokenize_for_balance(rows: list[dict]) -> list[dict]:
        sources = [
            f"translate English to {language_name}: {row['source']}" for row in rows
        ]
        targets = [row["target"] for row in rows]
        source_ids = tokenizer(
            sources,
            max_length=args.max_source_length,
            truncation=True,
            padding=False,
        )["input_ids"]
        target_ids = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
            padding=False,
        )["input_ids"]
        tokenized_rows = []
        for row, source_tokens, target_tokens in zip(rows, source_ids, target_ids):
            enriched = dict(row)
            enriched["_source_tokens"] = len(source_tokens)
            enriched["_target_tokens"] = len(target_tokens)
            enriched["_supervised_tokens"] = len(target_tokens)
            enriched["_combined_tokens"] = len(source_tokens) + len(target_tokens)
            tokenized_rows.append(enriched)
        return tokenized_rows

    scheduled_examples = (
        args.max_steps * args.batch_size * args.gradient_accumulation
    )
    tokenized_candidates = {
        arm: tokenize_for_balance(load_jsonl(data_dir / f"train_{arm}.jsonl"))
        for arm in ("A", "B", "C")
    }
    selected_train_rows, token_exposure = select_token_balanced_subsets(
        tokenized_candidates,
        selected_count=scheduled_examples,
        tolerance=args.token_exposure_tolerance,
    )

    results = {
        "classification": "Samanantar cleaning-policy micro-proxy; not 1B/3B",
        "world_size": world_size,
        "visible_cuda_devices": visible_cuda_devices,
        "scheduled_examples_per_arm": scheduled_examples,
        "dataset_id": manifest["dataset_id"],
        "requested_dataset_revision": manifest["requested_dataset_revision"],
        "resolved_dataset_revision": resolved_dataset_revision,
        "model_name": args.model_name,
        "requested_model_revision": args.model_revision,
        "resolved_model_revision": resolved_model_revision,
        "tokenizer_name": getattr(tokenizer, "name_or_path", args.model_name),
        "resolved_tokenizer_revision": resolved_tokenizer_revision,
        "library_versions": {
            name: package_metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "datasets",
                "sentencepiece",
                "sacrebleu",
            )
        },
        "max_steps": args.max_steps,
        "seed": args.seed,
        "language_pair": manifest["language_pair"],
        "token_exposure": token_exposure,
        "arms": {},
    }

    for arm in ("A", "B", "C"):
        set_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_rows = selected_train_rows[arm]
        train_data = Dataset.from_dict(encode_rows(train_rows))
        validation_data = Dataset.from_dict(encode_rows(validation_rows))
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name, revision=resolved_model_revision
        )
        loaded_revision = getattr(model.config, "_commit_hash", None)
        if loaded_revision != resolved_model_revision:
            raise RuntimeError(
                f"Arm {arm} resolved model revision {loaded_revision!r}; "
                f"expected {resolved_model_revision!r}."
            )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        arm_dir = output_dir / f"arm_{arm}"
        evaluation_argument = (
            {"eval_strategy": "no"}
            if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments).parameters
            else {"evaluation_strategy": "no"}
        )
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(arm_dir),
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            warmup_ratio=0.05,
            logging_steps=max(1, args.max_steps // 20),
            save_strategy="no",
            predict_with_generate=True,
            generation_max_length=args.max_target_length,
            fp16=torch.cuda.is_available(),
            optim="adafactor",
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
            **evaluation_argument,
        )
        trainer_tokenizer_argument = (
            {"processing_class": tokenizer}
            if "processing_class" in inspect.signature(Seq2SeqTrainer).parameters
            else {"tokenizer": tokenizer}
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=validation_data,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            **trainer_tokenizer_argument,
        )
        started = time.time()
        trainer.train()
        evaluation = trainer.evaluate()
        prediction = trainer.predict(validation_data)
        predictions = tokenizer.batch_decode(
            prediction.predictions, skip_special_tokens=True
        )
        references = [row["target"] for row in validation_rows]
        sources = [row["source"] for row in validation_rows]
        generated_token_ids = [
            token_id
            for text in predictions
            for token_id in tokenizer(
                text, add_special_tokens=False, truncation=False
            )["input_ids"]
        ]
        metrics = {
            "validation_cross_entropy": float(evaluation["eval_loss"]),
            "chrf": float(sacrebleu.corpus_chrf(predictions, [references]).score),
            "sacrebleu": float(sacrebleu.corpus_bleu(predictions, [references]).score),
            "wrong_script_output_rate": sum(
                script_fraction(text, language) < args.min_script_fraction
                for text in predictions
            )
            / len(predictions),
            "source_copy_rate": sum(
                SequenceMatcher(None, source.casefold(), output.casefold()).ratio() >= 0.90
                for source, output in zip(sources, predictions)
            )
            / len(predictions),
            "empty_output_rate": sum(not text.strip() for text in predictions)
            / len(predictions),
            "repeated_output_rate": sum(repeated_output(text) for text in predictions)
            / len(predictions),
            "unique_target_token_yield": (
                len(set(generated_token_ids)) / len(generated_token_ids)
                if generated_token_ids
                else 0.0
            ),
            "raw_pair_retention_percent": manifest["arm_raw_retention_percent"][arm],
            "train_examples": len(train_rows),
            "source_nonpadding_tokens": token_exposure[arm][
                "source_nonpadding_tokens"
            ],
            "target_nonpadding_tokens": token_exposure[arm][
                "target_nonpadding_tokens"
            ],
            "supervised_tokens": token_exposure[arm]["supervised_tokens"],
            "combined_nonpadding_tokens": token_exposure[arm][
                "combined_nonpadding_tokens"
            ],
            "optimizer_steps": args.max_steps,
            "wall_time_seconds": time.time() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
        }
        results["arms"][arm] = metrics
        (arm_dir / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
        (arm_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    a = results["arms"]["A"]
    c = results["arms"]["C"]
    relative_wrong_script_drop = (
        (a["wrong_script_output_rate"] - c["wrong_script_output_rate"])
        / a["wrong_script_output_rate"]
        if a["wrong_script_output_rate"] > 0
        else None
    )
    relative_copy_drop = (
        (a["source_copy_rate"] - c["source_copy_rate"]) / a["source_copy_rate"]
        if a["source_copy_rate"] > 0
        else None
    )
    loss_change = (
        (c["validation_cross_entropy"] - a["validation_cross_entropy"])
        / a["validation_cross_entropy"]
    )
    results["c_minus_a"] = {
        "chrf_points": c["chrf"] - a["chrf"],
        "wrong_script_relative_drop": relative_wrong_script_drop,
        "source_copy_relative_drop": relative_copy_drop,
        "validation_loss_relative_change": loss_change,
    }
    results["confirmation_checks"] = {
        "chrf_at_least_plus_1": c["chrf"] - a["chrf"] >= 1.0,
        "wrong_script_drop_at_least_25_percent": (
            relative_wrong_script_drop is not None
            and relative_wrong_script_drop >= 0.25
        ),
        "source_copy_drop_at_least_20_percent": (
            relative_copy_drop is not None and relative_copy_drop >= 0.20
        ),
        "loss_not_worse_than_2_percent": loss_change <= 0.02,
        "arm_c_retains_at_least_70_percent": c["raw_pair_retention_percent"] >= 70.0,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--week4-root", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--language", choices=sorted(LANGUAGE_NAMES), default="mr")
    prep.add_argument(
        "--revision",
        required=True,
        help=(
            "Requested immutable Samanantar revision. Mutable branches are rejected; "
            "the resolved commit is recorded."
        ),
    )
    prep.add_argument("--raw-limit", type=int, default=60_000)
    prep.add_argument("--train-per-arm", type=int, default=10_000)
    prep.add_argument("--heldout-size", type=int, default=2_000)
    prep.add_argument("--heldout-fraction", type=float, default=0.10)
    prep.add_argument("--cluster-threshold", type=float, default=0.80)
    prep.add_argument("--num-perm", type=int, default=112)
    prep.add_argument("--min-script-fraction", type=float, default=0.50)
    prep.add_argument("--seed", type=int, default=1729)
    prep.set_defaults(func=prepare)

    run = subparsers.add_parser("train")
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model-name", default="google/mt5-small")
    run.add_argument(
        "--model-revision",
        required=True,
        help="Pinned model/tokenizer revision; the hub must resolve it to an immutable commit.",
    )
    run.add_argument("--max-steps", type=int, default=500)
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--eval-batch-size", type=int, default=4)
    run.add_argument("--gradient-accumulation", type=int, default=16)
    run.add_argument("--learning-rate", type=float, default=3e-4)
    run.add_argument("--max-source-length", type=int, default=192)
    run.add_argument("--max-target-length", type=int, default=192)
    run.add_argument("--min-script-fraction", type=float, default=0.50)
    run.add_argument("--token-exposure-tolerance", type=float, default=0.01)
    run.add_argument("--seed", type=int, default=1729)
    run.set_defaults(func=train)

    self_test = subparsers.add_parser(
        "self-test",
        help="Run deterministic wrong-script and token-exposure behavioral tests.",
    )
    self_test.set_defaults(func=run_self_tests)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
