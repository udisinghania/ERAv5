#!/usr/bin/env python3
"""Validate Week 4 evidence and Session 5 percentage accounting.

This script is dependency-free and read-only with respect to Week 4. It does
not retokenize Parquet data and does not claim to be a model experiment.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys


getcontext().prec = 28
D = Decimal

PRE_MIX = {
    "general": D("53.00"),
    "science_math": D("10.00"),
    "code": D("10.00"),
    "reasoning": D("10.00"),
    "long": D("8.50"),
    "indic": D("8.00"),
    "agentic": D("0.50"),
}

EXPECTED_TOTAL_COLUMNS = {
    "general": D("51.940"),
    "science_math": D("9.800"),
    "code": D("9.800"),
    "reasoning": D("9.800"),
    "long": D("8.330"),
    "indic": D("7.840"),
    "agentic": D("0.490"),
}

STAGES = {
    "seed": {
        "total": D("5.000"),
        "general": D("2.500"),
        "science_math": D("0.750"),
        "code": D("0.750"),
        "reasoning": D("0.325"),
        "long": D("0.250"),
        "indic": D("0.400"),
        "agentic": D("0.025"),
    },
    "general": {
        "total": D("55.000"),
        "general": D("39.000"),
        "science_math": D("5.000"),
        "code": D("5.000"),
        "reasoning": D("0.750"),
        "long": D("0.575"),
        "indic": D("4.400"),
        "agentic": D("0.275"),
    },
    "reasoning": {
        "total": D("20.000"),
        "general": D("6.000"),
        "science_math": D("2.500"),
        "code": D("2.500"),
        "reasoning": D("7.000"),
        "long": D("0.300"),
        "indic": D("1.600"),
        "agentic": D("0.100"),
    },
    "long_context": {
        "total": D("18.000"),
        "general": D("4.440"),
        "science_math": D("1.550"),
        "code": D("1.550"),
        "reasoning": D("1.725"),
        "long": D("7.205"),
        "indic": D("1.440"),
        "agentic": D("0.090"),
    },
}

INDIC_SPLIT = {
    "verified_native": D("40"),
    "unverified_native": D("20"),
    "translated": D("30"),
    "synthetic": D("10"),
}

ANNEAL_SPLIT = {
    "tier_a_agentic": D("20"),
    "b5_reasoning": D("20"),
    "verified_native_indic": D("20"),
    "general_knowledge": D("20"),
    "code": D("10"),
    "long_context": D("10"),
}

INVENTORY_CLASSIFICATION = {
    "indic_verified_native": "Missing local supply",
    "indic_unverified_native": "Missing local supply",
    "indic_translated": "Partially supported by Week 4 AI4Bharat Samanantar",
    "indic_synthetic": "Missing local supply",
    "general_web": "Missing local supply",
    "code": "Missing local supply",
    "science_and_mathematics": "Missing local supply",
    "agentic": "Missing local supply",
    "explicit_reasoning": "Missing local supply",
    "long_context": "Missing local supply",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_json_number(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week4-root", type=Path, default=Path(r"D:\ERAv5\Week_4"))
    parser.add_argument("--week5-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    main_run = args.week4_root / "twm_pilot_500k_v1"
    report_path = main_run / "detailed_pipeline_report.json"
    manifest_path = main_run / "shard.manifest.json"
    split_path = main_run / "pretraining_split" / "split_manifest.json"
    run_state_path = main_run / "run_state.json"
    readme_path = args.week5_root / "README.md"
    results_template_path = args.week5_root / "RESULTS_TEMPLATE.md"
    audit_path = args.week5_root / "WEEK4_AUDIT.md"
    additional_path = args.week5_root / "ADDITIONAL_EXPERIMENTS.md"
    experiment_path = args.week5_root / "samanantar_cleaning_ablation.py"

    required = [
        report_path,
        manifest_path,
        split_path,
        run_state_path,
        readme_path,
        results_template_path,
        audit_path,
        additional_path,
        experiment_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(json.dumps({"status": "failed", "missing": missing}, indent=2))
        return 2

    report = load_json(report_path)
    manifest = load_json(manifest_path)
    split = load_json(split_path)
    state = load_json(run_state_path)
    readme_text = readme_path.read_text(encoding="utf-8")
    results_text = results_template_path.read_text(encoding="utf-8")
    audit_text = audit_path.read_text(encoding="utf-8")
    additional_text = additional_path.read_text(encoding="utf-8")

    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    check("pre_mix_sums_100", sum(PRE_MIX.values()) == D("100"), str(sum(PRE_MIX.values())))
    check("indic_split_sums_100", sum(INDIC_SPLIT.values()) == D("100"), str(sum(INDIC_SPLIT.values())))
    check("anneal_split_sums_100", sum(ANNEAL_SPLIT.values()) == D("100"), str(sum(ANNEAL_SPLIT.values())))
    reasoning_bands = [D("10"), D("20"), D("25"), D("25"), D("15"), D("5")]
    effort_tiers = [D("35"), D("35"), D("20"), D("10")]
    check(
        "reasoning_band_split_sums_100",
        sum(reasoning_bands) == D("100"),
        str(sum(reasoning_bands)),
    )
    check(
        "reasoning_effort_split_sums_100",
        sum(effort_tiers) == D("100"),
        str(sum(effort_tiers)),
    )
    check(
        "lane_budget_defense_sums_100",
        sum(PRE_MIX.values()) == D("100"),
        str(sum(PRE_MIX.values())),
    )
    indic_gap_components = [
        D("31.36"),
        D("15.68"),
        D("22.887186933"),
        D("7.84"),
    ]
    check(
        "indic_gap_decomposition_reconciles",
        sum(indic_gap_components) == D("77.767186933"),
        str(sum(indic_gap_components)),
    )
    supported_lanes = [
        lane
        for lane, status in INVENTORY_CLASSIFICATION.items()
        if status != "Missing local supply"
    ]
    check(
        "samanantar_only_indic_translated",
        supported_lanes == ["indic_translated"],
        supported_lanes,
    )
    benchmark_rows = [
        "MMLU-Pro, ARC-Challenge",
        "HumanEval+, LiveCodeBench",
        "GPQA, MATH",
        "IndicGenBench, IndicXTREME, IN22",
        "BFCL, GAIA, held-out SWE-style tasks",
        "verified code/science tasks",
        "RULER, LongBench v2",
    ]
    check(
        "named_benchmark_mapping_exists",
        all(row in readme_text for row in benchmark_rows),
        [row for row in benchmark_rows if row not in readme_text],
    )
    opus_definition = (
        "`u_i = -g_proxy^T Δθ_i`" in readme_text
        and "`Δθ_i = OptimizerUpdate(g_i, optimizer_state)`" in readme_text
        and "Boltzmann probabilities" in readme_text
    )
    check(
        "opus_optimizer_update_proxy_direction_exists",
        opus_definition,
        "optimizer-induced update projected onto proxy direction plus stochastic sampling",
    )
    check(
        "opus_starvation_o0_o1_o2_exists",
        all(f"| O{index} |" in readme_text for index in range(3)),
        ["O0", "O1", "O2"],
    )
    check(
        "reasoning_order_r0_r1_r2_exists",
        all(f"| R{index} |" in additional_text for index in range(3)),
        ["R0", "R1", "R2"],
    )
    check(
        "tier_a_agentic_anneal_exclusivity_exists",
        "**Tier-A records are physically inaccessible before Anneal**" in readme_text
        and "pre-Anneal floor may use only admitted Tier-B/C primitives or executable synthetic trajectories"
        in readme_text,
        "Tier-A inaccessible before Anneal; pre-Anneal limited to admitted Tier-B/C or executable synthetic",
    )
    result_seed_rows = [
        line for line in results_text.splitlines() if line.startswith("| 3B-")
    ]
    result_seeds = [line.split("|")[3].strip() for line in result_seed_rows]
    one_b_seed_rows = [
        line for line in results_text.splitlines() if line.startswith("| 1B-")
    ]
    one_b_seeds = [line.split("|")[3].strip() for line in one_b_seed_rows]
    selector_seed_rows = [
        line
        for line in results_text.splitlines()
        if line.startswith(("| O0 ", "| O1 ", "| O2 ", "| R0 ", "| R1 ", "| R2 "))
    ]
    selector_seeds = [line.split("|")[2].strip() for line in selector_seed_rows]
    check(
        "readme_results_template_seeds_agree",
        "using **seeds 17 and 29**" in readme_text
        and "**seed 17**" in readme_text
        and "**seed 29**" in readme_text
        and result_seeds == ["17", "29", "17", "29"]
        and one_b_seeds == ["17; 29 if promoted"] * 6
        and selector_seeds == ["17, 29"] * 6,
        {
            "readme": [17, 29],
            "results_3b": result_seeds,
            "results_1b": one_b_seeds,
            "results_selectors": selector_seeds,
        },
    )
    budget_shape_rows = [
        "| General | 53.00% |",
        "| Code | 10.00% |",
        "| Science/Math | 10.00% |",
        "| Reasoning | 10.00% |",
        "| Long-context | 8.50% |",
        "| Indic | **8.00% floor** |",
        "| Agentic | **0.50% floor** |",
    ]
    check(
        "lane_budget_training_shape_defense_exists",
        all(row in readme_text for row in budget_shape_rows)
        and "These numbers are starting hypotheses" in readme_text,
        [row for row in budget_shape_rows if row not in readme_text],
    )
    check(
        "hierarchical_quota_language_source_caps_exists",
        "A quota ledger tracks selected tokens and debt" in readme_text
        and "Indic is balanced by tier/language/source" in readme_text
        and "Agentic by tier/tool family/task subtype" in readme_text
        and "One source family is capped at 20%" in readme_text,
        "hierarchical subqueues, source cap, and bounded quota debt",
    )
    check(
        "opus_selector_scaling_gate_exists",
        "**K=4 candidate microbatches per committed update**" in readme_text
        and "**>=0.90 utility-rank correlation**" in readme_text
        and "no more than **0.5 pp**" in readme_text,
        "bounded exact selector plus 240B surrogate promotion gate",
    )
    check(
        "nonoverlapping_primary_lane_ledger_exists",
        "Global parent/near-duplicate clustering occurs before accounting" in readme_text
        and "cross-tags never offset two demands" in readme_text,
        "global cluster assignment plus deterministic primary-lane precedence",
    )
    check(
        "token_origin_deny_mask_exists",
        "Token-origin labels are stored before formatting" in readme_text
        and "deny-mask" in readme_text
        and "overrides serialized role" in readme_text,
        "origin mask overrides role after packing/truncation",
    )
    check(
        "no_forbidden_all_token_loss_path",
        "All-token loss" not in readme_text
        and "All-token loss" not in results_text
        and "Both arms retain the forbidden-origin deny-mask" in additional_text,
        "all formal arms preserve the forbidden-origin deny-mask",
    )
    check(
        "curriculum_transition_stability_gates_exist",
        "Linear boundary blends preserve integrated totals" in readme_text
        and "gradient-norm p99 below **2x**" in readme_text,
        "boundary blend, rate cap, and context stability gates",
    )
    check(
        "reasoning_order_same_candidates_only_order_changes",
        "reuses the same reasoning candidate IDs" in additional_text
        and "only reasoning-tier order changes" in additional_text,
        "same candidate IDs, counts, and non-reasoning order",
    )
    check(
        "three_b_locks_one_b_winner",
        "| 3B-B locked winner |" in readme_text
        and "primary 3B-B treatment is the O-series winner" in readme_text
        and "corresponding optional 1B experiments were actually executed and passed"
        in readme_text
        and "unexecuted secondary treatment remains fixed to the declared base recipe"
        in readme_text
        and "| 3B-B locked O-series winner |" in results_text
        and "must not be recorded as a 1B winner" in results_text,
        "O-series winner is primary; optional treatments require an executed, passed 1B gate",
    )
    threshold_tokens = [
        "99.0%",
        "0.05 pp",
        "0.01 pp",
        "3% relative",
        "+2.0 pp",
        "+1.5 pp",
        "+3.0 pp",
        "+1.0 pp",
        "+0.5 pp",
        "1% relative",
    ]
    threshold_document = readme_text + "\n" + additional_text
    check(
        "readme_results_thresholds_agree",
        all(
            token in threshold_document and token in results_text
            for token in threshold_tokens
        ),
        [
            token
            for token in threshold_tokens
            if token not in threshold_document or token not in results_text
        ],
    )
    placeholder = "".join(("T", "B", "D"))
    markdown_texts = {
        "README.md": readme_text,
        "RESULTS_TEMPLATE.md": results_text,
        "WEEK4_AUDIT.md": audit_text,
        "ADDITIONAL_EXPERIMENTS.md": additional_text,
    }
    check(
        "no_ambiguous_placeholder_remains",
        all(placeholder not in content for content in markdown_texts.values()),
        [name for name, content in markdown_texts.items() if placeholder in content],
    )
    missing_supply_rows = [
        "| General | 519.4B | 0 |",
        "| Science/Math | 98.0B | 0 |",
        "| Code | 98.0B | 0 |",
        "| Reasoning | 98.0B | 0 |",
        "| Long-context | 83.3B | 0 |",
        "| Agentic | 4.9B | 0 |",
        "| Anneal | 20.0B | 0 |",
    ]
    check(
        "missing_lanes_state_zero_locally_admitted_supply",
        all(row in readme_text for row in missing_supply_rows),
        [row for row in missing_supply_rows if row not in readme_text],
    )
    indic_supply_row = "| Indic | 78.4B | about 0.210937689B translated; exact count pending |"
    check(
        "pending_retokenization_distinct_from_absent_supply",
        indic_supply_row in readme_text
        and "| Indic | 78.4B | 0 |" not in readme_text
        and "Pending frozen-V5 retokenization" in readme_text,
        indic_supply_row,
    )
    check(
        "unexecuted_experiments_have_explicit_state",
        readme_text.count("**Planned—not executed**") >= 2
        and results_text.count("**Planned—not executed**") >= 3
        and "Not available—experiment not executed" in results_text
        and "Pending execution" in results_text
        and "Assigned at execution" in results_text,
        {
            "readme_status_count": readme_text.count("**Planned—not executed**"),
            "results_status_count": results_text.count("**Planned—not executed**"),
        },
    )
    blank_result_headers = ["| Observed |", "| 95% CI |", "| Run ID |", "| Decision |"]
    check(
        "readme_has_no_blank_observed_result_matrix",
        all(header not in readme_text for header in blank_result_headers),
        [header for header in blank_result_headers if header in readme_text],
    )
    zero_result_cells = ["| 0 |", "| 0.0 |", "| 0.00 |"]
    check(
        "results_template_does_not_encode_unexecuted_results_as_zero",
        all(cell not in results_text for cell in zero_result_cells),
        [cell for cell in zero_result_cells if cell in results_text],
    )
    evidence_states = [
        "**Measured**",
        "**Derived**",
        "**Target**",
        "**Missing local supply**",
        "**Pending frozen-V5 retokenization**",
        "**Planned—not executed**",
    ]
    check(
        "evidence_state_vocabulary_is_explicit",
        all(state in readme_text for state in evidence_states),
        [state for state in evidence_states if state not in readme_text],
    )
    design_marker = "### Design decisions and trade-offs"
    indic_marker = "### Indic tier split"
    design_section = (
        readme_text.split(design_marker, 1)[1].split(indic_marker, 1)[0]
        if design_marker in readme_text and indic_marker in readme_text
        else ""
    )
    design_words = len(design_section.split())
    check(
        "design_tradeoff_section_present_and_concise",
        250 <= design_words <= 400
        and all(
            token in design_section
            for token in (
                "53%",
                "10%",
                "8.5%",
                "8% protected floor",
                "0.5%",
                "2%",
                "40/20/30/10",
                "3x translated replay cap",
            )
        ),
        {"words": design_words},
    )
    check(
        "current_data_gate_is_explicit",
        "### Current data-gate status" in readme_text
        and "data gate is **not currently met**" in readme_text
        and "not authorization" in readme_text
        and "Production stays blocked" in readme_text,
        "blocked gate, local translated-only evidence, and no invented threshold",
    )
    check(
        "one_primary_1b_and_one_primary_3b_are_named",
        "### Primary 1B proxy: OPUS starvation" in readme_text
        and "### Primary 3B confirmation" in readme_text
        and all(f"| O{index} |" in readme_text for index in range(3)),
        "primary O0/O1/O2 hypothesis plus locked-winner 3B confirmation",
    )
    check(
        "secondary_experiments_preserved_in_supplement",
        all(f"| R{index} |" in additional_text for index in range(3))
        and all(f"| 1B-{letter} " in additional_text for letter in "ABCDEF")
        and "ADDITIONAL_EXPERIMENTS.md" in readme_text,
        "R0/R1/R2 and 1B-A through 1B-F",
    )
    check(
        "next_executable_cleaning_gate_exists",
        "### Next executable cleaning gate" in readme_text
        and "Samanantar frozen-tokenizer inventory gate" in readme_text
        and "`samanantar_frozen_token_inventory.json`" in readme_text
        and "all 11 language directions" in readme_text
        and "immutable Samanantar revision" in readme_text,
        "immutable dataset/tokenizer revisions and exact per-language token inventory",
    )
    research_marker = "## Research basis and alternatives considered"
    promotion_marker = "## Promotion gates"
    if research_marker in readme_text and promotion_marker in readme_text:
        research_section = readme_text.split(research_marker, 1)[1].split(
            promotion_marker, 1
        )[0]
    else:
        research_section = ""
    research_words = len(research_section.split())
    check(
        "research_basis_section_present_and_concise",
        bool(research_section)
        and 350 <= research_words <= 500,
        {"words": research_words},
    )
    core_paper_ids = [
        "2602.05400v2",
        "2508.17677v1",
        "2603.08022v2",
        "2601.21698v2",
        "2506.11300v2",
        "2604.20549v1",
        "2510.25947v1",
    ]
    check(
        "required_core_research_ids_and_curriculum_boundary",
        all(paper_id in research_section for paper_id in core_paper_ids),
        [paper_id for paper_id in core_paper_ids if paper_id not in research_section],
    )
    limitation_fragments = [
        "It does not establish that the proposed 8% Indic floor, 0.5% Agentic floor, 2% Anneal reserve, Indic tier split, 3x replay cap, or exact stage shares are optimal",
        "Citation does not convert a candidate dataset into admitted inventory",
        "published results do not replace local supply accounting",
        "benchmark-targeted selection requires strict contamination quarantine",
        "no citation is evidence that an unexecuted V5 proxy passed",
    ]
    check(
        "required_scientific_limitation_statement",
        all(fragment in research_section for fragment in limitation_fragments),
        [fragment for fragment in limitation_fragments if fragment not in research_section],
    )
    excluded_adjacent_ids = [
        "2508.11953",
        "2507.17702",
        "2508.09874",
    ]
    check(
        "adjacent_papers_excluded",
        all(paper_id not in research_section for paper_id in excluded_adjacent_ids),
        [paper_id for paper_id in excluded_adjacent_ids if paper_id in research_section],
    )
    readme_words = len(readme_text.split())
    check(
        "main_readme_within_concision_target",
        3000 <= readme_words <= 4000,
        {"words": readme_words},
    )

    self_test_run = subprocess.run(
        [sys.executable, str(experiment_path), "self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        self_test_payload = json.loads(self_test_run.stdout)
    except json.JSONDecodeError:
        self_test_payload = {}
    check(
        "microproxy_behavioral_self_tests_pass",
        self_test_run.returncode == 0
        and self_test_payload.get("status") == "passed"
        and all(
            self_test_payload.get("wrong_script_behavior", {}).values()
        )
        and all(self_test_payload.get("revision_handling", {}).values())
        and all(self_test_payload.get("execution_mode", {}).values())
        and self_test_payload.get("token_exposure_balance", {}).get("passed")
        is True,
        self_test_payload or self_test_run.stderr,
    )
    mutable_dataset_revision_run = subprocess.run(
        [
            sys.executable,
            str(experiment_path),
            "prepare",
            "--week4-root",
            "unused",
            "--output-dir",
            "unused",
            "--revision",
            "main",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "microproxy_rejects_mutable_dataset_revision",
        mutable_dataset_revision_run.returncode != 0
        and "mutable branch" in mutable_dataset_revision_run.stderr,
        {
            "return_code": mutable_dataset_revision_run.returncode,
            "stderr": mutable_dataset_revision_run.stderr[-300:],
        },
    )
    missing_revision_run = subprocess.run(
        [
            sys.executable,
            str(experiment_path),
            "train",
            "--data-dir",
            "unused",
            "--output-dir",
            "unused",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "microproxy_model_revision_is_required",
        missing_revision_run.returncode == 2
        and "--model-revision" in missing_revision_run.stderr,
        {
            "return_code": missing_revision_run.returncode,
            "stderr": missing_revision_run.stderr[-500:],
        },
    )
    experiment_text = experiment_path.read_text(encoding="utf-8")
    check(
        "microproxy_records_pinned_revisions_versions_and_tokens",
        all(
            token in experiment_text
            for token in (
                "requested_model_revision",
                "resolved_model_revision",
                "resolved_tokenizer_revision",
                "requested_dataset_revision",
                "resolved_dataset_revision",
                "library_versions",
                "world_size",
                "scheduled_examples_per_arm",
                "source_nonpadding_tokens",
                "target_nonpadding_tokens",
                "supervised_tokens",
                "combined_nonpadding_tokens",
            )
        ),
        "pinned dataset/model/tokenizer provenance, single-process schedule, and per-arm token exposure",
    )
    check(
        "microproxy_loads_from_resolved_commits",
        "revision=resolved_dataset_revision" in experiment_text
        and "revision=resolved_model_revision" in experiment_text
        and "immutable_hub_commit(resolved_dataset_revision)" in experiment_text
        and "immutable_hub_commit(resolved_model_revision)" in experiment_text,
        "dataset stream, tokenizer, and model loads are anchored to resolved commits",
    )

    stage_total = sum(stage["total"] for stage in STAGES.values())
    check("stage_totals_sum_98", stage_total == D("98"), str(stage_total))

    for stage_name, stage in STAGES.items():
        components = sum(value for key, value in stage.items() if key != "total")
        check(
            f"{stage_name}_row_reconciles",
            components == stage["total"],
            {"components": str(components), "declared": str(stage["total"])},
        )
        check(
            f"{stage_name}_indic_floor",
            stage["indic"] / stage["total"] == D("0.08"),
            str(stage["indic"] / stage["total"]),
        )
        check(
            f"{stage_name}_agentic_floor",
            stage["agentic"] / stage["total"] == D("0.005"),
            str(stage["agentic"] / stage["total"]),
        )

    stage_columns = {
        lane: sum(stage[lane] for stage in STAGES.values())
        for lane in EXPECTED_TOTAL_COLUMNS
    }
    for lane, expected in EXPECTED_TOTAL_COLUMNS.items():
        check(
            f"stage_column_{lane}",
            stage_columns[lane] == expected,
            {"actual": str(stage_columns[lane]), "expected": str(expected)},
        )
    check(
        "total_budget_reconciles",
        sum(stage_columns.values()) + D("2.000") == D("100.000"),
        str(sum(stage_columns.values()) + D("2.000")),
    )

    language_rows = report["stage_a_spokes"]
    raw = sum(int(row["total_raw"]) for row in language_rows.values())
    stage_a = sum(int(row["surviving_spokes"]) for row in language_rows.values())
    alignment = sum(int(row["dropped_alignment"]) for row in language_rows.values())
    lid = sum(int(row["dropped_lid"]) for row in language_rows.values())
    quality = sum(int(row["dropped_quality"]) for row in language_rows.values())
    decontam = sum(int(row["dropped_decontam"]) for row in language_rows.values())
    pii = sum(int(row["pii_redactions"]) for row in language_rows.values())
    stage_b_translations = sum(
        int(value)
        for value in report["stage_b_star_graph"]["retained_translations_by_language"].values()
    )
    hubs = int(report["stage_b_star_graph"]["total_english_hubs"])
    dedup = report["stage_c_deduplication"]
    final_hubs = int(dedup["surviving_records"])
    estimated_tokens = int(manifest["estimated_token_count"])

    check(
        "stage_a_drop_identity",
        raw - stage_a == alignment + lid + quality + decontam,
        {
            "raw_minus_survivors": raw - stage_a,
            "sum_drop_reasons": alignment + lid + quality + decontam,
        },
    )
    check("stage_c_input_matches_stage_b_hubs", int(dedup["input_records"]) == hubs, hubs)
    check(
        "stage_c_survivor_identity",
        int(dedup["input_records"]) - int(dedup["dropped_by_lsh"]) == final_hubs,
        final_hubs,
    )
    check("report_manifest_match", int(report["stage_d_manifest"]["record_count"]) == final_hubs, final_hubs)
    check("external_manifest_match", int(manifest["record_count"]) == final_hubs, final_hubs)
    check(
        "split_rows_match_manifest",
        int(split["train_rows"]) + int(split["validation_rows"]) == final_hubs,
        int(split["train_rows"]) + int(split["validation_rows"]),
    )
    check(
        "run_complete",
        bool(state["stage_c_completed"])
        and bool(state["stage_d_completed"])
        and len(state["stage_a_completed_languages"]) == 11
        and len(state["stage_b_completed_buckets"]) == 256,
        {
            "stage_a_languages": len(state["stage_a_completed_languages"]),
            "stage_b_buckets": len(state["stage_b_completed_buckets"]),
            "stage_c": state["stage_c_completed"],
            "stage_d": state["stage_d_completed"],
        },
    )

    train_fraction = D(str(split["train_fraction_actual"]))
    train_token_estimate = (D(estimated_tokens) * train_fraction).quantize(D("1"))
    translated_demand_per_1t = D("23520000000")
    indic_demand_per_1t = D("78400000000")
    translated_coverage = D(100) * train_token_estimate / translated_demand_per_1t
    indic_coverage = D(100) * train_token_estimate / indic_demand_per_1t

    result = {
        "classification": "evidence-and-accounting systems validation; not a model proxy",
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "week4_root": str(args.week4_root),
        "authoritative_run": str(main_run),
        "artifact_sha256": {
            "report": sha256_file(report_path),
            "manifest": sha256_file(manifest_path),
            "split_manifest": sha256_file(split_path),
            "run_state": sha256_file(run_state_path),
            "readme": sha256_file(readme_path),
            "results_template": sha256_file(results_template_path),
            "additional_experiments": sha256_file(additional_path),
            "week4_audit": sha256_file(audit_path),
            "microproxy_script": sha256_file(experiment_path),
        },
        "inventory_classification": INVENTORY_CLASSIFICATION,
        "measured": {
            "raw_translation_pairs": raw,
            "stage_a_surviving_pairs": stage_a,
            "stage_b_retained_translations": stage_b_translations,
            "pre_lsh_hubs": hubs,
            "final_hubs": final_hubs,
            "alignment_drops": alignment,
            "lid_drops": lid,
            "quality_drops": quality,
            "decontamination_drops": decontam,
            "pii_redactions": pii,
            "lsh_drops": int(dedup["dropped_by_lsh"]),
            "candidate_comparisons": int(dedup["candidate_comparisons"]),
            "candidate_limit_hits": int(dedup["candidate_limit_hits"]),
            "clean_token_estimator_output": estimated_tokens,
            "train_rows": int(split["train_rows"]),
            "validation_rows": int(split["validation_rows"]),
        },
        "derived": {
            "stage_a_retention_percent": as_json_number(D(100) * D(stage_a) / D(raw)),
            "stage_b_translations_over_raw_percent": as_json_number(
                D(100) * D(stage_b_translations) / D(raw)
            ),
            "lsh_drop_percent": as_json_number(
                D(100) * D(int(dedup["dropped_by_lsh"])) / D(hubs)
            ),
            "estimated_train_tokens_row_fraction_assumption": int(train_token_estimate),
            "translated_coverage_per_1t_percent": as_json_number(translated_coverage),
            "total_indic_coverage_per_1t_percent": as_json_number(indic_coverage),
        },
        "evidence_status": {
            "raw_word_count": "Not recorded",
            "raw_token_count": "Not recorded",
            "frozen_v5_clean_tokens": "Pending frozen-V5 retokenization",
            "unique_corpus_token_types": "Not measured",
            "verified_native_indic_supply": "Missing local supply",
            "unverified_native_indic_supply": "Missing local supply",
            "synthetic_indic_supply": "Missing local supply",
            "agentic_supply": "Missing local supply",
            "reasoning_supply": "Missing local supply",
            "long_context_supply": "Missing local supply",
        },
        "checks": checks,
    }

    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
