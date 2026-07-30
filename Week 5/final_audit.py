#!/usr/bin/env python3
"""Dependency-free final audit for the ERA V5 Session 5 deliverables."""

from __future__ import annotations

import argparse
import ast
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


D = Decimal


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add(checks: list[dict], name: str, passed: bool, detail: object) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week5-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--week4-root", type=Path, default=Path(r"D:\ERAv5\Week_4"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.week5_root.resolve()
    readme = root / "README.md"
    validator = root / "validate_plan.py"
    experiment = root / "samanantar_cleaning_ablation.py"
    results_template = root / "RESULTS_TEMPLATE.md"
    audit_file = root / "WEEK4_AUDIT.md"
    additional_file = root / "ADDITIONAL_EXPERIMENTS.md"
    required_files = [
        readme,
        validator,
        experiment,
        results_template,
        audit_file,
        additional_file,
    ]
    checks: list[dict] = []

    add(
        checks,
        "required_local_files_exist",
        all(path.is_file() for path in required_files),
        [str(path) for path in required_files if not path.is_file()],
    )
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    results_text = (
        results_template.read_text(encoding="utf-8")
        if results_template.is_file()
        else ""
    )
    audit_text = audit_file.read_text(encoding="utf-8") if audit_file.is_file() else ""
    additional_text = (
        additional_file.read_text(encoding="utf-8")
        if additional_file.is_file()
        else ""
    )

    pre_mix = [D("53"), D("10"), D("10"), D("10"), D("8.5"), D("8"), D("0.5")]
    total_mix = [D("51.94"), D("9.8"), D("9.8"), D("9.8"), D("8.33"), D("7.84"), D("0.49"), D("2")]
    indic = [D("40"), D("20"), D("30"), D("10")]
    anneal = [D("20"), D("20"), D("20"), D("20"), D("10"), D("10")]
    add(checks, "pre_anneal_mix_sums_100", sum(pre_mix) == D("100"), str(sum(pre_mix)))
    add(checks, "total_mix_sums_100", sum(total_mix) == D("100"), str(sum(total_mix)))
    add(checks, "indic_tiers_sum_100", sum(indic) == D("100"), str(sum(indic)))
    add(checks, "anneal_components_sum_100", sum(anneal) == D("100"), str(sum(anneal)))
    reasoning_bands = [D("10"), D("20"), D("25"), D("25"), D("15"), D("5")]
    effort_tiers = [D("35"), D("35"), D("20"), D("10")]
    add(
        checks,
        "reasoning_band_split_sums_100",
        sum(reasoning_bands) == D("100"),
        str(sum(reasoning_bands)),
    )
    add(
        checks,
        "reasoning_effort_split_sums_100",
        sum(effort_tiers) == D("100"),
        str(sum(effort_tiers)),
    )
    add(
        checks,
        "lane_budget_defense_sums_100",
        sum(pre_mix) == D("100"),
        str(sum(pre_mix)),
    )
    indic_gap_components = [D("31.36"), D("15.68"), D("22.887186933"), D("7.84")]
    add(
        checks,
        "indic_gap_decomposition_reconciles",
        sum(indic_gap_components) == D("77.767186933"),
        str(sum(indic_gap_components)),
    )

    stage_rows = {
        "Seed": [D("2.5"), D(".75"), D(".75"), D(".325"), D(".25"), D(".4"), D(".025")],
        "General": [D("39"), D("5"), D("5"), D(".75"), D(".575"), D("4.4"), D(".275")],
        "Reasoning": [D("6"), D("2.5"), D("2.5"), D("7"), D(".3"), D("1.6"), D(".1")],
        "Long-context": [D("4.44"), D("1.55"), D("1.55"), D("1.725"), D("7.205"), D("1.44"), D(".09")],
    }
    declared = {"Seed": D("5"), "General": D("55"), "Reasoning": D("20"), "Long-context": D("18")}
    for stage, values in stage_rows.items():
        add(checks, f"stage_{stage}_reconciles", sum(values) == declared[stage], str(sum(values)))
    columns = [sum(row[index] for row in stage_rows.values()) for index in range(7)]
    expected_columns = [D("51.94"), D("9.8"), D("9.8"), D("9.8"), D("8.33"), D("7.84"), D(".49")]
    add(checks, "phase_weighted_columns_reconcile", columns == expected_columns, [str(value) for value in columns])
    add(checks, "five_stage_total_sums_100", sum(declared.values()) + D("2") == D("100"), str(sum(declared.values()) + D("2")))

    invariant = D("38.5")
    one_b = {
        "A": [D("57.25"), D("4"), D(".25")],
        "B": [D("53.25"), D("8"), D(".25")],
        "C": [D("57"), D("4"), D(".5")],
        "D": [D("53"), D("8"), D(".5")],
        "E": [D("53"), D("8"), D(".5")],
        "F": [D("53"), D("8"), D(".5")],
    }
    for arm, varied in one_b.items():
        add(checks, f"1b_arm_{arm}_sums_100", sum(varied) + invariant == D("100"), str(sum(varied) + invariant))
    control_3b = [D("57.25"), D("10"), D("10"), D("10"), D("8.5"), D("4"), D(".25")]
    add(checks, "3b_control_sums_100", sum(control_3b) == D("100"), str(sum(control_3b)))

    required_phrases = {
        "samanantar_translated_only": "Samanantar is counted only as **Indic translated**",
        "protected_indic_queue": "Protected Indic and Agentic queues",
        "protected_agentic_queue": "Protected Indic and Agentic queues",
        "lane_local_selection": "normalized within a lane/subqueue",
        "quota_tracking": "quota ledger",
        "starvation_condition": "A lane is starved",
        "translation_gap": "translated acquisition",
        "synthetic_generation_gap": "synthetic generation",
        "agentic_environment_mask": "environment/execution state",
        "agentic_verifier_mask": "verifier, reward, grader",
        "formal_results_not_claimed": "No model result or promotion decision is claimed",
        "micro_proxy_not_formal_proxy": "not a 1B/3B mixture proxy",
        "seed_17": "**seed 17**",
        "seed_29": "**seed 29**",
    }
    for name, phrase in required_phrases.items():
        add(checks, name, phrase in text, phrase)
    benchmark_rows = [
        "MMLU-Pro, ARC-Challenge",
        "HumanEval+, LiveCodeBench",
        "GPQA, MATH",
        "IndicGenBench, IndicXTREME, IN22",
        "BFCL, GAIA, held-out SWE-style tasks",
        "verified code/science tasks",
        "RULER, LongBench v2",
    ]
    add(
        checks,
        "named_benchmark_mapping_exists",
        all(row in text for row in benchmark_rows),
        [row for row in benchmark_rows if row not in text],
    )
    add(
        checks,
        "opus_optimizer_update_proxy_direction_exists",
        "`u_i = -g_proxy^T Δθ_i`" in text
        and "`Δθ_i = OptimizerUpdate(g_i, optimizer_state)`" in text
        and "Boltzmann probabilities" in text,
        "optimizer-induced update projected onto proxy direction plus stochastic sampling",
    )
    add(
        checks,
        "opus_starvation_o0_o1_o2_exists",
        all(f"| O{index} |" in text for index in range(3)),
        ["O0", "O1", "O2"],
    )
    add(
        checks,
        "reasoning_order_r0_r1_r2_exists",
        all(f"| R{index} |" in additional_text for index in range(3)),
        ["R0", "R1", "R2"],
    )
    add(
        checks,
        "tier_a_agentic_anneal_exclusivity_exists",
        "**Tier-A records are physically inaccessible before Anneal**" in text
        and "pre-Anneal floor may use only admitted Tier-B/C primitives or executable synthetic trajectories"
        in text,
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
    add(
        checks,
        "readme_results_template_seeds_agree",
        "using **seeds 17 and 29**" in text
        and "**seed 17**" in text
        and "**seed 29**" in text
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
    add(
        checks,
        "lane_budget_training_shape_defense_exists",
        all(row in text for row in budget_shape_rows)
        and "These numbers are starting hypotheses" in text,
        [row for row in budget_shape_rows if row not in text],
    )
    add(
        checks,
        "hierarchical_quota_language_source_caps_exists",
        "A quota ledger tracks selected tokens and debt" in text
        and "Indic is balanced by tier/language/source" in text
        and "Agentic by tier/tool family/task subtype" in text
        and "One source family is capped at 20%" in text,
        "hierarchical subqueues, source cap, and bounded quota debt",
    )
    add(
        checks,
        "opus_selector_scaling_gate_exists",
        "**K=4 candidate microbatches per committed update**" in text
        and "**>=0.90 utility-rank correlation**" in text
        and "no more than **0.5 pp**" in text,
        "bounded exact selector plus 240B surrogate promotion gate",
    )
    add(
        checks,
        "nonoverlapping_primary_lane_ledger_exists",
        "Global parent/near-duplicate clustering occurs before accounting" in text
        and "cross-tags never offset two demands" in text,
        "global cluster assignment plus deterministic primary-lane precedence",
    )
    add(
        checks,
        "token_origin_deny_mask_exists",
        "Token-origin labels are stored before formatting" in text
        and "deny-mask" in text
        and "overrides serialized role" in text,
        "origin mask overrides role after packing/truncation",
    )
    add(
        checks,
        "no_forbidden_all_token_loss_path",
        "All-token loss" not in text
        and "All-token loss" not in results_text
        and "Both arms retain the forbidden-origin deny-mask" in additional_text,
        "all formal arms preserve the forbidden-origin deny-mask",
    )
    add(
        checks,
        "curriculum_transition_stability_gates_exist",
        "Linear boundary blends preserve integrated totals" in text
        and "gradient-norm p99 below **2x**" in text,
        "boundary blend, rate cap, and context stability gates",
    )
    add(
        checks,
        "reasoning_order_same_candidates_only_order_changes",
        "reuses the same reasoning candidate IDs" in additional_text
        and "only reasoning-tier order changes" in additional_text,
        "same candidate IDs, counts, and non-reasoning order",
    )
    add(
        checks,
        "three_b_locks_one_b_winner",
        "| 3B-B locked winner |" in text
        and "primary 3B-B treatment is the O-series winner" in text
        and "corresponding optional 1B experiments were actually executed and passed"
        in text
        and "unexecuted secondary treatment remains fixed to the declared base recipe"
        in text
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
    threshold_document = text + "\n" + additional_text
    add(
        checks,
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
        "README.md": text,
        "RESULTS_TEMPLATE.md": results_text,
        "WEEK4_AUDIT.md": audit_text,
        "ADDITIONAL_EXPERIMENTS.md": additional_text,
    }
    add(
        checks,
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
    add(
        checks,
        "missing_lanes_state_zero_locally_admitted_supply",
        all(row in text for row in missing_supply_rows),
        [row for row in missing_supply_rows if row not in text],
    )
    indic_supply_row = "| Indic | 78.4B | about 0.210937689B translated; exact count pending |"
    add(
        checks,
        "pending_retokenization_distinct_from_absent_supply",
        indic_supply_row in text
        and "| Indic | 78.4B | 0 |" not in text
        and "Pending frozen-V5 retokenization" in text,
        indic_supply_row,
    )
    add(
        checks,
        "unexecuted_experiments_have_explicit_state",
        text.count("**Planned—not executed**") >= 2
        and results_text.count("**Planned—not executed**") >= 3
        and "Not available—experiment not executed" in results_text
        and "Pending execution" in results_text
        and "Assigned at execution" in results_text,
        {
            "readme_status_count": text.count("**Planned—not executed**"),
            "results_status_count": results_text.count("**Planned—not executed**"),
        },
    )
    blank_result_headers = ["| Observed |", "| 95% CI |", "| Run ID |", "| Decision |"]
    add(
        checks,
        "readme_has_no_blank_observed_result_matrix",
        all(header not in text for header in blank_result_headers),
        [header for header in blank_result_headers if header in text],
    )
    zero_result_cells = ["| 0 |", "| 0.0 |", "| 0.00 |"]
    add(
        checks,
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
    add(
        checks,
        "evidence_state_vocabulary_is_explicit",
        all(state in text for state in evidence_states),
        [state for state in evidence_states if state not in text],
    )
    design_marker = "### Design decisions and trade-offs"
    indic_marker = "### Indic tier split"
    design_section = (
        text.split(design_marker, 1)[1].split(indic_marker, 1)[0]
        if design_marker in text and indic_marker in text
        else ""
    )
    design_words = len(design_section.split())
    add(
        checks,
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
    add(
        checks,
        "current_data_gate_is_explicit",
        "### Current data-gate status" in text
        and "data gate is **not currently met**" in text
        and "not authorization" in text
        and "Production stays blocked" in text,
        "blocked gate, local translated-only evidence, and no invented threshold",
    )
    add(
        checks,
        "one_primary_1b_and_one_primary_3b_are_named",
        "### Primary 1B proxy: OPUS starvation" in text
        and "### Primary 3B confirmation" in text
        and all(f"| O{index} |" in text for index in range(3)),
        "primary O0/O1/O2 hypothesis plus locked-winner 3B confirmation",
    )
    add(
        checks,
        "secondary_experiments_preserved_in_supplement",
        all(f"| R{index} |" in additional_text for index in range(3))
        and all(f"| 1B-{letter} " in additional_text for letter in "ABCDEF")
        and "ADDITIONAL_EXPERIMENTS.md" in text,
        "R0/R1/R2 and 1B-A through 1B-F",
    )
    add(
        checks,
        "next_executable_cleaning_gate_exists",
        "### Next executable cleaning gate" in text
        and "Samanantar frozen-tokenizer inventory gate" in text
        and "`samanantar_frozen_token_inventory.json`" in text
        and "all 11 language directions" in text
        and "immutable Samanantar revision" in text,
        "immutable dataset/tokenizer revisions and exact per-language token inventory",
    )
    research_marker = "## Research basis and alternatives considered"
    promotion_marker = "## Promotion gates"
    if research_marker in text and promotion_marker in text:
        research_section = text.split(research_marker, 1)[1].split(
            promotion_marker, 1
        )[0]
    else:
        research_section = ""
    research_words = len(research_section.split())
    add(
        checks,
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
    add(
        checks,
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
    add(
        checks,
        "required_scientific_limitation_statement",
        all(fragment in research_section for fragment in limitation_fragments),
        [fragment for fragment in limitation_fragments if fragment not in research_section],
    )
    excluded_adjacent_ids = [
        "2508.11953",
        "2507.17702",
        "2508.09874",
    ]
    add(
        checks,
        "adjacent_papers_excluded",
        all(paper_id not in research_section for paper_id in excluded_adjacent_ids),
        [paper_id for paper_id in excluded_adjacent_ids if paper_id in research_section],
    )
    readme_words = len(text.split())
    add(
        checks,
        "main_readme_within_concision_target",
        3000 <= readme_words <= 4000,
        {"words": readme_words},
    )
    for band in ("B0", "B1", "B2", "B3", "B4", "B5"):
        add(checks, f"difficulty_{band}_has_row", f"| {band} |" in text, band)
    for tier in ("Short", "Medium", "Long", "Ultra"):
        add(checks, f"reasoning_{tier}_has_row", f"| {tier} |" in text, tier)

    for script in (validator, experiment, Path(__file__).resolve()):
        try:
            ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
            add(checks, f"syntax_{script.name}", True, "AST parsed")
        except (OSError, SyntaxError) as exc:
            add(checks, f"syntax_{script.name}", False, str(exc))

    run_dir = args.week4_root / "twm_pilot_500k_v1"
    evidence_paths = [
        run_dir,
        run_dir / "detailed_pipeline_report.json",
        run_dir / "shard.manifest.json",
        run_dir / "pretraining_split" / "split_manifest.json",
        run_dir / "run_state.json",
    ]
    add(
        checks,
        "week4_evidence_paths_exist",
        all(path.exists() for path in evidence_paths),
        [str(path) for path in evidence_paths if not path.exists()],
    )

    smoke: dict[str, object] = {}
    if validator.is_file():
        with tempfile.TemporaryDirectory() as temp_dir:
            out1 = Path(temp_dir) / "one.json"
            out2 = Path(temp_dir) / "two.json"
            commands = [
                [
                    sys.executable,
                    str(validator),
                    "--week4-root",
                    str(args.week4_root),
                    "--week5-root",
                    str(root),
                    "--output",
                    str(out1),
                ],
                [
                    sys.executable,
                    str(validator),
                    "--week4-root",
                    str(args.week4_root),
                    "--week5-root",
                    str(root),
                    "--output",
                    str(out2),
                ],
            ]
            runs = [subprocess.run(command, capture_output=True, text=True, check=False) for command in commands]
            reproducible = (
                all(run.returncode == 0 for run in runs)
                and out1.is_file()
                and out2.is_file()
                and json.loads(out1.read_text(encoding="utf-8")) == json.loads(out2.read_text(encoding="utf-8"))
            )
            smoke["validator_return_codes"] = [run.returncode for run in runs]
            add(checks, "validator_reproducible_two_runs", reproducible, smoke)
            if out1.is_file():
                evidence = json.loads(out1.read_text(encoding="utf-8"))
                expected = {
                    "raw_translation_pairs": 5_141_227,
                    "stage_a_surviving_pairs": 4_763_996,
                    "stage_b_retained_translations": 4_020_129,
                    "pre_lsh_hubs": 2_977_126,
                    "final_hubs": 2_972_462,
                    "clean_token_estimator_output": 215_294_331,
                    "train_rows": 2_912_312,
                    "validation_rows": 60_150,
                }
                observed = evidence.get("measured", {})
                add(
                    checks,
                    "stated_samanantar_statistics_match_evidence",
                    all(observed.get(key) == value and f"{value:,}" in text for key, value in expected.items()),
                    {key: observed.get(key) for key in expected},
                )

    help_run = subprocess.run(
        [sys.executable, str(experiment), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    add(
        checks,
        "experiment_cli_smoke_test",
        help_run.returncode == 0
        and "{prepare,train,self-test}" in help_run.stdout,
        {"return_code": help_run.returncode},
    )
    self_test_run = subprocess.run(
        [sys.executable, str(experiment), "self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        self_test_payload = json.loads(self_test_run.stdout)
    except json.JSONDecodeError:
        self_test_payload = {}
    add(
        checks,
        "microproxy_behavioral_self_tests_pass",
        self_test_run.returncode == 0
        and self_test_payload.get("status") == "passed"
        and all(self_test_payload.get("wrong_script_behavior", {}).values())
        and all(self_test_payload.get("revision_handling", {}).values())
        and all(self_test_payload.get("execution_mode", {}).values())
        and self_test_payload.get("token_exposure_balance", {}).get("passed")
        is True,
        self_test_payload or self_test_run.stderr,
    )
    mutable_dataset_revision_run = subprocess.run(
        [
            sys.executable,
            str(experiment),
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
    add(
        checks,
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
            str(experiment),
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
    add(
        checks,
        "microproxy_model_revision_is_required",
        missing_revision_run.returncode == 2
        and "--model-revision" in missing_revision_run.stderr,
        {"return_code": missing_revision_run.returncode},
    )
    experiment_text = experiment.read_text(encoding="utf-8")
    add(
        checks,
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
    add(
        checks,
        "microproxy_loads_from_resolved_commits",
        "revision=resolved_dataset_revision" in experiment_text
        and "revision=resolved_model_revision" in experiment_text
        and "immutable_hub_commit(resolved_dataset_revision)" in experiment_text
        and "immutable_hub_commit(resolved_model_revision)" in experiment_text,
        "dataset stream, tokenizer, and model loads are anchored to resolved commits",
    )

    result = {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "classification": "final document, accounting, evidence, path, syntax, and smoke-test audit",
        "week5_root": str(root),
        "week4_root": str(args.week4_root),
        "artifact_sha256": {
            "readme": sha256_file(readme),
            "results_template": sha256_file(results_template),
            "additional_experiments": sha256_file(additional_file),
            "week4_audit": sha256_file(audit_file),
            "microproxy_script": sha256_file(experiment),
        },
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
