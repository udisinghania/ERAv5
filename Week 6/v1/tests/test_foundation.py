from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.canonical import canonical_json_bytes, read_jsonl_gz, sha256_bytes, write_jsonl_gz
from era6.chunking import boundary_aware_chunks, boundary_aware_chunks_v2
from era6.acquisition import (
    deterministic_block_offsets,
    deterministic_bucket_ids,
    load_source_lock,
    transform_record,
)
from era6.cleaning import (
    BasicQualityFilter,
    PIIScrubber,
    SourceAwarePIIScrubber,
    TextNormalizer,
    strip_wikitable_markup,
)
from era6.curation import build_source_locks
from era6.firewall import EvaluationRegistry, NGramDecontaminator
from era6.manifests import ShardManifest, SourceLock
from era6.quality import (
    DEFAULT_V2_RULES,
    extract_quality_signals,
    extract_quality_signals_v2,
    provisional_quality_flags,
    provisional_quality_flags_v2,
    quality_band_and_weight,
    quality_band_and_weight_v2,
)
from era6.splitting import deterministic_partition
from era6.tokenizer import MultilaneTokenizer, train_tokenizer
from era6.tokenization import supervised_byte_spans
from era6.performance import reconstructable_throughput
from era6.runtime import (
    capture_rng_state,
    choose_device_type,
    resolve_execution_device,
    restore_rng_state,
)
from era6.scheduling import allocate_integer, assign_difficulty_bands, enforce_integer_floors
from era6.packing import clip_record_to_loss, pack_group
from era6.batching import apply_opus_policy, select_opus_batches, zero_loss_slots_for_next_batch


class FoundationTests(unittest.TestCase):
    def test_runtime_auto_prefers_cuda_and_falls_back_to_cpu(self) -> None:
        self.assertEqual(choose_device_type("auto", cuda_available=True), "cuda")
        self.assertEqual(choose_device_type("auto", cuda_available=False), "cpu")
        self.assertEqual(choose_device_type("cpu", cuda_available=True), "cpu")
        with self.assertRaises(RuntimeError):
            choose_device_type("cuda", cuda_available=False)

    def test_forced_cpu_runtime_and_rng_round_trip(self) -> None:
        config = json.loads((ROOT / "configs" / "training_v1.json").read_text(encoding="utf-8"))
        device, descriptor = resolve_execution_device(config, "cpu")
        self.assertEqual(device.type, "cpu")
        self.assertEqual(descriptor["device_type"], "cpu")
        state = capture_rng_state(device)
        import torch

        expected = torch.rand(3)
        restore_rng_state(state, device)
        observed = torch.rand(3)
        self.assertTrue(torch.equal(expected, observed))

    def test_throughput_is_reconstructable_from_counts_and_elapsed_time(self) -> None:
        report = reconstructable_throughput(
            elapsed_nanoseconds=2_000_000_000,
            physical_tokens=2000,
            nonpadding_tokens=1800,
            loss_bearing_tokens=1500,
        )
        self.assertEqual(report["elapsed_seconds"], 2.0)
        self.assertEqual(report["physical_tokens_per_second"], 1000.0)
        self.assertEqual(report["nonpadding_tokens_per_second"], 900.0)
        self.assertEqual(report["useful_loss_bearing_tokens_per_second"], 750.0)
        self.assertEqual(report["packing_utilization"], 0.9)
        self.assertEqual(report["useful_loss_fraction"], 0.75)

    def test_opus_records_accept_reject_defer_and_floor_override(self) -> None:
        candidates = [
            {
                "candidate_id": 0,
                "proxy_score": 100,
                "metrics": {
                    "candidate_loss_tokens": 80,
                    "loss_density_ppm": 800000,
                    "protected_floor_deficit_tokens": 10,
                },
            },
            {
                "candidate_id": 1,
                "proxy_score": 90,
                "metrics": {
                    "candidate_loss_tokens": 75,
                    "loss_density_ppm": 750000,
                    "protected_floor_deficit_tokens": 0,
                },
            },
            {
                "candidate_id": 2,
                "proxy_score": 85,
                "metrics": {
                    "candidate_loss_tokens": 20,
                    "loss_density_ppm": 200000,
                    "protected_floor_deficit_tokens": 0,
                },
            },
            {
                "candidate_id": 3,
                "proxy_score": 70,
                "metrics": {
                    "candidate_loss_tokens": 70,
                    "loss_density_ppm": 700000,
                    "protected_floor_deficit_tokens": 0,
                },
            },
        ]
        config = {
            "acceptance_policy": {"minimum_loss_density_ppm": 300000},
            "protected_floor_policy": {"override_enabled": True},
        }
        selected, audited, policy = apply_opus_policy(candidates, config=config)
        self.assertEqual(selected["candidate_id"], 1)
        self.assertTrue(policy["protected_floor_override"])
        self.assertEqual(policy["normal_proxy_winner_candidate_id"], 0)
        self.assertEqual(policy["rejected_candidate_ids"], [2])
        self.assertEqual(policy["deferred_candidate_ids"], [0, 3])
        self.assertEqual(
            {row["candidate_id"]: row["outcome"] for row in audited},
            {0: "deferred", 1: "accepted", 2: "rejected", 3: "deferred"},
        )

    def test_zero_loss_context_is_spread_across_useful_batches(self) -> None:
        self.assertEqual(
            zero_loss_slots_for_next_batch(
                zero_remaining=3, nonzero_remaining=10, batch_size=4
            ),
            1,
        )

    def test_opus_batching_is_deterministic_and_consumes_once(self) -> None:
        sequences = []
        for index, (lane, loss) in enumerate(
            [("a", 1), ("a", 1), ("b", 1), ("b", 1), ("a", 0)]
        ):
            sequences.append(
                {
                    "sequence_index": index,
                    "sequence_length": 1,
                    "lane": lane,
                    "indic_tier": None,
                    "loss_bearing_tokens": loss,
                    "nonpadding_tokens": 1,
                    "fragments": [{"source_id": f"source-{index % 2}"}],
                }
            )
        stage = {
            "name": "toy",
            "sequence_length": 1,
            "target_loss_tokens": 4,
            "lane_targets": {"a": 2, "b": 2},
            "indic_tier_targets": {},
        }
        config = {
            "microbatch_physical_token_budget": 4,
            "candidate_microbatches": 4,
            "candidate_seed": "test",
            "acceptance_policy": {"minimum_loss_density_ppm": 0},
            "protected_floor_policy": {
                "lanes": ["b"],
                "override_enabled": True,
            },
            "score_weights": {
                "lane_error_ppm": -100,
                "indic_tier_error_ppm": -25,
                "loss_density_ppm": 2,
                "packing_utilization_ppm": 1,
                "source_diversity": 1000,
            },
        }
        first_batches, first_decisions = select_opus_batches(
            sequences, stage_schedule=stage, config=config
        )
        second_batches, second_decisions = select_opus_batches(
            sequences, stage_schedule=stage, config=config
        )
        self.assertEqual((first_batches, first_decisions), (second_batches, second_decisions))
        consumed = [index for batch in first_batches for index in batch["sequence_indices"]]
        self.assertEqual(sorted(consumed), list(range(5)))
        self.assertEqual(len(consumed), len(set(consumed)))
        self.assertTrue(all(batch["loss_bearing_tokens"] > 0 for batch in first_batches))

    def test_packing_preserves_loss_with_zero_loss_continuation_overlap(self) -> None:
        records = [
            {"record_id": "a", "source_id": "s", "selected_loss_tokens": 4},
            {"record_id": "b", "source_id": "s", "selected_loss_tokens": 2},
        ]
        payloads = {
            "a": ([0, 10, 11, 12, 13], [0, 1, 1, 1, 1]),
            "b": ([0, 20, 21], [0, 1, 1]),
        }
        sequences = pack_group(
            records,
            sequence_length=4,
            pad_token_id=0,
            load_payload=lambda row: payloads[row["record_id"]],
        )
        self.assertEqual(sum(sequence["loss_tokens"] for sequence in sequences), 6)
        self.assertEqual(sequences[1]["fragments"][0]["continuation_overlap"], 1)
        self.assertEqual(sequences[1]["loss"][0], 0)
        self.assertEqual(sequences[1]["positions"][0], 0)

    def test_final_record_clipping_hits_exact_loss_target(self) -> None:
        tokens, loss = clip_record_to_loss(
            [0, 1, 2, 3, 4], [0, 0, 1, 0, 1], selected_loss_tokens=1
        )
        self.assertEqual(tokens, [0, 1, 2])
        self.assertEqual(loss, [0, 0, 1])

    def test_integer_mixture_allocation_is_exact_and_deterministic(self) -> None:
        weights = {"general": 0.53, "indic": 0.08, "agentic": 0.005, "other": 0.385}
        first = allocate_integer(1001, weights)
        second = allocate_integer(1001, dict(reversed(list(weights.items()))))
        self.assertEqual(first, second)
        self.assertEqual(sum(first.values()), 1001)
        self.assertGreaterEqual(first["indic"], 80)
        self.assertGreaterEqual(first["agentic"], 5)
        protected = enforce_integer_floors(
            {"general": 91, "indic": 8, "agentic": 1},
            total=100,
            floors={"indic": 0.085, "agentic": 0.005},
        )
        self.assertEqual(sum(protected.values()), 100)
        self.assertEqual(protected["indic"], 9)

    def test_difficulty_is_separate_from_quality_and_covers_bands(self) -> None:
        rows = [
            {
                "record_id": f"r{index:02d}",
                "lane": "code",
                "source_id": "source",
                "token_count": index + 1,
                "quality_weight": 0.25 if index == 19 else 1.0,
            }
            for index in range(20)
        ]
        bands = [
            {"name": "B0", "maximum_fraction": 0.10},
            {"name": "B1", "maximum_fraction": 0.30},
            {"name": "B2", "maximum_fraction": 0.60},
            {"name": "B3", "maximum_fraction": 0.82},
            {"name": "B4", "maximum_fraction": 0.95},
            {"name": "B5", "maximum_fraction": 1.00},
        ]
        assigned = assign_difficulty_bands(rows, bands)
        by_id = {row["record_id"]: row for row in assigned}
        self.assertEqual(by_id["r00"]["difficulty_band"], "B0")
        self.assertEqual(by_id["r19"]["difficulty_band"], "B5")
        self.assertEqual(by_id["r19"]["quality_weight"], 0.25)

    def test_origin_loss_spans_select_only_model_outputs(self) -> None:
        reasoning = {
            "source_id": "gsm8k_reasoning_train",
            "capability_lane": "reasoning",
            "text": "<question>\n2+2?\n<reasoning>\n4",
        }
        translated = {
            "source_id": "assignment4_samanantar_translated",
            "capability_lane": "indic",
            "text": "<state> source </state> <action> translate </action> <observation> उत्तर </observation>",
        }
        tool_call = {
            "source_id": "hermes_function_calling",
            "capability_lane": "agentic",
            "text": '<tool_call>\n{"name":"search"}\n</tool_call>',
        }
        reasoning_spans = supervised_byte_spans(reasoning)
        translated_spans = supervised_byte_spans(translated)
        tool_spans = supervised_byte_spans(tool_call)
        self.assertEqual(reasoning["text"].encode("utf-8")[reasoning_spans[0][0] :].decode(), "4")
        self.assertEqual(
            translated["text"].encode("utf-8")[translated_spans[0][0] : translated_spans[0][1]].decode(),
            "उत्तर ",
        )
        self.assertEqual(len(tool_spans), 1)

    def test_multilane_tokenizer_is_lossless_and_deterministic(self) -> None:
        texts = [
            "<user> गणित 123 और हिन्दी पाठ </user>",
            "<assistant> def answer(x): return x + 1 </assistant>",
            "science α β γ; [EMAIL] and unseen🙂",
        ] * 20
        config = {
            "tokenizer_id": "test-tokenizer",
            "algorithm": "deterministic_greedy_unigram_with_byte_fallback",
            "vocab_size": 320,
            "maximum_unit_bytes": 64,
            "substring_max_characters": 5,
            "substring_source_units": 100,
            "reserved_character_pieces": 32,
            "special_tokens": ["<pad>", "<bos>", "<eos>", "<unk>", "<user>", "</user>", "[EMAIL]"],
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            payload_a = train_tokenizer(texts, config=config, corpus_hash="sha256:test", output_path=first)
            payload_b = train_tokenizer(texts, config=config, corpus_hash="sha256:test", output_path=second)
            self.assertEqual(payload_a["tokenizer_hash"], payload_b["tokenizer_hash"])
            tokenizer = MultilaneTokenizer.load(first)
            sample = "<user> नया code🙂 [EMAIL] </user>"
            self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_canonical_json_is_order_independent(self) -> None:
        self.assertEqual(canonical_json_bytes({"b": 2, "a": 1}), canonical_json_bytes({"a": 1, "b": 2}))

    def test_reproducible_gzip_jsonl(self) -> None:
        records = [{"id": 1, "text": "भारत"}, {"id": 2, "text": "code"}]
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.jsonl.gz"
            right = Path(directory) / "right.jsonl.gz"
            left_meta = write_jsonl_gz(left, records)
            right_meta = write_jsonl_gz(right, records)
            self.assertEqual(left.read_bytes(), right.read_bytes())
            self.assertEqual(left_meta, right_meta)
            self.assertEqual(list(read_jsonl_gz(left)), records)

    def test_normalizer_preserves_indic_joiners_and_newlines(self) -> None:
        value = "A\u200dB\n\n\nक्\u200dष <|endoftext|>"
        normalized = TextNormalizer().normalize(value)
        self.assertIn("\u200d", normalized)
        self.assertIn("\n\n", normalized)
        self.assertNotIn("endoftext", normalized)

    def test_pii_and_secret_masking(self) -> None:
        result = PIIScrubber().scrub("mail a@example.org phone +91 9876543210 api_key=abcdefghijklmnop")
        self.assertEqual(result.num_redactions, 3)
        self.assertIn("[EMAIL]", result.text)
        self.assertIn("[PHONE]", result.text)
        self.assertIn("[SECRET]", result.text)

    def test_public_reference_pii_preserves_identifiers_and_masks_contacts(self) -> None:
        value = (
            "OCLC 44090600; postcode 836100; patent 2825108; multicast 234.5.6.7; "
            "Phone: +91 98765 43210; mail editor@example.org"
        )
        result = SourceAwarePIIScrubber().scrub(value, source_class="public_reference")
        self.assertIn("44090600", result.text)
        self.assertIn("836100", result.text)
        self.assertIn("2825108", result.text)
        self.assertIn("234.5.6.7", result.text)
        self.assertIn("Phone: [PHONE]", result.text)
        self.assertIn("[EMAIL]", result.text)

    def test_structured_numeric_pii_preserves_all_numeric_values(self) -> None:
        value = "answer=884100; array=[1234567890]; api_key=abcdefghijklmnop; mail a@example.org"
        result = SourceAwarePIIScrubber().scrub(value, source_class="structured_numeric")
        self.assertIn("884100", result.text)
        self.assertIn("1234567890", result.text)
        self.assertIn("[SECRET]", result.text)
        self.assertIn("[EMAIL]", result.text)
        self.assertEqual(result.counts["phone"], 0)

    def test_public_reference_v2_masks_hindi_phone_and_bank_account(self) -> None:
        value = (
            "हमारे नए वॉट्सऐप नंबर (9958894163) से खबरें मिलेंगी। "
            "पीड़िता का खाता नंबर 10038954223 था।"
        )
        result = SourceAwarePIIScrubber().scrub(value, source_class="public_reference_v2")
        self.assertNotIn("9958894163", result.text)
        self.assertNotIn("10038954223", result.text)
        self.assertIn("[PHONE]", result.text)
        self.assertIn("[BANK_ACCOUNT]", result.text)
        self.assertEqual(result.counts["phone"], 1)
        self.assertEqual(result.counts["financial_account"], 1)

    def test_boundary_aware_chunks_share_title_and_obey_maximum(self) -> None:
        body = "\n\n".join(("Sentence one. Sentence two. " * 8).strip() for _ in range(8))
        chunks = boundary_aware_chunks("Example", body, maximum_characters=220)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.text.startswith("Example\n\n") for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 220 for chunk in chunks))
        self.assertNotIn("character", {chunk.end_boundary for chunk in chunks})

    def test_quality_filter(self) -> None:
        quality = BasicQualityFilter(min_characters=10, max_characters=100)
        self.assertTrue(quality.evaluate("A useful sentence.").admitted)
        self.assertFalse(quality.evaluate("tiny").admitted)

    def test_quality_signals_expose_repetition_and_boundaries(self) -> None:
        text = "Repeated sentence.\nRepeated sentence.\nRepeated sentence."
        signals = extract_quality_signals(text)
        self.assertGreater(signals["duplicate_line_fraction"], 0.5)
        self.assertTrue(signals["ends_at_boundary"])
        flags = provisional_quality_flags(signals, truncated=True, pii_redactions=1)
        self.assertIn("duplicate_lines", flags)
        self.assertIn("character_truncated", flags)
        self.assertIn("pii_pattern_redacted", flags)

    def test_low_prose_quality_is_retained_in_capped_b0(self) -> None:
        signals = extract_quality_signals("Name may refer to: One person. Another person.")
        flags = provisional_quality_flags(signals)
        band, weight, caps = quality_band_and_weight(signals, flags)
        self.assertEqual(band, "B0")
        self.assertLess(weight, 1.0)
        self.assertIn("general_disambiguation", caps)

    def test_v2_line_chunking_preserves_scientific_list_entries(self) -> None:
        body = "\n".join(
            f"Crassula example {index} L. var. sample, indigenous" for index in range(20)
        )
        chunks = boundary_aware_chunks_v2("Taxonomy", body, maximum_characters=220)
        self.assertGreater(len(chunks), 1)
        self.assertEqual({chunk.end_boundary for chunk in chunks}, {"line"})
        self.assertTrue(all(chunk.body.splitlines()[-1].endswith("indigenous") for chunk in chunks))

    def test_v2_detects_raw_wikitable_markup(self) -> None:
        text = "Example\n\nProse.\n|- align=\"center\" bgcolor=\"#ccffcc\"\n| 1 || Result\n|-"
        signals = extract_quality_signals_v2(text)
        flags = provisional_quality_flags_v2(signals)
        self.assertIn("raw_wikitable_markup", flags)

    def test_v2_detects_orphaned_repeated_table_footnotes(self) -> None:
        repeated = "\n".join(
            [
                "1. Alliance contested in several districts.",
                "2. Party contested as an independent group.",
            ]
            * 4
        )
        text = f"Election results\n\nNumber of votes\n\n{repeated}"
        signals = extract_quality_signals_v2(text)
        flags = provisional_quality_flags_v2(signals)
        self.assertIn("orphaned_table_footnotes", flags)

    def test_v2_category_tail_moves_to_capped_b0(self) -> None:
        text = "Composer\n\n1947 births\nLiving people\nFrench composers\nMusic critics"
        signals = extract_quality_signals_v2(text)
        flags = provisional_quality_flags_v2(signals, short_continuation=True)
        band, weight, caps = quality_band_and_weight_v2(signals, flags)
        self.assertIn("category_tail", flags)
        self.assertEqual((band, weight), ("B0", 0.25))
        self.assertIn("general_category_tail", caps)

    def test_v2_unbulleted_linewise_list_moves_to_capped_b0(self) -> None:
        text = "Taxa\n\n" + "\n".join(f"Species example {index} Author, 1900" for index in range(20))
        signals = extract_quality_signals_v2(text)
        flags = provisional_quality_flags_v2(signals)
        band, _weight, caps = quality_band_and_weight_v2(signals, flags)
        self.assertIn("linewise_list", flags)
        self.assertEqual(band, "B0")
        self.assertIn("general_linewise_list", caps)

    def test_v3_wikitable_cleanup_salvages_surrounding_prose(self) -> None:
        text = "Introduction prose.\n\n{| class=\"wikitable\"\n|-\n| 1 || result\n|}\n\nClosing prose."
        result = strip_wikitable_markup(text)
        self.assertEqual(result.removed_blocks, 1)
        self.assertGreaterEqual(result.removed_lines, 4)
        self.assertEqual(result.text, "Introduction prose.\n\nClosing prose.")

    def test_v3_stat_heavy_list_moves_to_capped_b0(self) -> None:
        text = "Club\n\nShort prose sentence.\nHonours\nCup: 10\n" + "\n".join(
            f"19{index:02d}-{index + 1:02d}" for index in range(10)
        )
        signals = extract_quality_signals_v2(text)
        rules = {
            **DEFAULT_V2_RULES,
            "stat_heavy_list": {
                "maximum_alpha_fraction": 0.60,
                "minimum_digit_fraction": 0.10,
                "minimum_short_line_fraction": 0.80,
                "minimum_nonempty_lines": 8,
            },
        }
        flags = provisional_quality_flags_v2(signals, rules=rules)
        band, _weight, caps = quality_band_and_weight_v2(signals, flags)
        self.assertIn("stat_heavy_list", flags)
        self.assertEqual(band, "B0")
        self.assertIn("general_stat_heavy_list", caps)

    def test_ngram_firewall_blocks_overlap(self) -> None:
        reference = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        firewall = NGramDecontaminator([reference])
        self.assertTrue(firewall.is_contaminated("prefix " + reference + " suffix"))
        self.assertFalse(firewall.is_contaminated("an unrelated training document"))

    def test_evaluation_registry_hash_and_canary(self) -> None:
        registry = EvaluationRegistry()
        entry = registry.register(
            evaluation_id="eval-1",
            benchmark_id="toy/test",
            version="1",
            content={"question": "held out"},
            canaries=["CANARY-NEVER-TRAIN"],
        )
        self.assertTrue(registry.blocks_hash(entry.content_hash))
        self.assertTrue(registry.blocks_canary("prefix canary-never-train suffix"))
        self.assertTrue(registry.registry_hash.startswith("sha256:"))

    def test_group_partition_is_deterministic(self) -> None:
        first = deterministic_partition("cluster-7", seed="v1")
        self.assertEqual(first, deterministic_partition("cluster-7", seed="v1"))
        self.assertIn(first, {"train", "validation", "anneal"})

    def test_source_lock_requires_lane_and_revision(self) -> None:
        lock = SourceLock(
            source_id="s1",
            source_url="https://example.test/data",
            revision="a" * 40,
            license_id="MIT",
            capability_lane="code",
            provenance_tier="verified",
        )
        self.assertTrue(lock.lock_hash.startswith("sha256:"))

    def test_acquisition_offsets_are_deterministic_and_distinct(self) -> None:
        first = deterministic_block_offsets("source-a", 10_000, 20)
        second = deterministic_block_offsets("source-a", 10_000, 20)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(offset % 100 == 0 for offset in first))

    def test_assignment4_bucket_selection_is_deterministic(self) -> None:
        first = deterministic_bucket_ids("assignment4", 16)
        self.assertEqual(first, deterministic_bucket_ids("assignment4", 16))
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(len(bucket) == 2 for bucket in first))

    def test_code_acquisition_rejects_non_permissive_row(self) -> None:
        target = {
            "source_id": "code",
            "revision": "abc",
            "license_id": "per-record",
            "allowed_row_licenses": ["mit"],
            "lane": "code",
            "provenance_tier": "curated",
            "permission": "train",
            "language": "python",
            "min_chars": 10,
            "max_chars": 1000,
            "transform": "codeparrot",
        }
        row = {
            "repo_name": "example/repo",
            "path": "a.py",
            "hash": 1,
            "license": "gpl-3.0",
            "content": "print('this code is long enough')",
        }
        self.assertIsNone(transform_record(target, row))

    def test_agentic_turns_have_explicit_roles(self) -> None:
        target = {
            "source_id": "agent",
            "revision": "abc",
            "license_id": "apache-2.0",
            "lane": "agentic",
            "provenance_tier": "synthetic",
            "permission": "train",
            "language": "en",
            "min_chars": 5,
            "max_chars": 1000,
            "transform": "hermes",
        }
        row = {
            "id": "1",
            "conversations": [
                {"from": "human", "value": "Find the weather"},
                {"from": "gpt", "value": "<tool_call>{}</tool_call>"},
                {"from": "tool", "value": "sunny"},
            ],
        }
        record = transform_record(target, row)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertIn("<user>", record["text"])
        self.assertIn("<assistant>", record["text"])
        self.assertIn("<tool>", record["text"])

    def test_sources_lock_has_complete_training_supply(self) -> None:
        lock = load_source_lock(ROOT / "configs" / "sources.lock.json")
        targets = lock["targets"]
        lanes = {target["lane"] for target in targets if target["permission"] == "train"}
        self.assertEqual(
            lanes,
            {"general", "science_math", "code", "reasoning", "long_context", "indic", "agentic"},
        )
        training_quota = sum(target["quota"] for target in targets if target["permission"] == "train")
        self.assertEqual(training_quota, 20_000)
        indic_tiers = {
            target["provenance_tier"]
            for target in targets
            if target["lane"] == "indic" and target["permission"] == "train"
        }
        self.assertEqual(indic_tiers, {"verified", "unverified", "translated", "synthetic"})

    def test_all_configured_sources_produce_immutable_locks(self) -> None:
        config = load_source_lock(ROOT / "configs" / "sources.lock.json")
        locks = build_source_locks(config, "parent-1")
        self.assertEqual(len(locks), len(config["targets"]))
        self.assertTrue(all(item["lock_hash"].startswith("sha256:") for item in locks))

    def test_never_train_manifest_cannot_have_loss(self) -> None:
        manifest = ShardManifest(
            shard_id="eval",
            content_hash="sha256:" + "1" * 64,
            tokenizer_hash="sha256:" + "2" * 64,
            cleaning_pipeline_hash="sha256:" + "3" * 64,
            capability_lane="reasoning",
            permission="never_train",
            record_count=1,
            token_count=10,
            loss_bearing_token_count=1,
            source_lock_hashes=("sha256:" + "4" * 64,),
            language_distribution={"en": 1},
            dedup_status="passed",
            pii_screen_status="passed",
            eval_overlap_status="registered",
            position_policy="causal",
            loss_mask_hash="sha256:" + "5" * 64,
        )
        with self.assertRaises(ValueError):
            manifest.validate()


if __name__ == "__main__":
    unittest.main()
