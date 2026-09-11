from __future__ import annotations

import math
import unittest

import torch

from experiment import (
    AdamConfig,
    DATASET_SNAPSHOT,
    EXPECTED_DATASET_CONTENT_HASH,
    TrainConfig,
    batch_sequence_hash,
    bias_correction_factor,
    bias_threshold_sensitivity,
    compare_adam_rows,
    cosine_multiplier,
    fit_log_lr_quadratic,
    hash_named_tensors,
    make_dataset,
    manual_adam_rows,
    pytorch_adam_rows,
    wsd_multiplier,
)


class AdamArithmeticTests(unittest.TestCase):
    def test_hand_computation_matches_pytorch(self) -> None:
        gradients = [0.50, 0.40, 0.60, 0.45, 0.55]
        compared, maximum_error = compare_adam_rows(
            manual_adam_rows(gradients), pytorch_adam_rows(gradients)
        )
        self.assertEqual(len(compared), 5)
        self.assertLess(maximum_error, 1e-12)

    def test_first_moments(self) -> None:
        first = manual_adam_rows([0.50], AdamConfig())[0]
        self.assertAlmostEqual(float(first["m"]), 0.05, places=14)
        self.assertAlmostEqual(float(first["v"]), 0.00025, places=14)
        self.assertAlmostEqual(float(first["m_hat"]), 0.50, places=14)
        self.assertAlmostEqual(float(first["v_hat"]), 0.25, places=14)

    def test_bias_factor_converges(self) -> None:
        self.assertLess(bias_correction_factor(1, 0.9, 0.999), 0.32)
        self.assertLess(abs(bias_correction_factor(50_000, 0.9, 0.999) - 1.0), 1e-12)

    def test_one_percent_threshold_matches_report(self) -> None:
        rows = bias_threshold_sensitivity(AdamConfig(), [0.01])
        self.assertEqual(rows[0]["first_step_permanently_below"], 3916)


class ScheduleTests(unittest.TestCase):
    def test_schedules_share_warmup(self) -> None:
        config = TrainConfig()
        for step in range(1, config.warmup_steps + 1):
            self.assertEqual(cosine_multiplier(step, config), wsd_multiplier(step, config))

    def test_wsd_is_stable_before_decay(self) -> None:
        config = TrainConfig()
        self.assertEqual(wsd_multiplier(config.warmup_steps, config), 1.0)
        self.assertEqual(wsd_multiplier(config.stop_step, config), 1.0)
        self.assertEqual(wsd_multiplier(config.wsd_decay_start, config), 1.0)

    def test_schedules_finish_at_minimum(self) -> None:
        config = TrainConfig()
        self.assertTrue(math.isclose(cosine_multiplier(300, config), config.min_lr_ratio))
        self.assertTrue(math.isclose(wsd_multiplier(300, config), config.min_lr_ratio))


class DefensibilityTests(unittest.TestCase):
    def test_dataset_has_independent_split_shapes(self) -> None:
        config = TrainConfig()
        data = make_dataset(config, torch.device("cpu"))
        self.assertEqual(data["train_x"].shape[0], config.train_examples)
        self.assertEqual(data["validation_x"].shape[0], config.validation_examples)
        self.assertEqual(data["test_x"].shape[0], config.test_examples)

    def test_bundled_dataset_has_canonical_content_hash(self) -> None:
        self.assertTrue(DATASET_SNAPSHOT.is_file())
        data = make_dataset(TrainConfig(), torch.device("cpu"))
        self.assertEqual(hash_named_tensors(data.items()), EXPECTED_DATASET_CONTENT_HASH)

    def test_batch_sequence_hash_is_reproducible(self) -> None:
        first = batch_sequence_hash(20, 16, 100, 7)
        self.assertEqual(first, batch_sequence_hash(20, 16, 100, 7))
        self.assertNotEqual(first, batch_sequence_hash(20, 16, 100, 8))

    def test_log_lr_quadratic_recovers_interior_minimum(self) -> None:
        center = 0.001
        rows = [
            {
                "learning_rate": center * (2.0**offset),
                "mean_validation_loss": 0.5 + 0.1 * offset * offset,
            }
            for offset in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]
        fitted = fit_log_lr_quadratic(rows)
        self.assertAlmostEqual(fitted["learning_rate"], center, places=12)


if __name__ == "__main__":
    unittest.main()
