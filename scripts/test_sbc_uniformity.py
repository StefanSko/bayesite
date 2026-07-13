#!/usr/bin/env python3
"""Stdlib self-tests for the SBC rank-uniformity conformance gate."""

from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_sbc_uniformity.py"
SPEC = importlib.util.spec_from_file_location("check_sbc_uniformity", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"could not load {SCRIPT}")
sbc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sbc
SPEC.loader.exec_module(sbc)


def scalar_report(parameter: str, ranks: list[int], support_max: int = 40) -> dict:
    histogram = [0] * (support_max + 1)
    for rank in ranks:
        histogram[rank] += 1
    replicate_count = len(ranks)
    return {
        "sbc_format": "v0-provisional",
        "workflow_format": "v0-provisional",
        "report_kind": "simulation_based_calibration_rank_facts",
        "report_scope": "replicated_simulated_datasets",
        "rank_statistic": "count_posterior_draws_less_than_truth",
        "rank_scope": "per_parameter_coordinate_marginal",
        "tie_statistic": "count_posterior_draws_equal_to_truth",
        "rank_draws": support_max,
        "rank_bounds": {"min": 0, "max": support_max},
        "rank_bin_count": support_max + 1,
        "rank_bin_order": list(range(support_max + 1)),
        "replicates": replicate_count,
        "replicate_count": replicate_count,
        "replicate_order": list(range(replicate_count)),
        "parameter_count": 1,
        "parameter_report_count": 1,
        "parameter_order": [parameter],
        "parameters": {
            parameter: {
                "replicate_count": replicate_count,
                "replicate_order": list(range(replicate_count)),
                "shape": [],
                "rank_statistic": "count_posterior_draws_less_than_truth",
                "rank_scope": "per_parameter_coordinate_marginal",
                "tie_statistic": "count_posterior_draws_equal_to_truth",
                "rank_histogram_statistic": "count_simulated_replicates_by_rank",
                "rank_histogram_scope": "per_parameter_coordinate_marginal",
                "rank_draws": support_max,
                "rank_bounds": {"min": 0, "max": support_max},
                "rank_bin_count": support_max + 1,
                "rank_bin_order": list(range(support_max + 1)),
                "coordinate_order": [[]],
                "ranks": ranks,
                "tie_counts": [0] * replicate_count,
                "rank_histogram": histogram,
            }
        },
    }


class SbcUniformityTests(unittest.TestCase):
    def test_exact_binomial_pvalues_and_frozen_calibration_count(self) -> None:
        self.assertEqual(sbc.MONTE_CARLO_SETS, 4000)
        self.assertEqual(sbc._binomial_pointwise_pvalues(2, 0.5), [0.5, 1.0, 0.5])
        expected = [1.0, 0.875, 0.125]
        actual = sbc._binomial_pointwise_pvalues(2, 0.25)
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want)

    def test_frozen_default_bonferroni_gamma(self) -> None:
        self.assertAlmostEqual(
            sbc.calibrated_gamma(100, 40, 0.002),
            0.0001286609422221573,
            places=18,
        )

    def test_calibration_rejection_rate_is_bounded(self) -> None:
        alpha = 0.05
        rng = random.Random(8147)
        rejections = 0
        for _ in range(200):
            ranks = [rng.randrange(41) for _ in range(100)]
            rejected, _, _ = sbc.test_rank_uniformity(ranks, 40, alpha)
            rejections += int(rejected)
        self.assertLess(rejections / 200, 3 * alpha)

    def test_larger_replicate_count_has_stable_binomial_calibration(self) -> None:
        ranks = [index % 41 for index in range(250)]
        rejected, margin, diagnosis = sbc.test_rank_uniformity(ranks, 40, 0.01)
        self.assertFalse(rejected)
        self.assertGreaterEqual(margin, 0.0)
        self.assertIsNone(diagnosis)

    def test_strong_linear_bias_is_rejected(self) -> None:
        rng = random.Random(2718)
        # P(rank=k) increases linearly with k through the sqrt inverse CDF.
        ranks = [min(40, int(41 * rng.random() ** 0.5)) for _ in range(200)]
        rejected, margin, diagnosis = sbc.test_rank_uniformity(ranks, 40, 0.01)
        self.assertTrue(rejected)
        self.assertLess(margin, 0.0)
        self.assertEqual(diagnosis, "ranks skew high (biased estimates)")

    def test_u_shaped_ranks_are_underdispersed(self) -> None:
        ranks = [0, 40] * 100
        rejected, _, diagnosis = sbc.test_rank_uniformity(ranks, 40, 0.01)
        self.assertTrue(rejected)
        self.assertEqual(
            diagnosis, "too many extreme ranks (posterior underdispersed)"
        )

    def test_tie_resolution_is_deterministic(self) -> None:
        ranks = [2, 5, 7, 0]
        ties = [3, 0, 2, 8]
        first = sbc.resolve_tied_ranks(ranks, ties, seed=991)
        second = sbc.resolve_tied_ranks(ranks, ties, seed=991)
        self.assertEqual(first, second)
        self.assertEqual(first, [3, 5, 9, 2])
        self.assertTrue(all(rank <= value <= rank + tie for rank, tie, value in zip(ranks, ties, first)))

    def test_parser_rejects_binary_side_verdicts(self) -> None:
        for field in [
            "success",
            "pass",
            "passed",
            "fail",
            "failed",
            "uniformity",
            "verdict",
            "p_value",
        ]:
            report = scalar_report("theta", [0, 1, 2, 3])
            report[field] = True
            with self.subTest(field=field):
                with self.assertRaisesRegex(sbc.ConformanceError, "verdict-free"):
                    sbc.parse_rank_facts(report, "doctored", seed=1)

        report = scalar_report("theta", [0, 1, 2, 3])
        report["sampler_summary"] = {"success": True}
        with self.assertRaisesRegex(sbc.ConformanceError, "verdict-free"):
            sbc.parse_rank_facts(report, "doctored", seed=1)

        report = scalar_report("theta", [0, 1, 2, 3])
        report["parameters"]["theta"]["verdict"] = "pass"
        with self.assertRaisesRegex(sbc.ConformanceError, "verdict-free"):
            sbc.parse_rank_facts(report, "doctored", seed=1)

        report = scalar_report("theta", [0, 1, 2, 3])
        report["replicate_reports"] = [
            {"parameters": {"theta": {"p_value": 0.5}}}
        ]
        with self.assertRaisesRegex(sbc.ConformanceError, "verdict-free"):
            sbc.parse_rank_facts(report, "doctored", seed=1)

    def test_requested_replicate_count_must_match_report(self) -> None:
        report = scalar_report("theta", [0])
        with self.assertRaisesRegex(sbc.ConformanceError, "expected the requested 100"):
            sbc.validate_requested_replicates(report, "doctored", 100)

    def test_parser_rejects_incoherent_artifact_identity_and_order(self) -> None:
        report = scalar_report("theta", [0, 1, 2, 3])
        report["workflow_format"] = "wrong"
        with self.assertRaisesRegex(sbc.ConformanceError, "workflow_format"):
            sbc.parse_rank_facts(report, "doctored", seed=1)

        report = scalar_report("theta", [0, 1, 2, 3])
        report["parameter_order"] = ["theta", "theta"]
        report["parameter_count"] = 2
        report["parameter_report_count"] = 2
        with self.assertRaisesRegex(sbc.ConformanceError, "duplicates"):
            sbc.parse_rank_facts(report, "doctored", seed=1)

    def test_doctored_report_failure_names_parameter_and_shape(self) -> None:
        report = scalar_report("biased_theta", [0] * 100)
        results = sbc.evaluate_reports(
            [("doctored", report)], alpha=0.01, seed=1234
        )
        messages = sbc.failure_messages(results)
        self.assertEqual(len(messages), 1)
        self.assertIn("parameter biased_theta", messages[0])
        self.assertIn("ranks skew low (biased estimates)", messages[0])


if __name__ == "__main__":
    unittest.main()
