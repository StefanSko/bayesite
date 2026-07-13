#!/usr/bin/env python3
"""Self-tests for the nuts-rs oracle aggregation statistics."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_oracle_script():
    path = REPO_ROOT / "scripts" / "check_nuts_rs_oracle.py"
    if not path.exists():
        raise AssertionError(f"missing oracle script {path}")
    module_name = "check_nuts_rs_oracle"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


oracle = load_oracle_script()


def estimate_z_scores(values: list[float], mcse: float = 1.0) -> list[float]:
    estimates = [oracle.Estimate(value=value, mcse=mcse, batch_count=32) for value in values]
    return [oracle.signed_z(estimate.value, estimate.mcse) for estimate in estimates]


def synthetic_result(target: str, stat: str, z: float) -> object:
    bayesite = oracle.Estimate(value=z, mcse=1.0, batch_count=8)
    nuts_rs = oracle.Estimate(value=0.0, mcse=1.0, batch_count=8)
    return oracle.StatResult(
        target=target,
        stat=stat,
        truth=0.0,
        bayesite=bayesite,
        nuts_rs=nuts_rs,
        bayesite_truth_z=z,
        nuts_rs_truth_z=0.0,
        cross_z=z / 2**0.5,
    )


class OracleStatsTests(unittest.TestCase):
    def test_unbiased_estimates_across_eight_seeds_pass(self) -> None:
        z_scores = estimate_z_scores([0.4, -0.5, 0.2, -0.1, -0.3, 0.5, -0.2, 0.0])

        self.assertTrue(all(oracle.per_check_passes(z) for z in z_scores))
        self.assertAlmostEqual(oracle.stouffer_z(z_scores), 0.0)
        self.assertTrue(oracle.combined_passes(z_scores))

    def test_small_bias_can_fail_only_the_stouffer_gate(self) -> None:
        bias_mcse = 0.3
        fixed_noise_draws = [1.1, 1.3, 1.1, 1.3, 1.1, 1.3, 1.1, 1.3]
        z_scores = estimate_z_scores(
            [bias_mcse + noise for noise in fixed_noise_draws]
        )

        self.assertTrue(all(oracle.per_check_passes(z) for z in z_scores))
        self.assertGreater(oracle.stouffer_z(z_scores), oracle.COMBINED_MAX_Z)
        self.assertFalse(oracle.combined_passes(z_scores))

    def test_opposite_signs_cancel_in_stouffer_gate(self) -> None:
        z_scores = estimate_z_scores([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0])

        self.assertAlmostEqual(oracle.stouffer_z(z_scores), 0.0)
        self.assertTrue(oracle.combined_passes(z_scores))

    def test_single_replicate_uses_only_existing_coarse_guard(self) -> None:
        self.assertTrue(oracle.per_check_passes(1.5))
        self.assertFalse(oracle.per_check_passes(5.1))
        self.assertTrue(oracle.combined_passes([5.1]))

    def test_advisory_t_is_not_available_for_zero_variance(self) -> None:
        self.assertIsNone(oracle.advisory_t_statistic([0.25] * 8))
        self.assertIsNotNone(oracle.advisory_t_statistic([0.2, 0.3, 0.2, 0.3]))

    def test_report_smoke_failure_highlight_and_byte_determinism(self) -> None:
        results = [
            synthetic_result("synthetic", "mean[0]", 3.0),
            synthetic_result("synthetic", "mean[0]", 3.0),
        ]
        aggregates = oracle.aggregate_stat_results(results)
        kwargs = {
            "target_count": 1,
            "replicates": 2,
            "draws": 20,
            "warmup": 10,
            "chains": 2,
            "seed": 17,
            "passed": False,
        }
        first = oracle.render_oracle_report(results, aggregates, **kwargs)
        second = oracle.render_oracle_report(results, aggregates, **kwargs)

        self.assertEqual(first.encode(), second.encode())
        self.assertEqual(first.count("<svg"), 3)
        self.assertIn('<div class="banner fail">FAIL</div>', first)
        self.assertIn("synthetic / mean[0]", first)
        self.assertIn('class="fail-row"', first)
        self.assertIn("Mean delta", first)

    def test_report_pass_banner(self) -> None:
        results = [
            synthetic_result("healthy", "var[0]", -0.2),
            synthetic_result("healthy", "var[0]", 0.2),
        ]
        report = oracle.render_oracle_report(
            results,
            oracle.aggregate_stat_results(results),
            target_count=1,
            replicates=2,
            draws=20,
            warmup=10,
            chains=2,
            seed=17,
            passed=True,
        )
        self.assertIn('<div class="banner pass">PASS</div>', report)


if __name__ == "__main__":
    unittest.main()
