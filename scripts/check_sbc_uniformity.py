#!/usr/bin/env python3
"""Check Bayesite SBC rank facts with simultaneous ECDF confidence bands.

This development-only conformance gate follows the discrete version of the
ECDF-band approach in Säilynoja, Bürkner & Vehtari (2021).  The Bayesite binary
continues to report facts only; this script owns the uniformity verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gate_report as report_html

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = REPO_ROOT / "scripts" / "sbc_scenarios"
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "bayesite"
MONTE_CARLO_SETS = 4000
CALIBRATION_SEED = 0x5BC2021
FORBIDDEN_VERDICT_FIELDS = {
    "success",
    "pass",
    "passed",
    "fail",
    "failed",
    "verdict",
    "uniformity",
    "p_value",
    "uniformity_p_value",
    "sampler_quality",
}


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    fixture: str
    scenario_path: Path


@dataclass(frozen=True)
class RankSeries:
    scenario: str
    parameter: str
    ranks: tuple[int, ...]
    support_max: int


@dataclass(frozen=True)
class UniformityResult:
    scenario: str
    parameter: str
    sample_count: int
    min_band_margin: float
    rejected: bool
    diagnosis: str | None


@dataclass(frozen=True)
class ThinningSelection:
    scenario: str
    thin: int
    ess_stat: float | None
    tau_hat: float | None
    main_seed_span: tuple[int, int]
    pilot_seed_span: tuple[int, int] | None


SCENARIOS = (
    ScenarioSpec("bounded_rates", "bounded_rates", SCENARIO_DIR / "bounded_rates.json"),
    ScenarioSpec(
        "linear_regression",
        "linear_regression",
        SCENARIO_DIR / "linear_regression.json",
    ),
)


class ConformanceError(ValueError):
    """Invalid command input or SBC rank-facts artifact."""


def _require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConformanceError(f"{path} must be a JSON integer")
    return value


def _require_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ConformanceError(f"{path} must be a JSON array")
    return value


def _flatten_ints(value: object, path: str) -> list[int]:
    if isinstance(value, list):
        flattened: list[int] = []
        for index, item in enumerate(value):
            flattened.extend(_flatten_ints(item, f"{path}[{index}]"))
        return flattened
    return [_require_int(value, path)]


def _derived_seed(seed: int, *parts: object) -> int:
    material = "\0".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def _coordinates_for_shape(shape: Sequence[int]) -> list[tuple[int, ...]]:
    coordinates = [()]
    for dimension in shape:
        coordinates = [
            prefix + (index,)
            for prefix in coordinates
            for index in range(dimension)
        ]
    return coordinates


def _reject_verdict_fields(value: object, path: str) -> None:
    if not isinstance(value, dict):
        return
    present = sorted(FORBIDDEN_VERDICT_FIELDS.intersection(value))
    if present:
        raise ConformanceError(
            f"{path}: SBC rank facts must remain verdict-free; remove fields "
            + ", ".join(present)
        )


def resolve_tied_ranks(
    ranks: Sequence[int], tie_counts: Sequence[int], *, seed: int
) -> list[int]:
    """Randomize rank ties uniformly over [rank, rank + tie_count]."""
    if len(ranks) != len(tie_counts):
        raise ConformanceError("rank and tie-count arrays must have equal lengths")
    rng = random.Random(seed)
    resolved: list[int] = []
    for index, (rank, tie_count) in enumerate(zip(ranks, tie_counts)):
        if tie_count < 0:
            raise ConformanceError(f"tie count at index {index} must be non-negative")
        resolved.append(rng.randint(rank, rank + tie_count))
    return resolved


def parse_rank_facts(
    report: object,
    scenario: str,
    *,
    seed: int,
    expected_chains: int | None = None,
    expected_draws: int | None = None,
    expected_thin: int | None = None,
) -> list[RankSeries]:
    """Extract coordinate-marginal ranks using documented SBC artifact fields."""
    if not isinstance(report, dict):
        raise ConformanceError(f"{scenario}: SBC report must be a JSON object")
    if report.get("sbc_format") != "v0-provisional":
        raise ConformanceError(f'{scenario}: sbc_format must be "v0-provisional"')
    if report.get("workflow_format") != "v0-provisional":
        raise ConformanceError(f'{scenario}: workflow_format must be "v0-provisional"')
    if report.get("report_kind") != "simulation_based_calibration_rank_facts":
        raise ConformanceError(
            f"{scenario}: report_kind must identify SBC aggregate rank facts"
        )
    if report.get("report_scope") != "replicated_simulated_datasets":
        raise ConformanceError(
            f"{scenario}: report_scope must identify replicated simulated datasets"
        )
    _reject_verdict_fields(report, scenario)
    _reject_verdict_fields(report.get("sampler_summary"), f"{scenario}/sampler_summary")

    expected_semantics = {
        "rank_statistic": "count_posterior_draws_less_than_truth",
        "rank_scope": "per_parameter_coordinate_marginal",
        "tie_statistic": "count_posterior_draws_equal_to_truth",
    }
    for field, expected in expected_semantics.items():
        if report.get(field) != expected:
            raise ConformanceError(f"{scenario}: {field} must be {expected!r}")

    report_thin = _require_int(report.get("thin"), "thin")
    settings = report.get("settings")
    if not isinstance(settings, dict):
        raise ConformanceError(f"{scenario}: settings must be an object")
    report_chains = _require_int(settings.get("chains"), "settings.chains")
    report_draws = _require_int(settings.get("num_draws"), "settings.num_draws")
    if report_thin < 1 or report_draws < 1 or report_draws % report_thin != 0:
        raise ConformanceError(
            f"{scenario}: thin must be positive and divide settings.num_draws"
        )
    for actual, expected, label in (
        (report_chains, expected_chains, "chains"),
        (report_draws, expected_draws, "draws"),
        (report_thin, expected_thin, "thin"),
    ):
        if expected is not None and actual != expected:
            raise ConformanceError(
                f"{scenario}: bayesite sbc reported {label}={actual}, expected "
                f"the requested {expected}; ensure sample.{label} is honored"
            )

    report_rank_draws = _require_int(report.get("rank_draws"), "rank_draws")
    expected_rank_draws = report_chains * (report_draws // report_thin)
    if report_rank_draws != expected_rank_draws:
        raise ConformanceError(
            f"{scenario}: rank_draws must equal chains * draws / thin "
            f"({expected_rank_draws})"
        )
    report_rank_bounds = report.get("rank_bounds")
    if not isinstance(report_rank_bounds, dict):
        raise ConformanceError(f"{scenario}: rank_bounds must be an object")
    report_rank_min = _require_int(report_rank_bounds.get("min"), "rank_bounds.min")
    report_rank_max = _require_int(report_rank_bounds.get("max"), "rank_bounds.max")
    report_rank_bin_count = _require_int(report.get("rank_bin_count"), "rank_bin_count")
    report_rank_bin_order = [
        _require_int(value, f"rank_bin_order[{index}]")
        for index, value in enumerate(
            _require_list(report.get("rank_bin_order"), "rank_bin_order")
        )
    ]
    if (
        report_rank_min != 0
        or report_rank_max != report_rank_draws
        or report_rank_bin_count != report_rank_draws + 1
        or report_rank_bin_order != list(range(report_rank_draws + 1))
    ):
        raise ConformanceError(f"{scenario}: report rank support metadata is inconsistent")

    replicate_count = _require_int(report.get("replicate_count"), "replicate_count")
    replicates = _require_int(report.get("replicates"), "replicates")
    if replicate_count < 1 or replicates != replicate_count:
        raise ConformanceError(
            f"{scenario}: replicates and replicate_count must be the same positive integer"
        )
    replicate_order = [
        _require_int(value, f"replicate_order[{index}]")
        for index, value in enumerate(
            _require_list(report.get("replicate_order"), "replicate_order")
        )
    ]
    if replicate_order != list(range(replicate_count)):
        raise ConformanceError(
            f"{scenario}: replicate_order must be contiguous zero-based replicate order"
        )

    raw_parameter_order = _require_list(report.get("parameter_order"), "parameter_order")
    if not all(isinstance(name, str) for name in raw_parameter_order):
        raise ConformanceError(f"{scenario}: parameter_order entries must be strings")
    parameter_order = list(raw_parameter_order)
    if len(set(parameter_order)) != len(parameter_order):
        raise ConformanceError(f"{scenario}: parameter_order must not contain duplicates")
    parameter_count = _require_int(report.get("parameter_count"), "parameter_count")
    parameter_report_count = _require_int(
        report.get("parameter_report_count"), "parameter_report_count"
    )
    if parameter_count != len(parameter_order) or parameter_report_count != len(parameter_order):
        raise ConformanceError(
            f"{scenario}: parameter counts must equal the length of parameter_order"
        )
    parameters = report.get("parameters")
    if not isinstance(parameters, dict):
        raise ConformanceError(f"{scenario}: parameters must be a JSON object")
    if set(parameters) != set(parameter_order):
        raise ConformanceError(
            f"{scenario}: parameters keys must exactly match parameter_order"
        )
    for name, parameter in parameters.items():
        _reject_verdict_fields(parameter, f"{scenario}/parameters/{name}")

    replicate_reports = report.get("replicate_reports")
    if replicate_reports is not None:
        for replicate_index, replicate_report in enumerate(
            _require_list(replicate_reports, "replicate_reports")
        ):
            replicate_path = f"{scenario}/replicate_reports/{replicate_index}"
            _reject_verdict_fields(replicate_report, replicate_path)
            if isinstance(replicate_report, dict):
                _reject_verdict_fields(
                    replicate_report.get("sampler_summary"),
                    f"{replicate_path}/sampler_summary",
                )
                replicate_parameters = replicate_report.get("parameters")
                if isinstance(replicate_parameters, dict):
                    for name, parameter in replicate_parameters.items():
                        _reject_verdict_fields(
                            parameter, f"{replicate_path}/parameters/{name}"
                        )

    series: list[RankSeries] = []
    for raw_name in parameter_order:
        parameter = parameters.get(raw_name)
        if not isinstance(parameter, dict):
            raise ConformanceError(f"{scenario}: missing parameter facts for {raw_name!r}")
        bounds = parameter.get("rank_bounds")
        if not isinstance(bounds, dict):
            raise ConformanceError(f"{scenario}/{raw_name}: rank_bounds must be an object")
        support_min = _require_int(bounds.get("min"), f"{raw_name}.rank_bounds.min")
        support_max = _require_int(bounds.get("max"), f"{raw_name}.rank_bounds.max")
        if support_min != 0 or support_max < 1:
            raise ConformanceError(
                f"{scenario}/{raw_name}: rank support must be {{0..S}} with S >= 1"
            )
        rank_draws = _require_int(parameter.get("rank_draws"), f"{raw_name}.rank_draws")
        if rank_draws != support_max or rank_draws != report_rank_draws:
            raise ConformanceError(
                f"{scenario}/{raw_name}: parameter rank support must match the report"
            )
        for field, expected in expected_semantics.items():
            if parameter.get(field) != expected:
                raise ConformanceError(
                    f"{scenario}/{raw_name}: {field} must be {expected!r}"
                )
        if parameter.get("rank_histogram_statistic") != "count_simulated_replicates_by_rank":
            raise ConformanceError(
                f"{scenario}/{raw_name}: rank_histogram_statistic has unsupported semantics"
            )
        if parameter.get("rank_histogram_scope") != "per_parameter_coordinate_marginal":
            raise ConformanceError(
                f"{scenario}/{raw_name}: rank_histogram_scope has unsupported semantics"
            )
        rank_bin_count = _require_int(
            parameter.get("rank_bin_count"), f"{raw_name}.rank_bin_count"
        )
        rank_bin_order = [
            _require_int(value, f"{raw_name}.rank_bin_order[{index}]")
            for index, value in enumerate(
                _require_list(parameter.get("rank_bin_order"), f"{raw_name}.rank_bin_order")
            )
        ]
        if rank_bin_count != support_max + 1 or rank_bin_order != list(
            range(support_max + 1)
        ):
            raise ConformanceError(
                f"{scenario}/{raw_name}: rank bins must enumerate the full rank support"
            )
        parameter_replicates = _require_int(
            parameter.get("replicate_count"), f"{raw_name}.replicate_count"
        )
        parameter_replicate_order = [
            _require_int(value, f"{raw_name}.replicate_order[{index}]")
            for index, value in enumerate(
                _require_list(parameter.get("replicate_order"), f"{raw_name}.replicate_order")
            )
        ]
        if (
            parameter_replicates != replicate_count
            or parameter_replicate_order != replicate_order
        ):
            raise ConformanceError(
                f"{scenario}/{raw_name}: replicate count/order must match the report"
            )

        shape = [
            _require_int(value, f"{raw_name}.shape[{index}]")
            for index, value in enumerate(
                _require_list(parameter.get("shape"), f"{raw_name}.shape")
            )
        ]
        if any(dimension < 0 for dimension in shape):
            raise ConformanceError(f"{scenario}/{raw_name}: shape dimensions must be non-negative")
        coordinate_order = _require_list(
            parameter.get("coordinate_order"), f"{raw_name}.coordinate_order"
        )
        coordinates: list[tuple[int, ...]] = []
        for coordinate_index, coordinate in enumerate(coordinate_order):
            coordinate_values = _flatten_ints(
                coordinate, f"{raw_name}.coordinate_order[{coordinate_index}]"
            )
            coordinates.append(tuple(coordinate_values))
        if coordinates != _coordinates_for_shape(shape):
            raise ConformanceError(
                f"{scenario}/{raw_name}: coordinate_order must be zero-based row-major for shape"
            )

        rank_rows = _require_list(parameter.get("ranks"), f"{raw_name}.ranks")
        tie_rows = _require_list(parameter.get("tie_counts"), f"{raw_name}.tie_counts")
        if len(rank_rows) != replicate_count or len(tie_rows) != replicate_count:
            raise ConformanceError(
                f"{scenario}/{raw_name}: ranks and tie_counts must each have "
                f"replicate_count ({replicate_count}) entries"
            )
        ranks_by_coordinate = [[] for _ in coordinates]
        ties_by_coordinate = [[] for _ in coordinates]
        for replicate, (rank_row, tie_row) in enumerate(zip(rank_rows, tie_rows)):
            row_ranks = _flatten_ints(rank_row, f"{raw_name}.ranks[{replicate}]")
            row_ties = _flatten_ints(tie_row, f"{raw_name}.tie_counts[{replicate}]")
            if len(row_ranks) != len(coordinates) or len(row_ties) != len(coordinates):
                raise ConformanceError(
                    f"{scenario}/{raw_name}: replicate {replicate} rank/tie shapes "
                    "must match coordinate_order"
                )
            for coordinate_index, (rank, tie_count) in enumerate(zip(row_ranks, row_ties)):
                if rank < 0 or tie_count < 0 or rank + tie_count > support_max:
                    raise ConformanceError(
                        f"{scenario}/{raw_name}: replicate {replicate} rank/tie range "
                        f"must lie in 0..{support_max}"
                    )
                ranks_by_coordinate[coordinate_index].append(rank)
                ties_by_coordinate[coordinate_index].append(tie_count)

        raw_histograms = _require_list(
            parameter.get("rank_histogram"), f"{raw_name}.rank_histogram"
        )
        if len(coordinates) == 1 and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw_histograms
        ):
            histogram_rows = [raw_histograms]
        else:
            if len(raw_histograms) != len(coordinates):
                raise ConformanceError(
                    f"{scenario}/{raw_name}: rank_histogram must have one row per coordinate"
                )
            histogram_rows = [
                _require_list(value, f"{raw_name}.rank_histogram[{index}]")
                for index, value in enumerate(raw_histograms)
            ]
        for coordinate_index, histogram_row in enumerate(histogram_rows):
            histogram = [
                _require_int(value, f"{raw_name}.rank_histogram[{coordinate_index}][{index}]")
                for index, value in enumerate(histogram_row)
            ]
            expected = [0] * (support_max + 1)
            for rank in ranks_by_coordinate[coordinate_index]:
                expected[rank] += 1
            if histogram != expected:
                raise ConformanceError(
                    f"{scenario}/{raw_name}: rank_histogram must match the reported ranks"
                )

        for coordinate_index, coordinate in enumerate(coordinates):
            display_name = raw_name
            if coordinate:
                display_name += "[" + ",".join(str(index) for index in coordinate) + "]"
            tie_seed = _derived_seed(seed, scenario, raw_name, *coordinate)
            resolved = resolve_tied_ranks(
                ranks_by_coordinate[coordinate_index],
                ties_by_coordinate[coordinate_index],
                seed=tie_seed,
            )
            series.append(
                RankSeries(
                    scenario=scenario,
                    parameter=display_name,
                    ranks=tuple(resolved),
                    support_max=support_max,
                )
            )
    return series


def _binomial_pointwise_pvalues(sample_count: int, probability: float) -> list[float]:
    """Equal-tailed exact binomial p-values for every possible observed count."""
    if probability >= 1.0:
        return [0.0] * sample_count + [1.0]
    if probability <= 0.0:
        return [1.0] + [0.0] * sample_count
    # Anchor the recurrence at the mode.  Starting at count zero underflows
    # for ordinary larger N when p is near one, making every mass zero.
    mode = min(sample_count, math.floor((sample_count + 1) * probability))
    masses = [0.0] * (sample_count + 1)
    masses[mode] = 1.0
    down_ratio = (1.0 - probability) / probability
    for count in range(mode, 0, -1):
        masses[count - 1] = (
            masses[count] * count / (sample_count - count + 1) * down_ratio
        )
    up_ratio = probability / (1.0 - probability)
    for count in range(mode + 1, sample_count + 1):
        masses[count] = (
            masses[count - 1] * (sample_count - count + 1) / count * up_ratio
        )
    total = sum(masses)
    masses = [mass / total for mass in masses]
    lower: list[float] = []
    running = 0.0
    for mass in masses:
        running += mass
        lower.append(min(1.0, running))
    upper = [0.0] * (sample_count + 1)
    running = 0.0
    for count in range(sample_count, -1, -1):
        running += masses[count]
        upper[count] = min(1.0, running)
    return [min(1.0, 2.0 * min(lower[count], upper[count])) for count in range(sample_count + 1)]


def _pointwise_tables(sample_count: int, support_max: int) -> list[list[float]]:
    support_size = support_max + 1
    return [
        _binomial_pointwise_pvalues(sample_count, (rank + 1) / support_size)
        for rank in range(support_max)
    ]


_CALIBRATION_CACHE: dict[tuple[int, int, int], tuple[float, ...]] = {}


def calibrated_gamma(
    sample_count: int,
    support_max: int,
    alpha: float,
    *,
    simulations: int = MONTE_CARLO_SETS,
) -> float:
    """Monte Carlo calibrate the pointwise level for simultaneous coverage."""
    if sample_count < 1 or support_max < 1 or simulations < 1:
        raise ValueError("sample_count, support_max, and simulations must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    key = (sample_count, support_max, simulations)
    minima = _CALIBRATION_CACHE.get(key)
    if minima is None:
        tables = _pointwise_tables(sample_count, support_max)
        rng = random.Random(_derived_seed(CALIBRATION_SEED, sample_count, support_max, simulations))
        simulated_minima: list[float] = []
        for _ in range(simulations):
            histogram = [0] * (support_max + 1)
            for _ in range(sample_count):
                histogram[rng.randrange(support_max + 1)] += 1
            cumulative = 0
            min_p = 1.0
            for grid_index, table in enumerate(tables):
                cumulative += histogram[grid_index]
                min_p = min(min_p, table[cumulative])
            simulated_minima.append(min_p)
        minima = tuple(sorted(simulated_minima))
        _CALIBRATION_CACHE[key] = minima
    quantile_index = max(0, math.ceil(alpha * simulations) - 1)
    return minima[quantile_index]


def _classify_deviation(
    above: list[int], below: list[int], ranks: Sequence[int], support_max: int
) -> str:
    if (above and not below) or (below and not above):
        mean_rank = sum(ranks) / len(ranks)
        standard_error = math.sqrt(support_max * (support_max + 2) / 12) / math.sqrt(
            len(ranks)
        )
        z_score = (mean_rank - support_max / 2) / standard_error
        if abs(z_score) < 3:
            return (
                "extreme ranks exceed envelope without a mean shift (dispersion or "
                "residual rank dependence; verify ESS-adaptive thinning)"
            )
        if above:
            return "ranks skew low (biased estimates)"
        return "ranks skew high (biased estimates)"
    if sum(above) / len(above) < sum(below) / len(below):
        return "too many extreme ranks (posterior underdispersed)"
    return "too few extreme ranks (posterior overdispersed)"


def accepted_count_bounds(
    sample_count: int,
    support_max: int,
    alpha: float,
    *,
    simulations: int = MONTE_CARLO_SETS,
) -> list[tuple[int, int]]:
    """Return the exact accepted cumulative-count bounds at each ECDF grid point."""
    gamma = calibrated_gamma(
        sample_count, support_max, alpha, simulations=simulations
    )
    bounds: list[tuple[int, int]] = []
    for table in _pointwise_tables(sample_count, support_max):
        accepted = [count for count, p_value in enumerate(table) if p_value >= gamma]
        bounds.append((accepted[0], accepted[-1]))
    return bounds


def test_rank_uniformity(
    ranks: Sequence[int],
    support_max: int,
    alpha: float,
    *,
    simulations: int = MONTE_CARLO_SETS,
) -> tuple[bool, float, str | None]:
    """Apply a calibrated simultaneous discrete ECDF band to one rank series."""
    sample_count = len(ranks)
    if sample_count < 1:
        raise ValueError("at least one rank is required")
    if support_max < 1 or any(rank < 0 or rank > support_max for rank in ranks):
        raise ValueError("ranks must lie in 0..support_max")
    bounds = accepted_count_bounds(
        sample_count, support_max, alpha, simulations=simulations
    )
    histogram = [0] * (support_max + 1)
    for rank in ranks:
        histogram[rank] += 1

    above: list[int] = []
    below: list[int] = []
    min_margin_count = sample_count
    cumulative = 0
    for grid_index, (lower, upper) in enumerate(bounds):
        cumulative += histogram[grid_index]
        min_margin_count = min(min_margin_count, cumulative - lower, upper - cumulative)
        if cumulative > upper:
            above.append(grid_index)
        elif cumulative < lower:
            below.append(grid_index)
    rejected = bool(above or below)
    diagnosis = (
        _classify_deviation(above, below, ranks, support_max) if rejected else None
    )
    return rejected, min_margin_count / sample_count, diagnosis


def rank_series_from_reports(
    reports: Sequence[tuple[str, object]], *, seed: int
) -> list[RankSeries]:
    all_series: list[RankSeries] = []
    for scenario, report in reports:
        all_series.extend(parse_rank_facts(report, scenario, seed=seed))
    if not all_series:
        raise ConformanceError("SBC reports contain no parameter rank arrays")
    return all_series


def evaluate_rank_series(
    all_series: Sequence[RankSeries],
    *,
    alpha: float,
    simulations: int = MONTE_CARLO_SETS,
) -> list[UniformityResult]:
    if not all_series:
        raise ConformanceError("SBC reports contain no parameter rank arrays")
    alpha_parameter = alpha / len(all_series)
    results: list[UniformityResult] = []
    for series in all_series:
        rejected, margin, diagnosis = test_rank_uniformity(
            series.ranks,
            series.support_max,
            alpha_parameter,
            simulations=simulations,
        )
        results.append(
            UniformityResult(
                scenario=series.scenario,
                parameter=series.parameter,
                sample_count=len(series.ranks),
                min_band_margin=margin,
                rejected=rejected,
                diagnosis=diagnosis,
            )
        )
    return results


def evaluate_reports(
    reports: Sequence[tuple[str, object]],
    *,
    alpha: float,
    seed: int,
    simulations: int = MONTE_CARLO_SETS,
) -> list[UniformityResult]:
    all_series = rank_series_from_reports(reports, seed=seed)
    return evaluate_rank_series(all_series, alpha=alpha, simulations=simulations)


def smallest_divisor_at_least(draws: int, required_stride: int) -> int:
    """Return the smallest divisor of draws no smaller than required_stride."""
    if draws < 1 or required_stride < 1:
        raise ValueError("draws and required_stride must be positive")
    for candidate in range(required_stride, draws + 1):
        if draws % candidate == 0:
            return candidate
    return draws


def _load_scenario(
    spec: ScenarioSpec,
    args: argparse.Namespace,
    *,
    seed: int,
    replicates: int,
    thin: int,
) -> dict:
    try:
        scenario = json.loads(spec.scenario_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConformanceError(f"cannot read scenario {spec.scenario_path}: {error}") from error
    if not isinstance(scenario, dict):
        raise ConformanceError(f"{spec.scenario_path} must contain a JSON object")
    scenario["seed"] = seed
    scenario["replicates"] = replicates
    scenario["sample"] = {
        "chains": args.chains,
        "warmup": args.warmup,
        "draws": args.draws,
        "thin": thin,
        "max_treedepth": args.max_treedepth,
        "target_accept": args.target_accept,
    }
    return scenario


def validate_requested_replicates(report: object, scenario: str, expected: int) -> None:
    if not isinstance(report, dict):
        raise ConformanceError(f"{scenario}: SBC report must be a JSON object")
    actual = _require_int(report.get("replicate_count"), "replicate_count")
    if actual != expected:
        raise ConformanceError(
            f"{scenario}: bayesite sbc reported {actual} replicates, expected "
            f"the requested {expected}; ensure --replicates is honored"
        )


def _run_sbc(
    binary: Path,
    model_path: Path,
    scenario_path: Path,
    *,
    scenario: str,
    replicates: int,
) -> object:
    command = [
        str(binary),
        "sbc",
        "--model",
        str(model_path),
        "--scenario",
        str(scenario_path),
        "--replicates",
        str(replicates),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ConformanceError(f"cannot execute {binary}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise ConformanceError(
            f"{scenario}: bayesite sbc failed with exit code "
            f"{completed.returncode}: {detail}"
        )
    if completed.stderr:
        raise ConformanceError(f"{scenario}: successful bayesite sbc wrote stderr")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ConformanceError(
            f"{scenario}: bayesite sbc stdout is not one JSON report: {error}"
        ) from error
    validate_requested_replicates(report, scenario, replicates)
    return report


def _flatten_finite_numbers(value: object, path: str) -> list[float]:
    if isinstance(value, list):
        flattened: list[float] = []
        for index, item in enumerate(value):
            flattened.extend(_flatten_finite_numbers(item, f"{path}[{index}]"))
        return flattened
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConformanceError(f"{path} must contain finite numeric ESS values")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ConformanceError(f"{path} must contain positive finite ESS values")
    return [number]


def pilot_ess_stat(report: object, scenario: str) -> float:
    if not isinstance(report, dict):
        raise ConformanceError(f"{scenario}: pilot SBC report must be a JSON object")
    parameter_order = _require_list(report.get("parameter_order"), "parameter_order")
    replicate_reports = _require_list(report.get("replicate_reports"), "replicate_reports")
    ess_by_parameter: dict[str, list[list[float]]] = {
        name: [] for name in parameter_order if isinstance(name, str)
    }
    if len(ess_by_parameter) != len(parameter_order) or not ess_by_parameter:
        raise ConformanceError(f"{scenario}: pilot report needs named parameters")
    for replicate_index, replicate in enumerate(replicate_reports):
        if not isinstance(replicate, dict) or not isinstance(replicate.get("parameters"), dict):
            raise ConformanceError(
                f"{scenario}: pilot replicate {replicate_index} needs parameter diagnostics"
            )
        parameters = replicate["parameters"]
        for name in ess_by_parameter:
            parameter = parameters.get(name)
            if not isinstance(parameter, dict):
                raise ConformanceError(
                    f"{scenario}: pilot replicate {replicate_index} is missing {name!r}"
                )
            values = _flatten_finite_numbers(
                parameter.get("ess"),
                f"{scenario}/pilot/replicate_reports/{replicate_index}/parameters/{name}/ess",
            )
            previous = ess_by_parameter[name]
            if previous and len(values) != len(previous[0]):
                raise ConformanceError(
                    f"{scenario}: pilot ESS shape for {name!r} changed across replicates"
                )
            previous.append(values)
    parameter_medians: list[float] = []
    for values_by_replicate in ess_by_parameter.values():
        for coordinate in range(len(values_by_replicate[0])):
            parameter_medians.append(
                statistics.median(
                    values[coordinate] for values in values_by_replicate
                )
            )
    return min(parameter_medians)


def _fixed_thin(value: str) -> int | None:
    if value == "auto":
        return None
    try:
        thin = int(value)
    except ValueError as error:
        raise ConformanceError("--thin must be 'auto' or a positive integer") from error
    if thin < 1:
        raise ConformanceError("--thin must be 'auto' or a positive integer")
    return thin


def run_scenarios(
    binary: Path, args: argparse.Namespace
) -> tuple[list[tuple[str, object]], list[ThinningSelection]]:
    if not binary.is_file():
        raise ConformanceError(
            f"Bayesite release binary not found at {binary}; run cargo build --release "
            "--manifest-path crates/core/Cargo.toml"
        )
    fixed_thin = _fixed_thin(args.thin)
    reports: list[tuple[str, object]] = []
    selections: list[ThinningSelection] = []
    with tempfile.TemporaryDirectory(prefix="bayesite-sbc-uniformity-") as tmp:
        tmp_path = Path(tmp)
        for scenario_index, spec in enumerate(SCENARIOS):
            fixture_path = REPO_ROOT / "tests" / "golden_ir" / "fixtures" / f"{spec.fixture}.json"
            try:
                fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
                model = fixture["ir"]
            except (OSError, json.JSONDecodeError, KeyError) as error:
                raise ConformanceError(f"cannot load model fixture {fixture_path}: {error}") from error
            model_path = tmp_path / f"{spec.name}-model.json"
            scenario_path = tmp_path / f"{spec.name}-scenario.json"
            model_path.write_text(json.dumps(model), encoding="utf-8")
            main_seed = args.seed + scenario_index * 1_000_000
            ess_stat: float | None = None
            tau_hat: float | None = None
            pilot_span: tuple[int, int] | None = None
            thin = fixed_thin
            if thin is None:
                pilot_seed = main_seed + 500_000
                pilot_span = (pilot_seed, pilot_seed + 2 * args.pilot_replicates - 1)
                scenario_path.write_text(
                    json.dumps(
                        _load_scenario(
                            spec,
                            args,
                            seed=pilot_seed,
                            replicates=args.pilot_replicates,
                            thin=1,
                        )
                    ),
                    encoding="utf-8",
                )
                pilot_report = _run_sbc(
                    binary,
                    model_path,
                    scenario_path,
                    scenario=f"{spec.name} pilot",
                    replicates=args.pilot_replicates,
                )
                parse_rank_facts(
                    pilot_report,
                    f"{spec.name} pilot",
                    seed=pilot_seed,
                    expected_chains=args.chains,
                    expected_draws=args.draws,
                    expected_thin=1,
                )
                ess_stat = pilot_ess_stat(pilot_report, spec.name)
                tau_hat = args.draws / ess_stat
                required_stride = math.ceil(args.ess_safety * tau_hat)
                thin = smallest_divisor_at_least(args.draws, required_stride)
                if required_stride > args.draws or thin == args.draws:
                    raise ConformanceError(
                        f"{spec.name}: ESS-adaptive thinning requires stride "
                        f"{required_stride}, leaving fewer than two rank draws; increase --draws"
                    )
            assert thin is not None
            scenario_path.write_text(
                json.dumps(
                    _load_scenario(
                        spec,
                        args,
                        seed=main_seed,
                        replicates=args.replicates,
                        thin=thin,
                    )
                ),
                encoding="utf-8",
            )
            report = _run_sbc(
                binary,
                model_path,
                scenario_path,
                scenario=spec.name,
                replicates=args.replicates,
            )
            parse_rank_facts(
                report,
                spec.name,
                seed=main_seed,
                expected_chains=args.chains,
                expected_draws=args.draws,
                expected_thin=thin,
            )
            reports.append((spec.name, report))
            selections.append(
                ThinningSelection(
                    scenario=spec.name,
                    thin=thin,
                    ess_stat=ess_stat,
                    tau_hat=tau_hat,
                    main_seed_span=(main_seed, main_seed + 2 * args.replicates - 1),
                    pilot_seed_span=pilot_span,
                )
            )
    return reports, selections


def _validate_args(args: argparse.Namespace) -> None:
    if args.replicates < 1:
        raise ConformanceError("--replicates must be positive")
    if args.pilot_replicates < 1:
        raise ConformanceError("--pilot-replicates must be positive")
    if 2 * args.replicates > 500_000:
        raise ConformanceError(
            "--replicates must be at most 250000 so the main SBC seed span ends "
            "before the pilot span; reduce --replicates"
        )
    if 2 * args.pilot_replicates > 499_999:
        raise ConformanceError(
            "--pilot-replicates must be at most 249999 so the pilot seed span stays "
            "inside its scenario block; reduce --pilot-replicates"
        )
    last_seed = args.seed + (len(SCENARIOS) - 1) * 1_000_000 + 999_998
    if args.seed < 0 or last_seed > 2**63 - 1:
        raise ConformanceError(
            "--seed must leave room for every main and pilot SBC seed span; reduce --seed"
        )
    if not 0.0 < args.alpha < 1.0:
        raise ConformanceError("--alpha must be in (0, 1)")
    if args.chains < 1:
        raise ConformanceError("--chains must be positive")
    if args.warmup < 0:
        raise ConformanceError("--warmup must be non-negative")
    if args.draws < 4:
        raise ConformanceError("--draws must be at least 4 because SBC reports diagnostics")
    fixed_thin = _fixed_thin(args.thin)
    if fixed_thin is not None and args.draws % fixed_thin != 0:
        raise ConformanceError(
            "--thin must divide --draws exactly; pick a positive --thin that divides --draws"
        )
    if not math.isfinite(args.ess_safety) or args.ess_safety <= 0.0:
        raise ConformanceError("--ess-safety must be a positive finite number")
    if not 1 <= args.max_treedepth <= 20:
        raise ConformanceError("--max-treedepth must be in 1..=20")
    if not 0.0 < args.target_accept < 1.0:
        raise ConformanceError("--target-accept must be in (0, 1)")


def failure_messages(results: Sequence[UniformityResult]) -> list[str]:
    return [
        f"G11 SBC rank uniformity failed: scenario {result.scenario}, "
        f"parameter {result.parameter}: {result.diagnosis}; inspect the rank "
        "histogram and sampler facts, then repair calibration or sampling."
        for result in results
        if result.rejected
    ]


def _print_thinning(selections: Sequence[ThinningSelection]) -> None:
    print("ESS-adaptive rank thinning:")
    for selection in selections:
        ess = "not run" if selection.ess_stat is None else f"{selection.ess_stat:.6g}"
        tau = "not run" if selection.tau_hat is None else f"{selection.tau_hat:.6g}"
        pilot = (
            "not run"
            if selection.pilot_seed_span is None
            else f"[{selection.pilot_seed_span[0]}, {selection.pilot_seed_span[1]}]"
        )
        print(
            f"  {selection.scenario}: thin={selection.thin}, ess_stat={ess}, "
            f"tau_hat={tau}, main_seed_span=[{selection.main_seed_span[0]}, "
            f"{selection.main_seed_span[1]}], pilot_seed_span={pilot}"
        )


def _print_results(results: Sequence[UniformityResult]) -> None:
    print(f"{'scenario':20s} {'parameter':18s} {'N':>5s} {'min band margin':>15s}")
    for result in results:
        print(
            f"{result.scenario:20s} {result.parameter:18s} "
            f"{result.sample_count:5d} {result.min_band_margin:15.4f}"
        )


def _histogram_figure(series: RankSeries, result: UniformityResult) -> str:
    bin_count = 20
    histogram = [0] * bin_count
    support_size = series.support_max + 1
    for rank in series.ranks:
        histogram[min(bin_count - 1, rank * bin_count // support_size)] += 1
    expected = len(series.ranks) / bin_count
    maximum = max(max(histogram), expected * 1.1, 1.0)
    width = 520
    height = 245
    left = 50
    right = 500
    top = 38
    bottom = 205

    def y_position(value: float) -> float:
        return bottom - value / maximum * (bottom - top)

    elements = [
        report_html.svg_text(
            12,
            20,
            f"{series.scenario} / {series.parameter}",
            size=13,
            weight="bold",
        ),
        report_html.svg_axes(left, bottom, right, top),
    ]
    slot_width = (right - left) / bin_count
    color = report_html.RED if result.rejected else report_html.BLUE
    for index, count in enumerate(histogram):
        x = left + index * slot_width + 1
        y = y_position(count)
        elements.append(
            report_html.svg_rect(
                x, y, max(1.0, slot_width - 2), bottom - y, fill=color
            )
        )
    expected_y = y_position(expected)
    elements.append(
        report_html.svg_line(
            left,
            expected_y,
            right,
            expected_y,
            stroke=report_html.INK,
            dashed=True,
        )
    )
    elements.append(
        report_html.svg_text(
            right,
            expected_y - 4,
            f"expected {expected:.1f}",
            size=9,
            anchor="end",
        )
    )
    elements.extend(
        (
            report_html.svg_text(left, bottom + 18, "0", size=10, anchor="middle"),
            report_html.svg_text(right, bottom + 18, "1", size=10, anchor="middle"),
            report_html.svg_text(right, height - 5, "rank quantile", size=10, anchor="end"),
        )
    )
    return report_html.svg_figure(
        width,
        height,
        f"Rank histogram for {series.scenario} / {series.parameter}",
        elements,
    )


def _ecdf_values(series: RankSeries) -> list[tuple[float, float, int]]:
    histogram = [0] * (series.support_max + 1)
    for rank in series.ranks:
        histogram[rank] += 1
    values: list[tuple[float, float, int]] = []
    cumulative = 0
    support_size = series.support_max + 1
    for grid_index in range(series.support_max):
        cumulative += histogram[grid_index]
        t = (grid_index + 1) / support_size
        values.append((t, cumulative / len(series.ranks) - t, cumulative))
    return values


def _svg_polyline(points: Sequence[tuple[float, float]]) -> str:
    return "M " + " L ".join(f"{x:.6g} {y:.6g}" for x, y in points)


def _ecdf_figure(
    series: Sequence[RankSeries],
    results: Sequence[UniformityResult],
    *,
    alpha: float,
    simulations: int,
) -> str:
    members = list(zip(series, results))
    support_keys = {
        (len(rank_series.ranks), rank_series.support_max)
        for rank_series, _ in members
    }
    panels = [members] if len(support_keys) == 1 else [[member] for member in members]

    width = 1000
    panel_height = 275
    height = panel_height * len(panels) + 20
    elements: list[str] = []
    alpha_parameter = alpha / len(series)
    for panel_index, panel_members in enumerate(panels):
        sample_count = len(panel_members[0][0].ranks)
        support_max = panel_members[0][0].support_max
        panel_top = panel_index * panel_height
        plot_top = panel_top + 48
        plot_bottom = panel_top + 225
        left = 72
        right = 965
        bounds = accepted_count_bounds(
            sample_count,
            support_max,
            alpha_parameter,
            simulations=simulations,
        )
        support_size = support_max + 1
        envelope = [
            (
                (index + 1) / support_size,
                lower / sample_count - (index + 1) / support_size,
                upper / sample_count - (index + 1) / support_size,
            )
            for index, (lower, upper) in enumerate(bounds)
        ]
        curve_values = [(item, _ecdf_values(item)) for item, _ in panel_members]
        extent_values = [value for _, lower, upper in envelope for value in (lower, upper)]
        extent_values.extend(value for _, values in curve_values for _, value, _ in values)
        y_limit = max(0.05, math.ceil(max(abs(value) for value in extent_values) * 20) / 20)

        def x_position(value: float) -> float:
            return left + value * (right - left)

        def y_position(value: float) -> float:
            return plot_bottom - (value + y_limit) / (2 * y_limit) * (plot_bottom - plot_top)

        elements.append(
            report_html.svg_text(
                12,
                panel_top + 20,
                f"ECDF deviation envelope (N={sample_count}, rank support 0..{support_max})",
                size=13,
                weight="bold",
            )
        )
        elements.append(report_html.svg_axes(left, plot_bottom, right, plot_top))
        zero_y = y_position(0.0)
        elements.append(
            report_html.svg_line(left, zero_y, right, zero_y, stroke=report_html.GRID)
        )
        upper_points = [(x_position(t), y_position(upper)) for t, _, upper in envelope]
        lower_points = [(x_position(t), y_position(lower)) for t, lower, _ in reversed(envelope)]
        band_path = _svg_polyline(upper_points + lower_points) + " Z"
        elements.append(
            report_html.svg_path(
                band_path,
                stroke=report_html.GRID,
                fill=report_html.GRID,
                opacity=0.75,
            )
        )
        for member_index, ((rank_series, result), (_, values)) in enumerate(
            zip(panel_members, curve_values)
        ):
            color = report_html.RED if result.rejected else report_html.BLUE
            curve_points = [(x_position(t), y_position(value)) for t, value, _ in values]
            elements.append(
                report_html.svg_path(
                    _svg_polyline(curve_points), stroke=color, width=1.8
                )
            )
            elements.append(
                report_html.svg_text(
                    left + member_index * 175,
                    panel_top + 36,
                    f"{rank_series.scenario} / {rank_series.parameter}",
                    size=9,
                    fill=color,
                )
            )
            if result.rejected:
                for grid_index, ((t, value, cumulative), (lower, upper)) in enumerate(
                    zip(values, bounds)
                ):
                    if cumulative < lower or cumulative > upper:
                        marker_x = x_position(t)
                        marker_y = y_position(value)
                        elements.append(
                            report_html.svg_circle(
                                marker_x, marker_y, 4.5, fill=report_html.RED
                            )
                        )
                        elements.append(
                            report_html.svg_text(
                                min(right - 4, marker_x + 7),
                                marker_y - 6,
                                f"{rank_series.scenario} / {rank_series.parameter} exits band",
                                size=9,
                                fill=report_html.RED,
                                anchor="end" if marker_x > right - 250 else "start",
                            )
                        )
                        break
        elements.extend(
            (
                report_html.svg_text(left, plot_bottom + 17, "0", size=10, anchor="middle"),
                report_html.svg_text(right, plot_bottom + 17, "1", size=10, anchor="middle"),
                report_html.svg_text(right, plot_bottom + 17, "rank quantile t", size=10, anchor="end"),
                report_html.svg_text(left - 8, plot_top + 4, f"+{y_limit:.2f}", size=9, anchor="end"),
                report_html.svg_text(left - 8, plot_bottom, f"-{y_limit:.2f}", size=9, anchor="end"),
            )
        )
    return report_html.svg_figure(
        width, height, "Calibrated ECDF deviation envelopes", elements
    )


def render_sbc_report(
    series: Sequence[RankSeries],
    results: Sequence[UniformityResult],
    *,
    selections: Sequence[ThinningSelection],
    scenario_count: int,
    replicates: int,
    draws: int,
    warmup: int,
    chains: int,
    seed: int,
    alpha: float,
    simulations: int = MONTE_CARLO_SETS,
) -> str:
    """Render the deterministic, self-contained G11 statistical report."""
    if not series:
        raise ValueError("at least one rank series is required")
    if len(series) != len(results):
        raise ValueError("rank series and uniformity results must have equal lengths")
    passed = not any(result.rejected for result in results)
    settings = (
        ("scenarios", scenario_count),
        ("replicates N", replicates),
        ("draws", draws),
        ("warmup", warmup),
        ("chains", chains),
        ("seed", seed),
        ("alpha", alpha),
        ("calibration set count", simulations),
        ("overall", "PASS" if passed else "FAIL"),
    )
    thinning_rows = []
    for selection in selections:
        thinning_rows.append(
            (
                (
                    selection.scenario,
                    selection.thin,
                    "not run"
                    if selection.ess_stat is None
                    else f"{selection.ess_stat:.6g}",
                    "not run"
                    if selection.tau_hat is None
                    else f"{selection.tau_hat:.6g}",
                    f"[{selection.main_seed_span[0]}, {selection.main_seed_span[1]}]",
                    "not run"
                    if selection.pilot_seed_span is None
                    else f"[{selection.pilot_seed_span[0]}, {selection.pilot_seed_span[1]}]",
                ),
                False,
            )
        )
    sections = [
        report_html.section_heading("ESS-adaptive rank thinning"),
        report_html.data_table(
            ("Scenario", "Thin", "ESS statistic", "Tau estimate", "Main seed span", "Pilot seed span"),
            thinning_rows,
        ),
        report_html.section_heading("Rank histograms"),
    ]
    sections.extend(
        _histogram_figure(rank_series, result)
        for rank_series, result in zip(series, results)
    )
    sections.extend(
        (
            report_html.section_heading("ECDF deviations"),
            _ecdf_figure(series, results, alpha=alpha, simulations=simulations),
            report_html.section_heading("Uniformity results"),
        )
    )
    rows = [
        (
            (
                result.scenario,
                result.parameter,
                result.sample_count,
                f"{result.min_band_margin:.4f}",
                "FAIL" if result.rejected else "PASS",
                result.diagnosis or "none",
            ),
            result.rejected,
        )
        for result in results
    ]
    sections.append(
        report_html.data_table(
            ("Scenario", "Parameter", "N", "Min band margin", "Verdict", "Diagnosis"),
            rows,
        )
    )
    return report_html.html_document(
        "G11 SBC rank uniformity", settings, passed, sections
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayesite-bin", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--replicates", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20240713)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--thin", default="auto")
    parser.add_argument("--pilot-replicates", type=int, default=16)
    parser.add_argument("--ess-safety", type=float, default=3.0)
    parser.add_argument("--max-treedepth", type=int, default=5)
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        _validate_args(args)
        reports, selections = run_scenarios(args.bayesite_bin, args)
        _print_thinning(selections)
        series = rank_series_from_reports(reports, seed=args.seed)
        results = evaluate_rank_series(series, alpha=args.alpha)
    except ConformanceError as error:
        sys.exit(f"G11 SBC rank uniformity failed: {error}")

    if args.report is not None:
        args.report.write_text(
            render_sbc_report(
                series,
                results,
                selections=selections,
                scenario_count=len(SCENARIOS),
                replicates=args.replicates,
                draws=args.draws,
                warmup=args.warmup,
                chains=args.chains,
                seed=args.seed,
                alpha=args.alpha,
            ),
            encoding="utf-8",
        )

    failures = [result for result in results if result.rejected]
    if failures:
        for message in failure_messages(results):
            print(message, file=sys.stderr)
        sys.exit(f"{len(failures)} SBC parameter rank uniformity check(s) failed")
    _print_results(results)
    print(
        f"\nG11 SBC rank uniformity passed: {len(results)} parameter-coordinate "
        f"series, alpha {args.alpha}, {MONTE_CARLO_SETS} calibration sets"
    )


if __name__ == "__main__":
    main()
