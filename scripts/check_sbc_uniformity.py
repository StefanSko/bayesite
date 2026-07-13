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
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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


def parse_rank_facts(report: object, scenario: str, *, seed: int) -> list[RankSeries]:
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

    report_rank_draws = _require_int(report.get("rank_draws"), "rank_draws")
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


def _classify_deviation(above: list[int], below: list[int]) -> str:
    if above and not below:
        return "ranks skew low (biased estimates)"
    if below and not above:
        return "ranks skew high (biased estimates)"
    if sum(above) / len(above) < sum(below) / len(below):
        return "too many extreme ranks (posterior underdispersed)"
    return "too few extreme ranks (posterior overdispersed)"


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
    gamma = calibrated_gamma(
        sample_count, support_max, alpha, simulations=simulations
    )
    tables = _pointwise_tables(sample_count, support_max)
    histogram = [0] * (support_max + 1)
    for rank in ranks:
        histogram[rank] += 1

    above: list[int] = []
    below: list[int] = []
    min_margin_count = sample_count
    cumulative = 0
    for grid_index, table in enumerate(tables):
        cumulative += histogram[grid_index]
        accepted = [count for count, p_value in enumerate(table) if p_value >= gamma]
        lower = accepted[0]
        upper = accepted[-1]
        min_margin_count = min(min_margin_count, cumulative - lower, upper - cumulative)
        if cumulative > upper:
            above.append(grid_index)
        elif cumulative < lower:
            below.append(grid_index)
    rejected = bool(above or below)
    diagnosis = _classify_deviation(above, below) if rejected else None
    return rejected, min_margin_count / sample_count, diagnosis


def evaluate_reports(
    reports: Sequence[tuple[str, object]],
    *,
    alpha: float,
    seed: int,
    simulations: int = MONTE_CARLO_SETS,
) -> list[UniformityResult]:
    all_series: list[RankSeries] = []
    for scenario, report in reports:
        all_series.extend(parse_rank_facts(report, scenario, seed=seed))
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


def _load_scenario(spec: ScenarioSpec, args: argparse.Namespace, scenario_index: int) -> dict:
    try:
        scenario = json.loads(spec.scenario_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConformanceError(f"cannot read scenario {spec.scenario_path}: {error}") from error
    if not isinstance(scenario, dict):
        raise ConformanceError(f"{spec.scenario_path} must contain a JSON object")
    scenario["seed"] = args.seed + scenario_index * 1_000_000
    scenario["replicates"] = args.replicates
    scenario["sample"] = {
        "chains": args.chains,
        "warmup": args.warmup,
        "draws": args.draws,
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


def run_scenarios(binary: Path, args: argparse.Namespace) -> list[tuple[str, object]]:
    if not binary.is_file():
        raise ConformanceError(
            f"Bayesite release binary not found at {binary}; run cargo build --release "
            "--manifest-path crates/core/Cargo.toml"
        )
    reports: list[tuple[str, object]] = []
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
            scenario_path.write_text(
                json.dumps(_load_scenario(spec, args, scenario_index)), encoding="utf-8"
            )
            command = [
                str(binary),
                "sbc",
                "--model",
                str(model_path),
                "--scenario",
                str(scenario_path),
                "--replicates",
                str(args.replicates),
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
                    f"{spec.name}: bayesite sbc failed with exit code "
                    f"{completed.returncode}: {detail}"
                )
            if completed.stderr:
                raise ConformanceError(f"{spec.name}: successful bayesite sbc wrote stderr")
            try:
                report = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                raise ConformanceError(
                    f"{spec.name}: bayesite sbc stdout is not one JSON report: {error}"
                ) from error
            validate_requested_replicates(report, spec.name, args.replicates)
            reports.append((spec.name, report))
    return reports


def _validate_args(args: argparse.Namespace) -> None:
    if args.replicates < 1:
        raise ConformanceError("--replicates must be positive")
    if args.seed < 0 or args.seed + (len(SCENARIOS) - 1) * 1_000_000 + 2 * args.replicates > 2**63 - 1:
        raise ConformanceError("--seed must leave room for every deterministic SBC replicate seed")
    if not 0.0 < args.alpha < 1.0:
        raise ConformanceError("--alpha must be in (0, 1)")
    if args.chains < 1:
        raise ConformanceError("--chains must be positive")
    if args.warmup < 0:
        raise ConformanceError("--warmup must be non-negative")
    if args.draws < 4:
        raise ConformanceError("--draws must be at least 4 because SBC reports diagnostics")
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


def _print_results(results: Sequence[UniformityResult]) -> None:
    print(f"{'scenario':20s} {'parameter':18s} {'N':>5s} {'min band margin':>15s}")
    for result in results:
        print(
            f"{result.scenario:20s} {result.parameter:18s} "
            f"{result.sample_count:5d} {result.min_band_margin:15.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayesite-bin", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20240713)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--max-treedepth", type=int, default=5)
    parser.add_argument("--target-accept", type=float, default=0.8)
    args = parser.parse_args()

    try:
        _validate_args(args)
        reports = run_scenarios(args.bayesite_bin, args)
        results = evaluate_reports(reports, alpha=args.alpha, seed=args.seed)
    except ConformanceError as error:
        sys.exit(f"G11 SBC rank uniformity failed: {error}")

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
