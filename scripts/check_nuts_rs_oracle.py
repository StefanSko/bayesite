#!/usr/bin/env python3
"""Mandatory nuts-rs statistical oracle for Bayesite NUTS.

This gate keeps the Bayesite runtime dependency-free while checking sampler
behavior against an independently implemented NUTS backend. It uses Gaussian
analytic targets and compares Bayesite and nuts-rs summary estimates using
batch Monte Carlo standard errors (MCSE), not draw-by-draw equality or broad
fixed tolerances.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Iterator, Sequence

import gate_report as report_html

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_MANIFEST = REPO_ROOT / "crates" / "core" / "Cargo.toml"
RUNNER_SRC = REPO_ROOT / "tools" / "oracles" / "nuts-rs-runner" / "src" / "main.rs"
NUTS_RS_REV = REPO_ROOT / "NUTS_RS_REV"
EXPECTED_NUTS_RS_REV = NUTS_RS_REV.read_text(encoding="utf-8").strip()
MAX_Z = 5.0
COMBINED_MAX_Z = 4.0
MAX_RHAT = 1.05
MAX_DIVERGENCE_RATE = 0.01

DrawsByChain = list[list[list[float]]]
StatFn = Callable[[list[list[float]]], float]


@dataclass(frozen=True)
class Target:
    name: str
    mean: list[float]
    covariance: list[list[float]]
    model: dict
    data: dict


@dataclass(frozen=True)
class Estimate:
    value: float
    mcse: float
    batch_count: int


@dataclass(frozen=True)
class StatResult:
    target: str
    stat: str
    truth: float
    bayesite: Estimate
    nuts_rs: Estimate
    bayesite_truth_z: float
    nuts_rs_truth_z: float
    cross_z: float


@dataclass(frozen=True)
class AggregateResult:
    target: str
    stat: str
    comparison: str
    mean_delta: float
    stouffer_z: float
    advisory_t: float | None
    passed: bool


def signed_z(delta: float, scale: float) -> float:
    """Return a signed standardized delta, failing closed for invalid scales."""
    if not math.isfinite(scale) or scale <= 0.0:
        return math.copysign(float("inf"), delta)
    return delta / scale


def stouffer_z(z_scores: Sequence[float]) -> float:
    """Combine independent signed z-scores with Stouffer's method."""
    if not z_scores:
        raise ValueError("Stouffer combination requires at least one z-score")
    return sum(z_scores) / math.sqrt(len(z_scores))


def advisory_t_statistic(deltas: Sequence[float]) -> float | None:
    """Return the one-sample t-statistic, or None when variance is unavailable."""
    if len(deltas) < 2:
        return None
    delta_mean = mean(deltas)
    variance = sum((delta - delta_mean) ** 2 for delta in deltas) / (len(deltas) - 1)
    if variance == 0.0:
        return None
    return delta_mean / (math.sqrt(variance) / math.sqrt(len(deltas)))


def per_check_passes(z: float, max_z: float = MAX_Z) -> bool:
    return math.isfinite(z) and abs(z) <= max_z


def combined_passes(
    z_scores: Sequence[float], max_z: float = COMBINED_MAX_Z
) -> bool:
    """Apply the aggregate gate, leaving the K=1 verdict to the coarse guard."""
    if len(z_scores) < 2:
        return True
    combined = stouffer_z(z_scores)
    return math.isfinite(combined) and abs(combined) <= max_z


def _command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def _run_capture(label: str, command: Sequence[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    print(f"\n== {label}\n$ {_command_text(command)}", flush=True)
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        sys.exit(f"missing executable for {label}: {error.filename}")
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(f"{label} failed with exit code {result.returncode}")
    return result


def _node_const(value: float) -> dict:
    return {"node": "ConstNode", "value": value}


def _node_data(name: str) -> dict:
    return {"node": "DataRef", "name": name}


def _normal(loc: dict, scale: dict) -> dict:
    return {"node": "Normal", "loc": loc, "scale": scale}


def _mvn(mean_node: dict, scale_tril_node: dict) -> dict:
    return {"node": "MultivariateNormal", "mean": mean_node, "scale_tril": scale_tril_node}


def _resolved_param(distribution: dict, size: int | None) -> dict:
    return {
        "node": "ResolvedParam",
        "distribution": distribution,
        "constraint": None,
        "size": size,
    }


def _resolved_data_shape(dims: list[int]) -> dict:
    return {
        "node": "ResolvedData",
        "schema": {"node": "ResolvedDataShapeSchema", "dims": dims},
    }


def _model(params: list[tuple[str, dict]], data: list[tuple[str, dict]]) -> dict:
    return {
        "bayeswire_ir": 1,
        "model": {
            "node": "ModelMeta",
            "params": [{"name": name, "value": value} for name, value in params],
            "data": [{"name": name, "value": value} for name, value in data],
            "observed_nodes": [],
            "expressions": [],
            "free_values": [],
            "stochastic_sites": [],
        },
    }


def _targets() -> list[Target]:
    return [
        Target(
            name="scalar_standard",
            mean=[0.0],
            covariance=[[1.0]],
            model=_model(
                [("x", _resolved_param(_normal(_node_const(0.0), _node_const(1.0)), None))],
                [],
            ),
            data={},
        ),
        Target(
            name="shifted_scaled",
            mean=[2.0],
            covariance=[[9.0]],
            model=_model(
                [("x", _resolved_param(_normal(_node_const(2.0), _node_const(3.0)), None))],
                [],
            ),
            data={},
        ),
        Target(
            name="vector_diagonal",
            mean=[0.0, 2.0, -1.0],
            covariance=[[1.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 4.0]],
            model=_model(
                [
                    (
                        "x",
                        _resolved_param(
                            _normal(_node_data("loc"), _node_data("scale")),
                            3,
                        ),
                    )
                ],
                [
                    ("loc", _resolved_data_shape([3])),
                    ("scale", _resolved_data_shape([3])),
                ],
            ),
            data={"loc": [0.0, 2.0, -1.0], "scale": [1.0, 0.5, 2.0]},
        ),
        Target(
            name="correlated_mvn",
            mean=[1.0, -0.5],
            covariance=[[1.0, 0.6], [0.6, 1.0]],
            model=_model(
                [
                    (
                        "x",
                        _resolved_param(
                            _mvn(_node_data("mean"), _node_data("scale_tril")),
                            2,
                        ),
                    )
                ],
                [
                    ("mean", _resolved_data_shape([2])),
                    ("scale_tril", _resolved_data_shape([2, 2])),
                ],
            ),
            data={"mean": [1.0, -0.5], "scale_tril": [[1.0, 0.0], [0.6, 0.8]]},
        ),
    ]


def build_bayesite() -> Path:
    if shutil.which("cargo") is None:
        sys.exit("cargo is required for the nuts-rs oracle gate")
    _run_capture(
        "build Bayesite release binary for nuts-rs oracle",
        ["cargo", "build", "--release", "--bin", "bayesite", "--manifest-path", str(CORE_MANIFEST)],
    )
    binary_name = "bayesite.exe" if sys.platform.startswith("win") else "bayesite"
    return REPO_ROOT / "target" / "release" / binary_name


def _parse_value_vector(value: object) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


def run_bayesite(
    binary: Path,
    target: Target,
    *,
    seed: int,
    chains: int,
    warmup: int,
    draws: int,
) -> tuple[DrawsByChain, int]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model_path = tmp_path / "model.json"
        data_path = tmp_path / "data.json"
        model_path.write_text(json.dumps(target.model), encoding="utf-8")
        data_path.write_text(json.dumps(target.data), encoding="utf-8")
        result = _run_capture(
            f"Bayesite sample oracle target {target.name}",
            [
                str(binary),
                "sample",
                "--model",
                str(model_path),
                "--data",
                str(data_path),
                "--seed",
                str(seed),
                "--chains",
                str(chains),
                "--warmup",
                str(warmup),
                "--draws",
                str(draws),
            ],
        )
    by_chain: DrawsByChain = [[] for _ in range(chains)]
    lines = result.stdout.splitlines()
    for line in lines[1:-1]:
        payload = json.loads(line)
        chain = int(payload["chain"])
        by_chain[chain].append(_parse_value_vector(payload["values"]["x"]))
    trailer = json.loads(lines[-1])["trailer"]
    divergences = sum(int(chain["divergences"]) for chain in trailer["chains"])
    for chain_index, chain_draws in enumerate(by_chain):
        if len(chain_draws) != draws:
            sys.exit(
                f"Bayesite {target.name} chain {chain_index} emitted {len(chain_draws)} draws, expected {draws}"
            )
    return by_chain, divergences


def check_nuts_rs_path(path: Path) -> None:
    if not path.exists():
        sys.exit(
            f"NUTS cross-engine oracle (nuts-rs) requires a nuts-rs checkout at {path}. "
            "Clone with: git clone https://github.com/pymc-devs/nuts-rs /tmp/nuts-rs"
        )
    if not (path / "Cargo.toml").exists():
        sys.exit(
            f"NUTS cross-engine oracle (nuts-rs) path {path} does not look "
            "like a nuts-rs checkout"
        )
    result = _run_capture("verify pinned nuts-rs revision", ["git", "-C", str(path), "rev-parse", "HEAD"])
    rev = result.stdout.strip()
    if rev != EXPECTED_NUTS_RS_REV:
        sys.exit(
            "NUTS cross-engine oracle (nuts-rs) expected nuts-rs revision "
            f"{EXPECTED_NUTS_RS_REV}, got {rev}. "
            f"Run: git -C {path} fetch && git -C {path} checkout {EXPECTED_NUTS_RS_REV}"
        )


@contextmanager
def nuts_rs_runner_project(nuts_rs_path: Path) -> Iterator[Path]:
    """Materialize the oracle runner cargo project once for all replicates."""
    check_nuts_rs_path(nuts_rs_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        cargo_config_dir = tmp_path / ".cargo"
        cargo_config_dir.mkdir()
        (cargo_config_dir / "config.toml").write_text(
            '[build]\nrustflags = ["-Awarnings"]\n', encoding="utf-8"
        )
        (src_dir / "main.rs").write_text(RUNNER_SRC.read_text(encoding="utf-8"), encoding="utf-8")
        cargo_toml = f"""
[workspace]

[package]
name = "bayesite-nuts-rs-oracle-runner"
version = "0.0.0"
edition = "2024"

[dependencies]
nuts-rs = {{ path = {json.dumps(str(nuts_rs_path))}, default-features = false }}
""".strip()
        (tmp_path / "Cargo.toml").write_text(cargo_toml + "\n", encoding="utf-8")
        _run_capture(
            "build nuts-rs oracle runner",
            [
                "cargo",
                "build",
                "--release",
                "--quiet",
                "--manifest-path",
                str(tmp_path / "Cargo.toml"),
            ],
            cwd=tmp_path,
        )
        yield tmp_path


def run_nuts_rs_runner(
    runner_project: Path,
    targets: list[Target],
    *,
    seed: int,
    chains: int,
    warmup: int,
    draws: int,
) -> tuple[dict[str, DrawsByChain], dict[str, int]]:
    result = _run_capture(
        "run nuts-rs oracle runner",
        [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--manifest-path",
            str(runner_project / "Cargo.toml"),
            "--",
            "--targets",
            ",".join(target.name for target in targets),
            "--seed",
            str(seed),
            "--chains",
            str(chains),
            "--warmup",
            str(warmup),
            "--draws",
            str(draws),
        ],
        cwd=runner_project,
    )
    payload = json.loads(result.stdout)
    by_target: dict[str, DrawsByChain] = {}
    divergences: dict[str, int] = {}
    for target_doc in payload["targets"]:
        name = target_doc["name"]
        target_chains: DrawsByChain = []
        total_divergences = 0
        for chain_doc in target_doc["chains"]:
            total_divergences += int(chain_doc["divergences"])
            target_chains.append([[float(x) for x in draw] for draw in chain_doc["draws"]])
        by_target[name] = target_chains
        divergences[name] = total_divergences
    return by_target, divergences


def flatten(draws_by_chain: DrawsByChain) -> list[list[float]]:
    return [draw for chain in draws_by_chain for draw in chain]


def sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    return sum((value - m) ** 2 for value in values) / (len(values) - 1)


def sample_covariance(draws: list[list[float]], i: int, j: int) -> float:
    if len(draws) < 2:
        return float("nan")
    mean_i = sum(draw[i] for draw in draws) / len(draws)
    mean_j = sum(draw[j] for draw in draws) / len(draws)
    return sum((draw[i] - mean_i) * (draw[j] - mean_j) for draw in draws) / (len(draws) - 1)


def stat_mean(dim: int) -> StatFn:
    return lambda draws: sum(draw[dim] for draw in draws) / len(draws)


def stat_variance(dim: int) -> StatFn:
    return lambda draws: sample_variance([draw[dim] for draw in draws])


def stat_covariance(i: int, j: int) -> StatFn:
    return lambda draws: sample_covariance(draws, i, j)


def estimate(draws_by_chain: DrawsByChain, stat: StatFn, batches_per_chain: int) -> Estimate:
    all_draws = flatten(draws_by_chain)
    value = stat(all_draws)
    batch_values: list[float] = []
    for chain in draws_by_chain:
        batch_size = len(chain) // batches_per_chain
        if batch_size < 2:
            sys.exit("not enough draws per batch to estimate MCSE")
        for batch_index in range(batches_per_chain):
            start = batch_index * batch_size
            end = start + batch_size
            batch_values.append(stat(chain[start:end]))
    batch_var = sample_variance(batch_values)
    mcse = math.sqrt(batch_var / len(batch_values))
    return Estimate(value=value, mcse=mcse, batch_count=len(batch_values))


def split_rhat(draws_by_chain: DrawsByChain, dim: int) -> float:
    split_chains: list[list[float]] = []
    for chain in draws_by_chain:
        half = len(chain) // 2
        split_chains.append([draw[dim] for draw in chain[:half]])
        split_chains.append([draw[dim] for draw in chain[half : 2 * half]])
    n = len(split_chains[0])
    means = [mean(chain) for chain in split_chains]
    variances = [sample_variance(chain) for chain in split_chains]
    within = mean(variances)
    between = n * sample_variance(means)
    if within <= 0.0:
        return float("inf")
    var_hat = ((n - 1) / n) * within + between / n
    return math.sqrt(var_hat / within)


def stat_plan(target: Target) -> list[tuple[str, float, StatFn]]:
    plan: list[tuple[str, float, StatFn]] = []
    for dim, true_mean in enumerate(target.mean):
        plan.append((f"mean[{dim}]", true_mean, stat_mean(dim)))
        plan.append((f"var[{dim}]", target.covariance[dim][dim], stat_variance(dim)))
    if target.name == "correlated_mvn":
        plan.append(("cov[0,1]", target.covariance[0][1], stat_covariance(0, 1)))
    return plan


def validate_target(
    target: Target,
    bayesite_draws: DrawsByChain,
    nuts_rs_draws: DrawsByChain,
    bayesite_divergences: int,
    nuts_rs_divergences: int,
    *,
    draws: int,
    chains: int,
    batches_per_chain: int,
) -> tuple[list[StatResult], list[str]]:
    failures: list[str] = []
    total_draws = draws * chains
    for label, count in [("Bayesite", bayesite_divergences), ("nuts-rs", nuts_rs_divergences)]:
        rate = count / total_draws
        if rate > MAX_DIVERGENCE_RATE:
            failures.append(
                f"{target.name} {label} divergence rate {count}/{total_draws} = {rate:.4f} > {MAX_DIVERGENCE_RATE}"
            )
    for dim in range(len(target.mean)):
        for label, draws_by_chain in [("Bayesite", bayesite_draws), ("nuts-rs", nuts_rs_draws)]:
            rhat = split_rhat(draws_by_chain, dim)
            if not math.isfinite(rhat) or rhat > MAX_RHAT:
                failures.append(f"{target.name} {label} dim {dim} split R-hat {rhat:.4f} > {MAX_RHAT}")

    results: list[StatResult] = []
    for stat_name, truth, stat in stat_plan(target):
        bayes = estimate(bayesite_draws, stat, batches_per_chain)
        nuts = estimate(nuts_rs_draws, stat, batches_per_chain)
        bayes_truth_z = signed_z(bayes.value - truth, bayes.mcse)
        nuts_truth_z = signed_z(nuts.value - truth, nuts.mcse)
        cross_z = signed_z(bayes.value - nuts.value, math.hypot(bayes.mcse, nuts.mcse))
        result = StatResult(
            target=target.name,
            stat=stat_name,
            truth=truth,
            bayesite=bayes,
            nuts_rs=nuts,
            bayesite_truth_z=bayes_truth_z,
            nuts_rs_truth_z=nuts_truth_z,
            cross_z=cross_z,
        )
        results.append(result)
        if not per_check_passes(bayes_truth_z):
            failures.append(
                f"{target.name} {stat_name} Bayesite truth z {bayes_truth_z:+.2f}, |z| > {MAX_Z}"
            )
        if not per_check_passes(nuts_truth_z):
            failures.append(
                f"{target.name} {stat_name} nuts-rs truth z {nuts_truth_z:+.2f}, |z| > {MAX_Z}"
            )
        if not per_check_passes(cross_z):
            failures.append(
                f"{target.name} {stat_name} cross z {cross_z:+.2f}, |z| > {MAX_Z}"
            )
    return results, failures


def print_target_diagnostics(
    target: Target,
    bayesite_draws: DrawsByChain,
    nuts_rs_draws: DrawsByChain,
    bayesite_divergences: int,
    nuts_rs_divergences: int,
    total_draws: int,
) -> None:
    bayes_rhats = [split_rhat(bayesite_draws, dim) for dim in range(len(target.mean))]
    nuts_rhats = [split_rhat(nuts_rs_draws, dim) for dim in range(len(target.mean))]
    print(
        f"target {target.name}: divergences Bayesite {bayesite_divergences}/{total_draws}, "
        f"nuts-rs {nuts_rs_divergences}/{total_draws}; "
        f"max split R-hat Bayesite {max(bayes_rhats):.4f}, nuts-rs {max(nuts_rhats):.4f}"
    )


def print_stat_result(result: StatResult) -> None:
    status = "ok"
    if not all(
        per_check_passes(z)
        for z in (result.bayesite_truth_z, result.nuts_rs_truth_z, result.cross_z)
    ):
        status = "FAIL"
    print(
        f"{status:4s} {result.target:16s} {result.stat:9s} "
        f"truth {result.truth:9.4f} | "
        f"Bayesite {result.bayesite.value:10.4f} ± {result.bayesite.mcse:8.4f} "
        f"z_truth {result.bayesite_truth_z:+6.2f} | "
        f"nuts-rs {result.nuts_rs.value:10.4f} ± {result.nuts_rs.mcse:8.4f} "
        f"z_truth {result.nuts_rs_truth_z:+6.2f} | "
        f"cross_z {result.cross_z:+6.2f}"
    )


def stat_comparisons(result: StatResult) -> list[tuple[str, float, float]]:
    return [
        ("bayesite-truth", result.bayesite.value - result.truth, result.bayesite_truth_z),
        ("nuts-rs-truth", result.nuts_rs.value - result.truth, result.nuts_rs_truth_z),
        (
            "bayesite-nuts-rs",
            result.bayesite.value - result.nuts_rs.value,
            result.cross_z,
        ),
    ]


def aggregate_stat_results(results: Sequence[StatResult]) -> list[AggregateResult]:
    grouped: dict[tuple[str, str, str], tuple[list[float], list[float]]] = {}
    for result in results:
        for comparison, delta, z in stat_comparisons(result):
            deltas, z_scores = grouped.setdefault(
                (result.target, result.stat, comparison), ([], [])
            )
            deltas.append(delta)
            z_scores.append(z)

    aggregates: list[AggregateResult] = []
    for (target, stat, comparison), (deltas, z_scores) in grouped.items():
        aggregates.append(
            AggregateResult(
                target=target,
                stat=stat,
                comparison=comparison,
                mean_delta=mean(deltas),
                stouffer_z=stouffer_z(z_scores),
                advisory_t=advisory_t_statistic(deltas),
                passed=combined_passes(z_scores),
            )
        )
    return aggregates


def print_aggregate_results(results: Sequence[AggregateResult]) -> None:
    print("\naggregate seed-replicated comparisons:")
    print(
        f"{'target':16s} {'statistic':9s} {'comparison':20s} "
        f"{'mean delta':>12s} {'Stouffer Z':>11s} {'advisory t':>11s} verdict"
    )
    for result in results:
        advisory = "n/a" if result.advisory_t is None else f"{result.advisory_t:+.3f}"
        verdict = "ok" if result.passed else "FAIL"
        print(
            f"{result.target:16s} {result.stat:9s} {result.comparison:20s} "
            f"{result.mean_delta:+12.5g} {result.stouffer_z:+11.3f} "
            f"{advisory:>11s} {verdict}"
        )


def _comparison_z(result: StatResult, comparison: str) -> float:
    return {
        "bayesite-truth": result.bayesite_truth_z,
        "nuts-rs-truth": result.nuts_rs_truth_z,
        "bayesite-nuts-rs": result.cross_z,
    }[comparison]


def _oracle_dot_plot(
    comparison: str,
    results: Sequence[StatResult],
    aggregates: Sequence[AggregateResult],
    replicates: int,
) -> str:
    labels: list[tuple[str, str]] = []
    grouped: dict[tuple[str, str], list[float]] = {}
    for result in results:
        key = (result.target, result.stat)
        if key not in grouped:
            labels.append(key)
            grouped[key] = []
        grouped[key].append(_comparison_z(result, comparison))
    aggregate_by_key = {
        (result.target, result.stat): result
        for result in aggregates
        if result.comparison == comparison
    }

    width = 1000
    left = 230
    right = 970
    top = 72
    row_height = 28
    bottom = top + row_height * len(labels)
    x_min = -6.5
    x_max = 6.5

    def x_position(value: float) -> float:
        clipped = max(x_min, min(x_max, value))
        return left + (clipped - x_min) / (x_max - x_min) * (right - left)

    elements: list[str] = []
    for row_index, key in enumerate(labels):
        y = top + row_index * row_height
        aggregate = aggregate_by_key.get(key)
        if aggregate is not None and not aggregate.passed:
            elements.append(
                report_html.svg_rect(0, y, width, row_height, fill="#fde9e8")
            )
        elements.append(
            report_html.svg_line(left, y + row_height, right, y + row_height, stroke=report_html.GRID)
        )
        elements.append(
            report_html.svg_text(8, y + 18, f"{key[0]} / {key[1]}", size=11)
        )

    for value, color, dashed, label in (
        (-5.0, report_html.RED, True, "per-seed limit"),
        (-4.0, report_html.RED, True, "combined gate"),
        (0.0, report_html.GRID, False, "0"),
        (4.0, report_html.RED, True, "combined gate"),
        (5.0, report_html.RED, True, "per-seed limit"),
    ):
        x = x_position(value)
        elements.append(
            report_html.svg_line(x, top, x, bottom, stroke=color, dashed=dashed)
        )
        elements.append(
            report_html.svg_text(x, 44 if abs(value) == 5 else 58, label, size=9, fill=color, anchor="middle")
        )
        elements.append(report_html.svg_text(x, bottom + 18, f"{value:+g}", size=10, anchor="middle"))

    for row_index, key in enumerate(labels):
        center_y = top + row_index * row_height + row_height / 2
        z_scores = grouped[key]
        jitter_step = min(3.0, 12.0 / max(1, len(z_scores)))
        for seed_index, z_score in enumerate(z_scores):
            y = center_y + (seed_index - (len(z_scores) - 1) / 2) * jitter_step
            finite = math.isfinite(z_score)
            x = x_position(z_score if finite else math.copysign(x_max, z_score))
            color = report_html.BLUE if per_check_passes(z_score) else report_html.RED
            elements.append(report_html.svg_circle(x, y, 3.2, fill=color))
            if not finite or z_score < x_min or z_score > x_max:
                marker = ">" if z_score > 0 else "<"
                marker_x = x - 7 if z_score > 0 else x + 7
                anchor = "end" if z_score > 0 else "start"
                elements.append(
                    report_html.svg_text(marker_x, y + 3, marker, size=10, fill=color, anchor=anchor, weight="bold")
                )
        aggregate = aggregate_by_key.get(key)
        if replicates >= 2 and aggregate is not None:
            color = report_html.BLUE if aggregate.passed else report_html.RED
            elements.append(
                report_html.svg_diamond(
                    x_position(aggregate.stouffer_z), center_y, 5.5, fill=color
                )
            )

    elements.append(report_html.svg_text(left, 18, comparison, size=15, weight="bold"))
    elements.append(
        report_html.svg_text(right, bottom + 18, "signed z", size=10, anchor="end")
    )
    return report_html.svg_figure(
        width, bottom + 32, f"{comparison} signed z dot plot", elements
    )


def render_oracle_report(
    results: Sequence[StatResult],
    aggregates: Sequence[AggregateResult],
    *,
    target_count: int,
    replicates: int,
    draws: int,
    warmup: int,
    chains: int,
    seed: int,
    passed: bool,
) -> str:
    """Render the deterministic, self-contained cross-engine report."""
    settings = (
        ("targets count", target_count),
        ("replicates K", replicates),
        ("draws", draws),
        ("warmup", warmup),
        ("chains", chains),
        ("base seed", seed),
        ("per-check threshold", MAX_Z),
        ("combined threshold", COMBINED_MAX_Z),
        ("overall", "PASS" if passed else "FAIL"),
    )
    sections = [report_html.section_heading("Signed z comparisons")]
    for comparison in ("bayesite-truth", "nuts-rs-truth", "bayesite-nuts-rs"):
        sections.append(_oracle_dot_plot(comparison, results, aggregates, replicates))
    sections.append(report_html.section_heading("Aggregate comparisons"))
    if replicates < 2:
        sections.append('<p class="note">aggregation disabled for K=1</p>')
    else:
        rows = []
        for result in aggregates:
            advisory = "n/a" if result.advisory_t is None else f"{result.advisory_t:+.3f}"
            rows.append(
                (
                    (
                        result.target,
                        result.stat,
                        result.comparison,
                        f"{result.mean_delta:+.5g}",
                        f"{result.stouffer_z:+.3f}",
                        advisory,
                        "PASS" if result.passed else "FAIL",
                    ),
                    not result.passed,
                )
            )
        sections.append(
            report_html.data_table(
                (
                    "Target",
                    "Statistic",
                    "Comparison",
                    "Mean delta",
                    "Stouffer Z",
                    "Advisory t",
                    "Verdict",
                ),
                rows,
            )
        )
    return report_html.html_document(
        "nuts-rs cross-engine statistical oracle", settings, passed, sections
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuts-rs-path", type=Path, default=Path("/tmp/nuts-rs"))
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20240621)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--batches-per-chain", type=int, default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.replicates < 1:
        sys.exit("nuts-rs oracle requires --replicates >= 1")
    if args.chains < 2:
        sys.exit("nuts-rs oracle requires --chains >= 2 for split R-hat")
    if args.draws < args.batches_per_chain * 2:
        sys.exit("nuts-rs oracle requires at least two draws per batch")

    targets = _targets()
    binary = build_bayesite()
    all_results: list[StatResult] = []
    all_failures: list[str] = []
    total_draws = args.draws * args.chains

    with nuts_rs_runner_project(args.nuts_rs_path) as runner_project:
        for replicate in range(args.replicates):
            seed = args.seed + replicate
            print(f"\n== replicate {replicate + 1}/{args.replicates}, seed {seed}", flush=True)
            nuts_draws, nuts_divergences = run_nuts_rs_runner(
                runner_project,
                targets,
                seed=seed,
                chains=args.chains,
                warmup=args.warmup,
                draws=args.draws,
            )
            for target in targets:
                bayes_draws, bayes_divergences = run_bayesite(
                    binary,
                    target,
                    seed=seed,
                    chains=args.chains,
                    warmup=args.warmup,
                    draws=args.draws,
                )
                print_target_diagnostics(
                    target,
                    bayes_draws,
                    nuts_draws[target.name],
                    bayes_divergences,
                    nuts_divergences[target.name],
                    total_draws,
                )
                results, failures = validate_target(
                    target,
                    bayes_draws,
                    nuts_draws[target.name],
                    bayes_divergences,
                    nuts_divergences[target.name],
                    draws=args.draws,
                    chains=args.chains,
                    batches_per_chain=args.batches_per_chain,
                )
                for result in results:
                    print_stat_result(result)
                all_results.extend(results)
                all_failures.extend(f"seed {seed}: {failure}" for failure in failures)

    per_check_total = 3 * len(all_results)
    per_check_failed = sum(
        not per_check_passes(z)
        for result in all_results
        for _, _, z in stat_comparisons(result)
    )
    diagnostic_failures = len(all_failures) - per_check_failed

    aggregate_results: list[AggregateResult] = []
    if args.replicates >= 2:
        aggregate_results = aggregate_stat_results(all_results)
        print_aggregate_results(aggregate_results)
        for result in aggregate_results:
            if not result.passed:
                all_failures.append(
                    f"{result.target} {result.stat} {result.comparison} Stouffer "
                    f"Z {result.stouffer_z:+.3f}, |Z| > {COMBINED_MAX_Z}"
                )

    aggregate_failed = sum(not result.passed for result in aggregate_results)
    aggregate_total = len(aggregate_results)
    if all_failures:
        print("\nstatistical oracle failures:", file=sys.stderr)
        for failure in all_failures:
            print(f"- {failure}", file=sys.stderr)

    status = "PASSED" if not all_failures else "FAILED"
    combined_summary = (
        f"combined |Z| threshold {COMBINED_MAX_Z}, pass/fail "
        f"{aggregate_total - aggregate_failed}/{aggregate_failed}"
        if args.replicates >= 2
        else f"combined |Z| threshold {COMBINED_MAX_Z} disabled for K=1, pass/fail 0/0"
    )
    print(
        f"\nnuts-rs statistical oracle {status}: {len(targets)} targets, "
        f"replicates {args.replicates}, per-check |z| threshold {MAX_Z}, pass/fail "
        f"{per_check_total - per_check_failed}/{per_check_failed}; {combined_summary}; "
        f"diagnostic failures {diagnostic_failures}; draws {args.draws} x chains {args.chains}",
        flush=True,
    )
    if args.report is not None:
        args.report.write_text(
            render_oracle_report(
                all_results,
                aggregate_results,
                target_count=len(targets),
                replicates=args.replicates,
                draws=args.draws,
                warmup=args.warmup,
                chains=args.chains,
                seed=args.seed,
                passed=not all_failures,
            ),
            encoding="utf-8",
        )
    if all_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
