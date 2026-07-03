#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blackjax>=1.2.0",
#   "jax>=0.6.0",
# ]
# ///
"""Differential posterior check: Rust backend vs JAX backend, golden corpus.

Builds ``bayesite`` (cargo required), then for every golden-corpus fixture
samples the posterior with both backends and compares per-parameter
posterior means and standard deviations within Monte Carlo tolerance,
plus convergence gates (R-hat, ESS, divergence rate) on the Rust run.

Run from the repository root:

    uv run scripts/check_rust_backend_posterior.py \
        --jaxstanv5-path ../jaxstanv5 [--draws N] [--warmup N]

Exits nonzero on the first failed comparison; output is one line per
parameter so failures are attributable.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _bootstrap_jaxstanv5_path() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--jaxstanv5-path", type=Path, default=None)
    args, _ = parser.parse_known_args()
    if args.jaxstanv5_path is None:
        return
    source_path = args.jaxstanv5_path / "src"
    sys.path.insert(0, str(source_path if source_path.exists() else args.jaxstanv5_path))


_bootstrap_jaxstanv5_path()

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent

from jaxstanv5.inference import sample  # noqa: E402
from bayeswire.ir import bindable_from_meta, meta_from_dict  # noqa: E402
from jaxstanv5.model import bind_model  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "golden_ir" / "fixtures"
CRATE_DIR = REPO_ROOT / "crates" / "core"

MAX_RHAT = 1.05
MIN_ESS = 100.0
MAX_DIVERGENCE_RATE = 0.02
# Each backend's posterior mean carries MC error ~ sd/sqrt(ESS); with ESS
# in the hundreds on both sides, 0.25 sd separates real bias from noise.
MEAN_TOLERANCE_SDS = 0.25

# Corpus fixtures whose models use Truncated, which this backend does not
# evaluate yet; mirrored from crates/core/tests/fixtures_eval.rs.
UNSUPPORTED_TRUNCATED_FIXTURES = ("bounded_rates", "linear_regression")
SD_RATIO_RANGE = (0.7, 1.4)


def build_bayesite() -> Path:
    if shutil.which("cargo") is None:
        sys.exit("cargo is required to build the Rust backend")
    result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "bayesite"],
        cwd=CRATE_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"cargo build failed:\n{result.stderr}")
    return REPO_ROOT / "target" / "release" / "bayesite"


def run_rust_backend(
    binary: Path,
    fixture: dict,
    *,
    seed: int,
    chains: int,
    warmup: int,
    draws: int,
) -> tuple[dict[str, list[float]], dict]:
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.json"
        data_path = Path(tmp) / "data.json"
        model_path.write_text(json.dumps(fixture["ir"]))
        data_path.write_text(json.dumps(fixture["data"]))
        result = subprocess.run(
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
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        sys.exit(f"bayesite failed on {fixture['name']}: {result.stderr}")
    lines = result.stdout.splitlines()
    trailer = json.loads(lines[-1])["trailer"]
    flat: dict[str, list[float]] = {}
    for line in lines[1:-1]:
        draw = json.loads(line)
        for name, value in draw["values"].items():
            values = value if isinstance(value, list) else [value]
            for coord, scalar in enumerate(values):
                flat.setdefault(f"{name}[{coord}]", []).append(scalar)
    return flat, trailer


def run_jax_backend(
    fixture: dict,
    *,
    seed: int,
    chains: int,
    warmup: int,
    draws: int,
) -> dict[str, list[float]]:
    meta = meta_from_dict(fixture["ir"])
    rebuilt = bindable_from_meta(meta)
    bind_values = {
        name: (
            jnp.asarray(spec["values"]).reshape(spec["shape"])
            if spec["shape"]
            else spec["values"][0]
        )
        for name, spec in fixture["data"].items()
    }
    bound = bind_model(rebuilt, bind_values)
    result = sample(
        bound,
        seed=seed,
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
    )
    flat: dict[str, list[float]] = {}
    for name, values in result.samples.items():
        arr = jnp.asarray(values)
        if arr.ndim == 2:
            arr = arr[..., None]
        coords = arr.shape[-1]
        for coord in range(coords):
            flat[f"{name}[{coord}]"] = [float(v) for v in arr[..., coord].reshape(-1)]
    return flat


def summarize(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20240608)
    parser.add_argument(
        "--jaxstanv5-path",
        type=Path,
        default=None,
        help="path to a jaxstanv5 checkout or importable source tree",
    )
    args = parser.parse_args()

    binary = build_bayesite()
    failures = 0
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        fixture = json.loads(path.read_text())
        name = fixture["name"]
        if name in UNSUPPORTED_TRUNCATED_FIXTURES:
            print(
                f"SKIP {name}: model uses Truncated, a core-profile tag this "
                "backend does not evaluate yet (see fixtures_eval.rs)"
            )
            continue
        rust, trailer = run_rust_backend(
            binary,
            fixture,
            seed=args.seed,
            chains=args.chains,
            warmup=args.warmup,
            draws=args.draws,
        )
        jax_flat = run_jax_backend(
            fixture,
            seed=args.seed + 1,
            chains=args.chains,
            warmup=args.warmup,
            draws=args.draws,
        )

        for param, rhat in trailer["rhat"].items():
            if rhat > MAX_RHAT:
                print(f"FAIL {name} {param}: rhat {rhat:.4f} > {MAX_RHAT}")
                failures += 1
        for param, ess in trailer["ess"].items():
            if ess < MIN_ESS:
                print(f"FAIL {name} {param}: ess {ess:.1f} < {MIN_ESS}")
                failures += 1
        total_draws = args.chains * args.draws
        divergences = sum(c["divergences"] for c in trailer["chains"])
        if divergences > MAX_DIVERGENCE_RATE * total_draws:
            print(f"FAIL {name}: {divergences}/{total_draws} divergences")
            failures += 1

        for param, values in sorted(rust.items()):
            rust_mean, rust_sd = summarize(values)
            jax_mean, jax_sd = summarize(jax_flat[param])
            tolerance = MEAN_TOLERANCE_SDS * jax_sd
            mean_ok = abs(rust_mean - jax_mean) < tolerance
            ratio = rust_sd / jax_sd if jax_sd > 0 else float("nan")
            sd_ok = SD_RATIO_RANGE[0] < ratio < SD_RATIO_RANGE[1]
            status = "ok  " if (mean_ok and sd_ok) else "FAIL"
            if not (mean_ok and sd_ok):
                failures += 1
            print(
                f"{status} {name:28s} {param:18s} "
                f"mean {rust_mean:9.4f} vs {jax_mean:9.4f} (tol {tolerance:.4f})  "
                f"sd {rust_sd:8.4f} vs {jax_sd:8.4f}"
            )

    if failures:
        sys.exit(f"{failures} comparison(s) failed")
    print("all posterior comparisons passed")


if __name__ == "__main__":
    main()
