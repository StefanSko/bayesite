#!/usr/bin/env python3
"""Run Bayesite's self-contained validation ladder.

The default path uses only Rust/Cargo tools and committed fixtures. Optional
oracle-backed gates may use Python/JAX/jaxstanv5, but those are never required
for the agent execution path.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_MANIFEST = REPO_ROOT / "crates" / "core" / "Cargo.toml"


def _command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def _run(label: str, command: Sequence[str]) -> None:
    print(f"\n== {label}\n$ {_command_text(command)}", flush=True)
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    except FileNotFoundError as error:
        sys.exit(f"missing executable for {label}: {error.filename}")
    if result.returncode != 0:
        sys.exit(f"{label} failed with exit code {result.returncode}")


def _check_zero_dependency_core() -> None:
    command = ["cargo", "tree", "--manifest-path", str(CORE_MANIFEST)]
    print(f"\n== G0 zero-dependency core\n$ {_command_text(command)}", flush=True)
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        sys.exit(f"missing executable for G0 zero-dependency core: {error.filename}")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(f"G0 zero-dependency core failed with exit code {result.returncode}")
    if result.stdout:
        print(result.stdout, end="")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].startswith("bayesite-core "):
        sys.exit(
            "G0 zero-dependency core failed: cargo tree must contain only "
            "the bayesite-core root package"
        )


def _posterior_command(args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "scripts/check_rust_backend_posterior.py",
        "--draws",
        str(args.posterior_draws),
        "--warmup",
        str(args.posterior_warmup),
        "--chains",
        str(args.posterior_chains),
    ]
    if args.jaxstanv5_path is not None:
        command.extend(["--jaxstanv5-path", str(args.jaxstanv5_path)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-wasm",
        action="store_true",
        help="skip the wasm build gate when the target is not installed",
    )
    parser.add_argument(
        "--posterior",
        action="store_true",
        help="also run the optional jaxstanv5/BlackJAX posterior oracle gate",
    )
    parser.add_argument(
        "--jaxstanv5-path",
        type=Path,
        default=None,
        help="path to a jaxstanv5 checkout for the optional posterior oracle gate",
    )
    parser.add_argument("--posterior-draws", type=int, default=1000)
    parser.add_argument("--posterior-warmup", type=int, default=500)
    parser.add_argument("--posterior-chains", type=int, default=4)
    args = parser.parse_args()

    _check_zero_dependency_core()
    _run(
        "format",
        ["cargo", "fmt", "--check", "--manifest-path", str(CORE_MANIFEST)],
    )
    _run(
        "lint",
        [
            "cargo",
            "clippy",
            "--all-targets",
            "--manifest-path",
            str(CORE_MANIFEST),
            "--",
            "-D",
            "warnings",
        ],
    )
    _run(
        "G1-G5 fixture, log-density, sampler, and protocol tests",
        ["cargo", "test", "--manifest-path", str(CORE_MANIFEST)],
    )
    if not args.skip_wasm:
        _run(
            "G0 wasm build",
            [
                "cargo",
                "build",
                "--target",
                "wasm32-unknown-unknown",
                "--manifest-path",
                str(CORE_MANIFEST),
            ],
        )

    if args.posterior:
        if shutil.which("uv") is None:
            sys.exit("G6 posterior oracle requires uv")
        _run("G6 jaxstanv5/BlackJAX posterior oracle", _posterior_command(args))

    print("\nvalidation ladder passed")


if __name__ == "__main__":
    main()
