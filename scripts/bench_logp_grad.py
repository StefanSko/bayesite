#!/usr/bin/env python3
"""Benchmark Bayesite Posterior::logp_grad for analytic IR targets."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_MANIFEST = REPO_ROOT / "tools" / "bench" / "logp-grad" / "Cargo.toml"


def _command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def main() -> None:
    command = ["cargo", "run", "--release", "--quiet", "--manifest-path", str(BENCH_MANIFEST)]
    print(f"$ {_command_text(command)}", flush=True)
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    except FileNotFoundError as error:
        sys.exit(f"missing executable for logp_grad benchmark: {error.filename}")
    if result.returncode != 0:
        sys.exit(f"logp_grad benchmark failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
