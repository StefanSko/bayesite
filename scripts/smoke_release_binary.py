#!/usr/bin/env python3
"""Smoke-test a Bayesite release binary's machine-readable CLI error surface."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def smoke_binary(binary: Path) -> None:
    if not binary.is_file():
        raise FileNotFoundError(f"release binary not found: {binary}")
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        raise RuntimeError("missing-command smoke must fail")
    if result.stdout:
        raise RuntimeError("missing-command smoke must keep stdout empty")
    try:
        payload = json.loads(result.stderr)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"missing-command stderr is not JSON: {error}") from error
    if payload.get("error_format") != "v0-provisional":
        raise RuntimeError('missing-command JSON needs error_format "v0-provisional"')
    if payload.get("error") != "InvalidSettings":
        raise RuntimeError('missing-command JSON needs error "InvalidSettings"')
    message = payload.get("message")
    if not isinstance(message, str):
        raise RuntimeError("missing-command JSON needs a message string")
    if "missing command" not in message or "bayesite sample" not in message:
        raise RuntimeError("missing-command message must name the missing command and sample usage")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    try:
        smoke_binary(args.binary)
    except (FileNotFoundError, RuntimeError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
