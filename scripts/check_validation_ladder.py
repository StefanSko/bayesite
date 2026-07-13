#!/usr/bin/env python3
"""Run Bayesite's development validation ladder.

The default path uses Rust/Cargo tools, committed fixtures, and a pinned
nuts-rs checkout for independent NUTS statistical validation. Optional
bayesjax gates may use Python/JAX/BlackJAX, but those are never required
for the agent execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_MANIFEST = REPO_ROOT / "crates" / "core" / "Cargo.toml"
VENDOR_MANIFEST = REPO_ROOT / "bayeswire-vendor.json"
BAYESWIRE_TAG = REPO_ROOT / "BAYESWIRE_TAG"


def _check_vendored_bayeswire() -> None:
    """Verify vendored spec/corpus bytes against the pinned bayeswire manifest."""
    print("\n== G0 vendored bayeswire spec and corpus", flush=True)
    manifest = json.loads(VENDOR_MANIFEST.read_text(encoding="utf-8"))
    pinned = BAYESWIRE_TAG.read_text(encoding="utf-8").strip()
    if manifest.get("bayeswire_ref") != pinned:
        sys.exit(
            "G0 vendored bayeswire failed: BAYESWIRE_TAG "
            f"({pinned}) does not match bayeswire-vendor.json "
            f"({manifest.get('bayeswire_ref')}); re-run scripts/vendor_bayeswire.py"
        )
    files = manifest.get("files", {})
    if not files:
        sys.exit("G0 vendored bayeswire failed: manifest lists no files")
    for rel_path, want in sorted(files.items()):
        path = REPO_ROOT / rel_path
        if not path.is_file():
            sys.exit(f"G0 vendored bayeswire failed: missing vendored file {rel_path}")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            sys.exit(
                f"G0 vendored bayeswire failed: {rel_path} does not match the "
                "pinned bytes; vendored files are generated, never hand-edited. "
                "Re-run scripts/vendor_bayeswire.py against the pinned checkout."
            )
    print(
        f"{len(files)} vendored files match bayeswire {pinned} "
        f"({manifest.get('bayeswire_commit')})"
    )


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


def _check_release_cli_binary() -> None:
    _run(
        "G0 release CLI binary build",
        [
            "cargo",
            "build",
            "--release",
            "--bin",
            "bayesite",
            "--manifest-path",
            str(CORE_MANIFEST),
        ],
    )
    binary_name = "bayesite.exe" if sys.platform.startswith("win") else "bayesite"
    binary = REPO_ROOT / "target" / "release" / binary_name
    command = [str(binary)]
    print(f"\n== G1 release CLI JSON error smoke\n$ {_command_text(command)}", flush=True)
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        sys.exit(f"missing executable for G1 release CLI JSON error smoke: {error.filename}")
    if result.returncode == 0:
        sys.exit("G1 release CLI JSON error smoke failed: missing-args run must fail")
    if result.stdout:
        sys.exit("G1 release CLI JSON error smoke failed: error path wrote stdout")
    try:
        payload = json.loads(result.stderr)
    except json.JSONDecodeError as error:
        sys.exit(f"G1 release CLI JSON error smoke failed: stderr is not JSON: {error}")
    if payload.get("error_format") != "v0-provisional":
        sys.exit(
            "G1 release CLI JSON error smoke failed: stderr JSON must include "
            'error_format "v0-provisional"'
        )
    if payload.get("error") != "InvalidSettings":
        sys.exit(
            "G1 release CLI JSON error smoke failed: stderr JSON must include "
            'error "InvalidSettings"'
        )
    message = payload.get("message")
    if not isinstance(message, str):
        sys.exit("G1 release CLI JSON error smoke failed: stderr JSON needs a message string")
    if "missing command" not in message:
        sys.exit(
            "G1 release CLI JSON error smoke failed: missing-args message must "
            "name the missing command"
        )
    for command_name in ["sample", "diagnose", "prior-predictive", "recover", "sbc"]:
        if f"bayesite {command_name}" not in message:
            sys.exit(
                "G1 release CLI JSON error smoke failed: usage message must list "
                f"bayesite {command_name}"
            )


def _nuts_rs_command(args: argparse.Namespace) -> list[str]:
    return [
        "python3",
        "scripts/check_nuts_rs_oracle.py",
        "--nuts-rs-path",
        str(args.nuts_rs_path),
        "--draws",
        str(args.nuts_rs_draws),
        "--warmup",
        str(args.nuts_rs_warmup),
        "--chains",
        str(args.nuts_rs_chains),
        "--replicates",
        str(args.nuts_rs_replicates),
        "--batches-per-chain",
        str(args.nuts_rs_batches_per_chain),
    ]


def _sbc_uniformity_command(args: argparse.Namespace) -> list[str]:
    binary_name = "bayesite.exe" if sys.platform.startswith("win") else "bayesite"
    return [
        "python3",
        "scripts/check_sbc_uniformity.py",
        "--bayesite-bin",
        str(REPO_ROOT / "target" / "release" / binary_name),
        "--replicates",
        str(args.sbc_replicates),
    ]


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
    if args.bayescycle_path is not None:
        command.extend(["--bayescycle-path", str(args.bayescycle_path)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-wasm",
        action="store_true",
        help="skip the wasm build gate when the target is not installed",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="skip the mandatory nuts-rs statistical oracle gate (G6)",
    )
    parser.add_argument(
        "--nuts-rs-path",
        type=Path,
        default=Path("/tmp/nuts-rs"),
        help="path to a pinned nuts-rs checkout for the mandatory NUTS oracle gate",
    )
    parser.add_argument("--nuts-rs-draws", type=int, default=1000)
    parser.add_argument("--nuts-rs-warmup", type=int, default=500)
    parser.add_argument("--nuts-rs-chains", type=int, default=4)
    parser.add_argument("--nuts-rs-replicates", type=int, default=8)
    parser.add_argument("--nuts-rs-batches-per-chain", type=int, default=8)
    parser.add_argument(
        "--skip-sbc-uniformity",
        action="store_true",
        help="skip the mandatory G11 SBC rank-uniformity gate",
    )
    parser.add_argument(
        "--sbc-replicates",
        type=int,
        default=100,
        help="replicates per scenario for the G11 SBC rank-uniformity gate",
    )
    parser.add_argument(
        "--posterior",
        action="store_true",
        help="also run the optional bayesjax/BlackJAX posterior oracle gate",
    )
    parser.add_argument(
        "--bayescycle-path",
        type=Path,
        default=None,
        help="override the pinned bayesjax release with a bayescycle checkout",
    )
    parser.add_argument("--posterior-draws", type=int, default=1000)
    parser.add_argument("--posterior-warmup", type=int, default=500)
    parser.add_argument("--posterior-chains", type=int, default=4)
    args = parser.parse_args()

    _check_zero_dependency_core()
    _check_vendored_bayeswire()
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
    _run("release packaging helper tests", ["python3", "scripts/test_release_tooling.py"])
    _check_release_cli_binary()
    if not args.skip_sbc_uniformity:
        _run("G11 SBC uniformity helper tests", ["python3", "scripts/test_sbc_uniformity.py"])
        _run("G11 SBC rank uniformity", _sbc_uniformity_command(args))
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

    if not args.skip_oracle:
        _run("G6 nuts-rs NUTS statistical oracle", _nuts_rs_command(args))

    if args.posterior:
        if shutil.which("uv") is None:
            sys.exit("G7 posterior oracle requires uv")
        _run("G7 bayesjax/BlackJAX posterior oracle", _posterior_command(args))

    print("\nvalidation ladder passed")


if __name__ == "__main__":
    main()
