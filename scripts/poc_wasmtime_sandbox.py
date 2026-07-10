#!/usr/bin/env python3
"""PoC for issue #32: host bayesite_core.wasm in wasmtime-py with fuel/memory limits.

Validates the three claims behind the sandboxed-engine plan
(https://github.com/StefanSko/bayesite/issues/32):

1. The wasm module is a sealed guest: its import section is empty, so the
   host mediates nothing beyond request bytes in / response bytes out.
2. NUTS sampling works under an enforced sandbox (deterministic fuel budget
   via `Config.consume_fuel`, hard linear-memory cap via `Store.set_limits`)
   at interactive speed on the golden corpus.
3. The limits actually bite: a starved fuel budget traps deterministically
   instead of hanging.

The host glue mirrors demo/glue.js byte for byte: bayesite_alloc -> write
request -> bayesite_run -> read 4-byte length slot + response -> dealloc.

Usage:
    cargo build --release --target wasm32-unknown-unknown \
        --manifest-path crates/core/Cargo.toml
    python3 -m venv .venv && .venv/bin/pip install wasmtime
    .venv/bin/python scripts/poc_wasmtime_sandbox.py

Requires the `wasmtime` package (PyPI, Bytecode Alliance); everything else
is stdlib. Exits non-zero if any claim fails to reproduce.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

try:
    from wasmtime import Config, Engine, Instance, Module, Store, Trap
except ImportError:  # pragma: no cover - environment guard
    sys.exit("missing dependency: pip install wasmtime")

REPO_ROOT = Path(__file__).resolve().parent.parent
WASM_PATH = (
    REPO_ROOT / "target" / "wasm32-unknown-unknown" / "release" / "bayesite_core.wasm"
)
GOLDEN = REPO_ROOT / "tests" / "golden_ir"

DEFAULT_FUEL = 500_000_000_000
DEFAULT_MEMORY_BYTES = 256 * 1024 * 1024
STARVED_FUEL = 100_000_000


def load_module(engine: Engine) -> Module:
    module = Module.from_file(engine, str(WASM_PATH))
    return module


def run_request(
    engine: Engine,
    module: Module,
    request: dict,
    fuel: int,
    memory_bytes: int,
) -> tuple[str, int]:
    """Run one request in a fresh limited Store; return (response, fuel used)."""
    store = Store(engine)
    store.set_limits(memory_size=memory_bytes)
    store.set_fuel(fuel)
    exports = Instance(store, module, []).exports(store)
    memory = exports["memory"]
    alloc = exports["bayesite_alloc"]
    dealloc = exports["bayesite_dealloc"]
    run = exports["bayesite_run"]

    request_bytes = json.dumps(request).encode()
    request_ptr = alloc(store, len(request_bytes))
    memory.write(store, request_bytes, request_ptr)
    length_ptr = alloc(store, 4)

    response_ptr = run(store, request_ptr, len(request_bytes), length_ptr)

    (response_length,) = struct.unpack(
        "<I", memory.read(store, length_ptr, length_ptr + 4)
    )
    response = memory.read(
        store, response_ptr, response_ptr + response_length
    ).decode()
    for pointer, length in (
        (request_ptr, len(request_bytes)),
        (length_ptr, 4),
        (response_ptr, response_length),
    ):
        dealloc(store, pointer, length)
    return response, fuel - store.get_fuel()


def sample_request(model_name: str, num_warmup: int, num_draws: int) -> dict:
    model = json.loads((GOLDEN / f"{model_name}.json").read_text())
    data = json.loads((GOLDEN / "data" / f"{model_name}.json").read_text())
    return {
        "command": "sample",
        "model": model,
        "data": data,
        "settings": {"num_warmup": num_warmup, "num_draws": num_draws},
        "seed": 20260710,
        "chain_id": 0,
    }


def check_zero_imports(module: Module) -> None:
    imports = module.imports
    print(f"claim 1 - sealed guest: import count = {len(imports)}")
    if imports:
        for item in imports:
            print(f"  unexpected import: {item.module}::{item.name}")
        sys.exit("FAIL: module is expected to import nothing from the host")


def check_sampling(engine: Engine, module: Module, model_name: str) -> None:
    request = sample_request(model_name, num_warmup=500, num_draws=500)
    started = time.perf_counter()
    response, fuel_used = run_request(
        engine, module, request, DEFAULT_FUEL, DEFAULT_MEMORY_BYTES
    )
    elapsed = time.perf_counter() - started

    lines = response.strip().split("\n")
    first = json.loads(lines[0])
    if "error" in first:
        sys.exit(f"FAIL: engine error for {model_name}: {first}")
    trailer = json.loads(lines[-1])["trailer"]
    print(
        f"claim 2 - sampling under limits: {model_name}: "
        f"{trailer['draws_per_chain']} draws in {elapsed:.2f}s, "
        f"fuel used {fuel_used:,}"
    )
    if trailer["draws_per_chain"] != 500:
        sys.exit(f"FAIL: expected 500 draws, got {trailer['draws_per_chain']}")


def check_fuel_trap(engine: Engine, module: Module, model_name: str) -> None:
    request = sample_request(model_name, num_warmup=500, num_draws=500)
    started = time.perf_counter()
    try:
        run_request(engine, module, request, STARVED_FUEL, DEFAULT_MEMORY_BYTES)
    except Trap:
        elapsed = time.perf_counter() - started
        print(
            f"claim 3 - fuel enforcement: {model_name} with "
            f"{STARVED_FUEL:,} fuel trapped after {elapsed:.2f}s"
        )
        return
    sys.exit("FAIL: starved fuel budget did not trap")


def main() -> None:
    if not WASM_PATH.exists():
        sys.exit(
            f"missing {WASM_PATH}; build it first:\n"
            "  cargo build --release --target wasm32-unknown-unknown "
            "--manifest-path crates/core/Cargo.toml"
        )
    config = Config()
    config.consume_fuel = True
    engine = Engine(config)
    module = load_module(engine)

    check_zero_imports(module)
    check_sampling(engine, module, "linear_regression")
    check_sampling(engine, module, "eight_schools_non_centered")
    check_fuel_trap(engine, module, "eight_schools_non_centered")
    print("all sandbox claims reproduced")


if __name__ == "__main__":
    main()
