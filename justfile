# Build/check entry points for Bayesite.

default: check

fmt:
    cargo fmt --manifest-path crates/core/Cargo.toml

# The self-contained gates that must pass before every commit.
check:
    python3 scripts/check_validation_ladder.py

# Raw cargo gates, useful when Python is not available in a dev shell.
check-cargo:
    cargo fmt --check --manifest-path crates/core/Cargo.toml
    cargo clippy --all-targets --manifest-path crates/core/Cargo.toml -- -D warnings
    cargo test --manifest-path crates/core/Cargo.toml
    cargo build --target wasm32-unknown-unknown --manifest-path crates/core/Cargo.toml

wasm-release:
    cargo build --release --target wasm32-unknown-unknown --manifest-path crates/core/Cargo.toml

# Put the wasm binary next to the demo page (run after wasm-release).
demo-assets:
    cp target/wasm32-unknown-unknown/release/bayesite_core.wasm demo/bayesite_core.wasm

# Serve the repo root so the demo can fetch the golden-corpus fixtures:
#   just wasm-release demo-assets demo
demo:
    python3 -m http.server 8000 --bind 127.0.0.1

# Optional cross-backend posterior comparison over the golden corpus (needs uv
# and a jaxstanv5 checkout or installation).
check-posterior jaxstanv5_path="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "{{jaxstanv5_path}}" ]]; then
        uv run scripts/check_rust_backend_posterior.py --jaxstanv5-path "{{jaxstanv5_path}}"
    else
        uv run scripts/check_rust_backend_posterior.py
    fi
