# SHA-256 dependency decision

## Decision

**Adopt RustCrypto `sha2` 0.11.0.** Bayesite now uses the exact-pinned,
default-feature-disabled crate behind the existing `sha256_bytes` and
`model_data_fingerprint` APIs. No hash input, framing, renderer, artifact, or
protocol changed. `model_data_fingerprint` updates the hasher with the prefix,
received model bytes, separator, and received data bytes, rather than building
a combined buffer.

Maintaining a cryptographic compression implementation is a greater correctness
and maintenance risk than this narrowly audited source boundary. This is not a
backend abstraction: there is one production implementation and no feature or
runtime selection.

## Call-site inventory

`rg -n 'sha256_bytes|model_data_fingerprint|hashlib\.sha256' crates/core scripts --glob '!target/**'`
was reviewed before the swap. Production Rust hashing remains centralized in
`crates/core/src/fingerprint.rs`: `generation.rs` re-exports `sha256_bytes` for
received generation documents; CLI, `protocol.rs`, and predictive workflows
call `model_data_fingerprint` before parsing model/data documents. The Python
`hashlib` uses are release, vendoring, validation, and SBC tooling with distinct
contracts and were not changed.

## Evidence

Baseline is `21008c5`; candidate is this worktree, built on the same
Apple aarch64 host with rustc 1.91.1 (ed61e7d7e 2025-11-07), LLVM 21.1.2.
Clean release builds used warm registry caches, fresh target directories,
`CARGO_INCREMENTAL=0`, and `--locked` three times. The committed harness is
`tools/bench/fingerprint`; it uses deterministic bytes, `black_box`, 16 warmup
calls, nine samples, and a median. Baseline used the same harness copied into
the baseline checkout.

| Evidence | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Normal dependency packages, excluding root | 0 | 8 native (9 on aarch64 Linux/Apple) / 7 wasm | target-conditioned |
| Median clean release build | 11.99 s | 13.04 s | +8.8% |
| Native release binary bytes | 1,747,520 | 1,747,792 | +0.02% |
| Raw release wasm bytes | 1,247,554 | 1,269,504 | +1.76% |
| `gzip -9 -n` wasm bytes | 381,239 | 386,810 | +1.46% |
| 1 MiB SHA-256 throughput | 444.9 MB/s | 2,950.1 MB/s | +563% |
| Framed corpus API throughput | 434.5 MB/s | 2,557.5 MB/s | +489% |
| Standard, boundary, framing vectors | PASS | PASS | — |
| Exact corpus model/data/key sets and vectors | PASS | PASS | — |
| Wasm build | PASS | PASS | — |

The build, native-size, and compressed-wasm changes are below their decision
gates. The harness also covers 0, 3, 55, 56, 63, 64, 65, 1,024, 65,536, and
1,048,576-byte inputs plus real framed `linear_regression` documents.

## Reviewed dependency surface

All packages are crates.io registry sources and their exact SHA-256 source
checksums are in `Cargo.lock`. All are `MIT OR Apache-2.0`:

| Package | Version | Repository | MSRV |
|---|---:|---|---:|
| sha2 | 0.11.0 | RustCrypto/hashes | 1.85 |
| digest | 0.11.3 | RustCrypto/traits | 1.85 |
| crypto-common | 0.2.2 | RustCrypto/traits | 1.85 |
| block-buffer | 0.12.1 | RustCrypto/utils | 1.85 |
| cpufeatures | 0.3.0 | RustCrypto/utils | 1.85 |
| hybrid-array | 0.4.13 | RustCrypto/hybrid-array | 1.85 |
| typenum | 1.20.1 | paholg/typenum | 1.41 |
| cfg-if | 1.0.4 | rust-lang/cfg-if | 1.32 |
| libc (native target only) | 0.2.186 | rust-lang/libc | 1.65 |

Defaults are disabled. The normal tree contains eight packages on ordinary
native targets. `sha2` selects `cpufeatures` only on aarch64, x86, and
x86_64; `cpufeatures` in turn adds target-only `libc` on aarch64 Linux/Apple.
`wasm32-unknown-unknown` excludes both. There are no
proc macros or native build steps except target-only `libc`'s `build.rs`.
Source inspection found unsafe code in the dependency boundary (notably
optimized buffer handling and target CPU detection); Bayesite's crate-wide
`#![deny(unsafe_code)]` does not apply to dependencies. `sha2` has optional
architecture-specific implementations, but the wasm build succeeds without new
host imports. `cargo-audit` 0.22.2 reported no vulnerabilities or warning-class
advisories against RustSec advisory database commit
`9f3e138091487e69144f536d36976e427a7a3307` (2026-07-13). CI repeats the audit
against the current database for dependency changes, weekly, and before every
release, recording the database revision and date.

## Policy

`sha2 = "=0.11.0"` is the sole direct dependency exception. Its transitives are
lock-pinned and the validation ladder compares exact normal dependency
allowlists for native and wasm targets. Cargo dependencies do not add runtime
I/O, a runtime package graph, filesystem access, clock, entropy, network, or
host capabilities; they do expand the source/audit boundary.

Any update requires explicit review of the lockfile, licenses, provenance,
features, build scripts, unsafe/FFI surface, a clean current RustSec audit,
native/wasm
trees and imports, independent vectors, artifact sizes, benchmark results, and
the full validation ladder. No other dependency is implicitly approved.
