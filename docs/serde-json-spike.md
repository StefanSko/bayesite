# Serde JSON decoder decision

## Decision

**Reject `serde` + `serde_json` for Bayesite JSON syntax parsing.** The
in-tree parser remains the only production parser and the dependency policy is
unchanged: exact-pinned RustCrypto `sha2` is the sole exception.

The mandatory early numeric feasibility gate failed before a broad adapter or
direct typed-IR prototype could be justified. A normal `DeserializeSeed`
visitor cannot distinguish the integer JSON lexeme `-0` from decimal `-0.0`:
`serde_json` invokes `visit_f64(-0.0)` for both. Bayesite's stable public
contract is respectively `Value::Int(0)` and `Value::Float(-0.0)`. Recovering
that distinction requires retaining or recreating a number lexer/raw-lexeme
protocol, so the candidate does not reduce the owned syntax risk. It also
would not eliminate the generic ordered, duplicate-preserving `Value` path
needed by data, protocol, artifacts, and public `decode_model(&Value)`.

This is a terminal rejection, not a multi-parser design. No Serde dependency,
feature, adapter, parser switch, or typed second decoder is retained.

## Scope and call-site inventory

Baseline: `82d5d7c399fa768e85aae582ada7ef40f8860e01` (2026-07-16), on Apple
`aarch64-apple-darwin`, rustc 1.91.1 (LLVM 21.1.2). The frozen inventory command
from the issue plan produced 623 matches:

```sh
rg -n 'json::parse|json::write|decode_model|json::Value' \
  crates/core/src crates/core/tests --glob '*.rs'
```

It covers CLI model/data/design/truth/scenario input; protocol root and embedded
documents; generation model/design/fit-data/parameter documents; fit NDJSON;
generic data/scenario/workflow decoding; the IR envelope; compact artifact and
error writers; and the wasm UTF-8/protocol boundary. The parser therefore
cannot be replaced with a typed IR-only route.

## Numeric stop-gate experiment

A throwaway, non-committed crate used exactly:

```toml
serde = { version = "=1.0.228", default-features = false, features = ["std"] }
serde_json = { version = "=1.0.150", default-features = false, features = ["std"] }
```

Its `deserialize_any` visitor recorded the callbacks below:

| JSON | Bayesite | serde_json callback |
|---|---|---|
| `-0` | `Int(0)` | `visit_f64`, bits `8000000000000000` |
| `-0.0` | `Float(-0.0)` | `visit_f64`, bits `8000000000000000` |
| `0` | `Int(0)` | `visit_u64(0)` |
| `9223372036854775807` | `Int(i64::MAX)` | `visit_u64` |
| `9223372036854775808` | finite `Float` | `visit_u64` |
| `18446744073709551616` | finite `Float` | `visit_f64`, bits `43f0000000000000` |

Thus the distinction is already lost at the public visitor boundary. Using
`arbitrary_precision`, a raw-lexeme pre-scanner, or Serde private synthetic-map
protocols would need a separate security and ambiguity review and would retain
the exact parser ownership the spike was meant to remove. Per the stop gate,
A and A+B were not implemented; doing so would create misleading evidence.

The candidate normal closure was `serde`, `serde_core`, `serde_json`, `itoa`,
`memchr`, and `zmij`; no derive feature was evaluated. Since the candidate was
rejected before production integration, there is no candidate artifact,
performance, RustSec, or wasm measurement to compare and no dependency update
to audit. The terminal normal and wasm trees remain the reviewed `sha2`
closures documented in `docs/sha2-fingerprint-spike.md`.

## Retained independent evidence

The rejection keeps dependency-free characterization tests for duplicate object
entries, strict malformed forms, integer/float boundaries, subnormal bits,
underflow, overflow, and signed zero. The benchmark at
`tools/bench/json-parse` uses only public Bayesite APIs, deterministic preloaded
inputs, 16 warmups, nine samples, fixed byte accounting, and median reporting.
Run it with:

```sh
cargo run --release --locked --manifest-path tools/bench/json-parse/Cargo.toml
```

On the baseline host, representative median results were:

| Workload | Bytes/s | Documents/s |
|---|---:|---:|
| tiny request | 285,176,305 | 10,968,319 |
| escaped UTF-8 | 375,636,966 | 13,912,480 |
| 64 KiB object | 302,203,252 | 4,612 |
| depth 256 | 64,614,489 | 125,954 |
| golden `linear_regression` parse | 570,467,989 | 95,893 |
| complete 33-document golden corpus parse | 526,012,574 | 117,762 |
| golden parse + `decode_model` | 456,739,643 | 76,776 |

The release artifact snapshot is native 1,747,792 bytes, raw wasm 1,269,504
bytes, and deterministic gzip wasm 386,810 bytes. The new stdlib-only
`scripts/check_wasm_json_boundary.mjs` invokes the actual release ABI and
requires typed, non-trapping results for depth 256, depth 257, 100,000-level
hostile input, and invalid UTF-8. Separately, `tools/bench/json-parse` emits
CSV rows with all nine raw samples plus MAD and range. The native `protocol`
integration test covers the three UTF-8 string cases (depth 256, depth 257,
and hostile depth); the wasm ABI harness covers those cases plus invalid UTF-8,
which only exists at the byte boundary. Both shells assert the same typed
outcomes for their shared UTF-8 request matrix.

## Validation and residual risk

Focused JSON tests and the release wasm build/boundary harness pass. `cargo
audit` was unavailable on this host (`cargo-audit` is not installed), so no new
RustSec claim is made; rejection leaves the locked dependency graph unchanged.
The existing full validation ladder remains required before merge.

The retained parser remains audited code, but it is smaller than an adapter
that must preserve raw lexical numeric class, explicit depth, duplicate ordered
maps, and Bayesite errors. Future reconsideration must begin with an independent
mechanism that proves `-0`/`-0.0` classification and exact numeric bits without
reintroducing an owned number lexer, then repeat the complete issue #45 plan.
