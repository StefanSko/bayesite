# `bayesite capabilities` (v0-provisional)

`bayesite capabilities` takes no arguments, prints one JSON document to
stdout, and exits 0. It replaces usage-text scraping as the way consumers
discover what an engine binary supports.

```json
{
  "capabilities_format": "v0-provisional",
  "version": "0.3.0",
  "commands": ["sample", "diagnose", "prior-predictive", "generate", "posterior-predictive", "posterior-check", "simulate", "recover-check", "recover", "sbc", "capabilities"],
  "ir": {"bayeswire_ir": 1},
  "schemas": {
    "recover_scenario": "v0-provisional",
    "sbc_scenario": "v0-provisional",
    "recover_check_targets": "v0-provisional",
    "error_format": "v0-provisional"
  }
}
```

## Fields

- `capabilities_format`: version marker for this document itself.
- `version`: the engine crate version (`CARGO_PKG_VERSION` of
  `bayesite-core`), e.g. `"0.3.0"`. An additive field; it does not bump
  `capabilities_format`.
- `commands`: every CLI subcommand this binary dispatches, in dispatch-table
  order. The list is derived from the dispatch table in the binary, never
  hand-maintained; consumers use it verbatim.
- `ir`: accepted IR envelope versions. `bayeswire_ir: 1` means the engine
  decodes `{"bayeswire_ir": 1, "model": ...}` documents.
- `schemas`: version markers for the machine-readable input schemas and the
  error format. Each key names a schema documented in this directory:
  - `recover_scenario`: [recover-scenario-v0.md](recover-scenario-v0.md)
  - `sbc_scenario`: [sbc-scenario-v0.md](sbc-scenario-v0.md)
  - `recover_check_targets`: [recover-check-targets-v0.md](recover-check-targets-v0.md)
  - `error_format`: the single-line JSON error object on stderr, see
    [artifacts-v0.md](artifacts-v0.md).

## Compatibility

The document is a wire contract, versioned like the error format:

- Field **additions** are backward compatible and do not bump
  `capabilities_format`.
- Field **renames or removals** bump `capabilities_format`.
- Consumers must ignore fields they do not recognize.

## Consumer contract

Consumers (bayescycle preflight) try `bayesite capabilities` first and use
`commands` verbatim. If the subcommand is unknown (older binaries exit
nonzero with an `unknown command` error), they fall back to probing usage
text.
