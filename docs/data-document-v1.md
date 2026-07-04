# bayeswire canonical data document, version 1

This document specifies the canonical data document: the JSON document that
carries concrete bind data alongside — never inside — a `bayeswire_ir` model
document. It is the single data artifact every tool in the toolchain reads:
workflow harnesses write it, engines bind from it, exporters decode it, and
the model/data fingerprint ([`model-data-fingerprint-v1.md`](model-data-fingerprint-v1.md))
hashes its exact bytes.

The format marker is `bayescycle.data.json.v1`, the name under which
bayescycle introduced the document before it was specified here. The marker
is an opaque version string; like `bayescycle-dims-v1` in the dimension
sidecar, the historical name is kept so that every existing document stays
valid.

## Document shape

```json
{
  "format": "bayescycle.data.json.v1",
  "variables": {
    "<name>": {"dtype": "<dtype>", "shape": [<int>, ...], "values": [<scalar>, ...]},
    ...
  }
}
```

- The document is a JSON object with exactly the keys `format` and
  `variables`; anything else is malformed. The `format` value must be the
  marker string above; any other value is malformed and the error names the
  unsupported format.
- `variables` maps each data variable name to a typed value object with
  exactly the semantics below. Variable names are non-empty strings; which
  names a model requires is defined by its IR document (`data` plus
  `observed_nodes`), not by this format.
- `dtype` is one of `bool`, `int32`, `int64`, `float32`, `float64`.
- `shape` is an array of non-negative integers; `[]` denotes a scalar.
- `values` is the flat row-major array of the variable's elements; its
  length must equal the product of `shape` (one element for `[]`).
- Float values are JSON numbers and must be finite: `NaN` and `Infinity`
  have no JSON encoding and must not be smuggled through non-standard
  emitters. Integer values are JSON integers within the dtype's range.
  `bool` values are JSON `true`/`false` and bind as integer-valued 1/0.

## Bare-map compatibility form

Engines and exporters also accept the historical bare form — the
`variables` object at the top level, without the wrapper — because the
evaluation fixtures in the conformance corpus predate the wrapper. The
`format` key is **reserved** at the top level to keep the two forms
unambiguous: a bare document must not contain a variable named `format`,
and decoders must fail explicitly (naming the value) rather than guess when
they meet an unrecognized `format` value. New producers write the wrapped
form only.

## Semantics

- Decoding executes no user code.
- The document carries values, not meaning: validation against a concrete
  model (required names, shape compatibility, integer-ness per site) is a
  bind-time concern for backends, not part of this format.
- Producers write the document once and fingerprint the exact bytes they
  wrote; every downstream tool receives the same file. Re-serializing the
  document (re-indenting, reordering keys) produces a semantically equal but
  byte-distinct document and therefore a different fingerprint.

## Relationship to run artifacts

Workflow harnesses conventionally store the document as `data.json` next to
`model.ir.json` in a run directory. That filename is a run-directory
convention, not part of this format. The conformance corpus ships one
canonical data document per reference model under `corpus/data/<model>.json`,
wrapping the same values as the evaluation fixture's `data` block.
