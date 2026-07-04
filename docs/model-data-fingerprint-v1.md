# bayeswire model/data fingerprint, version 1

This document specifies `model_data_fingerprint`: the string that binds a
fit artifact to the exact model and data documents it was sampled from, so
that posterior-conditioned tools (`posterior-predictive`, `posterior-check`)
can refuse a fit paired with the wrong inputs — across backends. The
reference implementation is `model_data_fingerprint` in `bayeswire.ir`.

The framing prefix is `bayescycle-model-data-v1`, the name under which
bayescycle and bayesite introduced the fingerprint before it was specified
here; the historical name is kept so that every existing fit stays
interpretable.

## Definition

```python
model_data_fingerprint(model_document, data_document) == "sha256:" + sha256(
    b"bayescycle-model-data-v1\n" + model_document + b"\n" + data_document
).hexdigest()
```

- `model_document` is the exact bytes of the `bayeswire_ir` model document
  as exchanged — for a run directory, the canonical bytes written to
  `model.ir.json`.
- `data_document` is the exact bytes of the canonical data document
  ([`data-document-v1.md`](data-document-v1.md)) as exchanged — for a run
  directory, the bytes written to `data.json`.
- The rendering is the literal prefix `sha256:` followed by 64 lowercase hex
  characters.

## Producer and verifier responsibilities

The fingerprint follows the same rule as the model hash
([`ir-format-v1.md`](ir-format-v1.md), "Canonical bytes and hashing"): the
**producer** computes it at write time over the exact bytes it writes, and
**verifiers hash the files as received and never re-serialize**. No tool
may recompute the fingerprint from parsed values; there is no canonical
re-serialization step, so cross-language float formatting never enters the
contract.

- A sampler that writes a fit stream computes the fingerprint from the model
  and data files it was invoked with and records it in the fit header and
  trailer (`model_data_fingerprint` fields).
- A verifier handed a model document, a data document, and a fit recomputes
  the fingerprint from the received file bytes and compares it with the
  recorded field. On mismatch it must fail explicitly; the repair is to
  re-sample against the current inputs.
- A byte-level change to either file — including whitespace — invalidates
  the fit on purpose. The run directory travels as a unit; tools that edit
  artifacts in place forfeit the fit.
- Transports that never materialize file bytes (for example bayesite's
  wasm/JSON-request path) omit the fingerprint; fit identity then falls back
  to the verifier's structural identity check. A fit without the field is
  valid; a fit with a mismatched field is not.

## Scope

The fingerprint is an integrity check for cooperating tools, not a security
boundary: the framed concatenation is not injective under adversarial
documents containing chosen newlines, and no key is involved. It answers
"does this fit belong to these files", nothing more.

## Conformance

The corpus pins the fingerprint two ways:

- `corpus/fingerprints.json` — test vectors: the fingerprint of each
  reference model's committed corpus document bytes (`corpus/<model>.json`)
  with its canonical data document bytes (`corpus/data/<model>.json`).
  Implementations reproduce these strings from the committed files
  byte-for-byte.
- `corpus/artifacts/<model>/` — the real path: each run directory's
  `posterior.ndjson` header and trailer record the fingerprint the engine
  computed from that directory's `model.ir.json` and `data.json`, and it
  must equal the reference implementation's output over those files.
