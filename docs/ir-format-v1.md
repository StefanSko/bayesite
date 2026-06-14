# jaxstanv5 IR format, version 1

This document specifies the serialized form of resolved model metadata
(`ModelMeta`), the `jaxstanv5_ir` version 1 wire format. It is the
cross-language contract between the Python library and any other consumer
(tooling, caches, non-Python sampling backends). The Python implementation
lives in `jaxstanv5.ir`; the built-in tag and field inventory is generated
into [`ir-v1-tags.md`](ir-v1-tags.md) and enforced by tests.

The serialized IR decouples model interpretation from model use, is the unit
of provenance (hash the bytes, record the hash in run manifests), makes
resolved declarations diffable, and provides a code-free construction path
(`meta_from_dict` runs no user code).

## Envelope

```json
{"jaxstanv5_ir": 1, "model": { ...encoded ModelMeta node... }}
```

Decoders reject a missing or unknown version with `UnsupportedIRVersion`.

## Encoding rules

1. **Node.** A registered dataclass encodes as a JSON object
   `{"node": "<Tag>", "<field>": <encoded>, ...}` with constructor fields in
   `dataclasses.fields()` order. The tag defaults to the Python class name,
   but the registry supports explicit overrides: the tag vocabulary, not
   Python class names, is the contract. A future Python rename must not
   change the wire format.
2. **Ordered map.** A field typed `dict[str, T]` encodes as a JSON array of
   entries `[{"name": "<key>", "value": <encoded T>}, ...]` in insertion
   order. Never sorted.
3. **Tuple.** A tuple field encodes as a JSON array of encoded items.
4. **Scalars.** `int`, `float`, `str`, `bool`, and `None` pass through.
   Int/float lexical identity is preserved exactly (`1` stays `1`, `1.0`
   stays `1.0`); hashes and diffs depend on it.
5. **Non-finite floats.** `inf`, `-inf`, and `nan` anywhere in the tree raise
   `NonFiniteConstant` at encode time. Strict JSON parsers reject those
   tokens, and a non-finite constant in a log density is an upstream bug.
6. **Union fields** such as `size: DataRef | int | None` need no special
   encoding: a JSON object with a `"node"` key decodes as a node, anything
   else passes through as a scalar or null.
7. **Distributions.** Built-in distributions are pre-registered. User
   packages opt in frozen-dataclass distributions with
   `jaxstanv5.ir.register_distribution(cls)`. A distribution that is not a
   registered dataclass raises `UnserializableDistribution`. This boundary is
   intentional: it coincides with what a code-free parser can parse and what
   a non-Python backend can evaluate.

## Decoding rules

The decoder looks up `"node"` tags in the registry, rebuilds field values
recursively, and calls the constructor. Unknown tags raise `UnknownNodeTag`;
nodes with missing or unexpected fields, map entries without exactly
`"name"`/`"value"`, duplicate entry names, and bare arrays outside map or
tuple fields raise `MalformedIRDocument`.

Empty containers are disambiguated by the *field kind* recorded at
registration time (`map`, `tuple`, or `value`, derived once from the
dataclass type hints), never guessed from the JSON shape. An empty array in
a `map` field decodes to an empty dict; in a `tuple` field, to an empty
tuple.

## `ModelMeta` field roles

`ModelMeta` carries both declaration metadata and the resolved execution
metadata needed by backends. Consumers must not re-run declaration
resolution from the declaration-shaped fields.

- `free_values` defines the flat unconstrained NUTS state layout. If it is
  empty, consumers use `params` as the legacy layout source.
- `stochastic_sites` defines the log-density factors and their value
  expressions. If it is empty, consumers may derive the legacy param and
  observed sites from `params` and `observed_nodes`.
- `data` defines schemas for declared `Data` inputs.
- `observed_nodes` records observed declarations and their required bind
  input names. When `stochastic_sites` is populated, it is not the source of
  log-density factors.
- `params` records declared `Param` metadata. When `free_values` is
  populated, it is not the source of flat parameter packing order.
- `expressions` records named derived expressions for metadata, inspection,
  and validation; stochastic-site expressions are self-contained.

## Canonical bytes and hashing

The canonical serialization is:

```python
canonical_bytes(meta) == json.dumps(
    meta_to_dict(meta),
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

The model hash is `sha256(canonical_bytes(meta))`, computed by the
**producer** at write time. **Consumers hash the file as received and never
re-serialize to hash.** This sidesteps cross-language float-formatting
divergence (Python `repr` and Rust `ryu` differ in corner cases such as
`1e-07` versus `1e-7`).

## Format guarantees

- **Packing order.** The order of entries in `free_values` (or `params` when
  `free_values` is empty) **is** the packing order of the flat unconstrained
  NUTS parameter vector. This is what makes cross-backend differential
  testing well-defined.
- **Order is semantic everywhere.** Entry arrays must never be reordered by
  tooling.
- **Core profile versus extended.** Documents confined to the built-in tag
  set ([`ir-v1-tags.md`](ir-v1-tags.md)) are consumable by all backends and
  tools. Registry-extended documents are consumable only by Python processes
  that imported the registering package; non-Python consumers fail on
  extended tags with their equivalent of `UnknownNodeTag`, by design.
- **Version policy.** Any change to the tag set, a field list, or an encoding
  rule requires a deliberate golden-file diff
  (`scripts/regenerate_ir_golden.py`, `tests/golden_ir/`) plus a version
  decision.

## Errors

| Error | Raised when |
|---|---|
| `NonFiniteConstant` | encoding finds `inf`/`-inf`/`nan` anywhere in the tree |
| `UnserializableDistribution` | a distribution is not a registered dataclass |
| `UnserializableValue` | a leaf value has no IR encoding |
| `UnsupportedIRVersion` | the document version is missing or unknown |
| `UnknownNodeTag` | a document tag is not in the registry |
| `MalformedIRDocument` | the document structure violates this format |

Error messages state what to change, not only what is wrong; they double as
repair instructions for agents producing IR documents.

## Out of scope

This format serializes resolved model metadata only. It does not cover the
declaration class, bound data arrays, `BoundModel`, or sampling results; the
fit artifact is a separate format. Cross-backend evaluation fixtures under
`tests/golden_ir/fixtures/` bundle IR documents with concrete data and
expected log-density values for differential testing, but their layout is a
test fixture convention, not part of this wire format.
