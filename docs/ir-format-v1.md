# bayeswire IR format, version 1

This document specifies the serialized form of resolved model metadata
(`ModelMeta`), the `bayeswire_ir` version 1 wire format. It is the normative
cross-language contract between every producer and consumer in the bayes\*
toolchain: the authoring frontend in this repository, sampling backends such
as jaxstanv5 (Python/JAX) and bayesite (Rust), and any tooling or caches in
between. The reference implementation lives in `bayeswire.ir`; the built-in
tag and field inventory is generated into [`ir-v1-tags.md`](ir-v1-tags.md)
and enforced by tests. Downstream repositories vendor these documents from a
pinned bayeswire release and hash-check them; only the copies in this
repository are normative.

The serialized IR decouples model interpretation from model use, is the unit
of provenance (hash the bytes, record the hash in run manifests), makes
resolved declarations diffable, and provides a code-free construction path
(`meta_from_dict` runs no user code). The authoring and IR path is
stdlib-only: sampling backends bring their own runtime dependencies for
binding, log-density evaluation, gradients, simulation, diagnostics, and
inference, none of which are needed to declare a model or to read or write
this JSON document.

## Envelope

```json
{"bayeswire_ir": 1, "model": { ...encoded ModelMeta node... }}
```

The two envelope fields must each appear exactly once. Fields outside
`bayeswire_ir` and `model` are malformed; the envelope shape is part of the
v1 wire contract. Decoders reject a missing or unknown version with
`UnsupportedIRVersion` (or their language's equivalent).

Consumers that parse from raw bytes must enforce the exactly-once rule
directly. Consumers built on JSON parsers that silently collapse duplicate
keys (for example Python's `json.loads`) cannot observe duplicates
post-parse; they enforce the remaining envelope rules (both keys present,
no others) on the parsed object and rely on hashing received bytes for
provenance.

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
   `bayeswire.ir.register_distribution(cls)`. A distribution that is not a
   registered dataclass raises `UnserializableDistribution`. This boundary is
   intentional: it coincides with what a code-free parser can parse and what
   a non-Python backend can evaluate. Distribution objects in IR are
   metadata; runtime behavior such as log-density evaluation is backend
   behavior, not part of the wire contract. Custom Python distributions with
   runtime methods are a single-backend interoperability path only; portable
   custom tags require explicit support in each non-Python backend that
   consumes them.

## Decoding rules

The decoder looks up `"node"` tags in the registry, rebuilds field values
recursively, and calls the constructor. Unknown tags raise `UnknownNodeTag`;
nodes with missing or unexpected fields, more than one `"node"` tag, map
entries without exactly `"name"`/`"value"`, duplicate entry names, and bare
arrays outside map or tuple fields raise `MalformedIRDocument`.

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
- **Version policy.** Any change to the tag set, a field list, or an
  encoding rule requires a deliberate corpus diff
  (`scripts/regenerate_corpus.py`), a changelog entry below, and
  a version decision. `bayeswire_ir` stays at 1 until a real format change;
  package versions move freely underneath it as long as produce-conformance
  bytes are stable.

## Conformance corpus and tolerance policy

The corpus (`src/bayeswire/corpus/`, shipped as package data) holds the golden documents (`<model>.json`), their canonical-bytes
hashes (`hashes.json`), and cross-backend evaluation fixtures
(`fixtures/<model>.json`) bundling each IR document with concrete bind data
and log-density plus gradient values at deterministic unconstrained points.
The recorded values are **JAX-oracle outputs**: produced by the jaxstanv5
backend in float64. Fixture file layout is a corpus convention, not part of
this wire format.

Producers conform by reproducing the corpus documents byte-for-byte from the
reference declarations. Consumers conform by decoding every corpus document
and reproducing the recorded evaluations within this tolerance policy:

- **float64 evaluation** (e.g. bayesite): log density within relative
  tolerance `1e-12`, gradient components within relative tolerance `1e-10`,
  each with an absolute magnitude floor of `1e-8` on the reference value.
- **float32 evaluation** (e.g. jaxstanv5's default test precision): log
  density and gradient components within relative and absolute tolerance
  `1e-3`.

Tolerances live here, in the spec, so that no consumer quietly weakens them.

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
declaration class, bound data arrays, bound-model state, or sampling
results; the fit artifact is a separate format. Dimension labels travel in a
separate sidecar document specified in
[`dimension-sidecar-v1.md`](dimension-sidecar-v1.md).

## Changelog

### 1 — envelope renamed to `bayeswire_ir` (bayeswire extraction)

The wire envelope is `{"bayeswire_ir": 1, "model": ...}`. The format was
extracted from jaxstanv5, where the identical encoding shipped under the
`{"jaxstanv5_ir": 1, "model": ...}` envelope. Renaming the envelope to a
producer-neutral key was the deliberate format decision that bayesite's
compatibility notes reserved for exactly this occasion; it changed the two
envelope strings and nothing else — tags, field lists, entry-array
semantics, and canonical-bytes rules are byte-identical (verified by
structural diff when the corpus was regenerated). The retired
`jaxstanv5_ir` key is not accepted anywhere, by any consumer, and no
producer emits it; documents carrying it fail with the standard
unsupported-version error. This entry is the only place the retired key is
recorded.
