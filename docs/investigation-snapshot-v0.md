# Investigation snapshots (`v0-provisional`)

Status: experimental format decision for the first Bayesite investigation
vertical slice. It wraps existing Bayeswire IR and v0 numerical artifacts; it
does not alter or reinterpret them. Consumers must reject an unknown
`investigation_snapshot` value.

## Bundle and identities

A portable directory is:

```text
bundle/
  manifest.json
  objects/sha256/<64 lowercase hexadecimal characters>
```

Objects contain exact bytes and are addressed by SHA-256. Each reference also
carries byte length, a closed artifact kind, and a format label. The path comes
only from a validated lowercase digest. Hash equality does not establish kind,
authorship, availability, privacy, or scientific validity.

Three identities remain separate:

1. **Artifact/result identity:** SHA-256 of exact object bytes.
2. **Recipe identity:** SHA-256 over the ASCII domain
   `bayesite-investigation-recipe-v0\0` followed by unsigned 64-bit big-endian
   length-framed fields. Fields, in order, are operation; model reference
   (digest, length, kind, format); data reference; either `fit` plus the fit
   reference or `no-fit`; engine binary reference; capabilities reference;
   target; profile; and deterministic compact JSON for normalized settings.
   Sample settings include the resolved initial step size as well as chains,
   warmup, draws, tree depth, target acceptance, and seed.
   The human recipe ID is not part of this identity.
3. **Snapshot identity:** SHA-256 over
   `bayesite-investigation-snapshot-v0\0` followed by the exact received
   `manifest.json` bytes. It is emitted externally as `sha256:<hex>` and is not
   self-referential. Equivalent JSON encodings can have different IDs.

Fixed vectors live in `crates/core/tests/investigation.rs`. Model and data are
separate references, so ambiguous string concatenation is not used.

## Manifest records

The manifest has exactly these top-level fields:

- `investigation_snapshot`: `"v0-provisional"`;
- `question`;
- `estimand`: free-text `description` and target `parameter`;
- `source`: null, or parent `snapshot_id`, parent-manifest artifact reference,
  and local branch-point `decision` ID in that parent;
- `inputs`: current model and data references;
- `decisions`: ordered `{id,parent,reason,cites,kind}` records. `kind` is
  `note`, `agent_recommendation`, or `human_approval`. A human approval is only
  recorded from explicit input; it is not inferred or authenticated by a hash.
  Citation-only bytes are retained in the snapshot closure even if no recipe
  references them;
- `recipes`: one of `inspect`, `sample`, `diagnose`, or `posterior-check`, with
  typed inputs, normalized settings, recipe identity, and an exact engine
  identity (binary, verbatim capabilities object, target, profile);
- `executions`: recipe identity plus `completed`, `failed`, `cancelled`,
  `unsupported`, or `incomplete`. Completed attempts require an output and no
  error. Incomplete attempts have neither output nor an invented cancellation.
  Other non-completed states retain an error and cannot carry a successful
  output;
- `evidence`: named execution selections labeled `current` or `historical`;
- `interpretation` and `unresolved_questions`.

Numerical dependencies and decision reasons are intentionally distinct. There
is no generic relation ontology and no merge operation.

A current result must come from a completed execution whose recipe model/data
match the snapshot inputs. Current downstream evidence must name the exact fit
selected by current sample evidence. The format rejects multiple current
results for one operation. Historical results remain available. A completed
posterior check can expose a limitation without becoming an execution failure.

## Verification

`bayesite investigation verify BUNDLE` performs no recipe execution and no
network access. It reports separately:

- schema/current-selection validity;
- reference closure and exact-byte object integrity;
- availability of bundled engine artifacts;
- whether a replay report is recorded; and
- the computed snapshot ID.

It additionally rejects duplicate JSON fields and parses model, data,
inspection, fit, diagnostics, check, capabilities, and replay artifacts to the
depth their existing provisional contracts support. Investigation fits must
carry the older exact-input model/data compatibility fingerprint, agree with
the recipe's seed/chains/sampler settings, and have the parameter layout
resolved from the separately hashed recipe inputs. That
fingerprint is not promoted to snapshot identity. A manifest execution claim
is not authenticated proof that an execution occurred.

A continuation includes its parent manifest as an object and all required
ancestral objects. Verification follows only local object references, detects
cycles, confirms the branch decision exists, and is bounded to 16 manifests,
256 objects, a 1 MiB manifest, 64 MiB per object, and 256 records per record
array.

## Storage and recovery model

Host storage inserts bytes with create-new/non-clobber semantics and verifies a
pre-existing object before reuse. Export creates a fresh directory, copies
objects, and publishes `manifest.json` last. A leftover temporary file is not an
object or manifest and cannot make a bundle valid. Supported operations never
replace a snapshot.

The first slice assumes a trusted, operator-controlled workspace. It bounds
sizes and derives object paths from validated digests, but does not claim
protection from owner-level mutation, symlink attacks, hostile-neighbour races,
or copy-on-write filesystem behavior.

## Static publication wrapper

`bayesite investigation export <bundle> --viewer --public-data-confirmed --out
<fresh-directory>` creates a non-normative publication wrapper around an exact
snapshot. `entry.json` records the snapshot ID, one pinned engine download for
the saved target, and SHA-256/length pairs for viewer assets. The wrapper also
contains `bundle/`, license/notice, protocol/continuation documents, and the
explicit public-data confirmation. It is not part of snapshot identity.

The static viewer checks the raw manifest bytes under the snapshot domain and
checks displayed objects when WebCrypto is available. It deliberately claims
only that partial coverage. CLI verification remains the authoritative full
recursive check; neither route authenticates an author or asserts scientific
quality.

## Deliberate deferrals

No signatures, private data, authentication, remote store, URL fetching,
compression/extraction, garbage collection, generic cache, concurrent writer,
cross-target numerical comparison, arbitrary-code recipe, scientific merge, or
stable v1 promise is introduced here. Verification, same-target exact replay,
and future cross-target numerical agreement remain different outcomes.
