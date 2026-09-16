# Investigation workspace and CLI (`v0-provisional`)

The workspace is mutable; a snapshot bundle is immutable through supported
commands. All commands emit one bounded JSON fact document and JSON errors.
They perform no network access and run no arbitrary scripts.

## Author path

```sh
bayesite investigation init \
  --metadata examples/investigation-counts/metadata.json \
  --model examples/investigation-counts/poisson.json \
  --data examples/investigation-counts/data.json --out study/
bayesite investigation inspect study/
bayesite investigation run study/ --recipe inspect-initial
bayesite investigation run study/ --recipe sample-initial
bayesite investigation run study/ --recipe diagnose-initial
bayesite investigation run study/ --recipe check-initial
bayesite investigation snapshot study/ --out original/
bayesite investigation verify original/
```

`init` copies editable model/data bytes to `inputs/`, pins the exact running
executable plus verbatim capabilities document in the object store, and writes
`investigation.json`. Every inspect, run, and snapshot hashes the actual working
bytes again; timestamps and remembered invalidation commands are not trusted.

A recipe is one closed operation: `inspect`, `sample`, `diagnose`, or
`posterior-check`. Sample settings explicitly include chains, warmup, draws,
maximum tree depth, target acceptance, initial step size, and seed. Work is
sequential and bounded to eight chains and 10,000 warmup/draws per chain. No
scheduler, retry, cache lookup, executable launch, or upstream auto-run occurs.

Before numerical execution, `run` persists an `incomplete` attempt. Success
stores exact output bytes and marks it complete. A typed failure records
`failed` and the error, without replacing valid evidence. A first successful
result is selected. Additional outputs remain distinct and unselected until the
operator explicitly edits the workspace `selections` array.

## Fresh continuation

```sh
bayesite investigation replay original/ --recipe sample-initial --out replay/
bayesite investigation fork original/ --at initial-likelihood --out alternative/
# Edit alternative/inputs/model.json and alternative/investigation.json.
bayesite investigation inspect alternative/
# The source evidence now reports historical before any new sample exists.
bayesite investigation run alternative/ --recipe sample-alternative
bayesite investigation run alternative/ --recipe diagnose-alternative
bayesite investigation run alternative/ --recipe check-alternative
bayesite investigation snapshot alternative/ --out continuation/
bayesite investigation verify continuation/
```

`fork` verifies first, copies working input bytes (never writable hardlinks),
imports the local source closure, and records both source snapshot and branch
decision. It never writes the source bundle. Inherited evidence is shown as
historical even before the model changes.

The mutable workspace accepts `"model"` and `"data"` in a decision's `cites`
array and resolves them from the actual working bytes on the next command; an
editor need not calculate content hashes. Add a new decision with its explicit
kind and reason, and use a new recipe ID when execution-relevant inputs change.
Reusing one recipe ID for a different identity is rejected to preserve history.
Human approval is recorded only when the editor explicitly supplies
`"kind":"human_approval"`.

Changing model, data, settings, seed, engine, or an upstream fit makes affected
selections historical by recipe identity. A text-only reason change does not.
Snapshotting an unrun but resolvable recipe records it as `incomplete`; it does
not invent a completion or cancellation.

## Replay and verification

Replay verifies the source, checks the running executable/target/capabilities,
runs one exact saved recipe into a separate directory, and reports input
integrity, engine match, execution outcome, actual and expected output hashes,
and exact-byte agreement separately. Cross-target numerical comparison is
`unsupported` in this format.

Verification never invokes an engine or follows a URL. A fit inside an
investigation must carry the existing exact-input model/data fingerprint and
its parameter layout must match the separately hashed recipe model/data. This
is a compatibility contradiction check, not use of the older combined
fingerprint as snapshot identity.
