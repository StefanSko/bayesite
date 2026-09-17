# Bayesite investigation agent (experimental)

This self-contained TypeScript package is the `v0-experimental` host spike from
[`docs/contract.md`](docs/contract.md). It embeds Pi for constrained scientific
chat, delegates all scientific execution and verification to the Bayesite Rust
CLI, records durable proposals/reviews/attempts, exposes a human-only CLI, and
installs a dependency-free read-only proposal viewer in each root's
`.investigation-agent/` directory.

## Authority

**The agent proposes.**
**The human authorizes.**
**The host executes and records.**

The model receives only `read_investigation`, `read_evidence`,
`prepare_candidate`, and `submit_proposal`. It receives no shell, filesystem,
approval, or execution tool.

## Build and test

Build the engine once from the repository root, then install and test the
package:

```sh
cargo build --release --locked --bin bayesite
cd experiments/investigation-agent
npm install
npm test
```

Tests use the real `target/release/bayesite` (or `BAYESITE_BIN`) and Pi's faux
provider. They require no model credentials or network access.

## Run

```sh
# Machine-readable human channel
npm run host -- orient /path/to/root study
npm run host -- proposals /path/to/root --all
npm run host -- show /path/to/root p-0123456789abcdef
npm run host -- approve /path/to/root p-0123456789abcdef --note "approved"
npm run host -- execute /path/to/root p-0123456789abcdef

# Pi-embedded terminal chat
npm run chat -- /path/to/root --public-data-confirmed
npm run chat -- /path/to/root --public-data-confirmed \
  --model openai-codex/gpt-5.6-sol --thinking medium \
  --engine ../../target/release/bayesite
```

The chat refuses to initialize a model without `--public-data-confirmed`,
because workspace model, data, and evidence can be sent to the selected model
provider. Its slash commands are `/proposals`, `/show`, `/approve`, `/reject`,
`/execute`, `/orient`, and `/quit`; they are handled by the host and are never
sent to the model.

`BAYESITE_BIN` overrides the host CLI's engine path. Without it, the host uses
the repository's `target/release/bayesite` relative to this package.

## Approval and execution

Submitting creates an immutable pending proposal and returns immediately.
Approval and execution are separate operations. `approve` never executes.
Execution checks the exact proposal encoding, review hash, engine target, and
current workspace or snapshot state before writing an `incomplete` attempt.
It then performs exactly one action and atomically replaces that attempt with a
`completed` or `failed` record, so interruption during final publication leaves
either the readable incomplete record or the readable final record. Any existing
attempt, including an interrupted `incomplete` attempt, prevents a rerun.
Failure to refresh the auxiliary read-only browser index is never allowed to
block the action or rewrite its authoritative outcome.

For candidate adoption, `--record-human-approval` stores that review choice and
causes execution to append a second `human_approval` decision. Without the flag,
no human-approval decision is inferred.

To use the read-only viewer, serve `<root>/.investigation-agent/` with any local
static-file server and open `index.html`. The host copies the packaged viewer
there and refreshes `index.json` whenever it writes a proposal, review, or
attempt. The browser cannot approve or execute anything.

## Deliberate limits

- No browser approval, MCP server, remote storage, automatic execution,
  compaction policy, or prior-predictive investigation recipe.
- The host and owner-controlled root are trusted. This does not protect against
  an owner process mutating files or racing the host.
- There is no concurrent-writer protocol.
- Sessions are disposable conversation state; deleting `sessions/` does not
  remove proposal authority records.
- Engine and artifact formats remain provisional. The Rust CLI remains the
  verifier and executor.
- A verified snapshot whose top-level manifest has no local recipes (for
  example, snapshotting an untouched fork) can be oriented but cannot currently
  be forked by the Rust CLI: the engine reports that no pinned recipe engine is
  available. The host does not bypass or reimplement that engine refusal; an
  approved attempt is recorded as failed. Resolving this requires an engine or
  frozen-contract decision outside this package.

## Choices where the contract is silent

- Hashes exposed as artifact/state identifiers use `sha256:<hex>`. The three
  component hashes inside the workspace state-hash input are bare lowercase
  hexadecimal strings.
- Host review/attempt timestamps use the local process clock in ISO-8601 UTC.
  Proposal identity does not include a timestamp.
- An identical proposal submission is idempotent. It does not rewrite the
  immutable proposal file and still returns `pending_human_review`; its actual
  review/attempt status is available from `show` or `proposals --all`.
- Proposal files use recursively key-sorted compact JSON plus a trailing newline
  and must retain those exact bytes; even whitespace or key-order-only edits are
  treated as tampering and refused at review or execution.
- The optional candidate `note` is request context for the agent and is not
  persisted. The inspection sidecar records the canonical preparation workspace
  and exact data hash, and adoption requires both to match; recommendation
  reasons belong in proposals. A truncated candidate-ID collision is refused
  without replacing the bytes or inspection already stored under that ID.
- A recorded human-approval decision uses
  `<recommendation-id>-human-approval`. Approval refuses if that ID already
  exists. The review file carries `record_human_approval: true` so the choice
  survives process restart.
- Repeating `approve` or `reject` with the same existing disposition returns the
  original review and ignores a new note. Cross-disposition review is refused.
- Workspace evidence origins are reported as `workspace` or `source_snapshot`;
  bundle origins are `bundle` or `source_snapshot`.
- Workspace orientation runs the required engine inspection against an exact
  temporary copy, then removes it. This preserves the contract's read-only rule
  when the engine binds previously unbound decision inputs during inspection.
- Fit evidence truncation keeps its header, the first 50 draw lines, and every
  trailer line. Other UTF-8 evidence is byte-truncated at 256 KiB.
- Orientation lists only proposals whose action names the oriented path as its
  workspace, source bundle, or output.
- Paths in actions and orientation are stored/reported in canonical root-relative
  form after resolution through their nearest existing ancestor, including
  filesystem case aliases. Fork and snapshot outputs may be nested elsewhere
  under the root, but never inside their own source bundle or workspace.
  Inherited recipe IDs are reserved
  in forks to prevent local/inherited execution-ID collisions.
- Proposal citations accept engine-style evidence identifiers or exact
  `sha256:<64 lowercase hex>` identifiers. Candidate-decision citations accept
  `model`, `data`, or a bare 64-character engine digest.
- Sample recipe proposals must supply all seven engine settings. Inspect and
  diagnose settings must be empty; posterior-check requires `seed`.
- Interpretation and unresolved-question strings are checked against the
  engine's 16,384 UTF-8-byte bound before a proposal is persisted.
- Pi validates tool arguments before calling a tool. Each tool therefore maps a
  pre-validation failure to a fresh, tracked UUID sentinel, then consumes that
  marker to throw the required typed `MalformedArguments` JSON error without
  invoking the host. Ordinary schema-valid strings are never fixed sentinels.
- Pi append-system-prompt discovery is explicitly overridden to an empty list in
  addition to disabling extensions, skills, templates, themes, and context files.
- CLI successes are written to stdout and typed refusals to stderr. Browser
  assets are copied into host state when a `HostApi` is opened; this never
  changes accepted scientific workspace state.
