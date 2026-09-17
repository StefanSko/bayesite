# Investigation handoff experiment report

This report separates automated evidence from observations that still require
an independent participant. It does not claim scientific validation of either
likelihood.

## Automated integrity results

The repository test suite exercises the frozen protocol in fresh temporary
directories:

- authoring and snapshotting the initial Poisson investigation;
- recursive verification without executing recipes;
- exact replay of the saved sample recipe;
- forking at `initial-likelihood` without modifying source bytes;
- changing the model and observing inherited evidence as `historical` before
  any replacement sample exists;
- running inspection, sampling, diagnostics, and posterior checks for the
  continuation;
- snapshotting and verifying the continuation with one ancestry edge;
- exporting the real continuation contract and pinned engine to a fresh static
  directory;
- retaining unrun work as `incomplete`, rejecting conflicting selections,
  tampered objects, fit/input or seed/settings contradictions, and a
  prospective seventeenth ancestry manifest.

The integration assertion also rereads the original manifest and every source
object after continuation and requires exact byte equality. Viewer parser
smoke tests reject duplicate JSON fields, malformed JSON, and shell-active
identifiers while preserving script-like text as inert data.

The complete validation ladder passed with the external `nuts-rs` oracle
explicitly skipped because its optional checkout was absent. This included
formatting, Clippy with warnings denied, all Cargo tests, release build, SBC
calibration, Wasm release build, and the Wasm JSON boundary. The unmodified
default ladder reached the external oracle and stopped only because
`/tmp/nuts-rs` was not present.

## Committed initial evidence

`scripts/generate_investigation_counts_evidence.sh` generated the committed
inspection, fit, diagnostics, posterior-check, and engine records twice with
byte-identical outputs. The retained check reports the observed standard
deviation and maximum outside the generated Poisson ranges in this bounded
run. These are factual discrepancy summaries, not a model-quality verdict.

## Observed handoff status

The automated recipient path completes using one Bayesite executable and plain
file edits. It requires no producer environment, Python, package manager,
network fetch, framework, or browser-side sampling. Exported material includes
the frozen task, continuation guide, raw Bayeswire format/tag references,
license/notice, exact bundle, and inert engine download.

No independent human participant has yet performed the frozen protocol, and no
manual desktop/mobile browser walkthrough has been recorded. Therefore these
failure conditions remain **unresolved rather than passed**:

- whether a fresh person needs undocumented help;
- time to identify the retained limitation and complete the continuation;
- usability and layout in ordinary desktop and mobile browsers;
- whether the published raw-IR references are sufficient for a person to
  choose and encode a meaningful alternative without hints.

A future participant report must record commands, errors, help requests,
source verification before and after, original/continuation IDs, the
participant's own explanation, and evidence that old results became historical
before resampling. Until then, this slice establishes engineering behavior and
a portable handoff candidate—not successful human handoff.
