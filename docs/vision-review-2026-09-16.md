# Vision review: synthesis of two independent assessments

Reviewed: `vision.md` at commit `6ce0e55` (2026-09-14). Reviewers: Claude
Fable 5.1 (first pass, risk elaboration) and GPT-6 Astra via Pi at high
reasoning (second pass, adversarial verification). Both had read-only access
to this repository and the sibling `bayescycle` checkout at `67be105`.
Every factual claim below was checked against source by at least one
reviewer and the disputed ones by both. Synthesized 2026-09-16.

This is a review, not a rewrite. Proposed edits are listed so they can be
applied or rejected one at a time.

## Verdict

The vision is the right direction and the experiments support the reset it
describes. Its central idea, that the unit of sharing is an inspectable and
continuable investigation rather than a notebook or a posterior, is the one
place the custom stack can do what plain scripts cannot. The document is
honest about what is unproven.

Three things need to change before the first demonstration is planned:

1. The vision must promise inspection of the *effective* model, not just of
   model bytes. Today nothing exposes what the engine actually sampled.
2. The first demonstration must be a falsifiable recipient challenge, not a
   six-step publication checklist the author completes alone.
3. Two evidence claims that both the vision's sources and the first review
   repeat are wrong and must not be cited as support.

## What the evidence actually supports

Corrections to claims made in the experiment reports or the first review.
Sources are in `bayescycle/experiments/`.

| Claim in circulation | What the sources show |
|---|---|
| The fifteen MVP fits were bit-identical. | The 4000 draw lines hash identically across sessions. Header and trailer differ because they carry the model/data fingerprint, and each agent's model bytes differ. Draws are deterministic; files are not identical. `mvp-agent-binary/FINDINGS.md` F6 says "only the fit headers differ", which is also inaccurate. |
| The fingerprint caught the two transient edits of `model.json`. | It did not. The evaluator found them by scanning tool-call records for write/edit events on protected files (`scripts/evaluate.py` lines 1738 to 1779). The fingerprint only confirmed the final bytes were intact. A final-state hash cannot detect an edit that was undone. FINDINGS F9 overclaims this. |
| `diagnose` reports no divergence count. | The report copies per-chain trailer stats, which include `divergences`. It lacks posterior summaries, not divergence counts. FINDINGS F7 and card correction F13 are wrong on this detail. |
| Plain study notes were enough for the hosted handoff. | The notes sat beside executable models, saved draws, provenance hashes, and preservation checks (`workflow-value-pilot/a-numpyro/STUDY.md`). Free text carried the reasons; artifacts carried the evidence. |
| Rejecting the silent full-form IR modes is a cheap engine fix. | The IR spec makes populated `stochastic_sites` the sole source of density factors and forbids re-deriving from declaration fields (`docs/ir-format-v1.md` lines 91 to 111). A declaration/site mismatch is spec-conformant. Refusing it in `sample` is a bayeswire contract decision, not a bug fix. |
| The fit header pins the engine. | It records fingerprint, seed, settings, layout, and chain metadata. No engine name, version, or target. `CARGO_PKG_VERSION` is emitted only by `capabilities` (`crates/core/src/bin/bayesite.rs` line 1559). Header construction is in `crates/core/src/protocol.rs`, not `artifact.rs`. |
| The playground share link is a starting point for snapshot links. | It is a deflate-compressed URL fragment carrying Python source, data documents, and authoring state, capped at 65,536 characters and 1 MiB decompressed. Saved fits and evidence are outside its payload schema. It is source-first and the opposite of what the vision needs. |
| The hosted pilot's offline HTML report is unrelated. | It is a self-contained evidence viewer that survives being copied elsewhere. It has no execution or fork, but it is the closest existing thing to "open saved evidence first". |

What survives intact: all fifteen MVP sessions authored valid IR on the first
attempt; thirteen passed every check; sessions took 54 to 185 seconds with
no Python; no agent reimplemented the likelihood; every reported number was
reproducible from the session's own artifacts. Neither Python pilot executed
Bayesite. Human-controlled scientific iteration remains unmeasured by every
experiment so far.

## What the vision gets right and should keep

- The investigation as the unit of sharing. Scripts pass a cooperative
  handoff. They do not give a reader a branch point at an earlier decision, a
  retained failed check with its stated reason, or a citable snapshot.
- Stepping back from the five-package ecosystem. The hosted pilot found no
  unique advantage for the CLI layer; the binary alone ran the whole loop.
- The two graphs. Nix's derivation graph is only the computational graph.
  Keeping the investigation graph separate stops decisions being encoded as
  fake build dependencies.
- Three identities, and the separation of exact verification, byte replay,
  and numerical comparison. The run-directory spec already documents
  native/Wasm last-bit differences in model-prior draws.
- The "what the link must not promise" section. Keep it verbatim.
- Nix as inspiration for content-addressing discipline, and nothing more. The
  document says this in two sentences. It should stay that small.

## Proposed edits to `vision.md`

Ordered by importance. Each is a paragraph-scale change.

### E1. Add effective-model inspection as a stated requirement

Under "Why Bayesite" or "One object, several ways to work with it", add:

> A model document carries declaration metadata and execution metadata. The
> engine samples from the execution metadata when it is present. Inspecting
> an investigation means seeing the model that was actually executed: the
> resolved parameter layout and the density factors the sampler used, with
> any discrepancy from the declared priors made visible. Showing a reader the
> declared model beside a valid hash is not inspection.

Why: the IR spec gives populated execution fields authority. A viewer that
renders `params` can show priors that were not sampled, and every hash will
verify. This defeats "identify what they disagree with" more thoroughly than
any identity gap. No current verb exposes the resolved model; the resolver
functions exist in the decoder (`crates/core/src/ir.rs` lines 223 to 266)
but nothing prints them.

### E2. Replace demo steps 3 to 6 with a recipient challenge

Keep steps 1 and 2 for the author. Replace the rest with:

> 3. Publish an immutable snapshot with public data, the pinned engine, and
>    the retained limitation.
> 4. A fresh recipient, without the author's help, identifies the retained
>    limitation, changes one scientifically meaningful assumption at an
>    earlier decision point, and produces a linked continuation with its own
>    snapshot. The original is unchanged.
>
> The demonstration fails if the recipient needs undocumented help, cannot
> determine the effective model, or presents stale results as current. An
> honest continuation may conclude that the evidence does not justify a
> revision; a better model is not the required outcome.

Why: the current list can be completed by the author alone, which the
hosted pilot shows any stack can do. The product hypothesis is about the
recipient. Moving the branch to the recipient also tests forking at an
earlier decision, which is the distinctive claim. The hosted study's own
recovery result, where every arm missed truth and the tighter prior did not
fix it, is the model for a limitation that does not resolve into a story.

### E3. Correct the evidence section

Replace the MVP bullet's evidence framing with the verified facts above.
Cite draw-level determinism as the property that makes content addressing
cheap. State that the transient edits were detected from tool-call records,
and draw the right lesson: final-state hashing cannot enforce "the original
remains unchanged" against a cooperating but careless agent. Snapshot
storage must be append-only or copy-on-write, which is what content-addressed
stores provide.

### E4. State the default reproduction claim and the cache rule

Under "Immutable snapshots and unique links", add:

> A snapshot claims that it contains saved, hash-verified evidence. Whether
> a computation has been replayed, and whether a replay agreed, are separate
> recorded outcomes. Cache reuse requires recipe identity including the
> execution target. Cross-target reproduction compares named quantities
> under stated tolerances; it is never a cache hit, and agreement is never
> permission to substitute a result from a different recipe.

### E5. Fence the fingerprint out of snapshot identity

Add one sentence: `model_data_fingerprint` is a cooperative integrity check
between tools, documented as non-injective under chosen newlines, and the
Wasm fallback identity is a non-cryptographic structural hash. Snapshot
identity hashes model bytes and data bytes independently. This is a design
fence for new work, not an instruction to migrate existing fits.

### E6. Decide what is typed in the investigation graph from day one

Under "Two connected graphs", add:

> Stated reasons begin as free text with a parent pointer and hash
> references to the artifacts they cite. Three things are typed from the
> start because the vision already requires them to be distinguished:
> whether a result is current or superseded, whether an operation completed,
> was cancelled, or is unsupported, and whether a record is a human approval,
> an agent recommendation, or a note. Everything else earns a field only when
> a view cannot be built without it.

Why: this resolves the tension between "do not design a second notebook
format" and "human approvals must never be invented". The hosted handoff
worked with free-text supersession, but the evaluator had to check it by
hand.

### E7. Smaller edits

- Under "Branching and forking": "There is no merge operation. A synthesis
  is a new decision that references both branches and executes fresh
  computations." One sentence, no new verb.
- Under "Evidence so far": note that the existing playground share link is
  source-first and size-capped, and that the hosted pilot's offline HTML
  report is the nearer starting point for a saved-evidence viewer.
- Under "What the link must not promise": one sentence that restricted or
  private-data investigations are outside the first demonstration and need
  their own design decision. Step 4 already says public data; make the
  exclusion explicit.

## Engineering order before the demonstration

Each item names what it does not require, to keep the list from growing.

1. **An `inspect` verb.** Reads a model document, prints the resolved free
   value layout and the resolved stochastic sites, states whether the legacy
   or full layout is in use, and flags two discrepancies as warnings: a
   populated site list that omits an observed declaration, and a site whose
   distribution disagrees with its `params` entry. Pure, no contract change,
   no sampling. The demo's supported subset refuses inputs with warnings
   before publication. Acceptance fixtures: the MVP minimal layout, an
   equivalent full layout, a full layout omitting the observed factor, a
   declaration/site prior mismatch. Does not require changing `sample`.

2. **A pre-registered recipient protocol.** Written before the demo in the
   style of the existing experiment protocols, with the failure conditions
   from E2 frozen. Does not require any new code.

3. **Engine identity in the fit header.** Name, version, and build target,
   proposed to bayescycle as a format decision under the v0-provisional
   rules. Additive. For the demo itself, the snapshot manifest pins the
   executable and Wasm digests externally, so this does not block the demo.

4. **Snapshot as a directory with a manifest.** The link is a git commit or
   a static-host path. The viewer starts from the offline HTML evidence
   report pattern, extended to open a snapshot by URL and show saved
   evidence before offering execution. Does not require extending the
   Python-authoring playground.

Deferred, with reasons:

- A `summary` verb. Useful for agents; not required to inspect saved
  evidence, since the snapshot carries summaries.
- Refusing full-form ambiguity in `sample`. Needs a bayeswire decision; the
  `inspect` warnings cover the demo.
- Caching, MCP, and any resolver beyond a static path.

## Points where the reviewers still differ

Recorded so the decision is visible.

- **Engine version field.** First review: a prerequisite. Second review:
  pin digests externally and change the contract deliberately later.
  Recommendation: item 3 above, not blocking.
- **Author branches in the demo.** First review: keep two author branches.
  Second review: one claim slice. Recommendation: the author publishes one
  model and one retained limitation; the recipient makes the branch. This
  drops the author's two-branch comparison from the first demo and tests
  forking where it matters.
- **Typed fields in the investigation graph.** First review: free text only
  until a view demands more. Second review: status, completion, and approval
  provenance cannot wait. Recommendation: E6.
- **Nix.** First review made the analogy load-bearing; second review notes
  the document does not. Recommendation: keep Nix as the origin of the
  content-addressing discipline and do not use it to justify any particular
  mechanism.

## Explicitly not recommended

- Rewriting `vision.md` wholesale. The edits above are paragraph-scale.
- Designing the manifest format before the `inspect` verb and the recipient
  protocol exist.
- Citing the fingerprint or file-level bit identity as evidence for the
  vision. Cite draw-level determinism and the transcript-detected edits
  instead.
