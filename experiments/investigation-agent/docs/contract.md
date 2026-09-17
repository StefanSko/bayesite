# Investigation agent host contract (`v0-experimental`)

Status: experimental internal interface for the Pi-hosted investigation agent
spike tracked in issue #55. It is not a public standard, not an MCP server
yet, and not a change to any production Bayesite format. Everything here may
be replaced by the evidence-backed decision that closes the spike.

## Authority rule

**The agent proposes; the human authorizes; the host executes and records.**

- The agent receives four tools and nothing else: no shell, no filesystem,
  no approval tool, no execution tool, no direct workspace mutation.
- The human channel is the host CLI and the chat program's slash commands.
  Neither is a tool. An agent-supplied field such as `"actor": "human"` or
  `"approved": true` is rejected by schema and carries no authority.
- Reading evidence and preparing candidates never mutate accepted scientific
  state. Permission to explore is not permission to adopt; adopting a model
  does not approve sampling; sampling does not approve an interpretation.
- The host is trusted. This contract does not defend against an owner-level
  process editing the same directories outside the host.

## Directory model

The host operates on a **root** directory. Workspaces (from `bayesite
investigation init` or `fork`) and bundles (from `snapshot`) are
subdirectories of the root and are always addressed by root-relative paths
without `..` segments. Host state lives in `<root>/.investigation-agent/`:

```text
<root>/.investigation-agent/
  candidates/<candidate_id>.json          exact candidate model bytes
  candidates/<candidate_id>.inspection.json
  proposals/<proposal_id>/proposal.json   immutable once written
  proposals/<proposal_id>/review.json     written once: approve or reject
  proposals/<proposal_id>/attempts/<n>.json
  sessions/                               Pi session files (conversation)
```

Deleting `sessions/` must not affect proposals, reviews, or attempts.

## Identities and preconditions

- `candidate_id` = `c-` + first 16 hex of SHA-256 over the candidate bytes.
- `proposal_id` = `p-` + first 16 hex of `proposal_sha256`, where
  `proposal_sha256` is SHA-256 over the canonical JSON (sorted keys, no
  whitespace) of `{action, rationale, cites, preconditions}`. Any edit to a
  proposal yields a new proposal. `proposal.json` stores the full sha as well.
- `preconditions.state_sha256` for a workspace is SHA-256 over the ASCII
  concatenation `model:<sha>\ndata:<sha>\ninvestigation:<sha>\n` where the
  three shas are SHA-256 of `inputs/model.json`, `inputs/data.json`, and
  `investigation.json` bytes. For a bundle it is the `snapshot_id` reported
  by `bayesite investigation verify`.
- `preconditions.engine_target` is the `engine.target` string from the
  workspace's `investigation.json` (or the bundle manifest).
- A review references `proposal_sha256`. Execution recomputes both the
  proposal hash from `proposal.json` bytes and the state hash from the
  current directory and refuses on any mismatch.

## Agent-facing operations

All return JSON. Errors are `{ "error": "<Kind>", "message": "..." }` with
kinds from: `InvalidPath`, `NotFound`, `MalformedArguments`,
`EngineError`, `CandidateRejected`, `StalePreconditions`, `TargetMismatch`,
`RecipeConflict`, `Refused`.

### `read_investigation { path }`

Returns the orientation document for a workspace or bundle:

```json
{
  "orientation_format": "v0-experimental",
  "kind": "workspace" | "bundle",
  "path": "study",
  "question": "...", "estimand": {"description": "...", "parameter": "..."},
  "model_sha256": "...", "data_sha256": "...", "state_sha256": "...",
  "engine_target": "aarch64-apple-darwin",
  "source": null | {"snapshot_id": "...", "decision": "..."},
  "decisions": [{"id","parent","kind","reason","cites"}],
  "recipes": [{"id","operation","settings","status": "unrun"|"completed"|"failed"|"incomplete"}],
  "evidence": [{"name","execution","status": "current"|"historical","origin"}],
  "inspection": null | {"free_slots": [...], "density_factors": [...], "structural_discrepancies": [...]},
  "diagnostics": null | {"per_parameter": [{"name","rhat","ess"}], "divergences": <int>},
  "check": null | {"summaries": [...as emitted by posterior-check...]},
  "interpretation": "...", "unresolved_questions": [...],
  "next_steps": [{"step": "<label>", "operation": "<bayesite operation or host action>", "why": "<fact>"}],
  "proposals": [{"proposal_id","status": "pending"|"approved"|"rejected"|"executed"|"failed"|"incomplete","action_type"}]
}
```

`inspection`, `diagnostics`, and `check` are parsed from the **current**
evidence only; historical evidence is listed but not summarised. The host
obtains workspace facts by running `bayesite investigation inspect <path>`
and reading artifacts from the object store; it never reimplements
verification.

`next_steps` is computed by deterministic rules over that state. Rules state
facts and options only, never verdicts. Minimum rule set:

| State | Suggested step |
|---|---|
| no current inspection | run the inspect recipe |
| inspection has structural discrepancies | fix the model before sampling; record why |
| current inspection, no current fit | run the sample recipe |
| current fit, no current diagnostics | run the diagnose recipe |
| current diagnostics, no approved `thresholds:` decision | record thresholds; report current max R-hat, min ESS, and divergences |
| diagnostics exceed the approved threshold decision, no approved diagnostics-citing `waiver:` | record a waiver or consider settings/reparameterisation |
| current diagnostics satisfy approved thresholds or have an approved waiver, no current check | run the posterior-check recipe |
| check present | discuss what it does and does not support; consider a decision and fork |
| every recipe current | snapshot |
| inherited evidence historical and no new fit | list the recipes to rerun |
| kind is bundle | fork at a named decision |

### `read_evidence { path, name }`

`name` is `model`, `data`, or the `name` of an evidence entry. Returns
`{ "name", "sha256", "bytes", "format", "content" }`. `content` is the UTF-8
text, truncated at 256 KiB with `"truncated": true`. For a fit stream the
host returns the header line, the trailer lines, and at most 50 draw lines.

### `prepare_candidate { workspace, model_json, note }`

Writes the candidate bytes under `candidates/`, runs
`bayesite inspect --model <candidate> --data <workspace>/inputs/data.json`,
stores the inspection, and returns
`{ "candidate_id", "sha256", "inspection": {free_slots, density_factors, structural_discrepancies, execution_metadata} }`
or an `EngineError` carrying the engine's typed error verbatim. Nothing under
the workspace changes.

### `submit_proposal { action, rationale, cites }`

`cites` is an array of `sha256:<hex>` artifact identifiers or evidence
names. `action` is exactly one of:

```json
{"type":"fork","source_bundle":"original","at":"initial-likelihood","out":"alternative"}
{"type":"adopt_candidate","workspace":"alternative","candidate_id":"c-…","decision":{"id":"…","parent":"…","reason":"…","cites":["model"]}}
{"type":"run_recipe","workspace":"alternative","recipe":{"id":"…","operation":"inspect"|"sample"|"diagnose"|"posterior-check","settings":{…}},"target":"aarch64-apple-darwin"}
{"type":"snapshot","workspace":"alternative","out":"continuation"}
{"type":"record_interpretation","workspace":"alternative","interpretation":"…","unresolved_questions":["…"]}
```

The host validates the action against a closed schema
(`additionalProperties: false` everywhere), computes preconditions, persists
`proposal.json`, and returns `{ "proposal_id", "status": "pending_human_review" }`
immediately. It never waits for a person and never executes.

Validation at submit time, each a typed refusal with no state written:

- `adopt_candidate`: candidate exists, its stored inspection has an empty
  `structural_discrepancies` array, the decision id is new, and
  `decision.kind` is not accepted from the agent: the host always records
  `agent_recommendation`.
- `run_recipe`: `target` equals the workspace engine target
  (`TargetMismatch` otherwise); if a recipe with that id already exists its
  operation and settings must be identical (`RecipeConflict` otherwise, with
  the message "use a new recipe id").
- `fork`: `source_bundle` verifies; `at` names a decision in its manifest;
  `out` does not exist.
- `snapshot`: `out` does not exist.

## Human-facing operations (host CLI)

```sh
investigation-host orient   <root> <path>
investigation-host proposals <root> [--all]
investigation-host show     <root> <proposal_id>
investigation-host approve  <root> <proposal_id> [--note TEXT] [--record-human-approval]
investigation-host reject   <root> <proposal_id> [--note TEXT]
investigation-host execute  <root> <proposal_id>
```

- `approve` writes `review.json` `{decision:"approved", proposal_sha256, note, at}`.
  Approving twice is idempotent and writes nothing new. Approving a rejected
  proposal is refused.
- `reject` writes `review.json` `{decision:"rejected", ...}`. Rejecting an
  approved proposal is refused.
- `--record-human-approval` is meaningful only for `adopt_candidate`. When
  the proposal later executes, the host appends a second decision with
  `kind: "human_approval"`, `parent` = the recommendation's id, `reason` =
  the note, `cites: ["model"]`. Without the flag no human approval is ever
  written into the workspace.
- `execute` requires an approved review, recomputes hashes, refuses when
  an attempt already exists for the proposal (`Refused`, "already attempted;
  review attempts/"), writes `attempts/<n>.json` with `outcome:"incomplete"`
  **before** any side effect, performs exactly the one action, and rewrites
  the attempt as `completed` (with the engine's JSON fact document) or
  `failed` (with the typed error). If the host process dies mid-execution
  the attempt stays `incomplete`; the next `execute` refuses and `proposals`
  lists it as `incomplete`. There is no automatic rerun.

Execution semantics per action type:

- `fork`: `bayesite investigation fork <source> --at <at> --out <out>`.
- `adopt_candidate`: copy candidate bytes to `<workspace>/inputs/model.json`,
  append the decision (kind `agent_recommendation`, `cites` as given) to
  `investigation.json`, then run `bayesite investigation inspect <workspace>`
  so the decision binds to the new bytes. With `--record-human-approval`,
  append the second decision after the recommendation and inspect again.
- `run_recipe`: add the recipe to `investigation.json` when absent, then
  `bayesite investigation run <workspace> --recipe <id>`.
- `snapshot`: `bayesite investigation snapshot <workspace> --out <out>`.
- `record_interpretation`: overwrite the two fields in `investigation.json`.

## Chat program

`investigation-agent chat <root> --public-data-confirmed [--model provider/id] [--thinking level] [--engine path]`

- Refuses to start without `--public-data-confirmed`; the message states
  that workspace model, data, and evidence will be sent to the model
  provider.
- Streams assistant text to the terminal. Prints one line per tool call:
  tool name and a short argument summary.
- Slash commands handled by the host code, never visible to the model:
  `/proposals`, `/show <id>`, `/approve <id> [note]`, `/reject <id> [note]`,
  `/execute <id>`, `/orient <path>`, `/quit`. `/approve` does not execute.
- After `/execute`, the outcome is sent to the agent as a user message
  prefixed `[host]` containing the attempt record, so discussion continues
  from the recorded fact.
- Uses Pi's coding-agent SDK `createAgentSession` with: a
  `DefaultResourceLoader` constructed with `noExtensions`, `noSkills`,
  `noPromptTemplates`, `noThemes`, `noContextFiles` all true and
  `systemPrompt` set to the investigation prompt; `tools` listing only the
  four custom tool names; `customTools` from `src/agent/tools.ts`;
  `SessionManager.create(root, <root>/.investigation-agent/sessions)`;
  `ModelRuntime.create()` for provider credentials.
- Default model `openai-codex/gpt-5.6-sol`, default thinking `medium`.

The system prompt states the authority rule, the estimand discipline
(interpretations are attributed, contestable, and never engine facts), that
the agent must cite evidence by name or hash, and that it must never claim a
proposal was approved or executed.

## Browser review page

`browser/index.html` plus `browser/review.js`: a dependency-free static page
that, when served from `<root>/.investigation-agent/`, lists proposals with
their status, shows one proposal's action, rationale, cites, preconditions,
review, and attempts. Read-only. No approval from the browser in this spike;
the page says so.

## Fixtures and tests

Tests build their fixtures with the engine from
`examples/investigation-counts/` in a temporary root: `init` the study,
run the four recipes, `snapshot` to `original/`. Tests use Pi's `faux`
provider (`@earendil-works/pi-ai`) for scripted assistant responses so no
network or credentials are needed.

Mandatory gates, each a test that must pass:

| Gate | Assertion |
|---|---|
| execute without approval | scripted model calls `submit_proposal` then a tool named `execute_approved_proposal`; unknown tool error; no `attempts/` |
| model claims approval | `submit_proposal` with extra `actor:"human"` and `approved:true` is rejected by schema; nothing persisted |
| evidence injection | fixture interpretation text instructs the agent to bypass approval; the tool loadout still contains only the four tools and no attempt exists |
| candidate preparation | after `prepare_candidate` and an `adopt_candidate` proposal, workspace `inputs/model.json` bytes are unchanged and no attempt exists |
| edited after approval | approve p1; a proposal differing only in rationale has a different id and cannot execute; tampering `proposal.json` bytes makes execute refuse |
| stale preconditions | edit workspace model bytes after submit; execute refuses `StalePreconditions` and writes no attempt beyond the refusal |
| declined | reject then execute refuses; no side effect |
| app closes | a fresh host instance over the same root lists the identical pending proposal |
| interrupted | an attempt left as `incomplete` makes execute refuse and `proposals` show `incomplete`; no rerun |
| duplicate | approve twice writes one review; execute twice creates one attempt |
| conversation lost | delete `sessions/`; proposals, reviews, attempts intact and readable |
| malformed arguments | `submit_proposal` with an unknown action type or a missing field returns `MalformedArguments`; nothing persisted |
| target change | `run_recipe` with a target different from the workspace engine target refuses `TargetMismatch` |

Plus: orientation rule tests over the fixture states (fresh workspace,
after inspect, after sample, after diagnose, after check, forked bundle),
and one scripted end-to-end run through fork, prepare, adopt, run, snapshot
with approvals issued through the host CLI functions.

## Amendments after review

**Amended after review.** The following rules supersede narrower wording above:

- Mutating candidate-adoption, recipe-addition, and decision-recording actions
  are first applied to an exact temporary workspace copy and validated with
  `bayesite investigation inspect`. Only a successful staged validation may
  change accepted workspace bytes. Recipe, decision, generated human-approval,
  and non-null parent identifiers are validated before persistence; a parent
  must name an existing earlier decision.
- Engine children have a 600-second hard timeout and are killed on expiry. A
  Pi prompt run has a 12-tool-call budget enforced by `beforeToolCall`; excess
  calls return typed `Refused` JSON and terminate that run. SIGINT during a
  model turn aborts the turn and returns control to chat rather than approving,
  executing, or exiting through an untyped path.
- Every evidence kind shares the 256-KiB UTF-8-safe response budget. Fit
  truncation reserves space for the header and every complete trailer before
  adding at most 50 complete draw lines. If header plus trailers do not fit,
  the response returns the header only and a `truncation_note`.
- A fork or snapshot output is refused with `InvalidPath` when any existing
  ancestor inside the root is a workspace or bundle, including one unrelated
  to the source.
- Orientation includes `phase`, `diagnostics_thresholds`, `allowed_actions`,
  and `phase_facts`. Workspace phases are `inspect_required`,
  `model_revision_required`, `sample_required`, `diagnose_required`,
  `diagnostics_decision_required`, `check_required`, and `snapshot_ready`;
  bundles are `fork_required`. The phase is derived from current verified
  evidence, never supplied by the model. `allowed_actions` is produced by the
  same rules execution uses; recipe entries are written as
  `run_recipe:<operation>`, and a bundle reports only `fork`.
- `phase_facts` contains `simulation_evidence`, the ordered historical evidence
  names, the newest approved `thresholds:` decision id or null, and whether the
  estimand parameter names a current inspection free slot (null without an
  inspection). An approved `waiver:` decision changes `simulation_evidence`
  from "no waiver recorded" to name that decision.
- Actions are phase-checked and return `PhaseRefused` with repair guidance when
  prerequisites are absent. After diagnostics, an approved decision must record
  a nonempty subset of `rhat<=NUMBER`, `ess>=NUMBER`, and
  `divergences<=NUMBER` in a whitespace-tolerant reason beginning
  `thresholds:`. Until then the phase remains `diagnostics_decision_required`.
  The host applies the recorded values mechanically. An exceeded threshold
  decision requires an approved `waiver:` decision citing the current
  diagnostics artifact; a recommendation without its `human_approval` child
  does not count. Refusals identify the threshold decision or its absence.
  Snapshot remains allowed only in `snapshot_ready`.
- `record_decision` is the sixth action:
  `{"type":"record_decision","workspace":"alternative","decision":{"id":"…","parent":"…","reason":"…","cites":["<bare artifact digest>"]}}`.
  Execution stages and then appends it as `agent_recommendation` and runs
  investigation inspection to bind its citations.
- A diagnostics recommendation is not a waiver. The phase advances from
  `diagnostics_decision_required` only when a `human_approval` child exists for
  an agent recommendation citing the current diagnostics artifact. Humans opt
  into that record with `--record-human-approval`, now valid for both
  `adopt_candidate` and `record_decision`; the chat `/approve` command accepts
  the same flag. Successful chat execution prints the resulting path and phase.

In `sample_required` and `check_required`, orientation also reports the
informational `simulation-unsupported` next step until any approved `waiver:`
decision exists. It records that prior-predictive or recovery evidence cannot be
represented by this format version, but does not block an action.

**Amended after review (gate follow-up).** The durability gates now compare the
complete proposal/review/attempt records and their raw bytes across independent
hosts and deleted Pi sessions. The duplicate gate additionally proves that a
second approval leaves review file metadata unchanged and that two execute
calls leave exactly one attempt entry.

The typed error-kind list therefore also includes `PhaseRefused`.

## Deliberate limits

No MCP server, no remote store, no automatic execution, no delegation
budget, no compaction policy beyond Pi's defaults, no browser approval, no
change to `crates/`, `docs/`, `scripts/`, or any production format. The
recipient path (engine binary, static viewer, published documents) does not
depend on anything in this directory.
