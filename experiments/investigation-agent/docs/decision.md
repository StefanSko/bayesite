# Investigation agent runtime decision (`v0-experimental`)

Status: evidence-backed architecture note for issue #55, written 2026-09-17
against Pi 0.85.1 and the engine at the `feature/investigation-agent-pi`
branch. It closes the Pi-versus-Rust comparison and records what the spike
showed, what it cost, and what it did not test.

## Decision

Embed Pi's coding-agent SDK for the author-side investigation agent. Do not
build a Rust agent runtime. The recipient path (engine binary, static viewer,
published documents) stays independent of this decision and of this package.

## What was built and what proves it

`docs/contract.md` is the frozen boundary; `README.md` lists every choice made
where it was silent. The host is runtime-independent TypeScript; the agent
receives four closed-schema tools through Pi with discovery disabled and our
own system prompt. Evidence:

- 52 offline tests against the real engine and Pi's faux model provider, one
  named test per contract gate, plus orientation, end-to-end, and regression
  tests from an automated review loop (six gate reviews at xhigh reasoning).
- An independent read-only review (GPT-6 Astra, medium reasoning) found no
  model-only approval or mutation bypass and five defects, listed below.
- One live run with a real model (GPT-5.6 Sol, medium reasoning) on the
  counts example, with a scripted human proxy approving through the host API.

## Live run, measured

| Quantity | Value |
|---|---|
| user turns | 8 |
| host outcome messages | 9 |
| tool calls | 27 |
| proposals submitted and executed | 9: fork, adopt, inspect, sample, diagnose, check, interpretation, snapshot, interpretation |
| typed refusals returned to the model | 5, all self-corrected in the same turn |
| wall time | about 4.5 minutes, of which engine execution was under 2 seconds |

The refusals were: one `NotFound` (evidence addressed by hash instead of
name), two `EngineError` from `prepare_candidate` carrying the engine's own
repair messages (`NegativeBinomial2` is not a core tag; `NegativeBinomial`
needs `overdispersion`), one `MalformedArguments` (decision citation format),
one `TargetMismatch` (a hash passed as the execution target). Each was
followed by a corrected call. No proposal was executed without a host-channel
approval, and the original bundle's bytes were unchanged at the end.

Scientifically, the agent stated the effective model from the inspection
object rather than the declared priors, identified the retained limitation
from the check (observed sd 4.47 against replicated 1.02 to 2.62; observed max
18 against 4 to 15), prepared a negative-binomial alternative, and recorded an
interpretation that named what the new check still failed (zero counts:
observed 4 against a replicated mean of 7.8) and six unresolved questions. It
did not claim a verdict. Whether a person would judge the interpretation
adequate is untested; this run measured the host, not the science.

## Review findings and their disposition

From the independent review of commit `b5dbe58`:

1. Adoption mutated the workspace before the engine validated the result.
   Fixed by staging in a temporary copy and validating decision parents at
   submit.
2. No per-turn control: a model could loop on read tools and the person could
   not interrupt; engine subprocesses had no timeout. Fixed with a per-turn
   tool budget, an engine timeout, and abort on interrupt. Host commands typed
   during a turn are still processed after it; a resumable turn boundary is
   the remaining gap for MCP or browser adapters.
3. Fit evidence escaped the byte budget. Fixed with one budget for all kinds.
4. Outputs could be written inside an unrelated bundle. Fixed by refusing any
   workspace or bundle ancestor.
5. Four gates asserted less than their rows. Strengthened.

## Post-review lifecycle amendment

**Amended after review.** The measured chat exposed an implicit lifecycle: the
host suggested next operations but did not make prerequisites enforceable. The
runtime now derives an explicit phase from current engine-verified evidence and
rejects out-of-phase posterior checking and snapshotting with `PhaseRefused`.
The added `record_decision` action separates an agent's diagnostics
thresholds and recommendations from a human waiver. Thresholds are themselves
approved decisions rather than hard-coded verdicts. A recommendation alone
leaves exceeded recorded thresholds blocked; only its explicitly requested
`human_approval` child advances the phase. The same review pass added staged
mutation validation, identifier/parent checks, subprocess and tool budgets,
SIGINT turn abort, bounded UTF-8 evidence, and output-ancestor protection.

## Rust runtime, evaluated on paper against the same gates

| Concern | Pi, measured | Rust, estimated |
|---|---|---|
| Authority boundary and gates | Host code, runtime-independent | Same host code needed; no difference |
| Provider transport and OAuth | Inherited from one catalogue | One adapter, HTTP, SSE, token refresh: each a dependency allowlist decision |
| Streaming, partial tool calls, retries | Inherited | Owned permanently |
| Session persistence, compaction | Inherited | Owned; the plan already conceded stopping at a context budget |
| Author-side delivery | Node plus `npm install` | One extra binary beside the engine |
| Recipient-side delivery | Not on the path | Not on the path, by the issue's condition |
| Browser | Provider proxy needed regardless | Provider proxy needed regardless; Rust adds no browser networking or storage |
| Second provider | Configuration | A second adapter |

Rust wins only author-side packaging, the smallest criterion, and that
advantage disappears once the operations are served over MCP, where the host
is what gets packaged.

## Twelve-factor fit

Owned by the host: prompts (verified: Pi uses our prompt as the preamble and
adds only the tool declarations), tools as structured outputs, human contact
through a durable proposal record, pause and resume across processes, small
focused agent, stateless reducer over on-disk records. Deliberately not
unified: execution state (Pi transcript, disposable) and business state
(workspace and snapshots, authoritative), because the vision makes the
investigation the durable object. Leaned on Pi: the model-tool loop and the
transcript format. The exit, if ever needed, is an owned loop over `pi-ai`
for providers and streaming, roughly 150 lines.

## Costs and limitations observed

- Every `read_investigation` on a workspace copies it to a temporary
  directory to run the engine's inspect without binding unbound decisions.
  Cheap today; a format-level read-only inspect would remove it.
- One proposal per workspace state. A second proposal submitted before the
  first executes becomes stale by design. The prompt and the live driver had
  to say so; the phase state machine makes it visible.
- The investigation format records only inspect, sample, diagnose, and
  posterior-check. The workflow's simulation phase cannot be recorded; see
  the prior-predictive recipe work on `feature/investigation-prior-predictive`.
- A snapshot of a fork that ran no recipes carries no engine pin and cannot be
  forked by the engine. Same branch.
- The browser page is read-only. Approval from the browser needs a backend.
- Trusted host. No protection from an owner process editing the same files.

## Conditions under which this is revisited

- The recipient demonstration in `examples/investigation-counts/PROTOCOL.md`
  cannot succeed without an agent. That is first a finding about the
  published material, then a runtime question.
- Pi's SDK stops supporting the no-discovery, custom-tools-only configuration.
  Versions are pinned exactly; upstream documentation is not a guarantee.
