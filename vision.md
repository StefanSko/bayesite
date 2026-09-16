# Bayesite: the Bayesian investigation as a first-class object

> Do not just publish a Bayesian answer. Publish the investigation in a form
> someone else can inspect, reproduce, challenge, and continue.

This is the product direction we want to reach, as of 2026-09-16. It is
not a description of completed functionality, a new wire-format specification,
or approval to expand the numerical core. Implementation decisions must preserve
[the project invariants](docs/invariants.md) and earn their complexity through
small, concrete experiments.

## The thing we want to exist

You receive a link to a Bayesian investigation.

Opening it shows the question, the estimand, the models considered, the data used,
the checks performed, and the current conclusion. A suspicious result leads back
to its computation. A model revision leads back to the observation or decision
that motivated it. Failed models and unresolved problems have not disappeared
from the story.

You can inspect the saved evidence without executing anything. You can choose to
reproduce supported computations locally. If you disagree with a prior, a
likelihood, or the treatment of the observation process, you can branch from that
point and explore an alternative. Your continuation receives its own link, with
an explicit relationship to the original.

The original investigation remains unchanged. You can send the author more than
an objection: you can send an executable alternative.

**The unit of sharing is an investigation, not a notebook environment, a chat
session, or a final posterior.**

## Why an investigation, not a reusable analysis recipe

Bayesian workflow is deeply dependent on the question, the data, the measurement
process, and what the investigator learns along the way. A procedure appropriate
for one problem may be misleading for another. Good workflow includes changing
the question, revisiting assumptions, constructing checks, and deciding not to
trust a result.

The primary vision is therefore not a library of one-size-fits-all pipelines
that accept arbitrary datasets. It is a faithful, executable record of a
particular investigation, together with the ability to continue it.

Reusable techniques, model fragments, and teaching examples may emerge from
investigations. Applying an investigation to new private data may become useful
later. Neither is the organizing principle, and neither should turn scientific
judgment into an invisible default.

## Why Bayesite

Bayesite's distinctive foundation is a small, prebuilt, agent-operable engine
that consumes models as data. It can execute supported Bayesian models without
running their producer's code or requiring a model-specific native toolchain.
WebAssembly makes the same numerical core available to browser hosts.

This changes what it costs to carry a computation with a claim. Instead of
requiring the reader to reconstruct a scientific programming environment, a
bounded investigation can carry explicit model and data artifacts, a pinned
engine, and the settings needed to execute them.

The ecological niche is the intersection of:

- **Portable execution:** useful Bayesian computation in constrained environments.
- **Model-as-data:** explicit assumptions that can be transported and inspected.
- **Checkable iteration:** artifacts and decisions that preserve how an answer
  was reached and where it remains uncertain.

This is not a claim to support every probabilistic program, every dataset size,
or every inference algorithm. Nor is it a claim that agents can replace
scientific judgment. Bayesite is the instrument; the investigation gives the
instrument's outputs context.

A model document carries declaration metadata and execution metadata. The
engine samples from the execution metadata when it is present. Inspecting an
investigation means seeing the model that was actually executed: the resolved
parameter layout and the density factors the sampler used, with any
discrepancy from the declared priors made visible. Showing a reader the
declared model beside a valid hash is not inspection.

The execution path should remain free of Python and package-manager setup.
That is distinct from having no third-party source dependencies: the core has
an explicit, audited SHA-256 dependency exception.

## Step back from the ecosystem; rebuild from the instrument

The five-package Bayescycle ecosystem is not the architecture this vision
assumes or seeks to complete. We return to Bayesite, the small standalone
engine, as the starting point and add only what concrete investigations
demonstrate is necessary. This is a reconsideration of the stack from the
vision, not a new product layer placed on top of the existing stack.

Preserve valuable foundations: the model specification, conformance corpus,
numerical validation, and lessons from the existing artifacts and experiments.
But existing package boundaries, Python authoring layers, orchestration,
visualization integrations, and lockstep releases are not commitments for the
future product. A development-time oracle or optional producer need not become
a dependency on the user's execution path.

The investigation layer must earn its existence through a minimal end-to-end
demonstration. Start with the engine and the smallest explicit artifacts and
operations that let someone inspect, reproduce, and continue one investigation.
Reuse existing machinery where it reduces that burden; do not retain it merely
because it already exists, or replace it with an equally elaborate new ecosystem.

This is a deliberate simplification of direction, not an instruction to
immediately delete existing code or silently break published contracts. Any
retirement, migration, or change in responsibility requires its own explicit
decision. The experiments motivate this reset; they do not establish that every
existing component is unnecessary.

## Two connected graphs

Making workflow first-class requires preserving two kinds of relationship.

### The computational graph: what produced this artifact?

A fit depends on exact model and data bytes, a pinned engine and execution
configuration, sampler settings, and seeds. A predictive check depends on its
specified procedure and inputs. An estimand summary depends on its definition,
transformation, and the relevant draws.

```text
Model + data + engine + settings + seeds
                    |
                    v
                   Fit
                 /     \
                v       v
       Diagnostics     Predictive checks
                \       /
                 v     v
              Saved evidence
```

These dependencies support reproduction, verification, caching, and explicit
invalidation. Supported derived computations need equally explicit recipes;
recording a prose description of an unrecorded script is not enough to make
that script reproducible.

### The investigation graph: why did we do this next?

A prior prediction reveals implausible values. A predictive check misses the
long tail. A conversation clarifies that the original estimand was not the
quantity of interest. Each may motivate a new branch of investigation.

```text
Initial model --> prior prediction --> revised prior
                                           |
                                           v
                                      fit and check
                                           |
                                  "The tail is wrong"
                                      /         \
                                     v           v
                               Alternative A  Alternative B
                                      \         /
                                       comparison
                                           |
                                   current conclusion
```

A check can motivate a new model without being a numerical input to that model.
An explicit decision can change the direction of a study without pretending to
be a statistical result. These relationships must not be collapsed into ordinary
build dependencies.

The goal is an inspectable account of actions, evidence, and stated reasons,
not a claim to capture an investigator's private reasoning or prove that their
explanation is correct. Human approvals, when present, must be distinguished
from agent recommendations and ordinary notes; they must never be invented.

Stated reasons begin as free text with a parent pointer and hash references
to the artifacts they cite. Three things are typed from the start because
this vision already requires them to be distinguished: whether a result is
current or superseded, whether an operation completed, was cancelled, or is
unsupported, and whether a record is a human approval, an agent
recommendation, or a note. Everything else earns a field only when a view
cannot be built without it.

## Immutable snapshots and unique links

Borrow the content-addressing discipline of systems like Nix without assuming
that Bayesite must use Nix or become a general build system.

Distinguish three identities:

1. **Recipe identity:** the exact operation and all execution-relevant inputs.
2. **Result identity:** the actual artifact bytes produced by an execution.
3. **Investigation snapshot identity:** a manifest connecting the question,
   estimand, recipes, evidence, decisions, branch relationships, and current
   interpretation.

A content-addressed snapshot can be resolved through a link. Its identity should
not depend on one website, account, or hosting location. Different viewers and
the CLI should be able to inspect the same object. An identifier verifies what
was retrieved; storage and retrieval infrastructure must separately keep it
available.

A working investigation remains open-ended. Publishing or checkpointing it
creates an immutable snapshot. A convenient project name or "latest" link may
move forward; a citation must continue to identify its original snapshot.

A result cache may reuse an artifact only when the relevant recipe identity
matches. Different execution targets or floating-point behavior can yield
different output bytes. Exact artifact verification, byte-identical replay,
and a defined numerical comparison are separate outcomes, not interchangeable
meanings of "reproduced."

A snapshot claims that it contains saved, hash-verified evidence. Whether a
computation has been replayed, and whether a replay agreed, are separate
recorded outcomes. Cache reuse requires recipe identity including the
execution target. Cross-target reproduction compares named quantities under
stated tolerances; it is never a cache hit, and agreement is never permission
to substitute a result from a different recipe.

The engine already writes a model/data fingerprint into each fit: one hash
over the concatenated model and data bytes, so that posterior-conditioned
tools can refuse a fit paired with the wrong inputs. It is a cooperative
integrity check between tools, not an identity: the concatenation is not
injective under adversarial inputs, and the fallback identity used when no
file bytes exist is a non-cryptographic structural hash. Snapshot identity
must hash model bytes and data bytes independently. This is a design fence
for new work, not an instruction to migrate existing fits.

The detailed manifest, identity rules, storage, and publication protocol require
an explicit design and compatibility decision. This vision does not itself
change the existing Bayeswire and Bayescycle contracts or introduce a competing
standard. Preserve the model boundary and assess which artifact contracts serve
the smaller product; any evolution or replacement must be deliberate rather
than an accidental consequence of stepping back from the ecosystem.

## Branching and forking are fundamental operations

The initial vocabulary should be small:

- **Snapshot:** preserve the current investigation without overwriting its past.
- **Branch:** explore an alternative from any recorded point in the investigation.
- **Fork:** continue a shared investigation in another workspace while retaining
  its source relationship.
- **Compare:** inspect changes in assumptions, evidence, estimands, and conclusions.
- **Cite:** refer to an exact snapshot or artifact rather than a moving project.

A branch records its parent, its changes, and their stated motivation. Unchanged
artifacts can be shared rather than copied or recomputed. Changing an input must
not leave an old downstream result presented as current, although that result
remains valid evidence about the earlier branch.

Forking must be possible before the final model. A reader should be able to
revisit an earlier modeling decision and explore the road not taken.

Do not make scientific merging look like automatic file merging. A branch that
changes the likelihood and a branch that changes the estimand cannot have their
conclusions mechanically combined. There is no merge operation. A synthesis
is a new decision that references both branches and executes fresh
computations.

**Failure belongs in the object.** A retained failed check can explain the
investigation better than the final fit alone. Cancellation, unsupported
operations, and incomplete work must also have honest visible states.

## One object, several ways to work with it

A report is a view of the investigation, not its authoritative definition. So
are a timeline, a branch comparison, and a command-line interface.

- An agent operates explicit commands and receives bounded, machine-readable
  facts and artifact references. Its chat context is not the durable study.
- A human sees assumptions, evidence, alternatives, and unresolved questions,
  and can intervene at meaningful decisions.
- A browser reader opens saved evidence first and explicitly opts into bounded
  local execution or a fork.
- An optional MCP adapter connects compatible agent hosts to the same
  operations. It does not define a second scientific workflow.

CLI, MCP, and browser execution are complementary access paths. They are not
competing definitions of the product. The protocol is replaceable; the
investigation should outlive the interface that created it.

## Keep the instrument small

The vision needs a division of responsibility, not a larger numerical core:

- **Bayesite core:** explicit numerical operations over supported model IR, with
  typed failures and inspectable artifacts.
- **A thin investigation layer:** identities, snapshots, artifact relationships,
  branches, and recorded decisions.
- **Adapters and viewers:** storage access, sharing, authentication, consent,
  resource enforcement, presentation, and host integration.

These are responsibilities, not a prescribed set of packages or services.
Where the investigation layer lives, and how much of it the CLI exposes, remain
implementation questions. It is not presumed to be the current Bayescycle layer
with more features. Reuse run-directory and provenance work only where it helps
the minimal investigation object without pulling the broader stack back onto
the execution path. Do not create another orchestration framework or broaden
package responsibilities merely because this vision is ambitious.

WebAssembly is useful because it provides an explicit execution boundary, not
because it makes every surrounding component safe. Hosts must constrain
capabilities and resources. Native process isolation is another deployment
option. Optional arbitrary-code authoring remains separate from IR execution.

## What the link must not promise

- **Availability without preservation.** Content addressing does not keep an
  object online or archive its engine and dependencies.
- **Privacy through hashing.** Data and data-derived outputs require explicit
  publication decisions and access controls. Even private data hashes can leak
  information. Restricted investigations must make reproduction limits visible.
  Restricted or private-data investigations are outside the first
  demonstration and need their own design decision.
- **Authenticity through byte identity.** Hashes alone do not authenticate an
  author, a human approval, or the claim that an execution actually occurred.
- **Safety through protocol choice.** MCP is not a sandbox; Wasm does not make
  its host, parser, or all computation immune to exploitation or exhaustion.
- **Scientific truth through reproducibility.** A reproducible mistake remains
  a mistake. Good chain diagnostics do not establish model adequacy or causal
  identification. Agent agreement does not constitute independent evidence.

## Evidence so far, and the next decisive step

The experiments in the Bayescycle repository support a narrow foundation:

- The [hosted workflow pilot](https://github.com/StefanSko/bayescycle/blob/67be105/experiments/workflow-value-pilot/README.md)
  completed the prescribed study with all three Python setups. It did not show
  a unique handoff advantage for the custom workflow layer, and it did not
  execute Bayesite.
- The [local Gemma pilot](https://github.com/StefanSko/bayescycle/blob/67be105/experiments/gemma-workflow-pilot/README.md)
  did not complete the full studies. A narrower interface alone did not establish
  reliable local-agent scientific work.
- The [binary MVP](https://github.com/StefanSko/bayescycle/blob/67be105/experiments/mvp-agent-binary/README.md)
  showed that five agents could operate the engine on one supplied model:
  15 of 15 sessions authored valid IR on the first attempt, 13 of 15 passed
  every workflow check, sessions took 54 to 185 seconds with no Python, and
  no agent reimplemented the likelihood. Because the engine is deterministic
  given seed and model semantics, the retained draws of all 15 initial fits
  are identical even though the agents wrote their model files
  independently; that draw-level determinism is what makes content
  addressing cheap. The two failures were transient edits of a protected
  model file: the agent edited the original in place, copied it, then
  restored it byte-exact. The final bytes verified; only the session's
  record of tool calls revealed the edit. A hash of final state cannot
  enforce "the original remains unchanged" against a cooperating but
  careless agent. Snapshot storage must therefore be append-only or
  copy-on-write, which is what content-addressed stores provide. This is
  evidence for an agent-operable instrument, not autonomous scientific
  judgment or an enforced immutable history.

Content-addressed, shareable investigation snapshots and their branch/fork
experience are still to be built and tested. Existing engine and browser
capabilities are foundations, not proof of the whole vision. The existing
browser playground can share a project through a link, but that link carries
Python source and authoring state inside the URL, with a size cap that
excludes saved fits; it is source-first, the reverse of what a snapshot link
needs. The hosted pilot produced a self-contained offline HTML report that
embeds its evidence and survives being copied elsewhere. That report, not
the share link, is the nearer starting point for a saved-evidence viewer.

The first end-to-end demonstration should be deliberately small. It should
start from the standalone engine and a minimal investigation representation,
not require completing or installing the broader Bayescycle ecosystem. Pinned
browser assets and thin host code are legitimate parts of the demonstration;
one small numerical binary does not by itself supply a sharing interface.

1. Conduct one concrete investigation with a clear question and estimand.
2. Preserve an initial model and a check that exposes a real limitation.
3. Publish an immutable snapshot with public data, the pinned engine, and
   the retained limitation.
4. A fresh recipient, without the author's help, identifies the retained
   limitation, changes one scientifically meaningful assumption at an earlier
   decision point, and produces a linked continuation with its own snapshot.
   The original is unchanged.

The demonstration fails if the recipient needs undocumented help, cannot
determine the effective model, or presents stale results as current. An
honest continuation may conclude that the evidence does not justify a
revision; a better model is not the required outcome.

Success is not merely that a hash verifies or a browser runs the sampler.
Success is that another person can understand what was done, identify what they
disagree with, and continue the investigation without reconstructing the author's
session or erasing its history.

**A small engine. A durable investigation. A link someone else can continue.**
