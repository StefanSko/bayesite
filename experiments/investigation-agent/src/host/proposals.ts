import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve, sep } from "node:path";
import { CandidateStore } from "./candidates.js";
import { Engine } from "./engine.js";
import { assertActionAllowed, type InvestigationPhase } from "./phase.js";
import {
  HostError,
  SubmitProposalArgumentsSchema,
  malformedUnless,
  type InvestigationAction,
  type ProposalDocument,
  type ReviewDocument,
  type SubmitProposalArguments,
} from "./types.js";
import {
  RootStore,
  arrayAt,
  canonicalJson,
  objectAt,
  proposalDigest,
  shaIdentifier,
  stringAt,
  type JsonObject,
} from "./workspace.js";

export interface ProposalSummary {
  proposal_id: string;
  status: "pending" | "approved" | "rejected" | "executed" | "failed" | "incomplete";
  action_type: string;
}

export class ProposalStore {
  constructor(
    private readonly store: RootStore,
    private readonly engine: Engine,
    private readonly candidates: CandidateStore,
    private readonly resolvePhase: (path: string) => Promise<InvestigationPhase>,
  ) {}

  async submit(arguments_: unknown): Promise<{ proposal_id: string; status: "pending_human_review" }> {
    malformedUnless(SubmitProposalArgumentsSchema, arguments_);
    const received: SubmitProposalArguments = arguments_;
    const args: SubmitProposalArguments = {
      ...received,
      action: canonicalizeActionPaths(this.store, received.action),
    };
    args.cites.forEach((citation, index) => assertProposalCitation(citation, `cites[${index}]`));
    if (args.action.type === "adopt_candidate" || args.action.type === "record_decision") {
      assertIdentifier(args.action.decision.id, "decision.id");
      if (args.action.decision.parent !== null) assertIdentifier(args.action.decision.parent, "decision.parent");
      assertEngineText(args.action.decision.reason, "decision.reason");
      args.action.decision.cites.forEach((citation, index) =>
        assertDecisionCitation(citation, `decision.cites[${index}]`),
      );
    }
    if (args.action.type === "run_recipe") {
      assertIdentifier(args.action.recipe.id, "recipe.id");
    }
    if (args.action.type === "record_interpretation") {
      assertEngineText(args.action.interpretation, "interpretation");
      args.action.unresolved_questions.forEach((question, index) =>
        assertEngineText(question, `unresolved_questions[${index}]`),
      );
    }
    const preconditions = await this.validateForSubmit(args.action);
    const digest = proposalDigest({ ...args, preconditions });
    const proposalId = `p-${digest.slice(0, 16)}`;
    const document: ProposalDocument = {
      proposal_id: proposalId,
      proposal_sha256: digest,
      action: args.action,
      rationale: args.rationale,
      cites: args.cites,
      preconditions,
    };
    const directory = this.proposalDirectory(proposalId);
    const path = resolve(directory, "proposal.json");
    mkdirSync(directory, { recursive: true });
    if (existsSync(path)) {
      const existing = this.readProposal(proposalId);
      if (canonicalJson(existing) !== canonicalJson(document)) {
        throw new HostError("Refused", `proposal id collision for ${proposalId}`);
      }
    } else {
      writeFileSync(path, `${canonicalJson(document)}\n`, { flag: "wx" });
      this.writeIndex();
    }
    return { proposal_id: proposalId, status: "pending_human_review" };
  }

  list(all = false): ProposalSummary[] {
    const base = resolve(this.store.stateDir, "proposals");
    const summaries: ProposalSummary[] = [];
    for (const id of safeDirectories(base).sort()) {
      if (!/^p-[0-9a-f]{16}$/.test(id) || !existsSync(resolve(base, id, "proposal.json"))) continue;
      const proposal = this.readProposal(id);
      const summary: ProposalSummary = {
        proposal_id: id,
        status: this.status(id),
        action_type: proposal.action.type,
      };
      if (all || summary.status === "pending") summaries.push(summary);
    }
    return summaries;
  }

  show(proposalId: string): Record<string, unknown> {
    const proposal = this.readProposal(proposalId);
    const review = this.readReview(proposalId);
    const attempts = this.readAttempts(proposalId);
    return { proposal, review, attempts, status: this.status(proposalId) };
  }

  approve(proposalId: string, note = "", recordHumanApproval = false): ReviewDocument {
    const proposal = this.readProposal(proposalId);
    this.assertProposalIntegrity(proposal);
    if (
      recordHumanApproval &&
      proposal.action.type !== "adopt_candidate" &&
      proposal.action.type !== "record_decision"
    ) {
      throw new HostError(
        "Refused",
        "--record-human-approval is only valid for adopt_candidate or record_decision proposals",
      );
    }
    const current = this.readReview(proposalId);
    if (current) {
      if (current.decision === "rejected") throw new HostError("Refused", "cannot approve a rejected proposal");
      return current;
    }
    if (
      recordHumanApproval &&
      (proposal.action.type === "adopt_candidate" || proposal.action.type === "record_decision")
    ) {
      const { document } = this.store.workspaceDocument(proposal.action.workspace);
      const humanId = `${proposal.action.decision.id}-human-approval`;
      assertIdentifier(humanId, "human approval decision id");
      if (arrayAt(document, "decisions").some((decision) => decision.id === humanId)) {
        throw new HostError("Refused", `human approval decision id already exists: ${humanId}`);
      }
    }
    const review: ReviewDocument = {
      decision: "approved",
      proposal_sha256: proposal.proposal_sha256,
      note,
      at: new Date().toISOString(),
      ...(recordHumanApproval ? { record_human_approval: true as const } : {}),
    };
    this.store.writeJson(resolve(this.proposalDirectory(proposalId), "review.json"), review, "wx");
    this.writeIndex();
    return review;
  }

  reject(proposalId: string, note = ""): ReviewDocument {
    const proposal = this.readProposal(proposalId);
    this.assertProposalIntegrity(proposal);
    const current = this.readReview(proposalId);
    if (current) {
      if (current.decision === "approved") throw new HostError("Refused", "cannot reject an approved proposal");
      return current;
    }
    const review: ReviewDocument = {
      decision: "rejected",
      proposal_sha256: proposal.proposal_sha256,
      note,
      at: new Date().toISOString(),
    };
    this.store.writeJson(resolve(this.proposalDirectory(proposalId), "review.json"), review, "wx");
    this.writeIndex();
    return review;
  }

  async execute(proposalId: string): Promise<JsonObject> {
    const proposal = this.readProposal(proposalId);
    this.assertProposalIntegrity(proposal);
    const review = this.readReview(proposalId);
    if (!review || review.decision !== "approved") {
      throw new HostError("Refused", review?.decision === "rejected" ? "proposal was rejected" : "proposal is not approved");
    }
    if (review.proposal_sha256 !== proposal.proposal_sha256) {
      throw new HostError("Refused", "review does not reference the current proposal hash");
    }
    const attempts = this.readAttempts(proposalId);
    if (attempts.length > 0) {
      throw new HostError("Refused", "already attempted; review attempts/");
    }
    await this.assertCurrentPreconditions(proposal);

    const attemptsDirectory = resolve(this.proposalDirectory(proposalId), "attempts");
    mkdirSync(attemptsDirectory, { recursive: true });
    const attemptPath = resolve(attemptsDirectory, "1.json");
    const attempt: JsonObject = {
      proposal_id: proposalId,
      proposal_sha256: proposal.proposal_sha256,
      attempt: 1,
      outcome: "incomplete",
      started_at: new Date().toISOString(),
      fact: null,
      error: null,
    };
    this.store.writeJson(attemptPath, attempt, "wx");
    try {
      this.writeIndex();
    } catch {
      // Read-only viewer publication cannot consume or block an authorized action.
    }

    let fact: JsonObject;
    try {
      fact = await this.perform(proposal.action, review);
    } catch (error) {
      const typed = error instanceof HostError ? error : new HostError("Refused", error instanceof Error ? error.message : String(error));
      const failed: JsonObject = {
        ...attempt,
        outcome: "failed",
        completed_at: new Date().toISOString(),
        error: typed.toJSON(),
      };
      this.store.replaceJsonAtomically(attemptPath, failed);
      try {
        this.writeIndex();
      } catch {
        // The typed action failure and durable attempt remain authoritative.
      }
      throw typed;
    }

    const completed: JsonObject = {
      ...attempt,
      outcome: "completed",
      completed_at: new Date().toISOString(),
      fact,
    };
    try {
      this.store.replaceJsonAtomically(attemptPath, completed);
    } catch (error) {
      throw new HostError(
        "Refused",
        `action completed but its attempt remains incomplete; review attempts/: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    try {
      this.writeIndex();
    } catch (error) {
      throw new HostError(
        "Refused",
        `action completed but browser index refresh failed; review attempts/: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    return completed;
  }

  writeIndex(): void {
    const proposals = this.list(true).map((proposal) => ({
      proposal_id: proposal.proposal_id,
      attempt_count: this.readAttempts(proposal.proposal_id).length,
    }));
    this.store.writeJson(resolve(this.store.stateDir, "index.json"), { proposals });
  }

  private async validateForSubmit(action: InvestigationAction) {
    switch (action.type) {
      case "fork": {
        const { path, manifest } = this.store.bundleManifest(action.source_bundle);
        const verification = await this.engine.run(["investigation", "verify", path]);
        if (!arrayAt(manifest, "decisions").some((decision) => decision.id === action.at)) {
          throw new HostError("Refused", `bundle has no decision named ${action.at}`);
        }
        const out = this.store.path(action.out);
        this.store.assertOutputOutsideInvestigation(out);
        if (isDescendantPath(out, path)) {
          throw new HostError("InvalidPath", "fork output must not be inside its immutable source bundle");
        }
        if (existsSync(out)) throw new HostError("Refused", `output already exists: ${action.out}`);
        assertActionAllowed("fork_required", action);
        return {
          state_sha256: stringAt(verification.json, "snapshot_id"),
          engine_target: this.store.bundleTarget(manifest, path),
        };
      }
      case "adopt_candidate": {
        const { path, document } = this.store.workspaceDocument(action.workspace);
        const candidate = this.candidates.get(action.candidate_id);
        const preparedWorkspace = candidate.inspection.candidate_workspace;
        const preparedData = candidate.inspection.candidate_data_sha256;
        const currentData = shaIdentifier(readFileSync(resolve(path, "inputs", "data.json")));
        if (preparedWorkspace !== action.workspace || preparedData !== currentData) {
          throw new HostError(
            "CandidateRejected",
            "candidate inspection is not bound to this workspace and its current data; prepare it again",
          );
        }
        const discrepancies = candidate.inspection.structural_discrepancies;
        if (!Array.isArray(discrepancies) || discrepancies.length > 0) {
          throw new HostError("CandidateRejected", "candidate inspection has structural discrepancies");
        }
        const decisions = arrayAt(document, "decisions");
        if (decisions.some((decision) => decision.id === action.decision.id)) {
          throw new HostError("CandidateRejected", `decision id already exists: ${action.decision.id}`);
        }
        if (
          action.decision.parent !== null &&
          !decisions.some((decision) => decision.id === action.decision.parent)
        ) {
          throw new HostError("CandidateRejected", `decision parent does not exist: ${action.decision.parent}`);
        }
        assertActionAllowed(await this.resolvePhase(action.workspace), action);
        return this.store.workspacePreconditions(action.workspace);
      }
      case "run_recipe": {
        const { path, document } = this.store.workspaceDocument(action.workspace);
        const target = stringAt(objectAt(document, "engine"), "target");
        if (action.target !== target) {
          throw new HostError("TargetMismatch", `recipe target ${action.target} does not match workspace target ${target}`);
        }
        if (sourceManifestHasRecipe(this.store, path, document, action.recipe.id)) {
          throw new HostError(
            "RecipeConflict",
            `recipe ${action.recipe.id} is inherited from the source; use a new recipe id`,
          );
        }
        const existing = arrayAt(document, "recipes").find((recipe) => recipe.id === action.recipe.id);
        if (
          existing &&
          (existing.operation !== action.recipe.operation || canonicalJson(existing.settings) !== canonicalJson(action.recipe.settings))
        ) {
          throw new HostError("RecipeConflict", `recipe ${action.recipe.id} conflicts; use a new recipe id`);
        }
        assertActionAllowed(await this.resolvePhase(action.workspace), action);
        return this.store.workspacePreconditions(action.workspace);
      }
      case "snapshot": {
        const workspace = this.store.workspaceDocument(action.workspace).path;
        const out = this.store.path(action.out);
        this.store.assertOutputOutsideInvestigation(out);
        if (isDescendantPath(out, workspace)) {
          throw new HostError("InvalidPath", "snapshot output must not be inside its mutable workspace");
        }
        if (existsSync(out)) throw new HostError("Refused", `output already exists: ${action.out}`);
        assertActionAllowed(await this.resolvePhase(action.workspace), action);
        return this.store.workspacePreconditions(action.workspace);
      }
      case "record_decision": {
        const { document } = this.store.workspaceDocument(action.workspace);
        const decisions = arrayAt(document, "decisions");
        if (decisions.some((decision) => decision.id === action.decision.id)) {
          throw new HostError("Refused", `decision id already exists: ${action.decision.id}`);
        }
        if (
          action.decision.parent !== null &&
          !decisions.some((decision) => decision.id === action.decision.parent)
        ) {
          throw new HostError("Refused", `decision parent does not exist: ${action.decision.parent}`);
        }
        assertActionAllowed(await this.resolvePhase(action.workspace), action);
        return this.store.workspacePreconditions(action.workspace);
      }
      case "record_interpretation":
        assertActionAllowed(await this.resolvePhase(action.workspace), action);
        return this.store.workspacePreconditions(action.workspace);
    }
  }

  private async assertCurrentPreconditions(proposal: ProposalDocument): Promise<void> {
    let current;
    if (proposal.action.type === "fork") {
      const { path, manifest } = this.store.bundleManifest(proposal.action.source_bundle);
      const verification = await this.engine.run(["investigation", "verify", path]);
      current = {
        state_sha256: stringAt(verification.json, "snapshot_id"),
        engine_target: this.store.bundleTarget(manifest, path),
      };
    } else {
      current = this.store.workspacePreconditions(proposal.action.workspace);
    }
    if (current.engine_target !== proposal.preconditions.engine_target) {
      throw new HostError("TargetMismatch", "engine target changed after proposal submission");
    }
    if (current.state_sha256 !== proposal.preconditions.state_sha256) {
      throw new HostError("StalePreconditions", "investigation state changed after proposal submission");
    }
  }

  private async perform(action: InvestigationAction, review: ReviewDocument): Promise<JsonObject> {
    switch (action.type) {
      case "fork": {
        const source = this.store.path(action.source_bundle, { mustExist: true });
        const out = this.store.path(action.out);
        return (await this.engine.run(["investigation", "fork", source, "--at", action.at, "--out", out])).json;
      }
      case "adopt_candidate": {
        await this.validateStagedMutation(action, review);
        const { path, document } = this.store.workspaceDocument(action.workspace);
        const candidate = this.candidates.get(action.candidate_id);
        copyFileSync(candidate.path, resolve(path, "inputs", "model.json"));
        const decisions = arrayAt(document, "decisions");
        decisions.push({
          ...action.decision,
          kind: "agent_recommendation",
        });
        document.decisions = decisions;
        this.store.writeJson(resolve(path, "investigation.json"), document);
        const firstInspection = await this.engine.run(["investigation", "inspect", path]);
        if (review.record_human_approval) {
          const refreshed = this.store.readJson(resolve(path, "investigation.json"));
          const refreshedDecisions = arrayAt(refreshed, "decisions");
          refreshedDecisions.push({
            id: `${action.decision.id}-human-approval`,
            parent: action.decision.id,
            reason: review.note,
            cites: ["model"],
            kind: "human_approval",
          });
          refreshed.decisions = refreshedDecisions;
          this.store.writeJson(resolve(path, "investigation.json"), refreshed);
          return (await this.engine.run(["investigation", "inspect", path])).json;
        }
        return firstInspection.json;
      }
      case "run_recipe": {
        await this.validateStagedMutation(action, review);
        const { path, document } = this.store.workspaceDocument(action.workspace);
        const recipes = arrayAt(document, "recipes");
        if (!recipes.some((recipe) => recipe.id === action.recipe.id)) {
          recipes.push(action.recipe as unknown as JsonObject);
          document.recipes = recipes;
          this.store.writeJson(resolve(path, "investigation.json"), document);
        }
        return (await this.engine.run(["investigation", "run", path, "--recipe", action.recipe.id])).json;
      }
      case "snapshot": {
        return (
          await this.engine.run([
            "investigation",
            "snapshot",
            this.store.path(action.workspace, { mustExist: true }),
            "--out",
            this.store.path(action.out),
          ])
        ).json;
      }
      case "record_decision": {
        await this.validateStagedMutation(action, review);
        const { path, document } = this.store.workspaceDocument(action.workspace);
        const decisions = arrayAt(document, "decisions");
        decisions.push({ ...action.decision, kind: "agent_recommendation" });
        document.decisions = decisions;
        this.store.writeJson(resolve(path, "investigation.json"), document);
        const firstInspection = await this.engine.run(["investigation", "inspect", path]);
        if (review.record_human_approval) {
          const refreshed = this.store.readJson(resolve(path, "investigation.json"));
          const refreshedDecisions = arrayAt(refreshed, "decisions");
          refreshedDecisions.push({
            id: `${action.decision.id}-human-approval`,
            parent: action.decision.id,
            reason: review.note,
            cites: ["model"],
            kind: "human_approval",
          });
          refreshed.decisions = refreshedDecisions;
          this.store.writeJson(resolve(path, "investigation.json"), refreshed);
          return (await this.engine.run(["investigation", "inspect", path])).json;
        }
        return firstInspection.json;
      }
      case "record_interpretation": {
        const { path, document } = this.store.workspaceDocument(action.workspace);
        document.interpretation = action.interpretation;
        document.unresolved_questions = action.unresolved_questions;
        this.store.writeJson(resolve(path, "investigation.json"), document);
        return {
          investigation_command: "record_interpretation",
          workspace: action.workspace,
          interpretation: action.interpretation,
          unresolved_questions: action.unresolved_questions,
        };
      }
    }
  }

  private async validateStagedMutation(
    action: Extract<InvestigationAction, { type: "adopt_candidate" | "run_recipe" | "record_decision" }>,
    review: ReviewDocument,
  ): Promise<void> {
    const source = this.store.workspaceDocument(action.workspace).path;
    const temporaryRoot = mkdtempSync(resolve(tmpdir(), "bayesite-agent-stage-"));
    const staged = resolve(temporaryRoot, "workspace");
    try {
      cpSync(source, staged, { recursive: true, force: false, errorOnExist: true });
      const documentPath = resolve(staged, "investigation.json");
      const document = this.store.readJson(documentPath, "staged investigation.json");
      if (action.type === "run_recipe") {
        const recipes = arrayAt(document, "recipes");
        if (!recipes.some((recipe) => recipe.id === action.recipe.id)) {
          recipes.push(action.recipe as unknown as JsonObject);
          document.recipes = recipes;
        }
      } else {
        if (action.type === "adopt_candidate") {
          const candidate = this.candidates.get(action.candidate_id);
          copyFileSync(candidate.path, resolve(staged, "inputs", "model.json"));
        }
        const decisions = arrayAt(document, "decisions");
        decisions.push({ ...action.decision, kind: "agent_recommendation" });
        if (review.record_human_approval) {
          decisions.push({
            id: `${action.decision.id}-human-approval`,
            parent: action.decision.id,
            reason: review.note,
            cites: ["model"],
            kind: "human_approval",
          });
        }
        document.decisions = decisions;
      }
      this.store.writeJson(documentPath, document);
      await this.engine.run(["investigation", "inspect", staged]);
    } finally {
      rmSync(temporaryRoot, { recursive: true, force: true });
    }
  }

  private proposalDirectory(proposalId: string): string {
    if (!/^p-[0-9a-f]{16}$/.test(proposalId)) throw new HostError("MalformedArguments", `invalid proposal id: ${proposalId}`);
    return resolve(this.store.stateDir, "proposals", proposalId);
  }

  private readProposal(proposalId: string): ProposalDocument {
    const path = resolve(this.proposalDirectory(proposalId), "proposal.json");
    if (!existsSync(path)) throw new HostError("NotFound", `proposal does not exist: ${proposalId}`);
    const value = this.store.readJson(path, `${proposalId}/proposal.json`);
    const keys = Object.keys(value).sort();
    const expectedKeys = ["action", "cites", "preconditions", "proposal_id", "proposal_sha256", "rationale"].sort();
    if (
      canonicalJson(keys) !== canonicalJson(expectedKeys) ||
      value.proposal_id !== proposalId ||
      typeof value.proposal_sha256 !== "string" ||
      typeof value.rationale !== "string" ||
      !Array.isArray(value.cites) ||
      typeof value.action !== "object" ||
      value.action === null ||
      typeof value.preconditions !== "object" ||
      value.preconditions === null
    ) {
      throw new HostError("Refused", `proposal is malformed: ${proposalId}`);
    }
    return value as unknown as ProposalDocument;
  }

  private assertProposalIntegrity(proposal: ProposalDocument): void {
    const digest = proposalDigest(proposal);
    const path = resolve(this.proposalDirectory(proposal.proposal_id), "proposal.json");
    const hostEncoded: ProposalDocument = {
      proposal_id: proposal.proposal_id,
      proposal_sha256: proposal.proposal_sha256,
      action: proposal.action,
      rationale: proposal.rationale,
      cites: proposal.cites,
      preconditions: proposal.preconditions,
    };
    const expectedBytes = `${canonicalJson(hostEncoded)}\n`;
    let exactBytes = false;
    try {
      exactBytes = readFileSync(path).equals(Buffer.from(expectedBytes));
    } catch {
      // The identity refusal below covers an unreadable proposal file.
    }
    if (
      !exactBytes ||
      proposal.proposal_sha256 !== digest ||
      proposal.proposal_id !== `p-${digest.slice(0, 16)}`
    ) {
      throw new HostError("Refused", "proposal.json bytes or hash do not match its recorded identity");
    }
  }

  private readReview(proposalId: string): ReviewDocument | null {
    const path = resolve(this.proposalDirectory(proposalId), "review.json");
    if (!existsSync(path)) return null;
    const value = this.store.readJson(path, `${proposalId}/review.json`);
    if (
      (value.decision !== "approved" && value.decision !== "rejected") ||
      typeof value.proposal_sha256 !== "string" ||
      typeof value.note !== "string" ||
      typeof value.at !== "string"
    ) {
      throw new HostError("Refused", `review is malformed: ${proposalId}`);
    }
    return value as unknown as ReviewDocument;
  }

  private readAttempts(proposalId: string): JsonObject[] {
    const directory = resolve(this.proposalDirectory(proposalId), "attempts");
    if (!existsSync(directory)) return [];
    return readdirSync(directory)
      .filter((name) => /^\d+\.json$/.test(name))
      .sort((a, b) => Number.parseInt(a) - Number.parseInt(b))
      .map((name) => this.store.readJson(resolve(directory, name), `${proposalId}/attempts/${name}`));
  }

  private status(proposalId: string): ProposalSummary["status"] {
    const attempts = this.readAttempts(proposalId);
    if (attempts.length > 0) {
      const outcome = attempts.at(-1)?.outcome;
      if (outcome === "completed") return "executed";
      if (outcome === "failed") return "failed";
      return "incomplete";
    }
    const review = this.readReview(proposalId);
    if (!review) return "pending";
    return review.decision === "approved" ? "approved" : "rejected";
  }
}

function canonicalizeActionPaths(store: RootStore, action: InvestigationAction): InvestigationAction {
  switch (action.type) {
    case "fork":
      return {
        ...action,
        source_bundle: store.relative(store.path(action.source_bundle, { mustExist: true })),
        out: store.relative(store.path(action.out)),
      };
    case "snapshot":
      return {
        ...action,
        workspace: store.relative(store.path(action.workspace, { mustExist: true })),
        out: store.relative(store.path(action.out)),
      };
    case "adopt_candidate":
    case "run_recipe":
    case "record_decision":
    case "record_interpretation":
      return {
        ...action,
        workspace: store.relative(store.path(action.workspace, { mustExist: true })),
      };
  }
}

function isDescendantPath(candidate: string, parent: string): boolean {
  return candidate.toLowerCase().startsWith(`${parent.toLowerCase()}${sep}`);
}

function sourceManifestHasRecipe(store: RootStore, workspacePath: string, document: JsonObject, id: string): boolean {
  const source = document.source;
  if (typeof source !== "object" || source === null || Array.isArray(source)) return false;
  const manifestRef = (source as JsonObject).manifest;
  if (typeof manifestRef !== "object" || manifestRef === null || Array.isArray(manifestRef)) return false;
  const digest = (manifestRef as JsonObject).sha256;
  if (typeof digest !== "string") return false;
  let manifest: JsonObject;
  try {
    manifest = JSON.parse(store.objectBytes(workspacePath, digest).toString("utf8")) as JsonObject;
  } catch {
    throw new HostError("Refused", `source manifest sha256:${digest} is malformed`);
  }
  return store.manifestChain(manifest, workspacePath).some((item) =>
    arrayAt(item, "recipes").some((recipe) => recipe.id === id),
  );
}

function assertIdentifier(value: string, field: string): void {
  if (!/^[A-Za-z0-9._-]{1,128}$/.test(value)) {
    throw new HostError("MalformedArguments", `${field} must be a 1..=128 character evidence identifier`);
  }
}

function assertProposalCitation(value: string, field: string): void {
  if (/^sha256:[0-9a-f]{64}$/.test(value) || /^[A-Za-z0-9._-]{1,128}$/.test(value)) return;
  throw new HostError("MalformedArguments", `${field} must be an evidence name or sha256:<64 lowercase hex>`);
}

function assertDecisionCitation(value: string, field: string): void {
  if (value === "model" || value === "data" || /^[0-9a-f]{64}$/.test(value)) return;
  throw new HostError("MalformedArguments", `${field} must be model, data, or 64 lowercase hex`);
}

function assertEngineText(value: string, field: string): void {
  if (Buffer.byteLength(value, "utf8") > 16_384) {
    throw new HostError("MalformedArguments", `${field} must be at most 16384 UTF-8 bytes`);
  }
}

function safeDirectories(path: string): string[] {
  if (!existsSync(path)) return [];
  return readdirSync(path, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);
}
