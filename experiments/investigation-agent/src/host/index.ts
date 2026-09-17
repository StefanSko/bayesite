import { cpSync, existsSync, mkdtempSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { CandidateStore, inspectionSummary } from "./candidates.js";
import { Engine } from "./engine.js";
import { nextSteps } from "./orientation.js";
import { ProposalStore } from "./proposals.js";
import { HostError, ReadEvidenceArgumentsSchema, ReadInvestigationArgumentsSchema, malformedUnless } from "./types.js";
import {
  RootStore,
  arrayAt,
  artifactReference,
  objectAt,
  readFileOrNotFound,
  sha256,
  stringAt,
  type ArtifactReference,
  type JsonObject,
} from "./workspace.js";

interface EvidenceRecord {
  name: string;
  execution: string;
  status: "current" | "historical";
  origin: string;
  operation?: string;
  output?: ArtifactReference;
}

export interface HostOptions {
  engine?: string;
}

export class HostApi {
  readonly root: RootStore;
  readonly engine: Engine;
  readonly candidates: CandidateStore;
  readonly proposals: ProposalStore;

  constructor(root: string, options: HostOptions = {}) {
    this.root = new RootStore(root);
    this.engine = new Engine(options.engine ?? process.env.BAYESITE_BIN ?? defaultEnginePath());
    this.candidates = new CandidateStore(this.root, this.engine);
    this.proposals = new ProposalStore(this.root, this.engine, this.candidates);
  }

  async readInvestigation(arguments_: unknown): Promise<Record<string, unknown>> {
    malformedUnless(ReadInvestigationArgumentsSchema, arguments_);
    const requestedPath = arguments_.path;
    const absolute = this.root.path(requestedPath, { mustExist: true });
    const pathArg = this.root.relative(absolute);
    if (existsSync(resolve(absolute, "investigation.json"))) return await this.workspaceOrientation(pathArg);
    if (existsSync(resolve(absolute, "manifest.json"))) return await this.bundleOrientation(pathArg);
    throw new HostError("NotFound", `${requestedPath} is neither an investigation workspace nor a bundle`);
  }

  async readEvidence(arguments_: unknown): Promise<Record<string, unknown>> {
    malformedUnless(ReadEvidenceArgumentsSchema, arguments_);
    const absolute = this.root.path(arguments_.path, { mustExist: true });
    const orientation = await this.load(this.root.relative(absolute));
    let ref: ArtifactReference;
    let bytes: Buffer;
    if (arguments_.name === "model" || arguments_.name === "data") {
      ref = orientation.inputs[arguments_.name];
      if (orientation.kind === "workspace") {
        bytes = readFileOrNotFound(resolve(orientation.containerPath, "inputs", `${arguments_.name}.json`));
      } else {
        bytes = this.root.objectBytes(orientation.containerPath, ref.sha256);
      }
    } else {
      const evidence = orientation.evidence.find((entry) => entry.name === arguments_.name);
      if (!evidence?.output) throw new HostError("NotFound", `evidence does not exist: ${arguments_.name}`);
      ref = evidence.output;
      bytes = this.root.objectBytes(orientation.containerPath, ref.sha256);
    }
    const contentResult = evidenceContent(bytes, ref.kind);
    return {
      name: arguments_.name,
      sha256: `sha256:${ref.sha256}`,
      bytes: bytes.length,
      format: ref.format,
      content: contentResult.content,
      ...(contentResult.truncated ? { truncated: true } : {}),
    };
  }

  prepareCandidate(arguments_: unknown) {
    return this.candidates.prepare(arguments_);
  }

  submitProposal(arguments_: unknown) {
    return this.proposals.submit(arguments_);
  }

  listProposals(all = false) {
    return this.proposals.list(all);
  }

  showProposal(proposalId: string) {
    return this.proposals.show(proposalId);
  }

  approve(proposalId: string, note = "", recordHumanApproval = false) {
    return this.proposals.approve(proposalId, note, recordHumanApproval);
  }

  reject(proposalId: string, note = "") {
    return this.proposals.reject(proposalId, note);
  }

  execute(proposalId: string) {
    return this.proposals.execute(proposalId);
  }

  private async workspaceOrientation(pathArg: string): Promise<Record<string, unknown>> {
    const loaded = await this.load(pathArg);
    const document = loaded.document;
    const engineInspection = loaded.engineInspection as JsonObject;
    return this.orientationDocument(pathArg, loaded, {
      question: document.question,
      estimand: document.estimand,
      source: publicSource(document.source),
      decisions: publicDecisions(document),
      recipes: recipeStatuses(document, loaded.evidence, false),
      interpretation: document.interpretation ?? "",
      unresolved_questions: document.unresolved_questions ?? [],
      model_sha256: stringAt(engineInspection, "model_sha256"),
      data_sha256: stringAt(engineInspection, "data_sha256"),
    });
  }

  private async bundleOrientation(pathArg: string): Promise<Record<string, unknown>> {
    const loaded = await this.load(pathArg);
    const manifest = loaded.document;
    return this.orientationDocument(pathArg, loaded, {
      question: manifest.question,
      estimand: manifest.estimand,
      source: publicSource(manifest.source),
      decisions: publicDecisions(manifest),
      recipes: recipeStatuses(manifest, loaded.evidence, true),
      interpretation: manifest.interpretation ?? "",
      unresolved_questions: manifest.unresolved_questions ?? [],
      model_sha256: `sha256:${loaded.inputs.model.sha256}`,
      data_sha256: `sha256:${loaded.inputs.data.sha256}`,
    });
  }

  private orientationDocument(
    pathArg: string,
    loaded: LoadedInvestigation,
    fields: Record<string, unknown> & { recipes: Array<Record<string, unknown>> },
  ): Record<string, unknown> {
    const current = loaded.evidence.filter((entry) => entry.status === "current");
    const byOperation = (operation: string) => current.find((entry) => entry.operation === operation);
    const inspection = parseArtifact(this.root, loaded.containerPath, byOperation("inspect")?.output);
    const diagnosticsArtifact = parseArtifact(this.root, loaded.containerPath, byOperation("diagnose")?.output);
    const checkArtifact = parseArtifact(this.root, loaded.containerPath, byOperation("posterior-check")?.output);
    const diagnostics = diagnosticsArtifact ? diagnosticsSummary(diagnosticsArtifact) : null;
    const check = checkArtifact
      ? { summaries: Array.isArray(checkArtifact.checks) ? checkArtifact.checks : [] }
      : null;
    const inspectionWithMetadata = inspection ? inspectionSummary(inspection) : null;
    const inspectionPublic = inspectionWithMetadata
      ? {
          free_slots: Array.isArray(inspectionWithMetadata.free_slots) ? inspectionWithMetadata.free_slots : [],
          density_factors: Array.isArray(inspectionWithMetadata.density_factors)
            ? inspectionWithMetadata.density_factors
            : [],
          structural_discrepancies: Array.isArray(inspectionWithMetadata.structural_discrepancies)
            ? inspectionWithMetadata.structural_discrepancies
            : [],
        }
      : null;
    const publicEvidence = loaded.evidence.map(({ operation: _operation, output: _output, ...entry }) => entry);
    const recipesCurrent = allRecipesAreCurrent(loaded);
    const historicalOperations = loaded.evidence
      .filter((entry) => entry.status === "historical" && entry.operation)
      .map((entry) => ({ name: entry.name, operation: entry.operation as string }));
    const steps = nextSteps({
      kind: loaded.kind,
      inspection: inspectionPublic,
      hasFit: Boolean(byOperation("sample")),
      diagnostics,
      check,
      allRecipesCurrent: recipesCurrent,
      inheritedHistorical: historicalOperations.length > 0,
      historicalOperations,
    });
    const proposalSummaries = this.proposals
      .list(true)
      .filter((summary) => proposalTouchesPath(this.proposals.show(summary.proposal_id).proposal as JsonObject, pathArg));
    return {
      orientation_format: "v0-experimental",
      kind: loaded.kind,
      path: pathArg,
      ...fields,
      state_sha256: loaded.preconditions.state_sha256,
      engine_target: loaded.preconditions.engine_target,
      evidence: publicEvidence,
      inspection: inspectionPublic,
      diagnostics,
      check,
      next_steps: steps,
      proposals: proposalSummaries,
    };
  }

  private async load(pathArg: string): Promise<LoadedInvestigation> {
    const absolute = this.root.path(pathArg, { mustExist: true });
    if (existsSync(resolve(absolute, "investigation.json"))) {
      const { document } = this.root.workspaceDocument(pathArg);
      const engineInspection = await this.inspectWorkspaceReadOnly(absolute);
      const inputs = {
        model: workspaceRef(resolve(absolute, "inputs", "model.json"), "model_ir", "bayeswire-ir-v1"),
        data: workspaceRef(resolve(absolute, "inputs", "data.json"), "data", "bayesite-data-json-v1"),
      };
      return {
        kind: "workspace",
        containerPath: absolute,
        document,
        engineInspection,
        inputs,
        evidence: workspaceEvidence(this.root, absolute, document, engineInspection),
        preconditions: this.root.workspacePreconditions(pathArg),
      };
    }
    if (existsSync(resolve(absolute, "manifest.json"))) {
      const { manifest } = this.root.bundleManifest(pathArg);
      const verification = (await this.engine.run(["investigation", "verify", absolute])).json;
      const inputsObject = objectAt(manifest, "inputs");
      return {
        kind: "bundle",
        containerPath: absolute,
        document: manifest,
        inputs: {
          model: artifactReference(inputsObject.model),
          data: artifactReference(inputsObject.data),
        },
        evidence: manifestEvidence(this.root, absolute, manifest),
        preconditions: {
          state_sha256: stringAt(verification, "snapshot_id"),
          engine_target: this.root.bundleTarget(manifest, absolute),
        },
      };
    }
    throw new HostError("NotFound", `${pathArg} is neither an investigation workspace nor a bundle`);
  }

  private async inspectWorkspaceReadOnly(workspace: string): Promise<JsonObject> {
    const temporary = mkdtempSync(resolve(tmpdir(), "bayesite-investigation-read-"));
    const copy = resolve(temporary, "workspace");
    try {
      cpSync(workspace, copy, { recursive: true });
      return (await this.engine.run(["investigation", "inspect", copy])).json;
    } finally {
      rmSync(temporary, { recursive: true, force: true });
    }
  }
}

interface LoadedInvestigation {
  kind: "workspace" | "bundle";
  containerPath: string;
  document: JsonObject;
  engineInspection?: JsonObject;
  inputs: { model: ArtifactReference; data: ArtifactReference };
  evidence: EvidenceRecord[];
  preconditions: { state_sha256: string; engine_target: string };
}

function workspaceRef(path: string, kind: string, format: string): ArtifactReference {
  const bytes = readFileOrNotFound(path);
  return { sha256: sha256(bytes), bytes: bytes.length, kind, format };
}

function workspaceEvidence(store: RootStore, path: string, document: JsonObject, inspection: JsonObject): EvidenceRecord[] {
  const localAttempts = arrayAt(document, "attempts");
  let sourceManifest: JsonObject | undefined;
  const source = document.source;
  if (typeof source === "object" && source !== null && !Array.isArray(source)) {
    const manifestRef = (source as JsonObject).manifest;
    if (typeof manifestRef === "object" && manifestRef !== null && !Array.isArray(manifestRef)) {
      const digest = (manifestRef as JsonObject).sha256;
      if (typeof digest === "string") {
        sourceManifest = JSON.parse(store.objectBytes(path, digest).toString("utf8")) as JsonObject;
      }
    }
  }
  return arrayAt(inspection, "evidence").map((entry) => {
    const name = stringAt(entry, "name");
    const executionId = stringAt(entry, "execution");
    const status = entry.status === "historical" ? "historical" : "current";
    const local = localAttempts.find((attempt) => {
      const execution = attempt.execution;
      return typeof execution === "object" && execution !== null && (execution as JsonObject).id === executionId;
    });
    let execution: JsonObject | undefined;
    let recipe: JsonObject | undefined;
    if (status === "historical" && sourceManifest) {
      const found = store.findExecution(sourceManifest, path, executionId);
      execution = found?.execution;
      recipe = found?.recipe;
    }
    if (!execution && local) {
      execution = objectAt(local, "execution");
      recipe = objectAt(local, "recipe");
    }
    if (!execution && sourceManifest) {
      const found = store.findExecution(sourceManifest, path, executionId);
      execution = found?.execution;
      recipe = found?.recipe;
    }
    const output = execution?.output ? artifactReference(execution.output) : undefined;
    return {
      name,
      execution: executionId,
      status,
      origin: typeof entry.origin === "string" ? entry.origin : status === "current" ? "workspace" : "source_snapshot",
      ...(typeof recipe?.operation === "string" ? { operation: recipe.operation } : {}),
      ...(output ? { output } : {}),
    };
  });
}

function manifestEvidence(store: RootStore, path: string, manifest: JsonObject): EvidenceRecord[] {
  return arrayAt(manifest, "evidence").map((entry) => {
    const executionId = stringAt(entry, "execution");
    const found = store.findExecution(manifest, path, executionId);
    const output = found?.execution.output ? artifactReference(found.execution.output) : undefined;
    const status = entry.status === "historical" ? "historical" : "current";
    return {
      name: stringAt(entry, "name"),
      execution: executionId,
      status,
      origin: status === "current" ? "bundle" : "source_snapshot",
      ...(typeof found?.recipe?.operation === "string" ? { operation: found.recipe.operation } : {}),
      ...(output ? { output } : {}),
    };
  });
}

function recipeStatuses(document: JsonObject, _evidence: EvidenceRecord[], bundle: boolean): Array<Record<string, unknown>> {
  const executions = bundle ? arrayAt(document, "executions") : arrayAt(document, "attempts").map((entry) => objectAt(entry, "execution"));
  return arrayAt(document, "recipes").map((recipe) => {
    const id = stringAt(recipe, "id");
    const related = executions.filter((execution) => execution.recipe === id);
    const latest = related.at(-1);
    let status = "unrun";
    if (latest?.outcome === "completed") status = "completed";
    else if (latest?.outcome === "failed") status = "failed";
    else if (latest) status = "incomplete";
    return {
      id,
      operation: recipe.operation,
      settings: recipe.settings,
      status,
    };
  });
}

function publicSource(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const source = value as JsonObject;
  const snapshot = source.snapshot_id;
  const decision = source.decision;
  if (typeof snapshot !== "string" || typeof decision !== "string") return null;
  return {
    snapshot_id: snapshot.startsWith("sha256:") ? snapshot : `sha256:${snapshot}`,
    decision,
  };
}

function allRecipesAreCurrent(loaded: LoadedInvestigation): boolean {
  const recipes = arrayAt(loaded.document, "recipes");
  if (recipes.length === 0) return false;
  const executions = loaded.kind === "bundle"
    ? arrayAt(loaded.document, "executions")
    : arrayAt(loaded.document, "attempts").map((entry) => objectAt(entry, "execution"));
  const currentIds = new Set(loaded.evidence.filter((entry) => entry.status === "current").map((entry) => entry.execution));
  return recipes.every((recipe) =>
    executions.some(
      (execution) => execution.recipe === recipe.id && execution.outcome === "completed" && currentIds.has(String(execution.id)),
    ),
  );
}

function publicDecisions(document: JsonObject): Array<Record<string, unknown>> {
  return arrayAt(document, "decisions").map((decision) => ({
    id: decision.id,
    parent: decision.parent ?? null,
    kind: decision.kind,
    reason: decision.reason,
    cites: decision.cites,
  }));
}

function parseArtifact(store: RootStore, path: string, ref: ArtifactReference | undefined): JsonObject | null {
  if (!ref) return null;
  try {
    const value = JSON.parse(store.objectBytes(path, ref.sha256).toString("utf8")) as unknown;
    return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as JsonObject) : null;
  } catch {
    return null;
  }
}

function diagnosticsSummary(artifact: JsonObject) {
  const rhat = typeof artifact.rhat === "object" && artifact.rhat !== null ? (artifact.rhat as JsonObject) : {};
  const ess = typeof artifact.ess === "object" && artifact.ess !== null ? (artifact.ess as JsonObject) : {};
  const names = new Set([...Object.keys(rhat), ...Object.keys(ess)]);
  const per_parameter = [...names].map((name) => ({
    name,
    rhat: typeof rhat[name] === "number" ? rhat[name] : null,
    ess: typeof ess[name] === "number" ? ess[name] : null,
  }));
  const divergences = arrayAt(artifact, "chains").reduce(
    (sum, chain) => sum + (typeof chain.divergences === "number" ? chain.divergences : 0),
    0,
  );
  return { per_parameter, divergences };
}

function evidenceContent(bytes: Buffer, kind: string): { content: string; truncated: boolean } {
  const text = bytes.toString("utf8");
  if (kind === "posterior_draws") {
    const lines = text.trimEnd().split("\n");
    if (lines.length <= 52) return { content: text, truncated: false };
    const trailers = lines.filter((line) => {
      try {
        const parsed = JSON.parse(line) as JsonObject;
        return typeof parsed.trailer === "object" && parsed.trailer !== null;
      } catch {
        return false;
      }
    });
    return { content: [...lines.slice(0, 51), ...trailers].join("\n") + "\n", truncated: true };
  }
  const limit = 256 * 1024;
  if (bytes.length <= limit) return { content: text, truncated: false };
  return { content: bytes.subarray(0, limit).toString("utf8"), truncated: true };
}

function proposalTouchesPath(proposal: JsonObject, path: string): boolean {
  const action = proposal.action;
  if (typeof action !== "object" || action === null || Array.isArray(action)) return false;
  const value = action as JsonObject;
  return value.workspace === path || value.source_bundle === path || value.out === path;
}

function defaultEnginePath(): string {
  return fileURLToPath(new URL("../../../../../target/release/bayesite", import.meta.url));
}

export { HostError } from "./types.js";
export type { InvestigationAction, ProposalDocument, ReviewDocument } from "./types.js";
