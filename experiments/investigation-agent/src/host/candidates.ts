import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { Engine } from "./engine.js";
import { HostError, PrepareCandidateArgumentsSchema, malformedUnless, type PrepareCandidateArguments } from "./types.js";
import { RootStore, sha256, shaIdentifier, type JsonObject } from "./workspace.js";

export class CandidateStore {
  constructor(
    private readonly store: RootStore,
    private readonly engine: Engine,
  ) {}

  async prepare(arguments_: unknown): Promise<Record<string, unknown>> {
    malformedUnless(PrepareCandidateArgumentsSchema, arguments_);
    const args: PrepareCandidateArguments = arguments_;
    const workspace = this.store.workspaceDocument(args.workspace).path;
    const bytes = Buffer.from(args.model_json, "utf8");
    const digest = sha256(bytes);
    const candidateId = `c-${digest.slice(0, 16)}`;
    const candidatePath = resolve(this.store.stateDir, "candidates", `${candidateId}.json`);
    if (existsSync(candidatePath)) {
      const existing = readFileSync(candidatePath);
      if (!existing.equals(bytes)) {
        throw new HostError("CandidateRejected", `candidate id collision for ${candidateId}; existing bytes were preserved`);
      }
    } else {
      writeFileSync(candidatePath, bytes, { flag: "wx" });
    }
    const dataPath = resolve(workspace, "inputs", "data.json");
    const result = await this.engine.run([
      "inspect",
      "--model",
      candidatePath,
      "--data",
      dataPath,
    ]);
    const inspectionPath = resolve(this.store.stateDir, "candidates", `${candidateId}.inspection.json`);
    writeFileSync(
      inspectionPath,
      `${JSON.stringify({
        ...result.json,
        candidate_workspace: this.store.relative(workspace),
        candidate_data_sha256: shaIdentifier(readFileSync(dataPath)),
      })}\n`,
    );
    return {
      candidate_id: candidateId,
      sha256: `sha256:${digest}`,
      inspection: inspectionSummary(result.json),
    };
  }

  get(candidateId: string): { bytes: Buffer; inspection: JsonObject; path: string } {
    if (!/^c-[0-9a-f]{16}$/.test(candidateId)) {
      throw new HostError("MalformedArguments", `invalid candidate id: ${candidateId}`);
    }
    const path = resolve(this.store.stateDir, "candidates", `${candidateId}.json`);
    const inspectionPath = resolve(this.store.stateDir, "candidates", `${candidateId}.inspection.json`);
    if (!existsSync(path) || !existsSync(inspectionPath)) {
      throw new HostError("NotFound", `candidate does not exist: ${candidateId}`);
    }
    const bytes = readFileSync(path);
    if (`c-${sha256(bytes).slice(0, 16)}` !== candidateId) {
      throw new HostError("CandidateRejected", `candidate bytes do not match ${candidateId}`);
    }
    return { bytes, inspection: this.store.readJson(inspectionPath, `${candidateId} inspection`), path };
  }
}

export function inspectionSummary(inspection: JsonObject): Record<string, unknown> {
  return {
    free_slots: Array.isArray(inspection.free_slots) ? inspection.free_slots : [],
    density_factors: Array.isArray(inspection.density_factors) ? inspection.density_factors : [],
    structural_discrepancies: Array.isArray(inspection.structural_discrepancies)
      ? inspection.structural_discrepancies
      : [],
    execution_metadata:
      typeof inspection.execution_metadata === "object" && inspection.execution_metadata !== null
        ? inspection.execution_metadata
        : {},
  };
}
