import { randomUUID } from "node:crypto";
import { defineTool } from "@earendil-works/pi-coding-agent";
import type { Static, TSchema } from "typebox";
import * as Value from "typebox/value";
import type { HostApi } from "../host/index.js";
import { HostError } from "../host/index.js";
import {
  PrepareCandidateArgumentsSchema,
  ReadEvidenceArgumentsSchema,
  ReadInvestigationArgumentsSchema,
  SubmitProposalArgumentsSchema,
} from "../host/types.js";

function result(value: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(value) }], details: value };
}

const malformedMarkers = new Set<string>();

function prepared<T extends TSchema>(schema: T, args: unknown, fallback: (marker: string) => Static<T>): Static<T> {
  if (Value.Check(schema, args)) return args as Static<T>;
  const marker = `__bayesite_malformed_${randomUUID()}__`;
  malformedMarkers.add(marker);
  return fallback(marker);
}

function refusePreparedMalformed(value: string): void {
  if (!malformedMarkers.delete(value)) return;
  throw new HostError("MalformedArguments", "arguments do not match the closed tool schema");
}

async function invoke(operation: () => unknown | Promise<unknown>) {
  try {
    return result(await operation());
  } catch (error) {
    if (error instanceof HostError) throw new Error(JSON.stringify(error.toJSON()));
    throw error;
  }
}

export function investigationTools(host: HostApi) {
  return [
    defineTool({
      name: "read_investigation",
      label: "Read investigation",
      description: "Read verified orientation facts for one root-relative workspace or bundle.",
      parameters: ReadInvestigationArgumentsSchema,
      prepareArguments: (args) => prepared(ReadInvestigationArgumentsSchema, args, (marker) => ({ path: marker })),
      execute: async (_id, params) =>
        await invoke(() => {
          refusePreparedMalformed(params.path);
          return host.readInvestigation(params);
        }),
    }),
    defineTool({
      name: "read_evidence",
      label: "Read evidence",
      description: "Read model, data, or named investigation evidence by root-relative investigation path.",
      parameters: ReadEvidenceArgumentsSchema,
      prepareArguments: (args) =>
        prepared(ReadEvidenceArgumentsSchema, args, (marker) => ({ path: marker, name: marker })),
      execute: async (_id, params) =>
        await invoke(() => {
          refusePreparedMalformed(params.path);
          return host.readEvidence(params);
        }),
    }),
    defineTool({
      name: "prepare_candidate",
      label: "Prepare candidate",
      description: "Store and inspect candidate model JSON without changing a workspace.",
      parameters: PrepareCandidateArgumentsSchema,
      prepareArguments: (args) =>
        prepared(PrepareCandidateArgumentsSchema, args, (marker) => ({ workspace: marker, model_json: "", note: "" })),
      execute: async (_id, params) =>
        await invoke(() => {
          refusePreparedMalformed(params.workspace);
          return host.prepareCandidate(params);
        }),
    }),
    defineTool({
      name: "submit_proposal",
      label: "Submit proposal",
      description: "Persist one closed action proposal for later human review; never approves or executes it.",
      parameters: SubmitProposalArgumentsSchema,
      prepareArguments: (args) =>
        prepared(SubmitProposalArgumentsSchema, args, (marker) => ({
          action: {
            type: "record_interpretation" as const,
            workspace: marker,
            interpretation: "",
            unresolved_questions: [],
          },
          rationale: marker,
          cites: [],
        })),
      execute: async (_id, params) =>
        await invoke(() => {
          refusePreparedMalformed(params.rationale);
          return host.submitProposal(params);
        }),
    }),
  ];
}
