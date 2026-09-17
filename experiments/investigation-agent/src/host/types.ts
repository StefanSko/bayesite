import { Type, type Static, type TSchema } from "typebox";
import * as Value from "typebox/value";

const closed = { additionalProperties: false } as const;
const id = Type.String({ minLength: 1 });
const settingsInspect = Type.Object({}, closed);
const settingsSample = Type.Object(
  {
    chains: Type.Integer({ minimum: 1, maximum: 8 }),
    warmup: Type.Integer({ minimum: 0, maximum: 10_000 }),
    draws: Type.Integer({ minimum: 1, maximum: 10_000 }),
    max_treedepth: Type.Integer({ minimum: 1 }),
    target_accept: Type.Number({ exclusiveMinimum: 0, exclusiveMaximum: 1 }),
    initial_step_size: Type.Number({ exclusiveMinimum: 0 }),
    seed: Type.Integer(),
  },
  closed,
);
const settingsCheck = Type.Object({ seed: Type.Integer() }, closed);
const decision = Type.Object(
  {
    id,
    parent: Type.Union([Type.String(), Type.Null()]),
    reason: Type.String(),
    cites: Type.Array(Type.String({ minLength: 1 })),
  },
  closed,
);

export const RecipeSchema = Type.Union([
  Type.Object({ id, operation: Type.Literal("inspect"), settings: settingsInspect }, closed),
  Type.Object({ id, operation: Type.Literal("sample"), settings: settingsSample }, closed),
  Type.Object({ id, operation: Type.Literal("diagnose"), settings: settingsInspect }, closed),
  Type.Object({ id, operation: Type.Literal("posterior-check"), settings: settingsCheck }, closed),
]);

export const ActionSchema = Type.Union([
  Type.Object(
    { type: Type.Literal("fork"), source_bundle: id, at: id, out: id },
    closed,
  ),
  Type.Object(
    {
      type: Type.Literal("adopt_candidate"),
      workspace: id,
      candidate_id: Type.String({ pattern: "^c-[0-9a-f]{16}$" }),
      decision,
    },
    closed,
  ),
  Type.Object(
    { type: Type.Literal("run_recipe"), workspace: id, recipe: RecipeSchema, target: id },
    closed,
  ),
  Type.Object(
    { type: Type.Literal("snapshot"), workspace: id, out: id },
    closed,
  ),
  Type.Object(
    { type: Type.Literal("record_decision"), workspace: id, decision },
    closed,
  ),
  Type.Object(
    {
      type: Type.Literal("record_interpretation"),
      workspace: id,
      interpretation: Type.String(),
      unresolved_questions: Type.Array(Type.String()),
    },
    closed,
  ),
]);

export const ReadInvestigationArgumentsSchema = Type.Object({ path: id }, closed);
export const ReadEvidenceArgumentsSchema = Type.Object({ path: id, name: id }, closed);
export const PrepareCandidateArgumentsSchema = Type.Object(
  { workspace: id, model_json: Type.String(), note: Type.String() },
  closed,
);
export const SubmitProposalArgumentsSchema = Type.Object(
  { action: ActionSchema, rationale: Type.String(), cites: Type.Array(Type.String({ minLength: 1 })) },
  closed,
);

export type InvestigationAction = Static<typeof ActionSchema>;
export type Recipe = Static<typeof RecipeSchema>;
export type SubmitProposalArguments = Static<typeof SubmitProposalArgumentsSchema>;
export type PrepareCandidateArguments = Static<typeof PrepareCandidateArgumentsSchema>;

export const TOOL_NAMES = [
  "read_investigation",
  "read_evidence",
  "prepare_candidate",
  "submit_proposal",
] as const;

export type ErrorKind =
  | "InvalidPath"
  | "NotFound"
  | "MalformedArguments"
  | "EngineError"
  | "CandidateRejected"
  | "StalePreconditions"
  | "TargetMismatch"
  | "RecipeConflict"
  | "PhaseRefused"
  | "Refused";

export class HostError extends Error {
  constructor(
    readonly kind: ErrorKind,
    message: string,
    readonly engineError?: unknown,
  ) {
    super(message);
    this.name = "HostError";
  }

  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = { error: this.kind, message: this.message };
    if (this.engineError !== undefined) result.engine_error = this.engineError;
    return result;
  }
}

export function malformedUnless<T extends TSchema>(schema: T, value: unknown): asserts value is Static<T> {
  if (!Value.Check(schema, value)) {
    const first = [...Value.Errors(schema, value)][0];
    const detail = first ? first.message : "arguments do not match the schema";
    throw new HostError("MalformedArguments", detail);
  }
}

export interface Preconditions {
  state_sha256: string;
  engine_target: string;
}

export interface ProposalDocument {
  proposal_id: string;
  proposal_sha256: string;
  action: InvestigationAction;
  rationale: string;
  cites: string[];
  preconditions: Preconditions;
}

export interface ReviewDocument {
  decision: "approved" | "rejected";
  proposal_sha256: string;
  note: string;
  at: string;
  record_human_approval?: true;
}
