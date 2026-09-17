import { HostError, type InvestigationAction } from "./types.js";
import type { JsonObject } from "./workspace.js";

export type InvestigationPhase =
  | "fork_required"
  | "inspect_required"
  | "model_revision_required"
  | "sample_required"
  | "diagnose_required"
  | "diagnostics_decision_required"
  | "check_required"
  | "snapshot_ready";

export interface PhaseFacts {
  kind: "workspace" | "bundle";
  inspection: JsonObject | null;
  hasFit: boolean;
  diagnostics: JsonObject | null;
  check: JsonObject | null;
  decisions: JsonObject[];
  diagnosticsSha256?: string;
}

export interface PhaseResult {
  phase: InvestigationPhase;
  diagnostics_thresholds: {
    max_rhat: number | null;
    divergences: number;
    exceeded: boolean;
    human_waiver_recorded: boolean;
  } | null;
}

export function derivePhase(facts: PhaseFacts): PhaseResult {
  if (facts.kind === "bundle") return { phase: "fork_required", diagnostics_thresholds: null };
  if (!facts.inspection) return { phase: "inspect_required", diagnostics_thresholds: null };
  const discrepancies = facts.inspection.structural_discrepancies;
  if (Array.isArray(discrepancies) && discrepancies.length > 0) {
    return { phase: "model_revision_required", diagnostics_thresholds: null };
  }
  if (!facts.hasFit) return { phase: "sample_required", diagnostics_thresholds: null };
  if (!facts.diagnostics) return { phase: "diagnose_required", diagnostics_thresholds: null };

  const perParameter = Array.isArray(facts.diagnostics.per_parameter)
    ? (facts.diagnostics.per_parameter as JsonObject[])
    : [];
  const rhats = perParameter
    .map((entry) => entry.rhat)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const maxRhat = rhats.length === 0 ? null : Math.max(...rhats);
  const divergences = typeof facts.diagnostics.divergences === "number"
    ? facts.diagnostics.divergences
    : 0;
  const exceeded = (maxRhat !== null && maxRhat > 1.01) || divergences > 0;
  const humanWaiverRecorded = exceeded && facts.diagnosticsSha256 !== undefined
    ? hasHumanWaiver(facts.decisions, facts.diagnosticsSha256)
    : false;
  const thresholds = {
    max_rhat: maxRhat,
    divergences,
    exceeded,
    human_waiver_recorded: humanWaiverRecorded,
  };
  if (exceeded && !humanWaiverRecorded) {
    return { phase: "diagnostics_decision_required", diagnostics_thresholds: thresholds };
  }
  if (!facts.check) return { phase: "check_required", diagnostics_thresholds: thresholds };
  return { phase: "snapshot_ready", diagnostics_thresholds: thresholds };
}

export function assertActionAllowed(phase: InvestigationPhase, action: InvestigationAction): void {
  if (action.type === "fork") {
    if (phase === "fork_required") return;
    refuse(phase, action.type, "fork requires a verified bundle");
  }
  if (phase === "fork_required") refuse(phase, action.type, "choose a workspace by forking the bundle first");

  if (action.type === "snapshot" && phase !== "snapshot_ready") {
    refuse(phase, action.type, "complete inspection, sampling, diagnostics, threshold review, and checking first");
  }
  if (action.type !== "run_recipe") return;

  switch (action.recipe.operation) {
    case "inspect":
      return;
    case "sample":
      if (phase !== "inspect_required" && phase !== "model_revision_required") return;
      refuse(phase, action.recipe.operation, "obtain a discrepancy-free current inspection first");
    case "diagnose":
      if (["diagnose_required", "diagnostics_decision_required", "check_required", "snapshot_ready"].includes(phase)) return;
      refuse(phase, action.recipe.operation, "obtain a current fit first");
    case "posterior-check":
      if (phase === "check_required" || phase === "snapshot_ready") return;
      if (phase === "diagnostics_decision_required") {
        refuse(
          phase,
          action.recipe.operation,
          "diagnostic thresholds were exceeded; execute a record_decision proposal citing the current diagnostics with recorded human approval, or rerun sampling",
        );
      }
      refuse(phase, action.recipe.operation, "obtain current diagnostics first");
  }
}

function hasHumanWaiver(decisions: JsonObject[], diagnosticsSha256: string): boolean {
  const digest = diagnosticsSha256.replace(/^sha256:/, "");
  const recommendations = new Set(
    decisions
      .filter(
        (decision) =>
          decision.kind === "agent_recommendation" &&
          typeof decision.id === "string" &&
          Array.isArray(decision.cites) &&
          decision.cites.some((citation) => citation === digest || citation === `sha256:${digest}`),
      )
      .map((decision) => decision.id as string),
  );
  return decisions.some(
    (decision) => decision.kind === "human_approval" && typeof decision.parent === "string" && recommendations.has(decision.parent),
  );
}

function refuse(phase: InvestigationPhase, action: string, repair: string): never {
  throw new HostError("PhaseRefused", `${action} is not allowed in phase ${phase}; ${repair}`);
}
