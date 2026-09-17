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

export interface RecordedThresholds {
  rhat_max?: number;
  ess_min?: number;
  divergences_max?: number;
}

export interface ApprovedDecision<T = undefined> {
  id: string;
  value: T;
}

export interface DecisionFacts {
  threshold: ApprovedDecision<RecordedThresholds> | null;
  waiver: ApprovedDecision | null;
  diagnosticsWaiver: ApprovedDecision | null;
}

export interface DiagnosticsThresholds {
  max_rhat: number | null;
  min_ess: number | null;
  divergences: number;
  threshold_decision: string | null;
  limits: RecordedThresholds | null;
  exceeded: boolean | null;
  human_waiver_recorded: boolean;
  reason: string;
}

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
  diagnostics_thresholds: DiagnosticsThresholds | null;
  decision_facts: DecisionFacts;
}

const WORKSPACE_BASE_ACTIONS = [
  "adopt_candidate",
  "record_decision",
  "record_interpretation",
  "run_recipe:inspect",
] as const;

export function derivePhase(facts: PhaseFacts): PhaseResult {
  const decisionFacts = analyzeDecisions(facts.decisions, facts.diagnosticsSha256);
  if (facts.kind === "bundle") {
    return { phase: "fork_required", diagnostics_thresholds: null, decision_facts: decisionFacts };
  }
  if (!facts.inspection) {
    return { phase: "inspect_required", diagnostics_thresholds: null, decision_facts: decisionFacts };
  }
  const discrepancies = facts.inspection.structural_discrepancies;
  if (Array.isArray(discrepancies) && discrepancies.length > 0) {
    return { phase: "model_revision_required", diagnostics_thresholds: null, decision_facts: decisionFacts };
  }
  if (!facts.hasFit) {
    return { phase: "sample_required", diagnostics_thresholds: null, decision_facts: decisionFacts };
  }
  if (!facts.diagnostics) {
    return { phase: "diagnose_required", diagnostics_thresholds: null, decision_facts: decisionFacts };
  }

  const perParameter = Array.isArray(facts.diagnostics.per_parameter)
    ? (facts.diagnostics.per_parameter as JsonObject[])
    : [];
  const rhats = perParameter
    .map((entry) => entry.rhat)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const esses = perParameter
    .map((entry) => entry.ess)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const maxRhat = rhats.length === 0 ? null : Math.max(...rhats);
  const minEss = esses.length === 0 ? null : Math.min(...esses);
  const divergences = typeof facts.diagnostics.divergences === "number"
    ? facts.diagnostics.divergences
    : 0;

  if (!decisionFacts.threshold) {
    return {
      phase: "diagnostics_decision_required",
      diagnostics_thresholds: {
        max_rhat: maxRhat,
        min_ess: minEss,
        divergences,
        threshold_decision: null,
        limits: null,
        exceeded: null,
        human_waiver_recorded: false,
        reason: "no recorded threshold decision",
      },
      decision_facts: decisionFacts,
    };
  }

  const limits = decisionFacts.threshold.value;
  const exceeded =
    (limits.rhat_max !== undefined && maxRhat !== null && maxRhat > limits.rhat_max) ||
    (limits.ess_min !== undefined && minEss !== null && minEss < limits.ess_min) ||
    (limits.divergences_max !== undefined && divergences > limits.divergences_max);
  const humanWaiverRecorded = exceeded && decisionFacts.diagnosticsWaiver !== null;
  const thresholds: DiagnosticsThresholds = {
    max_rhat: maxRhat,
    min_ess: minEss,
    divergences,
    threshold_decision: decisionFacts.threshold.id,
    limits,
    exceeded,
    human_waiver_recorded: humanWaiverRecorded,
    reason: exceeded
      ? `recorded threshold decision ${decisionFacts.threshold.id} was exceeded`
      : `recorded threshold decision ${decisionFacts.threshold.id} was satisfied`,
  };
  if (exceeded && !humanWaiverRecorded) {
    return { phase: "diagnostics_decision_required", diagnostics_thresholds: thresholds, decision_facts: decisionFacts };
  }
  if (!facts.check) return { phase: "check_required", diagnostics_thresholds: thresholds, decision_facts: decisionFacts };
  return { phase: "snapshot_ready", diagnostics_thresholds: thresholds, decision_facts: decisionFacts };
}

export function allowedActions(phase: InvestigationPhase): string[] {
  if (phase === "fork_required") return ["fork"];
  const actions: string[] = [...WORKSPACE_BASE_ACTIONS];
  if (phase !== "inspect_required" && phase !== "model_revision_required") {
    actions.push("run_recipe:sample");
  }
  if (["diagnose_required", "diagnostics_decision_required", "check_required", "snapshot_ready"].includes(phase)) {
    actions.push("run_recipe:diagnose");
  }
  if (phase === "check_required" || phase === "snapshot_ready") {
    actions.push("run_recipe:posterior-check");
  }
  if (phase === "snapshot_ready") actions.push("snapshot");
  return actions;
}

export function assertActionAllowed(
  phase: InvestigationPhase,
  action: InvestigationAction,
  thresholdDecisionId: string | null = null,
): void {
  const key = action.type === "run_recipe" ? `run_recipe:${action.recipe.operation}` : action.type;
  if (allowedActions(phase).includes(key)) return;

  if (action.type === "fork") refuse(phase, action.type, "fork requires a verified bundle");
  if (phase === "fork_required") refuse(phase, action.type, "choose a workspace by forking the bundle first");
  if (action.type === "snapshot") {
    refuse(phase, action.type, "complete inspection, sampling, diagnostics, threshold review, and checking first");
  }
  if (action.type !== "run_recipe") refuse(phase, action.type, "the action is not available in this phase");
  switch (action.recipe.operation) {
    case "sample":
      refuse(phase, action.recipe.operation, "obtain a discrepancy-free current inspection first");
    case "diagnose":
      refuse(phase, action.recipe.operation, "obtain a current fit first");
    case "posterior-check":
      if (phase === "diagnostics_decision_required") {
        const thresholdFact = thresholdDecisionId === null
          ? "no recorded threshold decision"
          : `recorded threshold decision ${thresholdDecisionId} was exceeded`;
        refuse(
          phase,
          action.recipe.operation,
          `${thresholdFact}; execute an approved record_decision proposal for thresholds or a diagnostics-citing waiver, or rerun sampling`,
        );
      }
      refuse(phase, action.recipe.operation, "obtain current diagnostics first");
    case "inspect":
      refuse(phase, action.recipe.operation, "inspection is unavailable");
  }
}

export function analyzeDecisions(decisions: JsonObject[], diagnosticsSha256?: string): DecisionFacts {
  const approvedIds = new Set(
    decisions
      .filter((decision) => decision.kind === "human_approval" && typeof decision.parent === "string")
      .map((decision) => decision.parent as string),
  );
  let threshold: ApprovedDecision<RecordedThresholds> | null = null;
  let waiver: ApprovedDecision | null = null;
  let diagnosticsWaiver: ApprovedDecision | null = null;
  const digest = diagnosticsSha256?.replace(/^sha256:/, "");
  for (const decision of decisions) {
    if (typeof decision.id !== "string" || typeof decision.reason !== "string" || !approvedIds.has(decision.id)) continue;
    const parsed = parseThresholdReason(decision.reason);
    if (parsed) threshold = { id: decision.id, value: parsed };
    if (decision.reason.startsWith("waiver:")) {
      const approved = { id: decision.id, value: undefined };
      waiver = approved;
      if (
        digest !== undefined &&
        Array.isArray(decision.cites) &&
        decision.cites.some((citation) => citation === digest || citation === `sha256:${digest}`)
      ) {
        diagnosticsWaiver = approved;
      }
    }
  }
  return { threshold, waiver, diagnosticsWaiver };
}

export function parseThresholdReason(reason: string): RecordedThresholds | null {
  const prefix = /^thresholds\s*:\s*/.exec(reason);
  if (!prefix) return null;
  const rest = reason.slice(prefix[0].length).trim();
  if (!rest) return null;
  const result: RecordedThresholds = {};
  const seen = new Set<string>();
  const token = /^(rhat\s*<=\s*(\d+(?:\.\d+)?|\.\d+)|ess\s*>=\s*(\d+(?:\.\d+)?|\.\d+)|divergences\s*<=\s*(\d+(?:\.\d+)?|\.\d+))(?:\s+|$)/;
  let remaining = rest;
  while (remaining.length > 0) {
    const match = token.exec(remaining);
    if (!match) return null;
    if (match[2] !== undefined) {
      const value = Number(match[2]);
      if (seen.has("rhat") || !Number.isFinite(value)) return null;
      seen.add("rhat");
      result.rhat_max = value;
    } else if (match[3] !== undefined) {
      const value = Number(match[3]);
      if (seen.has("ess") || !Number.isFinite(value)) return null;
      seen.add("ess");
      result.ess_min = value;
    } else if (match[4] !== undefined) {
      const value = Number(match[4]);
      if (seen.has("divergences") || !Number.isFinite(value)) return null;
      seen.add("divergences");
      result.divergences_max = value;
    }
    remaining = remaining.slice(match[0].length).trimStart();
  }
  return seen.size > 0 ? result : null;
}

function refuse(phase: InvestigationPhase, action: string, repair: string): never {
  throw new HostError("PhaseRefused", `${action} is not allowed in phase ${phase}; ${repair}`);
}
