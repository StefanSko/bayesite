import type { DiagnosticsThresholds, InvestigationPhase } from "./phase.js";

export interface NextStep {
  step: string;
  operation: string;
  why: string;
}

export interface OrientationFacts {
  kind: "workspace" | "bundle";
  inspection: { structural_discrepancies?: unknown[] } | null;
  hasFit: boolean;
  diagnostics: { per_parameter: Array<{ name: string; rhat: number | null; ess: number | null }>; divergences: number } | null;
  check: { summaries: unknown[] } | null;
  allRecipesCurrent: boolean;
  inheritedHistorical: boolean;
  historicalOperations: Array<{ name: string; operation: string }>;
  phase?: InvestigationPhase;
  diagnosticsThresholds?: DiagnosticsThresholds | null;
  simulationWaiverRecorded?: boolean;
}

export function nextSteps(facts: OrientationFacts): NextStep[] {
  if (facts.kind === "bundle") {
    return [
      {
        step: "fork-bundle",
        operation: "fork",
        why: "A snapshot bundle is immutable; continue by forking it at a named decision.",
      },
    ];
  }

  const steps: NextStep[] = [];
  if (facts.inheritedHistorical && !facts.hasFit) {
    for (const item of facts.historicalOperations) {
      steps.push({
        step: `rerun-${item.name}`,
        operation: item.operation,
        why: `Inherited evidence ${item.name} is historical and there is no new current fit.`,
      });
    }
  }
  if (!facts.inspection) {
    steps.push({
      step: "inspect-model",
      operation: "inspect",
      why: "There is no current effective-model inspection.",
    });
  }

  const discrepancies = facts.inspection?.structural_discrepancies ?? [];
  if (discrepancies.length > 0) {
    steps.push({
      step: "fix-structural-discrepancies",
      operation: "record decision and prepare a candidate",
      why: "The current inspection reports structural discrepancies; fix the model and record why before sampling.",
    });
  } else if (facts.inspection && !facts.hasFit) {
    steps.push({
      step: "sample-model",
      operation: "sample",
      why: "A current inspection exists, but no current fit exists.",
    });
  }

  if (facts.hasFit && !facts.diagnostics) {
    steps.push({
      step: "diagnose-fit",
      operation: "diagnose",
      why: "A current fit exists, but no current diagnostics exist.",
    });
  }

  if (facts.diagnostics) {
    const thresholds = facts.diagnosticsThresholds;
    if (facts.phase === "diagnostics_decision_required" && thresholds?.threshold_decision === null) {
      steps.push({
        step: "record-thresholds",
        operation: "record_decision",
        why: `Current diagnostics report max R-hat ${formatMetric(thresholds.max_rhat)}, min ESS ${formatMetric(thresholds.min_ess)}, and ${thresholds.divergences} divergences; record approved thresholds, using the conventional values 1.01, 400, 0 as a starting point (rhat<=1.01 ess>=400 divergences<=0).`,
      });
    } else if (facts.phase === "diagnostics_decision_required") {
      steps.push({
        step: "resolve-diagnostics-thresholds",
        operation: "record_decision",
        why: `Current diagnostics exceed recorded threshold decision ${thresholds?.threshold_decision ?? "unknown"}; revise the run or record a diagnostics-citing waiver with explicit human approval.`,
      });
    }
    if (!facts.check && facts.phase === "check_required") {
      steps.push({
        step: "check-posterior",
        operation: "posterior-check",
        why: "Current diagnostics satisfy an approved threshold decision or have an approved diagnostics waiver, but no current posterior check exists.",
      });
    }
  }

  if (
    (facts.phase === "sample_required" || facts.phase === "check_required") &&
    facts.simulationWaiverRecorded !== true
  ) {
    steps.push({
      step: "simulation-unsupported",
      operation: "record_decision",
      why: "no prior-predictive or recovery evidence can be recorded by this format version; record a waiver decision (reason starting with 'waiver:') with human approval to proceed knowingly",
    });
  }

  if (facts.check) {
    steps.push({
      step: "discuss-check",
      operation: "record interpretation or decision; consider fork",
      why: "A current posterior check is present; discuss what it does and does not support.",
    });
  }

  if (facts.allRecipesCurrent && facts.phase !== "diagnostics_decision_required") {
    steps.push({
      step: "snapshot-investigation",
      operation: "snapshot",
      why: "Every declared recipe has current completed evidence.",
    });
  }
  return steps;
}

function formatMetric(value: number | null): string {
  return value === null ? "unavailable" : String(value);
}
