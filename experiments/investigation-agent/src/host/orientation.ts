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
    const highRhat = facts.diagnostics.per_parameter.some((item) => item.rhat !== null && item.rhat > 1.01);
    if (highRhat || facts.diagnostics.divergences > 0) {
      steps.push({
        step: "consider-sampler-revision",
        operation: "record decision; consider settings or reparameterisation",
        why: `Current diagnostics report ${facts.diagnostics.divergences} divergences${highRhat ? " and an R-hat above 1.01" : ""}.`,
      });
    }
    if (!facts.check) {
      steps.push({
        step: "check-posterior",
        operation: "posterior-check",
        why: "Current diagnostics exist, but no current posterior check exists.",
      });
    }
  }

  if (facts.check) {
    steps.push({
      step: "discuss-check",
      operation: "record interpretation or decision; consider fork",
      why: "A current posterior check is present; discuss what it does and does not support.",
    });
  }

  if (facts.allRecipesCurrent) {
    steps.push({
      step: "snapshot-investigation",
      operation: "snapshot",
      why: "Every declared recipe has current completed evidence.",
    });
  }
  return steps;
}
