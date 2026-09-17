import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { HostApi, HostError } from "../src/host/index.js";
import { assertActionAllowed, derivePhase, parseThresholdReason } from "../src/host/phase.js";
import type { InvestigationAction } from "../src/host/types.js";
import { engineBinary, initStudy, runEngine, temporaryRoot } from "./fixture.js";

process.env.BAYESITE_BIN = engineBinary;

const BASE_ACTIONS = [
  "adopt_candidate",
  "record_decision",
  "record_interpretation",
  "run_recipe:inspect",
];

function operations(orientation: Record<string, unknown>): string[] {
  return (orientation.next_steps as Array<{ operation: string }>).map((step) => step.operation);
}

function steps(orientation: Record<string, unknown>): Array<{ step: string; operation: string; why: string }> {
  return orientation.next_steps as Array<{ step: string; operation: string; why: string }>;
}

function assertAllowed(orientation: Record<string, unknown>, expected: string[]): void {
  assert.deepEqual(orientation.allowed_actions, expected);
  assert.ok(typeof orientation.phase_facts === "object" && orientation.phase_facts !== null);
}

test("recorded threshold and waiver decisions control diagnostics phase", () => {
  const digest = "a".repeat(64);
  const base = {
    kind: "workspace" as const,
    inspection: { structural_discrepancies: [] },
    hasFit: true,
    diagnostics: { per_parameter: [{ name: "rate", rhat: 1.02, ess: 100 }], divergences: 1 },
    check: null,
    diagnosticsSha256: digest,
  };
  const threshold = {
    id: "diagnostic-thresholds",
    parent: "initial-likelihood",
    kind: "agent_recommendation",
    reason: "thresholds:   rhat <= 1.01   ess >= 400 divergences <= 0",
    cites: [digest],
  };
  const thresholdApproval = {
    id: "diagnostic-thresholds-human-approval",
    parent: "diagnostic-thresholds",
    kind: "human_approval",
    reason: "Approved thresholds",
    cites: ["model"],
  };
  const waiver = {
    id: "diagnostics-waiver",
    parent: "diagnostic-thresholds",
    kind: "agent_recommendation",
    reason: "waiver: continue only for the bounded check",
    cites: [digest],
  };
  const waiverApproval = {
    id: "diagnostics-waiver-human-approval",
    parent: "diagnostics-waiver",
    kind: "human_approval",
    reason: "Approved waiver",
    cites: ["model"],
  };

  const missing = derivePhase({ ...base, decisions: [] });
  assert.equal(missing.phase, "diagnostics_decision_required");
  assert.equal(missing.diagnostics_thresholds?.reason, "no recorded threshold decision");
  assert.equal(missing.diagnostics_thresholds?.threshold_decision, null);

  const unapproved = derivePhase({ ...base, decisions: [threshold] });
  assert.equal(unapproved.phase, "diagnostics_decision_required");
  assert.equal(unapproved.decision_facts.threshold, null);

  const exceeded = derivePhase({ ...base, decisions: [threshold, thresholdApproval] });
  assert.equal(exceeded.phase, "diagnostics_decision_required");
  assert.equal(exceeded.diagnostics_thresholds?.threshold_decision, "diagnostic-thresholds");
  assert.equal(exceeded.diagnostics_thresholds?.exceeded, true);

  const waived = derivePhase({
    ...base,
    decisions: [threshold, thresholdApproval, waiver, waiverApproval],
  });
  assert.equal(waived.phase, "check_required");
  assert.equal(waived.diagnostics_thresholds?.human_waiver_recorded, true);
  assert.equal(waived.decision_facts.waiver?.id, "diagnostics-waiver");

  const satisfiedThreshold = {
    ...threshold,
    id: "satisfied-thresholds",
    reason: "thresholds: rhat<=1.03 ess>=50 divergences<=1",
  };
  const satisfied = derivePhase({
    ...base,
    decisions: [
      satisfiedThreshold,
      { ...thresholdApproval, parent: "satisfied-thresholds" },
    ],
  });
  assert.equal(satisfied.phase, "check_required");
  assert.equal(satisfied.diagnostics_thresholds?.exceeded, false);
  const newest = derivePhase({
    ...base,
    decisions: [
      threshold,
      thresholdApproval,
      satisfiedThreshold,
      { ...thresholdApproval, id: "satisfied-thresholds-human-approval", parent: "satisfied-thresholds" },
    ],
  });
  assert.equal(newest.decision_facts.threshold?.id, "satisfied-thresholds");

  assert.deepEqual(parseThresholdReason("thresholds: ess >= 400"), { ess_min: 400 });
  assert.equal(parseThresholdReason("thresholds: rhat<1.01"), null);

  const checkAction = {
    type: "run_recipe",
    workspace: "study",
    recipe: { id: "check", operation: "posterior-check", settings: { seed: 1 } },
    target: "target",
  } as InvestigationAction;
  assert.throws(
    () => assertActionAllowed("diagnostics_decision_required", checkAction),
    (error: unknown) =>
      error instanceof HostError &&
      error.kind === "PhaseRefused" &&
      error.message.includes("no recorded threshold decision"),
  );
  assert.throws(
    () => assertActionAllowed("diagnostics_decision_required", checkAction, "diagnostic-thresholds"),
    (error: unknown) =>
      error instanceof HostError &&
      error.kind === "PhaseRefused" &&
      error.message.includes("diagnostic-thresholds"),
  );
});

test("orientation rules over investigation lifecycle", async (suite) => {
  const root = temporaryRoot("orientation");
  try {
    initStudy(root);
    const host = new HostApi(root, { engine: engineBinary });

    await suite.test("orientation rule: fresh workspace suggests inspect", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.equal(orientation.phase, "inspect_required");
      assertAllowed(orientation, BASE_ACTIONS);
      assert.deepEqual(orientation.phase_facts, {
        simulation_evidence: "unsupported in investigation format; no waiver recorded",
        historical_evidence: [],
        threshold_decision: null,
        estimand_is_free_slot: null,
      });
      assert.ok(operations(orientation).includes("inspect"));
      await assert.rejects(
        host.submitProposal({
          action: { type: "snapshot", workspace: "study", out: "too-early" },
          rationale: "Snapshot must respect the explicit lifecycle.",
          cites: ["model"],
        }),
        (error: unknown) => error instanceof HostError && error.kind === "PhaseRefused",
      );
    });

    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", "inspect-initial"]);
    await suite.test("orientation rule: after inspect suggests sample and simulation disclosure", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.equal(orientation.phase, "sample_required");
      assertAllowed(orientation, [...BASE_ACTIONS, "run_recipe:sample"]);
      assert.equal((orientation.phase_facts as { estimand_is_free_slot: boolean }).estimand_is_free_slot, true);
      assert.ok(operations(orientation).includes("sample"));
      assert.ok(steps(orientation).some((step) => step.step === "simulation-unsupported"));
    });

    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", "sample-initial"]);
    await suite.test("orientation rule: after sample suggests diagnose", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.equal(orientation.phase, "diagnose_required");
      assertAllowed(orientation, [...BASE_ACTIONS, "run_recipe:sample", "run_recipe:diagnose"]);
      assert.ok(operations(orientation).includes("diagnose"));
    });

    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", "diagnose-initial"]);
    let diagnosticsDigest = "";
    await suite.test("orientation rule: diagnostics require an approved threshold decision", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.equal(orientation.phase, "diagnostics_decision_required");
      assertAllowed(orientation, [...BASE_ACTIONS, "run_recipe:sample", "run_recipe:diagnose"]);
      const thresholdStep = steps(orientation).find((step) => step.step === "record-thresholds");
      assert.ok(thresholdStep);
      assert.match(thresholdStep.why, /max R-hat .* min ESS .* divergences/);
      assert.match(thresholdStep.why, /rhat<=1\.01 ess>=400 divergences<=0/);
      diagnosticsDigest = ((await host.readEvidence({ path: "study", name: "diagnose-initial" })).sha256 as string)
        .replace(/^sha256:/, "");
    });

    const thresholdProposal = await host.submitProposal({
      action: {
        type: "record_decision",
        workspace: "study",
        decision: {
          id: "counts-diagnostic-thresholds",
          parent: "initial-likelihood",
          reason: "thresholds: rhat<=1.01 ess>=400 divergences<=0",
          cites: [diagnosticsDigest],
        },
      },
      rationale: "Record human-approved diagnostic thresholds.",
      cites: ["diagnose-initial"],
    });
    host.approve(thresholdProposal.proposal_id, "Approved thresholds.", true);
    await host.execute(thresholdProposal.proposal_id);

    await suite.test("orientation rule: approved satisfied thresholds permit posterior-check", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.equal(orientation.phase, "check_required");
      assertAllowed(orientation, [
        ...BASE_ACTIONS,
        "run_recipe:sample",
        "run_recipe:diagnose",
        "run_recipe:posterior-check",
      ]);
      const facts = orientation.phase_facts as {
        threshold_decision: string;
        simulation_evidence: string;
      };
      assert.equal(facts.threshold_decision, "counts-diagnostic-thresholds");
      assert.equal(facts.simulation_evidence, "unsupported in investigation format; no waiver recorded");
      assert.ok(operations(orientation).includes("posterior-check"));
      assert.ok(steps(orientation).some((step) => step.step === "simulation-unsupported"));
    });

    const waiverProposal = await host.submitProposal({
      action: {
        type: "record_decision",
        workspace: "study",
        decision: {
          id: "simulation-format-waiver",
          parent: "counts-diagnostic-thresholds",
          reason: "waiver: proceed knowing simulation evidence is unsupported by this format",
          cites: ["model"],
        },
      },
      rationale: "Record the format limitation explicitly.",
      cites: ["model"],
    });
    host.approve(waiverProposal.proposal_id, "Approved format waiver.", true);
    await host.execute(waiverProposal.proposal_id);

    await suite.test("orientation rule: approved waiver removes simulation disclosure row", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      const facts = orientation.phase_facts as { simulation_evidence: string };
      assert.equal(
        facts.simulation_evidence,
        "unsupported in investigation format; waiver recorded in decision simulation-format-waiver",
      );
      assert.equal(steps(orientation).some((step) => step.step === "simulation-unsupported"), false);
    });

    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", "check-initial"]);
    await suite.test("orientation rule: after check suggests discussion and snapshot", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.equal(orientation.phase, "snapshot_ready");
      assertAllowed(orientation, [
        ...BASE_ACTIONS,
        "run_recipe:sample",
        "run_recipe:diagnose",
        "run_recipe:posterior-check",
        "snapshot",
      ]);
      assert.ok(operations(orientation).includes("record interpretation or decision; consider fork"));
      assert.ok(operations(orientation).includes("snapshot"));
    });

    runEngine(["investigation", "snapshot", resolve(root, "study"), "--out", resolve(root, "original")]);
    runEngine([
      "investigation",
      "fork",
      resolve(root, "original"),
      "--at",
      "initial-likelihood",
      "--out",
      resolve(root, "forked"),
    ]);
    await suite.test("orientation rule: inherited historical evidence lists recipes to rerun", async () => {
      const orientation = await host.readInvestigation({ path: "forked" });
      assert.equal(orientation.phase, "inspect_required");
      assertAllowed(orientation, BASE_ACTIONS);
      const facts = orientation.phase_facts as { historical_evidence: string[] };
      assert.ok(facts.historical_evidence.includes("sample-initial"));
      assert.ok(steps(orientation).some((step) => step.step === "rerun-sample-initial"));
    });

    runEngine(["investigation", "snapshot", resolve(root, "forked"), "--out", resolve(root, "forked-bundle")]);
    await suite.test("orientation rule: forked bundle suggests a fork at a named decision", async () => {
      const orientation = await host.readInvestigation({ path: "forked-bundle" });
      assert.equal(orientation.phase, "fork_required");
      assertAllowed(orientation, ["fork"]);
      assert.deepEqual(operations(orientation), ["fork"]);
    });
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
