import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { HostApi, HostError } from "../src/host/index.js";
import { assertActionAllowed, derivePhase } from "../src/host/phase.js";
import type { InvestigationAction } from "../src/host/types.js";
import { engineBinary, initStudy, runEngine, temporaryRoot } from "./fixture.js";

process.env.BAYESITE_BIN = engineBinary;

function operations(orientation: Record<string, unknown>): string[] {
  return (orientation.next_steps as Array<{ operation: string }>).map((step) => step.operation);
}

test("phase thresholds require an explicit human waiver", () => {
  const base = {
    kind: "workspace" as const,
    inspection: { structural_discrepancies: [] },
    hasFit: true,
    diagnostics: { per_parameter: [{ name: "rate", rhat: 1.02, ess: 100 }], divergences: 0 },
    check: null,
    diagnosticsSha256: "a".repeat(64),
  };
  const recommendation = {
    id: "diagnostics-waiver",
    parent: "initial-likelihood",
    kind: "agent_recommendation",
    reason: "Continue only for the bounded check.",
    cites: ["a".repeat(64)],
  };
  assert.equal(derivePhase({ ...base, decisions: [] }).phase, "diagnostics_decision_required");
  assert.equal(derivePhase({ ...base, decisions: [recommendation] }).phase, "diagnostics_decision_required");
  assert.equal(
    derivePhase({
      ...base,
      decisions: [
        recommendation,
        {
          id: "diagnostics-waiver-human-approval",
          parent: "diagnostics-waiver",
          kind: "human_approval",
          reason: "Approved",
          cites: ["model"],
        },
      ],
    }).phase,
    "check_required",
  );
  assert.equal(
    derivePhase({
      ...base,
      diagnostics: { per_parameter: [{ name: "rate", rhat: 1.01, ess: 100 }], divergences: 0 },
      decisions: [],
    }).phase,
    "check_required",
  );
  const checkAction = {
    type: "run_recipe",
    workspace: "study",
    recipe: { id: "check", operation: "posterior-check", settings: { seed: 1 } },
    target: "target",
  } as InvestigationAction;
  assert.throws(
    () => assertActionAllowed("diagnostics_decision_required", checkAction),
    (error: unknown) => error instanceof HostError && error.kind === "PhaseRefused",
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
    await suite.test("orientation rule: after inspect suggests sample", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.equal(orientation.phase, "sample_required");
      assert.ok(operations(orientation).includes("sample"));
    });

    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", "sample-initial"]);
    await suite.test("orientation rule: after sample suggests diagnose", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.equal(orientation.phase, "diagnose_required");
      assert.ok(operations(orientation).includes("diagnose"));
    });

    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", "diagnose-initial"]);
    await suite.test("orientation rule: after diagnose suggests posterior-check", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.equal(orientation.phase, "check_required");
      assert.ok(operations(orientation).includes("posterior-check"));
    });

    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", "check-initial"]);
    await suite.test("orientation rule: after check suggests discussion and snapshot", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.equal(orientation.phase, "snapshot_ready");
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
      assert.ok((orientation.next_steps as Array<{ step: string }>).some((step) => step.step === "rerun-sample-initial"));
    });

    runEngine(["investigation", "snapshot", resolve(root, "forked"), "--out", resolve(root, "forked-bundle")]);
    await suite.test("orientation rule: forked bundle suggests a fork at a named decision", async () => {
      const orientation = await host.readInvestigation({ path: "forked-bundle" });
      assert.equal(orientation.phase, "fork_required");
      assert.deepEqual(operations(orientation), ["fork"]);
    });
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
