import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { HostApi } from "../src/host/index.js";
import { engineBinary, initStudy, runEngine, temporaryRoot } from "./fixture.js";

process.env.BAYESITE_BIN = engineBinary;

function operations(orientation: Record<string, unknown>): string[] {
  return (orientation.next_steps as Array<{ operation: string }>).map((step) => step.operation);
}

test("orientation rules over investigation lifecycle", async (suite) => {
  const root = temporaryRoot("orientation");
  try {
    initStudy(root);
    const host = new HostApi(root, { engine: engineBinary });

    await suite.test("orientation rule: fresh workspace suggests inspect", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.ok(operations(orientation).includes("inspect"));
    });

    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", "inspect-initial"]);
    await suite.test("orientation rule: after inspect suggests sample", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.ok(operations(orientation).includes("sample"));
    });

    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", "sample-initial"]);
    await suite.test("orientation rule: after sample suggests diagnose", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.ok(operations(orientation).includes("diagnose"));
    });

    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", "diagnose-initial"]);
    await suite.test("orientation rule: after diagnose suggests posterior-check", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
      assert.ok(operations(orientation).includes("posterior-check"));
    });

    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", "check-initial"]);
    await suite.test("orientation rule: after check suggests discussion and snapshot", async () => {
      const orientation = await host.readInvestigation({ path: "study" });
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
      assert.ok((orientation.next_steps as Array<{ step: string }>).some((step) => step.step === "rerun-sample-initial"));
    });

    runEngine(["investigation", "snapshot", resolve(root, "forked"), "--out", resolve(root, "forked-bundle")]);
    await suite.test("orientation rule: forked bundle suggests a fork at a named decision", async () => {
      const orientation = await host.readInvestigation({ path: "forked-bundle" });
      assert.deepEqual(operations(orientation), ["fork"]);
    });
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
