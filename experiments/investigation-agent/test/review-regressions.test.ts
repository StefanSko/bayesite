import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { handleSlashCommand } from "../src/agent/chat.js";
import { HostApi, HostError } from "../src/host/index.js";
import { nextSteps } from "../src/host/orientation.js";
import { alternativeModel, buildCompleteFixture, engineBinary, runEngine } from "./fixture.js";

process.env.BAYESITE_BIN = engineBinary;

test("read operations do not bind unbound decisions in accepted workspace state", async () => {
  const root = buildCompleteFixture();
  try {
    const path = resolve(root, "study/investigation.json");
    const document = JSON.parse(readFileSync(path, "utf8")) as { decisions: unknown[] };
    document.decisions.push({
      id: "unbound-read-regression",
      parent: "initial-likelihood",
      reason: "This decision must remain unbound during agent reads.",
      cites: ["model"],
      kind: "note",
    });
    writeFileSync(path, `${JSON.stringify(document)}\n`);
    const before = readFileSync(path);
    const host = new HostApi(root, { engine: engineBinary });
    await host.readInvestigation({ path: "study" });
    assert.deepEqual(readFileSync(path), before);
    await host.readEvidence({ path: "study", name: "model" });
    assert.deepEqual(readFileSync(path), before);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("orientation rule combines missing inspection with inherited rerun guidance", () => {
  const steps = nextSteps({
    kind: "workspace",
    inspection: null,
    hasFit: false,
    diagnostics: null,
    check: null,
    allRecipesCurrent: false,
    inheritedHistorical: true,
    historicalOperations: [{ name: "sample-old", operation: "sample" }],
  });
  assert.ok(steps.some((step) => step.operation === "inspect"));
  assert.ok(steps.some((step) => step.step === "rerun-sample-old"));
});

test("failed slash execute reports the recorded attempt to the agent and continues", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const proposal = await host.submitProposal({
      action: { type: "snapshot", workspace: "study", out: "appeared-after-submit" },
      rationale: "Exercise a recorded execution failure.",
      cites: ["check-initial"],
    });
    host.approve(proposal.proposal_id);
    mkdirSync(resolve(root, "appeared-after-submit"));
    const reports: unknown[] = [];
    const errors: string[] = [];
    const io = { stdout: (_text: string) => {}, stderr: (text: string) => errors.push(text) };
    const quit = await handleSlashCommand(
      `/execute ${proposal.proposal_id}`,
      host,
      async (attempt) => {
        reports.push(attempt);
      },
      io,
    );
    assert.equal(quit, false);
    assert.equal(reports.length, 1);
    assert.equal((reports[0] as { outcome: string }).outcome, "failed");
    assert.equal(host.showProposal(proposal.proposal_id).status, "failed");

    await handleSlashCommand(`/execute ${proposal.proposal_id}`, host, async (attempt) => {
      reports.push(attempt);
    }, io);
    assert.equal(reports.length, 1);
    assert.equal((JSON.parse(errors.at(-1) as string) as { error: string }).error, "Refused");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("dot-prefixed aliases cannot address reserved host state", () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    assert.throws(
      () => host.root.path("./.investigation-agent/sessions/continuation"),
      (error: unknown) => error instanceof HostError && error.kind === "InvalidPath",
    );
    if (existsSync(resolve(root, ".INVESTIGATION-AGENT"))) {
      assert.throws(
        () => host.root.path(".INVESTIGATION-AGENT/sessions/continuation"),
        (error: unknown) => error instanceof HostError && error.kind === "InvalidPath",
      );
    }
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("candidate inspection is bound to its workspace and current data bytes", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const candidate = await host.prepareCandidate({
      workspace: "study",
      model_json: alternativeModel(),
      note: "data binding regression",
    });
    const dataPath = resolve(root, "study/inputs/data.json");
    writeFileSync(dataPath, `${readFileSync(dataPath, "utf8")} `);
    await assert.rejects(
      host.submitProposal({
        action: {
          type: "adopt_candidate",
          workspace: "study",
          candidate_id: candidate.candidate_id,
          decision: {
            id: "candidate-after-data-change",
            parent: "initial-likelihood",
            reason: "must be inspected against current data",
            cites: ["model"],
          },
        },
        rationale: "must refuse",
        cites: ["model"],
      }),
      (error: unknown) => error instanceof HostError && error.kind === "CandidateRejected",
    );
    assert.equal(host.listProposals(true).length, 0);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("candidate-id collision never replaces existing candidate bytes", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const model = alternativeModel();
    const digest = createHash("sha256").update(model).digest("hex");
    const candidatePath = resolve(root, ".investigation-agent/candidates", `c-${digest.slice(0, 16)}.json`);
    writeFileSync(candidatePath, "different colliding bytes");
    await assert.rejects(
      host.prepareCandidate({ workspace: "study", model_json: model, note: "collision" }),
      (error: unknown) => error instanceof HostError && error.kind === "CandidateRejected",
    );
    assert.equal(readFileSync(candidatePath, "utf8"), "different colliding bytes");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("canonical workspace aliases share candidate binding and proposal orientation", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const candidate = await host.prepareCandidate({ workspace: "study", model_json: alternativeModel(), note: "alias" });
    const proposal = await host.submitProposal({
      action: {
        type: "adopt_candidate",
        workspace: "./study",
        candidate_id: candidate.candidate_id,
        decision: {
          id: "canonical-alias-candidate",
          parent: "initial-likelihood",
          reason: "Both spellings address the same workspace.",
          cites: ["model"],
        },
      },
      rationale: "canonical alias",
      cites: ["model"],
    });
    const orientation = await host.readInvestigation({ path: "./study" });
    assert.equal(orientation.path, "study");
    assert.ok(
      (orientation.proposals as Array<{ proposal_id: string }>).some(
        (item) => item.proposal_id === proposal.proposal_id,
      ),
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("viewer index failure after incomplete persistence cannot block the authorized action", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const proposal = await host.submitProposal({
      action: {
        type: "record_interpretation",
        workspace: "study",
        interpretation: "The read-only viewer cannot block execution.",
        unresolved_questions: [],
      },
      rationale: "first index fault injection",
      cites: ["model"],
    });
    host.approve(proposal.proposal_id);
    const original = host.proposals.writeIndex.bind(host.proposals);
    let calls = 0;
    host.proposals.writeIndex = () => {
      calls += 1;
      if (calls === 1) throw new Error("injected initial index failure");
      original();
    };
    const attempt = await host.execute(proposal.proposal_id);
    assert.equal(attempt.outcome, "completed");
    assert.equal(host.showProposal(proposal.proposal_id).status, "executed");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("viewer index failure cannot downgrade a completed action attempt", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const proposal = await host.submitProposal({
      action: {
        type: "record_interpretation",
        workspace: "study",
        interpretation: "The action completes before the auxiliary index refresh.",
        unresolved_questions: [],
      },
      rationale: "fault injection",
      cites: ["model"],
    });
    host.approve(proposal.proposal_id);
    const original = host.proposals.writeIndex.bind(host.proposals);
    let calls = 0;
    host.proposals.writeIndex = () => {
      calls += 1;
      if (calls === 2) throw new Error("injected index failure");
      original();
    };
    await assert.rejects(host.execute(proposal.proposal_id), (error: unknown) =>
      error instanceof HostError && error.message.includes("action completed"),
    );
    assert.equal(host.showProposal(proposal.proposal_id).status, "executed");
    const workspace = JSON.parse(readFileSync(resolve(root, "study/investigation.json"), "utf8")) as {
      interpretation: string;
    };
    assert.equal(workspace.interpretation, "The action completes before the auxiliary index refresh.");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("completed-attempt publication failure leaves the durable attempt incomplete", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const proposal = await host.submitProposal({
      action: {
        type: "record_interpretation",
        workspace: "study",
        interpretation: "The side effect completed but final publication was interrupted.",
        unresolved_questions: [],
      },
      rationale: "fault injection",
      cites: ["model"],
    });
    host.approve(proposal.proposal_id);
    host.root.replaceJsonAtomically = () => {
      throw new Error("injected completion publication failure");
    };
    await assert.rejects(host.execute(proposal.proposal_id), (error: unknown) =>
      error instanceof HostError && error.message.includes("remains incomplete"),
    );
    assert.equal(host.showProposal(proposal.proposal_id).status, "incomplete");
    const workspace = JSON.parse(readFileSync(resolve(root, "study/investigation.json"), "utf8")) as {
      interpretation: string;
    };
    assert.equal(workspace.interpretation, "The side effect completed but final publication was interrupted.");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("inherited recipe ids conflict before local execution ids can collide", async () => {
  const root = buildCompleteFixture();
  try {
    runEngine([
      "investigation",
      "fork",
      resolve(root, "original"),
      "--at",
      "initial-likelihood",
      "--out",
      resolve(root, "forked"),
    ]);
    const host = new HostApi(root, { engine: engineBinary });
    const document = JSON.parse(readFileSync(resolve(root, "forked/investigation.json"), "utf8")) as {
      engine: { target: string };
    };
    await assert.rejects(
      host.submitProposal({
        action: {
          type: "run_recipe",
          workspace: "forked",
          recipe: { id: "inspect-initial", operation: "inspect", settings: {} },
          target: document.engine.target,
        },
        rationale: "must use a new id",
        cites: ["inspect-initial"],
      }),
      (error: unknown) => error instanceof HostError && error.kind === "RecipeConflict",
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("outputs cannot be nested inside their source bundle or workspace", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    await assert.rejects(
      host.submitProposal({
        action: {
          type: "fork",
          source_bundle: "original",
          at: "initial-likelihood",
          out: "original/nested-fork",
        },
        rationale: "must not mutate bundle",
        cites: ["model"],
      }),
      (error: unknown) => error instanceof HostError && error.kind === "InvalidPath",
    );
    if (existsSync(resolve(root, "ORIGINAL"))) {
      await assert.rejects(
        host.submitProposal({
          action: {
            type: "fork",
            source_bundle: "ORIGINAL",
            at: "initial-likelihood",
            out: "original/case-aliased-nested-fork",
          },
          rationale: "case aliases must not mutate bundle",
          cites: ["model"],
        }),
        (error: unknown) => error instanceof HostError && error.kind === "InvalidPath",
      );
    }
    await assert.rejects(
      host.submitProposal({
        action: { type: "snapshot", workspace: "study", out: "study/nested-snapshot" },
        rationale: "must not nest output",
        cites: ["model"],
      }),
      (error: unknown) => error instanceof HostError && error.kind === "InvalidPath",
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("unknown proposal fields are refused as immutable-byte tampering", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const proposal = await host.submitProposal({
      action: {
        type: "record_interpretation",
        workspace: "study",
        interpretation: "unchanged",
        unresolved_questions: [],
      },
      rationale: "review exact bytes",
      cites: ["model"],
    });
    host.approve(proposal.proposal_id);
    const path = resolve(root, ".investigation-agent/proposals", proposal.proposal_id, "proposal.json");
    const document = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
    document.extra = true;
    writeFileSync(path, `${JSON.stringify(document)}\n`);
    await assert.rejects(host.execute(proposal.proposal_id), (error: unknown) =>
      error instanceof HostError && error.kind === "Refused",
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("proposal key reordering is refused as immutable-byte tampering", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const proposal = await host.submitProposal({
      action: {
        type: "record_interpretation",
        workspace: "study",
        interpretation: "unchanged",
        unresolved_questions: [],
      },
      rationale: "review exact encoding",
      cites: ["model"],
    });
    host.approve(proposal.proposal_id);
    const path = resolve(root, ".investigation-agent/proposals", proposal.proposal_id, "proposal.json");
    const original = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
    const reordered = Object.fromEntries(Object.entries(original).reverse());
    writeFileSync(path, `${JSON.stringify(reordered)}\n`);
    await assert.rejects(host.execute(proposal.proposal_id), (error: unknown) =>
      error instanceof HostError && error.kind === "Refused",
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("recorded human approval validates its generated decision id before review", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const candidate = await host.prepareCandidate({
      workspace: "study",
      model_json: alternativeModel(),
      note: "human id bound",
    });
    const proposal = await host.submitProposal({
      action: {
        type: "adopt_candidate",
        workspace: "study",
        candidate_id: candidate.candidate_id,
        decision: {
          id: "a".repeat(128),
          parent: "initial-likelihood",
          reason: "valid recommendation id, invalid generated approval id",
          cites: ["model"],
        },
      },
      rationale: "must refuse review",
      cites: ["model"],
    });
    await assert.rejects(
      Promise.resolve().then(() => host.approve(proposal.proposal_id, "yes", true)),
      (error: unknown) => error instanceof HostError && error.kind === "MalformedArguments",
    );
    assert.equal(host.showProposal(proposal.proposal_id).review, null);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("successful chat execution prints the resulting investigation phase", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const proposal = await host.submitProposal({
      action: {
        type: "record_interpretation",
        workspace: "study",
        interpretation: "Print the resulting phase after this host action.",
        unresolved_questions: [],
      },
      rationale: "phase feedback",
      cites: ["model"],
    });
    host.approve(proposal.proposal_id);
    const output: string[] = [];
    await handleSlashCommand(
      `/execute ${proposal.proposal_id}`,
      host,
      async () => {},
      { stdout: (text) => output.push(text), stderr: () => {} },
    );
    assert.equal((JSON.parse(output.at(-1) as string) as { phase: string }).phase, "snapshot_ready");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("chat approval accepts the explicit human-approval recording flag", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const proposal = await host.submitProposal({
      action: {
        type: "record_decision",
        workspace: "study",
        decision: {
          id: "chat-recorded-decision",
          parent: "initial-likelihood",
          reason: "Exercise the human-only chat flag.",
          cites: ["model"],
        },
      },
      rationale: "chat approval parsing",
      cites: ["model"],
    });
    await handleSlashCommand(
      `/approve ${proposal.proposal_id} --record-human-approval explicit waiver`,
      host,
      async () => {},
      { stdout: () => {}, stderr: () => {} },
    );
    const review = host.showProposal(proposal.proposal_id).review as {
      record_human_approval?: boolean;
      note: string;
    };
    assert.equal(review.record_human_approval, true);
    assert.equal(review.note, "explicit waiver");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("malformed proposal citations are refused before persistence", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    await assert.rejects(
      host.submitProposal({
        action: {
          type: "record_interpretation",
          workspace: "study",
          interpretation: "unchanged",
          unresolved_questions: [],
        },
        rationale: "bad citation",
        cites: ["sha256:not-a-digest"],
      }),
      (error: unknown) => error instanceof HostError && error.kind === "MalformedArguments",
    );
    assert.equal(host.listProposals(true).length, 0);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("oversized interpretation is refused before proposal or workspace mutation", async () => {
  const root = buildCompleteFixture();
  try {
    const host = new HostApi(root, { engine: engineBinary });
    const path = resolve(root, "study/investigation.json");
    const before = readFileSync(path);
    await assert.rejects(
      host.submitProposal({
        action: {
          type: "record_interpretation",
          workspace: "study",
          interpretation: "x".repeat(16_385),
          unresolved_questions: [],
        },
        rationale: "too large",
        cites: ["model"],
      }),
      (error: unknown) => error instanceof HostError && error.kind === "MalformedArguments",
    );
    assert.equal(host.listProposals(true).length, 0);
    assert.deepEqual(readFileSync(path), before);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
