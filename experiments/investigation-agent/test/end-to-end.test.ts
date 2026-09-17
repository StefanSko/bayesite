import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { InMemoryCredentialStore, fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { createInvestigationSession } from "../src/agent/session.js";
import { dispatchHostCommand } from "../src/host/cli.js";
import { alternativeModel, buildCompleteFixture, engineBinary, runEngine } from "./fixture.js";

process.env.BAYESITE_BIN = engineBinary;

test("scripted end-to-end: fork, prepare, adopt, run, and snapshot with human CLI approvals", async () => {
  const root = buildCompleteFixture();
  const faux = fauxProvider();
  const runtime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(),
    modelsPath: null,
    allowModelNetwork: false,
  });
  runtime.registerNativeProvider(faux.provider);
  const created = await createInvestigationSession(root, {
    modelRuntime: runtime,
    resolvedModel: faux.getModel(),
    thinking: "off",
    engine: engineBinary,
  });
  const { session, host } = created;

  async function callTool(name: string, args: Record<string, unknown>): Promise<void> {
    faux.setResponses([
      fauxAssistantMessage(fauxToolCall(name, args), { stopReason: "toolUse" }),
      fauxAssistantMessage("Recorded; human review remains separate."),
    ]);
    await session.prompt(`Use ${name} for the next investigation step.`);
    const lastResults = session.messages.filter((message) => message.role === "toolResult").slice(-1);
    assert.equal(lastResults[0]?.isError, false);
  }

  async function submitAndExecute(action: Record<string, unknown>, recordHuman = false): Promise<string> {
    const before = new Set(host.listProposals(true).map((proposal) => proposal.proposal_id));
    await callTool("submit_proposal", {
      action,
      rationale: "This step is supported by the named investigation evidence.",
      cites: ["model", "check-initial"],
    });
    const proposal = host.listProposals(true).find((item) => !before.has(item.proposal_id));
    assert.ok(proposal);
    await dispatchHostCommand([
      "approve",
      root,
      proposal.proposal_id,
      "--note",
      "Human authorizes this recorded step.",
      ...(recordHuman ? ["--record-human-approval"] : []),
    ]);
    const attempt = (await dispatchHostCommand(["execute", root, proposal.proposal_id])) as { outcome: string };
    assert.equal(attempt.outcome, "completed");
    return proposal.proposal_id;
  }

  try {
    assert.deepEqual(
      session.agent.state.tools.map((tool) => tool.name).sort(),
      ["prepare_candidate", "read_evidence", "read_investigation", "submit_proposal"],
    );

    await submitAndExecute({
      type: "fork",
      source_bundle: "original",
      at: "initial-likelihood",
      out: "alternative",
    });
    assert.ok(existsSync(resolve(root, "alternative/investigation.json")));

    await callTool("prepare_candidate", {
      workspace: "alternative",
      model_json: alternativeModel(),
      note: "Preserve the estimand while adding overdispersion.",
    });
    const candidateFile = readdirSync(resolve(root, ".investigation-agent/candidates")).find((name) => /^c-[0-9a-f]{16}\.json$/.test(name));
    assert.ok(candidateFile);
    const candidateId = candidateFile.slice(0, -5);

    const adoptionProposalId = await submitAndExecute(
      {
        type: "adopt_candidate",
        workspace: "alternative",
        candidate_id: candidateId,
        decision: {
          id: "alternative-likelihood",
          parent: "initial-likelihood",
          reason: "Evaluate an overdispersed count likelihood without changing the estimand.",
          cites: ["model"],
        },
      },
      true,
    );
    const adopted = JSON.parse(readFileSync(resolve(root, "alternative/investigation.json"), "utf8")) as {
      decisions: Array<{ kind: string }>;
    };
    assert.ok(adopted.decisions.some((decision) => decision.kind === "agent_recommendation"));
    assert.ok(adopted.decisions.some((decision) => decision.kind === "human_approval"));
    const adoptionReviewPath = resolve(
      root,
      ".investigation-agent/proposals",
      adoptionProposalId,
      "review.json",
    );
    const adoptionReviewBytes = readFileSync(adoptionReviewPath);
    await dispatchHostCommand([
      "approve",
      root,
      adoptionProposalId,
      "--note",
      "Human authorizes this recorded step.",
      "--record-human-approval",
    ]);
    assert.deepEqual(readFileSync(adoptionReviewPath), adoptionReviewBytes);

    const orientation = await host.readInvestigation({ path: "alternative" });
    const target = orientation.engine_target as string;
    const recipes = [
      { id: "inspect-alternative", operation: "inspect", settings: {} },
      {
        id: "sample-alternative",
        operation: "sample",
        settings: {
          chains: 1,
          warmup: 20,
          draws: 8,
          max_treedepth: 6,
          target_accept: 0.85,
          initial_step_size: 1,
          seed: 20260918,
        },
      },
      { id: "diagnose-alternative", operation: "diagnose", settings: {} },
      { id: "check-alternative", operation: "posterior-check", settings: { seed: 20260919 } },
    ];
    for (const recipe of recipes) {
      await submitAndExecute({ type: "run_recipe", workspace: "alternative", recipe, target });
    }

    await submitAndExecute({ type: "snapshot", workspace: "alternative", out: "continuation" });
    assert.ok(existsSync(resolve(root, "continuation/manifest.json")));
    const verification = runEngine(["investigation", "verify", resolve(root, "continuation")]);
    assert.equal(verification.verification_complete, true);
  } finally {
    session.dispose();
    rmSync(root, { recursive: true, force: true });
  }
});
