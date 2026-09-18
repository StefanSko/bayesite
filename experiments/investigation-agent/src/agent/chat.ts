#!/usr/bin/env node
import { createInterface } from "node:readline";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ModelThinkingLevel } from "@earendil-works/pi-ai";
import { HostError } from "../host/index.js";
import { createInvestigationSession } from "./session.js";

interface ChatArguments {
  root: string;
  model?: string;
  thinking?: ModelThinkingLevel;
  engine?: string;
  publicDataConfirmed: boolean;
}

export async function runChat(argv: readonly string[]): Promise<void> {
  const args = parseArguments(argv);
  if (!args.publicDataConfirmed) {
    throw new HostError(
      "Refused",
      "--public-data-confirmed is required: workspace model, data, and evidence will be sent to the model provider",
    );
  }
  const created = await createInvestigationSession(args.root, {
    ...(args.model ? { model: args.model } : {}),
    ...(args.thinking ? { thinking: args.thinking } : {}),
    ...(args.engine ? { engine: args.engine } : {}),
  });
  const { session, host } = created;
  let textStreaming = false;
  const unsubscribe = session.subscribe((event) => {
    if (event.type === "message_update" && event.assistantMessageEvent.type === "text_delta") {
      process.stdout.write(event.assistantMessageEvent.delta);
      textStreaming = true;
    } else if (event.type === "tool_execution_start") {
      if (textStreaming) process.stdout.write("\n");
      const summary = JSON.stringify(event.args);
      process.stdout.write(`[tool] ${event.toolName} ${summary.length > 180 ? `${summary.slice(0, 177)}...` : summary}\n`);
      textStreaming = false;
    } else if (event.type === "agent_end" && textStreaming) {
      process.stdout.write("\n");
      textStreaming = false;
    }
  });
  const readline = createInterface({ input: process.stdin, output: process.stdout, terminal: true });
  const modelLabel = `${session.model?.provider ?? "?"}/${session.model?.id ?? "?"}`;
  process.stdout.write(
    `investigation agent ready\n  root:  ${host.root.root}\n  model: ${modelLabel}\n` +
      `  type a message, or /orient <path>, /proposals, /show <id>, /approve <id> [--record-human-approval] [note], /reject <id> [note], /execute <id>, /quit\n`,
  );
  readline.setPrompt("> ");
  let closed = false;
  readline.on("close", () => {
    closed = true;
  });
  const showPrompt = (): void => {
    if (!closed) readline.prompt();
  };
  let turnActive = false;
  let turnInterrupted = false;
  const promptSession = async (text: string): Promise<void> => {
    turnActive = true;
    turnInterrupted = false;
    try {
      await session.prompt(text);
    } catch (error) {
      if (!turnInterrupted) throw error;
    } finally {
      turnActive = false;
    }
  };
  const onSigint = (): void => {
    if (!turnActive && !session.isStreaming) {
      readline.close();
      return;
    }
    turnInterrupted = true;
    process.stdout.write("\n[turn aborted]\n");
    void session.abort();
  };
  readline.on("SIGINT", onSigint);
  try {
    showPrompt();
    for await (const raw of readline) {
      const line = raw.trim();
      if (!line) {
        showPrompt();
        continue;
      }
      try {
        if (line.startsWith("/")) {
          const quit = await handleSlashCommand(line, host, async (attempt) => {
            await promptSession(
              `[host] Recorded outcome: ${JSON.stringify(attempt)}\n` +
                "Unless the person has said to stop, propose exactly one next action now with submit_proposal and explain it in at most three sentences.",
            );
            printPending(host);
          });
          if (quit) break;
        } else {
          await promptSession(line);
          printPending(host);
        }
      } catch (error) {
        const typed = error instanceof HostError
          ? error
          : new HostError("Refused", error instanceof Error ? error.message : String(error));
        process.stderr.write(`${JSON.stringify(typed.toJSON())}\n`);
      }
      showPrompt();
    }
  } finally {
    readline.off("SIGINT", onSigint);
    unsubscribe();
    readline.close();
    session.dispose();
  }
}

type Host = Awaited<ReturnType<typeof createInvestigationSession>>["host"];
type Io = { stdout: (text: string) => void; stderr: (text: string) => void };

const defaultIo: Io = {
  stdout: (text) => process.stdout.write(text),
  stderr: (text) => process.stderr.write(text),
};

function describeAction(action: Record<string, unknown> | undefined): string {
  if (!action) return "(no action)";
  const a = action as Record<string, any>;
  switch (a.type) {
    case "fork":
      return `fork ${a.source_bundle} at ${a.at} into ${a.out}`;
    case "adopt_candidate":
      return `adopt candidate ${a.candidate_id} in ${a.workspace} (decision ${a.decision?.id})`;
    case "run_recipe":
      return `run ${a.recipe?.id} (${a.recipe?.operation}) on ${a.workspace}`;
    case "snapshot":
      return `snapshot ${a.workspace} into ${a.out}`;
    case "record_decision":
      return `record decision ${a.decision?.id} in ${a.workspace}`;
    case "record_interpretation":
      return `record interpretation in ${a.workspace}`;
    default:
      return String(a.type);
  }
}

function proposalLine(host: Host, id: string): string {
  const shown = host.showProposal(id) as { proposal: { action?: Record<string, unknown> }; status: string };
  return `${id}  ${shown.status.padEnd(9)} ${describeAction(shown.proposal.action)}`;
}

/** Print pending proposals as the menu after a turn; returns true when any are pending. */
export function printPending(host: Host, io: Io = defaultIo): boolean {
  const pending = host.listProposals(false);
  if (pending.length === 0) return false;
  io.stdout(`\npending review:\n`);
  for (const p of pending) io.stdout(`  ${proposalLine(host, p.proposal_id)}\n`);
  io.stdout(`  /show <id> to read it, /approve <id> [note] then /execute <id>, or /reject <id> [reason]\n`);
  return true;
}

function renderOrientation(o: Record<string, any>): string {
  const lines: string[] = [];
  lines.push(`${o.kind} ${o.path}   phase: ${o.phase}`);
  lines.push(`question: ${o.question}`);
  lines.push(`estimand: ${o.estimand?.parameter}  (${o.estimand?.description})`);
  const evidence = (o.evidence ?? []) as Array<{ name: string; status: string }>;
  lines.push(`evidence: ${evidence.length === 0 ? "none" : evidence.map((e) => `${e.name}[${e.status}]`).join("  ")}`);
  if (o.diagnostics_thresholds) {
    const t = o.diagnostics_thresholds;
    lines.push(`diagnostics: max R-hat ${fmt(t.max_rhat)}, min ESS ${fmt(t.min_ess)}, divergences ${t.divergences}; ${t.reason}`);
  }
  if (o.phase_facts) {
    const f = o.phase_facts;
    lines.push(`facts: simulation ${f.simulation_evidence}; threshold decision ${f.threshold_decision ?? "none"}; historical ${JSON.stringify(f.historical_evidence ?? [])}`);
  }
  lines.push(`allowed now: ${(o.allowed_actions ?? []).join(", ")}`);
  lines.push(`next steps:`);
  for (const s of (o.next_steps ?? []) as Array<{ step: string; operation: string; why: string }>) {
    lines.push(`  - ${s.step} (${s.operation}): ${s.why}`);
  }
  return `${lines.join("\n")}\n`;
}

function fmt(value: unknown): string {
  return typeof value === "number" ? value.toFixed(3) : "n/a";
}

function renderShow(host: Host, id: string): string {
  const shown = host.showProposal(id) as {
    proposal: { action?: Record<string, unknown>; rationale?: string; cites?: string[]; preconditions?: Record<string, unknown> };
    review: { decision: string; note: string; at: string; record_human_approval?: true } | null;
    attempts: Array<{ outcome: string; error?: { message?: string } | null; completed_at?: string }>;
    status: string;
  };
  const lines = [
    `${id}  status: ${shown.status}`,
    `action:    ${describeAction(shown.proposal.action)}`,
    `rationale: ${shown.proposal.rationale ?? ""}`,
    `cites:     ${(shown.proposal.cites ?? []).join(", ") || "none"}`,
    `binds to:  state ${String(shown.proposal.preconditions?.state_sha256 ?? "").slice(0, 23)}…  target ${shown.proposal.preconditions?.engine_target ?? ""}`,
  ];
  if (shown.review) {
    lines.push(`review:    ${shown.review.decision} at ${shown.review.at}${shown.review.record_human_approval ? " (human approval will be recorded)" : ""}${shown.review.note ? `: ${shown.review.note}` : ""}`);
  }
  for (const [index, attempt] of shown.attempts.entries()) {
    lines.push(`attempt ${index + 1}: ${attempt.outcome}${attempt.error?.message ? ` — ${attempt.error.message}` : ""}`);
  }
  lines.push(`(full record: /show ${id} --json)`);
  return `${lines.join("\n")}\n`;
}

export async function handleSlashCommand(
  line: string,
  host: Host,
  reportExecution: (attempt: unknown) => Promise<void>,
  io: Io = defaultIo,
): Promise<boolean> {
  const [command, id, ...rest] = line.split(/\s+/);
  const wantJson = rest.includes("--json");
  switch (command) {
    case "/quit":
      return true;
    case "/help":
      io.stdout(`commands: /orient <path>, /proposals [--all], /show <id> [--json], /approve <id> [--record-human-approval] [note], /reject <id> [reason], /execute <id>, /quit\n`);
      return false;
    case "/proposals": {
      const all = id === "--all";
      const list = host.listProposals(all);
      if (wantJson || id === "--json") {
        io.stdout(`${JSON.stringify({ proposals: host.listProposals(true) })}\n`);
        return false;
      }
      if (list.length === 0) {
        io.stdout(all ? "no proposals recorded\n" : "no pending proposals; ask the agent to propose the next step (/proposals --all shows history)\n");
        return false;
      }
      for (const p of list) io.stdout(`${proposalLine(host, p.proposal_id)}\n`);
      return false;
    }
    case "/show":
      if (!id) throw new HostError("MalformedArguments", "/show expects a proposal id");
      io.stdout(wantJson ? `${JSON.stringify(host.showProposal(id))}\n` : renderShow(host, id));
      return false;
    case "/approve": {
      if (!id) throw new HostError("MalformedArguments", "/approve expects a proposal id");
      const status = (host.showProposal(id) as { status: string }).status;
      if (status !== "pending") {
        io.stdout(`${id} is already ${status}; nothing to approve\n`);
        return false;
      }
      const recordHumanApproval = rest.includes("--record-human-approval");
      const note = rest.filter((item) => item !== "--record-human-approval").join(" ");
      host.approve(id, note, recordHumanApproval);
      io.stdout(`approved ${id}${recordHumanApproval ? " (your approval will be recorded as a decision)" : ""}; run it with /execute ${id}\n`);
      return false;
    }
    case "/reject": {
      if (!id) throw new HostError("MalformedArguments", "/reject expects a proposal id");
      const status = (host.showProposal(id) as { status: string }).status;
      if (status !== "pending") {
        io.stdout(`${id} is already ${status}; nothing to reject\n`);
        return false;
      }
      host.reject(id, rest.join(" "));
      io.stdout(`rejected ${id}\n`);
      return false;
    }
    case "/execute": {
      if (!id) throw new HostError("MalformedArguments", "/execute expects a proposal id");
      const before = host.showProposal(id) as { proposal: { action?: Record<string, unknown> }; attempts: unknown[]; status: string };
      if (before.status === "pending") {
        io.stdout(`${id} is not approved yet; /approve ${id} first\n`);
        return false;
      }
      const description = describeAction(before.proposal.action);
      let attempt: Record<string, unknown>;
      try {
        attempt = (await host.execute(id)) as Record<string, unknown>;
      } catch (error) {
        const attempts = host.showProposal(id).attempts as Record<string, unknown>[];
        const recorded = attempts.length > before.attempts.length ? attempts.at(-1) : undefined;
        if (recorded !== undefined) {
          const message = (recorded.error as { message?: string } | null)?.message ?? "failed";
          io.stdout(`executed ${id}: ${description} → ${String(recorded.outcome)}: ${message}\n`);
          await reportExecution({ ...recorded, action: description });
        } else {
          const typed = error instanceof HostError
            ? error
            : new HostError("Refused", error instanceof Error ? error.message : String(error));
          io.stderr(`${JSON.stringify(typed.toJSON())}\n`);
        }
        return false;
      }
      const fact = attempt.fact as Record<string, unknown> | null;
      const output = typeof fact?.output_sha256 === "string" ? ` output ${String(fact.output_sha256).slice(0, 23)}…` : "";
      io.stdout(`executed ${id}: ${description} → ${String(attempt.outcome)}${output}\n`);
      const action = before.proposal.action as Record<string, unknown> | undefined;
      const phasePath = typeof action?.workspace === "string"
        ? action.workspace
        : typeof action?.out === "string"
          ? action.out
          : undefined;
      let orientation: Record<string, any> | undefined;
      if (phasePath) {
        orientation = await host.readInvestigation({ path: phasePath });
        const steps = (orientation.next_steps ?? []) as Array<{ step: string }>;
        io.stdout(`phase: ${orientation.phase}  next: ${steps.length === 0 ? "none" : steps.map((s) => s.step).join(", ")}\n`);
      }
      await reportExecution({
        ...attempt,
        action: description,
        ...(orientation
          ? {
              orientation: {
                path: phasePath,
                phase: orientation.phase,
                allowed_actions: orientation.allowed_actions,
                next_steps: orientation.next_steps,
                phase_facts: orientation.phase_facts,
                diagnostics_thresholds: orientation.diagnostics_thresholds,
              },
            }
          : {}),
      });
      return false;
    }
    case "/orient": {
      if (!id) throw new HostError("MalformedArguments", "/orient expects a root-relative path");
      const orientation = await host.readInvestigation({ path: id });
      io.stdout(wantJson ? `${JSON.stringify(orientation)}\n` : renderOrientation(orientation));
      return false;
    }
    default:
      throw new HostError("MalformedArguments", `unknown slash command: ${String(command)}; /help lists them`);
  }
}

function parseArguments(argv: readonly string[]): ChatArguments {
  const items = argv[0] === "chat" ? argv.slice(1) : [...argv];
  const root = items[0];
  if (!root || root.startsWith("--")) throw new HostError("MalformedArguments", "chat expects ROOT");
  let publicDataConfirmed = false;
  let model: string | undefined;
  let thinking: ModelThinkingLevel | undefined;
  let engine: string | undefined;
  for (let index = 1; index < items.length; index++) {
    const item = items[index];
    if (item === "--public-data-confirmed") publicDataConfirmed = true;
    else if (item === "--model" && items[index + 1]) model = items[++index];
    else if (item === "--thinking" && items[index + 1]) {
      const value = items[++index];
      if (!isThinking(value)) throw new HostError("MalformedArguments", `invalid thinking level: ${String(value)}`);
      thinking = value;
    } else if (item === "--engine" && items[index + 1]) engine = items[++index];
    else throw new HostError("MalformedArguments", `unknown or incomplete chat option: ${String(item)}`);
  }
  return {
    root,
    publicDataConfirmed,
    ...(model ? { model } : {}),
    ...(thinking ? { thinking } : {}),
    ...(engine ? { engine } : {}),
  };
}

function isThinking(value: string | undefined): value is ModelThinkingLevel {
  return value !== undefined && ["off", "minimal", "low", "medium", "high", "xhigh", "max"].includes(value);
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
  try {
    await runChat(argv);
  } catch (error) {
    const typed = error instanceof HostError ? error : new HostError("Refused", error instanceof Error ? error.message : String(error));
    process.stderr.write(`${JSON.stringify(typed.toJSON())}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
