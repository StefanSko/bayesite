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
    for await (const raw of readline) {
      const line = raw.trim();
      if (!line) continue;
      if (line.startsWith("/")) {
        const quit = await handleSlashCommand(line, host, async (attempt) => {
          await promptSession(`[host] ${JSON.stringify(attempt)}`);
        });
        if (quit) break;
      } else {
        await promptSession(line);
      }
    }
  } finally {
    readline.off("SIGINT", onSigint);
    unsubscribe();
    readline.close();
    session.dispose();
  }
}

export async function handleSlashCommand(
  line: string,
  host: Awaited<ReturnType<typeof createInvestigationSession>>["host"],
  reportExecution: (attempt: unknown) => Promise<void>,
  io: { stdout: (text: string) => void; stderr: (text: string) => void } = {
    stdout: (text) => process.stdout.write(text),
    stderr: (text) => process.stderr.write(text),
  },
): Promise<boolean> {
  const [command, id, ...rest] = line.split(/\s+/);
  let result: unknown;
  switch (command) {
    case "/quit":
      return true;
    case "/proposals":
      result = { proposals: host.listProposals(true) };
      break;
    case "/show":
      if (!id) throw new HostError("MalformedArguments", "/show expects a proposal id");
      result = host.showProposal(id);
      break;
    case "/approve": {
      if (!id) throw new HostError("MalformedArguments", "/approve expects a proposal id");
      const recordHumanApproval = rest.includes("--record-human-approval");
      const note = rest.filter((item) => item !== "--record-human-approval").join(" ");
      result = host.approve(id, note, recordHumanApproval);
      break;
    }
    case "/reject":
      if (!id) throw new HostError("MalformedArguments", "/reject expects a proposal id");
      result = host.reject(id, rest.join(" "));
      break;
    case "/execute": {
      if (!id) throw new HostError("MalformedArguments", "/execute expects a proposal id");
      const attemptsBefore = (host.showProposal(id).attempts as unknown[]).length;
      try {
        result = await host.execute(id);
      } catch (error) {
        const attempts = host.showProposal(id).attempts as unknown[];
        const recorded = attempts.length > attemptsBefore ? attempts.at(-1) : undefined;
        if (recorded !== undefined) {
          io.stdout(`${JSON.stringify(recorded)}\n`);
          await reportExecution(recorded);
        } else {
          const typed = error instanceof HostError
            ? error
            : new HostError("Refused", error instanceof Error ? error.message : String(error));
          io.stderr(`${JSON.stringify(typed.toJSON())}\n`);
        }
        return false;
      }
      io.stdout(`${JSON.stringify(result)}\n`);
      const shown = host.showProposal(id);
      const action = (shown.proposal as { action?: Record<string, unknown> }).action;
      const phasePath = typeof action?.workspace === "string"
        ? action.workspace
        : typeof action?.out === "string"
          ? action.out
          : undefined;
      if (phasePath) {
        const orientation = await host.readInvestigation({ path: phasePath });
        io.stdout(`${JSON.stringify({ path: phasePath, phase: orientation.phase })}\n`);
      }
      await reportExecution(result);
      return false;
    }
    case "/orient":
      if (!id) throw new HostError("MalformedArguments", "/orient expects a root-relative path");
      result = await host.readInvestigation({ path: id });
      break;
    default:
      throw new HostError("MalformedArguments", `unknown slash command: ${String(command)}`);
  }
  io.stdout(`${JSON.stringify(result)}\n`);
  return false;
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
