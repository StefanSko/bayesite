#!/usr/bin/env node
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { HostApi, HostError } from "./index.js";

export async function dispatchHostCommand(argv: readonly string[]): Promise<unknown> {
  const [command, root, ...rest] = argv;
  if (!command || !root) throw new HostError("MalformedArguments", "expected COMMAND ROOT and command arguments");
  const host = new HostApi(root);
  switch (command) {
    case "orient": {
      if (rest.length !== 1) throw new HostError("MalformedArguments", "orient expects PATH");
      return await host.readInvestigation({ path: rest[0] });
    }
    case "proposals": {
      if (rest.length > 1 || (rest.length === 1 && rest[0] !== "--all")) {
        throw new HostError("MalformedArguments", "proposals accepts only --all");
      }
      return { proposals: host.listProposals(rest[0] === "--all") };
    }
    case "show": {
      if (rest.length !== 1) throw new HostError("MalformedArguments", "show expects PROPOSAL_ID");
      return host.showProposal(rest[0] as string);
    }
    case "approve": {
      const parsed = parseReviewArguments(rest, true);
      return host.approve(parsed.id, parsed.note, parsed.recordHumanApproval);
    }
    case "reject": {
      const parsed = parseReviewArguments(rest, false);
      return host.reject(parsed.id, parsed.note);
    }
    case "execute": {
      if (rest.length !== 1) throw new HostError("MalformedArguments", "execute expects PROPOSAL_ID");
      return await host.execute(rest[0] as string);
    }
    default:
      throw new HostError("MalformedArguments", `unknown command: ${command}`);
  }
}

function parseReviewArguments(args: readonly string[], allowRecord: boolean): {
  id: string;
  note: string;
  recordHumanApproval: boolean;
} {
  const id = args[0];
  if (!id) throw new HostError("MalformedArguments", "review command expects PROPOSAL_ID");
  let note = "";
  let recordHumanApproval = false;
  for (let index = 1; index < args.length; index++) {
    const item = args[index];
    if (item === "--note" && args[index + 1] !== undefined) {
      note = args[++index] as string;
    } else if (allowRecord && item === "--record-human-approval") {
      recordHumanApproval = true;
    } else {
      throw new HostError("MalformedArguments", `unknown or incomplete review option: ${String(item)}`);
    }
  }
  return { id, note, recordHumanApproval };
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
  try {
    const result = await dispatchHostCommand(argv);
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (error) {
    const typed = error instanceof HostError ? error : new HostError("Refused", error instanceof Error ? error.message : String(error));
    process.stderr.write(`${JSON.stringify(typed.toJSON())}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
