import type { Model, ModelThinkingLevel } from "@earendil-works/pi-ai";
import {
  createAgentSession,
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
} from "@earendil-works/pi-coding-agent";
import { resolve } from "node:path";
import { HostApi, HostError } from "../host/index.js";
import { TOOL_NAMES } from "../host/types.js";
import { INVESTIGATION_SYSTEM_PROMPT } from "./prompt.js";
import { investigationTools } from "./tools.js";

export interface InvestigationSessionOptions {
  model?: string;
  thinking?: ModelThinkingLevel;
  engine?: string;
  modelRuntime?: ModelRuntime;
  resolvedModel?: Model<any>;
  toolCallBudget?: number;
}

export async function createInvestigationSession(root: string, options: InvestigationSessionOptions = {}) {
  const host = new HostApi(root, options.engine ? { engine: options.engine } : {});
  const modelRuntime = options.modelRuntime ?? (await ModelRuntime.create());
  const model = options.resolvedModel ?? resolveModel(modelRuntime, options.model ?? "openai-codex/gpt-5.6-sol");
  const stateDir = host.root.stateDir;
  const resourceLoader = new DefaultResourceLoader({
    cwd: host.root.root,
    agentDir: stateDir,
    noExtensions: true,
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    noContextFiles: true,
    systemPrompt: INVESTIGATION_SYSTEM_PROMPT,
    appendSystemPromptOverride: () => [],
  });
  await resourceLoader.reload();
  const created = await createAgentSession({
    cwd: host.root.root,
    agentDir: stateDir,
    model,
    thinkingLevel: options.thinking ?? "medium",
    modelRuntime,
    resourceLoader,
    tools: [...TOOL_NAMES],
    customTools: investigationTools(host),
    sessionManager: SessionManager.create(host.root.root, resolve(stateDir, "sessions")),
  });
  const toolCallBudget = options.toolCallBudget ?? 12;
  if (!Number.isSafeInteger(toolCallBudget) || toolCallBudget <= 0) {
    created.session.dispose();
    throw new HostError("MalformedArguments", "tool-call budget must be a positive integer");
  }
  let toolCalls = 0;
  created.session.agent.subscribe((event) => {
    if (event.type === "agent_start") toolCalls = 0;
  });
  created.session.agent.beforeToolCall = async () => {
    toolCalls += 1;
    if (toolCalls <= toolCallBudget) return undefined;
    return {
      block: true,
      terminate: true,
      reason: JSON.stringify(
        new HostError("Refused", `tool-call budget exceeded (${toolCallBudget} per turn)`).toJSON(),
      ),
    };
  };
  return { ...created, host, modelRuntime };
}

function resolveModel(runtime: ModelRuntime, spec: string): Model<any> {
  const slash = spec.indexOf("/");
  if (slash < 1 || slash === spec.length - 1) {
    throw new HostError("MalformedArguments", `model must be provider/id: ${spec}`);
  }
  const model = runtime.getModel(spec.slice(0, slash), spec.slice(slash + 1));
  if (!model) throw new HostError("Refused", `model is not available: ${spec}`);
  return model;
}
