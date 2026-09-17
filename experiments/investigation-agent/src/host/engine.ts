import { spawn } from "node:child_process";
import { HostError } from "./types.js";

export interface EngineResult {
  json: Record<string, unknown>;
  stdout: string;
}

export class Engine {
  constructor(readonly binary: string) {}

  async run(args: readonly string[]): Promise<EngineResult> {
    return await new Promise((resolve, reject) => {
      const child = spawn(this.binary, [...args], {
        shell: false,
        stdio: ["ignore", "pipe", "pipe"],
      });
      const stdout: Buffer[] = [];
      const stderr: Buffer[] = [];
      child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
      child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
      child.on("error", (error) => {
        reject(new HostError("EngineError", `could not start bayesite: ${error.message}`));
      });
      child.on("close", (code) => {
        const out = Buffer.concat(stdout).toString("utf8");
        const err = Buffer.concat(stderr).toString("utf8").trim();
        if (code !== 0) {
          let typed: unknown = err;
          try {
            typed = JSON.parse(err);
          } catch {
            // Preserve non-JSON process failures as text.
          }
          const message =
            typeof typed === "object" && typed !== null && "message" in typed
              ? String((typed as { message: unknown }).message)
              : err || `bayesite exited with status ${String(code)}`;
          reject(new HostError("EngineError", message, typed));
          return;
        }
        try {
          const parsed = JSON.parse(out) as unknown;
          if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
            throw new Error("expected a JSON object");
          }
          resolve({ json: parsed as Record<string, unknown>, stdout: out });
        } catch (error) {
          reject(
            new HostError(
              "EngineError",
              `bayesite emitted malformed JSON: ${error instanceof Error ? error.message : String(error)}`,
            ),
          );
        }
      });
    });
  }
}
