import { cpSync, mkdtempSync, mkdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
export const repositoryRoot = resolve(packageRoot, "../..");
export const engineBinary = process.env.BAYESITE_BIN ?? resolve(repositoryRoot, "target/release/bayesite");
export const exampleDirectory = resolve(repositoryRoot, "examples/investigation-counts");

export function temporaryRoot(label: string): string {
  return mkdtempSync(resolve(tmpdir(), `bayesite-agent-${label}-`));
}

export function runEngine(args: string[], cwd = repositoryRoot): Record<string, unknown> {
  const result = spawnSync(engineBinary, args, { cwd, encoding: "utf8", shell: false });
  if (result.status !== 0) throw new Error(`engine failed (${args.join(" ")}): ${result.stderr}`);
  return JSON.parse(result.stdout) as Record<string, unknown>;
}

export function initStudy(root: string, name = "study"): void {
  mkdirSync(root, { recursive: true });
  runEngine([
    "investigation",
    "init",
    "--metadata",
    resolve(exampleDirectory, "metadata.json"),
    "--model",
    resolve(exampleDirectory, "poisson.json"),
    "--data",
    resolve(exampleDirectory, "data.json"),
    "--out",
    resolve(root, name),
  ]);
}

export function buildCompleteFixture(): string {
  const root = temporaryRoot("complete");
  initStudy(root);
  for (const recipe of ["inspect-initial", "sample-initial", "diagnose-initial", "check-initial"]) {
    runEngine(["investigation", "run", resolve(root, "study"), "--recipe", recipe]);
  }
  runEngine(["investigation", "snapshot", resolve(root, "study"), "--out", resolve(root, "original")]);
  return root;
}

export function copyFixture(source: string, label: string): string {
  const target = temporaryRoot(label);
  cpSync(resolve(source, "study"), resolve(target, "study"), { recursive: true });
  cpSync(resolve(source, "original"), resolve(target, "original"), { recursive: true });
  return target;
}

export function alternativeModel(): string {
  return readFileSync(resolve(exampleDirectory, "negative-binomial.json"), "utf8");
}
