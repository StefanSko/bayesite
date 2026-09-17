import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { createHash, randomUUID } from "node:crypto";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { HostError, type Preconditions, type ProposalDocument } from "./types.js";

export type JsonObject = Record<string, unknown>;
export interface ArtifactReference {
  sha256: string;
  bytes: number;
  kind: string;
  format: string;
}

export function sha256(bytes: Buffer | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

export function shaIdentifier(bytes: Buffer | string): string {
  return `sha256:${sha256(bytes)}`;
}

export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const object = value as JsonObject;
  return `{${Object.keys(object)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`)
    .join(",")}}`;
}

export function proposalPayload(document: Pick<ProposalDocument, "action" | "rationale" | "cites" | "preconditions">): JsonObject {
  return {
    action: document.action,
    rationale: document.rationale,
    cites: document.cites,
    preconditions: document.preconditions,
  };
}

export function proposalDigest(document: Pick<ProposalDocument, "action" | "rationale" | "cites" | "preconditions">): string {
  return sha256(canonicalJson(proposalPayload(document)));
}

function parseObject(bytes: Buffer, description: string): JsonObject {
  try {
    const value = JSON.parse(bytes.toString("utf8")) as unknown;
    if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("expected an object");
    return value as JsonObject;
  } catch (error) {
    throw new HostError(
      "Refused",
      `${description} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

export class RootStore {
  readonly root: string;
  readonly stateDir: string;

  constructor(root: string) {
    const absolute = resolve(root);
    try {
      if (!statSync(absolute).isDirectory()) throw new Error("not a directory");
      this.root = realpathSync(absolute);
    } catch (error) {
      throw new HostError("InvalidPath", `root is not a readable directory: ${absolute}`);
    }
    this.stateDir = resolve(this.root, ".investigation-agent");
    for (const path of [
      this.stateDir,
      resolve(this.stateDir, "candidates"),
      resolve(this.stateDir, "proposals"),
      resolve(this.stateDir, "sessions"),
    ]) {
      mkdirSync(path, { recursive: true });
    }
    this.installBrowserAssets();
  }

  private installBrowserAssets(): void {
    for (const name of ["index.html", "review.js"]) {
      const source = fileURLToPath(new URL(`../../../browser/${name}`, import.meta.url));
      if (existsSync(source)) copyFileSync(source, resolve(this.stateDir, name));
    }
  }

  path(relativePath: string, options: { mustExist?: boolean } = {}): string {
    if (
      typeof relativePath !== "string" ||
      relativePath.length === 0 ||
      isAbsolute(relativePath) ||
      relativePath.split(/[\\/]/).some((part) => part === "..") ||
      relativePath === "." ||
      relativePath.split(/[\\/]/)[0]?.toLowerCase() === ".investigation-agent"
    ) {
      throw new HostError("InvalidPath", `path must be a root-relative workspace or bundle without '..': ${relativePath}`);
    }
    const lexical = resolve(this.root, relativePath);
    if (lexical !== this.root && !lexical.startsWith(`${this.root}${sep}`)) {
      throw new HostError("InvalidPath", `path escapes the root: ${relativePath}`);
    }
    const candidate = this.canonicalThroughExistingAncestor(lexical);
    if (candidate !== this.root && !candidate.startsWith(`${this.root}${sep}`)) {
      throw new HostError("InvalidPath", "path resolves outside the root");
    }
    const candidateKey = candidate.toLowerCase();
    const stateKey = this.stateDir.toLowerCase();
    if (candidateKey === stateKey || candidateKey.startsWith(`${stateKey}${sep}`)) {
      throw new HostError("InvalidPath", `workspace and bundle paths cannot use reserved host state: ${relativePath}`);
    }
    if (options.mustExist && !existsSync(candidate)) {
      throw new HostError("NotFound", `path does not exist: ${relativePath}`);
    }
    return candidate;
  }

  private canonicalThroughExistingAncestor(candidate: string): string {
    let cursor = candidate;
    while (!existsSync(cursor)) {
      const parent = dirname(cursor);
      if (parent === cursor) break;
      cursor = parent;
    }
    try {
      const actual = realpathSync(cursor);
      const suffix = relative(cursor, candidate);
      return suffix ? resolve(actual, suffix) : actual;
    } catch {
      throw new HostError("InvalidPath", `cannot resolve path under root: ${relative(this.root, candidate)}`);
    }
  }

  relative(absolute: string): string {
    return relative(this.root, absolute).split(sep).join("/");
  }

  readJson(path: string, description = path): JsonObject {
    try {
      return parseObject(readFileSync(path), description);
    } catch (error) {
      if (error instanceof HostError) throw error;
      throw new HostError("NotFound", `cannot read ${description}`);
    }
  }

  writeJson(path: string, value: unknown, flag?: "wx"): void {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${JSON.stringify(value)}\n`, flag ? { flag } : undefined);
  }

  replaceJsonAtomically(path: string, value: unknown): void {
    const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
    try {
      writeFileSync(temporary, `${JSON.stringify(value)}\n`, { flag: "wx" });
      renameSync(temporary, path);
    } finally {
      rmSync(temporary, { force: true });
    }
  }

  workspaceDocument(relativePath: string): { path: string; document: JsonObject } {
    const path = this.path(relativePath, { mustExist: true });
    const investigation = resolve(path, "investigation.json");
    if (!existsSync(investigation)) throw new HostError("NotFound", `${relativePath} is not an investigation workspace`);
    return { path, document: this.readJson(investigation, `${relativePath}/investigation.json`) };
  }

  bundleManifest(relativePath: string): { path: string; manifest: JsonObject } {
    const path = this.path(relativePath, { mustExist: true });
    const manifestPath = resolve(path, "manifest.json");
    if (!existsSync(manifestPath)) throw new HostError("NotFound", `${relativePath} is not an investigation bundle`);
    return { path, manifest: this.readJson(manifestPath, `${relativePath}/manifest.json`) };
  }

  objectBytes(containerPath: string, digest: string): Buffer {
    if (!/^[0-9a-f]{64}$/.test(digest)) throw new HostError("Refused", `invalid object digest: ${digest}`);
    const path = resolve(containerPath, "objects", "sha256", digest);
    try {
      return readFileSync(path);
    } catch {
      throw new HostError("NotFound", `object sha256:${digest} is missing`);
    }
  }

  workspacePreconditions(relativePath: string): Preconditions {
    const { path, document } = this.workspaceDocument(relativePath);
    const model = readFileOrNotFound(resolve(path, "inputs", "model.json"));
    const data = readFileOrNotFound(resolve(path, "inputs", "data.json"));
    const investigation = readFileOrNotFound(resolve(path, "investigation.json"));
    const state = `model:${sha256(model)}\ndata:${sha256(data)}\ninvestigation:${sha256(investigation)}\n`;
    return {
      state_sha256: shaIdentifier(state),
      engine_target: stringAt(objectAt(document, "engine"), "target"),
    };
  }

  bundleTarget(manifest: JsonObject, containerPath: string): string {
    const targets = new Set<string>();
    for (const current of this.manifestChain(manifest, containerPath)) {
      for (const recipe of arrayAt(current, "recipes")) {
        const engine = objectAt(recipe, "engine");
        const target = engine.target;
        if (typeof target === "string") targets.add(target);
      }
    }
    if (targets.size !== 1) throw new HostError("Refused", "bundle does not identify exactly one engine target");
    return [...targets][0] as string;
  }

  manifestChain(manifest: JsonObject, containerPath: string): JsonObject[] {
    const result: JsonObject[] = [];
    let current: JsonObject | undefined = manifest;
    for (let depth = 0; current && depth < 16; depth++) {
      result.push(current);
      const source = current.source;
      if (!source || typeof source !== "object" || Array.isArray(source)) break;
      const ref = (source as JsonObject).manifest;
      if (!ref || typeof ref !== "object" || Array.isArray(ref)) break;
      const digest = (ref as JsonObject).sha256;
      if (typeof digest !== "string") break;
      current = parseObject(this.objectBytes(containerPath, digest), `source manifest sha256:${digest}`);
    }
    return result;
  }

  findExecution(manifest: JsonObject, containerPath: string, id: string): { execution: JsonObject; recipe?: JsonObject } | undefined {
    for (const current of this.manifestChain(manifest, containerPath)) {
      const execution = arrayAt(current, "executions").find((entry) => entry.id === id);
      if (execution) {
        const recipeId = execution.recipe;
        const recipe = arrayAt(current, "recipes").find((entry) => entry.id === recipeId);
        return { execution, ...(recipe ? { recipe } : {}) };
      }
    }
    return undefined;
  }
}

export function readFileOrNotFound(path: string): Buffer {
  try {
    return readFileSync(path);
  } catch {
    throw new HostError("NotFound", `cannot read ${path}`);
  }
}

export function objectAt(object: JsonObject, key: string): JsonObject {
  const value = object[key];
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new HostError("Refused", `expected object field ${key}`);
  }
  return value as JsonObject;
}

export function arrayAt(object: JsonObject, key: string): JsonObject[] {
  const value = object[key];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is JsonObject => typeof item === "object" && item !== null && !Array.isArray(item));
}

export function stringAt(object: JsonObject, key: string): string {
  const value = object[key];
  if (typeof value !== "string") throw new HostError("Refused", `expected string field ${key}`);
  return value;
}

export function artifactReference(value: unknown): ArtifactReference {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new HostError("Refused", "malformed artifact reference");
  }
  const ref = value as JsonObject;
  const bytes = ref.bytes;
  if (typeof bytes !== "number") throw new HostError("Refused", "malformed artifact byte length");
  return {
    sha256: stringAt(ref, "sha256"),
    bytes,
    kind: stringAt(ref, "kind"),
    format: stringAt(ref, "format"),
  };
}
