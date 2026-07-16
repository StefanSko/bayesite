#!/usr/bin/env node
// Exercise the release wasm ABI with hostile JSON bytes. No npm dependencies.
import { readFile } from "node:fs/promises";

const [wasmPath] = process.argv.slice(2);
if (!wasmPath) throw new Error("usage: check_wasm_json_boundary.mjs <bayesite_core.wasm>");
const { instance } = await WebAssembly.instantiate(await readFile(wasmPath), {});
const { memory, bayesite_alloc: alloc, bayesite_dealloc: dealloc, bayesite_run: run } = instance.exports;
if (!(memory instanceof WebAssembly.Memory) || !alloc || !dealloc || !run) {
  throw new Error("wasm ABI exports memory, bayesite_alloc, bayesite_dealloc, and bayesite_run are required");
}
const encoder = new TextEncoder();
const decoder = new TextDecoder();

function invoke(bytes) {
  const input = alloc(bytes.length);
  new Uint8Array(memory.buffer, input, bytes.length).set(bytes);
  const outLen = alloc(4);
  const output = run(input, bytes.length, outLen);
  const length = new DataView(memory.buffer).getUint32(outLen, true);
  const text = decoder.decode(new Uint8Array(memory.buffer, output, length));
  dealloc(input, bytes.length);
  dealloc(outLen, 4);
  dealloc(output, length);
  return JSON.parse(text);
}

function expectError(name, bytes, kind) {
  const payload = invoke(bytes);
  if (payload.error !== kind || typeof payload.message !== "string") {
    throw new Error(`${name}: expected typed ${kind} error, got ${JSON.stringify(payload)}`);
  }
}

// The root request object consumes one of json::MAX_DEPTH's 256 levels.
const validDepth = `{"command":"capabilities","padding":${"[".repeat(255)}0${"]".repeat(255)}}`;
const valid = invoke(encoder.encode(validDepth));
if (valid.error === "MalformedJson") {
  throw new Error(`depth 256 was rejected as malformed JSON: ${JSON.stringify(valid)}`);
}
expectError("depth 257", encoder.encode(`{"command":"capabilities","padding":${"[".repeat(256)}0${"]".repeat(256)}}`), "MalformedJson");
expectError("hostile depth", encoder.encode("[".repeat(100_000)), "MalformedJson");
expectError("invalid UTF-8", Uint8Array.from([0xff]), "MalformedJson");
console.log("wasm JSON boundary passed");
