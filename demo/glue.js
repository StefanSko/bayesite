// Zero-dependency JS glue for the Bayesite wasm ABI.
//
// Memory contract (see crates/core/src/wasm_abi.rs):
// request bytes in via bayesite_alloc, response out of bayesite_run with the
// length written to a 4-byte slot, all buffers released via bayesite_dealloc.

export async function loadBayesite(wasmUrl) {
  const response = await fetch(wasmUrl);
  let instance;
  try {
    ({ instance } = await WebAssembly.instantiateStreaming(response, {}));
  } catch {
    // Server did not send application/wasm; fall back to ArrayBuffer.
    const bytes = await (await fetch(wasmUrl)).arrayBuffer();
    ({ instance } = await WebAssembly.instantiate(bytes, {}));
  }
  const { memory, bayesite_alloc, bayesite_dealloc, bayesite_run } = instance.exports;

  function run(requestObject) {
    const requestBytes = new TextEncoder().encode(JSON.stringify(requestObject));
    const requestPtr = bayesite_alloc(requestBytes.length);
    new Uint8Array(memory.buffer, requestPtr, requestBytes.length).set(requestBytes);
    const lengthPtr = bayesite_alloc(4);
    const responsePtr = bayesite_run(requestPtr, requestBytes.length, lengthPtr);
    // memory.buffer may have been detached by growth during the call;
    // re-create views afterwards.
    const responseLength = new DataView(memory.buffer).getUint32(lengthPtr, true);
    const responseBytes = new Uint8Array(memory.buffer, responsePtr, responseLength).slice();
    bayesite_dealloc(requestPtr, requestBytes.length);
    bayesite_dealloc(lengthPtr, 4);
    bayesite_dealloc(responsePtr, responseLength);
    return new TextDecoder().decode(responseBytes);
  }

  return { run };
}
