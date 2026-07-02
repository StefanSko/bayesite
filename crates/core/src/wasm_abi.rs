//! The wasm C-ABI boundary: pointer/length handoff across linear memory.
//!
//! This is the only module in the crate allowed to use `unsafe`, and it
//! does nothing but move bytes: allocate a buffer, run the pure
//! `protocol::handle_request` on it, and hand the response buffer back.
//! All semantics live behind the safe `handle_request` seam, which is
//! natively unit-tested.
//!
//! Memory contract for the JS glue (`demo/glue.js`):
//! 1. `bayesite_alloc(len)` -> request buffer; write UTF-8 request bytes.
//! 2. `bayesite_run(ptr, len, out_len_ptr)` -> response buffer pointer; the
//!    response byte length is written to `out_len_ptr` (a 4-byte little
//!    endian u32 slot also obtained via `bayesite_alloc`).
//! 3. `bayesite_dealloc(ptr, len)` both buffers when done.

use crate::protocol;

/// Allocate `len` bytes of zeroed linear memory owned by the caller.
/// Buffers are boxed slices, so length and capacity always agree.
#[no_mangle]
pub extern "C" fn bayesite_alloc(len: usize) -> *mut u8 {
    let buffer = vec![0u8; len.max(1)].into_boxed_slice();
    Box::into_raw(buffer) as *mut u8
}

/// Release a buffer previously returned by `bayesite_alloc` or `bayesite_run`.
///
/// # Safety contract (enforced by the glue, not the type system)
/// `ptr` must come from this module with exactly this `len`.
#[no_mangle]
pub extern "C" fn bayesite_dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    #[allow(unsafe_code)]
    unsafe {
        let slice = std::slice::from_raw_parts_mut(ptr, len.max(1));
        drop(Box::from_raw(slice as *mut [u8]));
    }
}

/// Run one JSON request; returns the response buffer and writes its
/// length to `out_len`. A malformed or non-UTF-8 request yields a JSON
/// error object, never a trap. The recursive consumers behind this seam
/// are depth-bounded (`json::MAX_DEPTH`, `ir::MAX_EXPR_DEPTH`), so deeply
/// nested untrusted input yields a typed error instead of exhausting the
/// stack.
#[no_mangle]
pub extern "C" fn bayesite_run(ptr: *const u8, len: usize, out_len: *mut u32) -> *mut u8 {
    let request_bytes = if ptr.is_null() {
        &[][..]
    } else {
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(ptr, len)
        }
    };
    let response = match std::str::from_utf8(request_bytes) {
        Ok(text) => protocol::handle_request(text),
        Err(_) => "{\"error\":\"MalformedJson\",\"message\":\"request is not UTF-8\"}".to_string(),
    };
    let bytes = response.into_bytes().into_boxed_slice();
    let response_len = bytes.len();
    let response_ptr = Box::into_raw(bytes) as *mut u8;
    // The length slot is a byte buffer from `bayesite_alloc`; only byte
    // alignment is guaranteed, so the write must be unaligned.
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::write_unaligned(out_len, response_len as u32);
    }
    response_ptr
}
