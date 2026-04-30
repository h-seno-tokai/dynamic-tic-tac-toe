/**
 * Configure onnxruntime-web's wasm asset paths so they resolve under any
 * Vite base URL.
 *
 * Without this, ORT-Web fetches its `.wasm` / `.mjs` siblings from a path
 * relative to the *importing document*, which resolves to `/ort-wasm-...`.
 * In a Vite SPA the dev server (and GitHub Pages in production) serves the
 * SPA fallback HTML for any unknown path, so ORT-Web ends up "fetching"
 * `text/html` and `WebAssembly.instantiate` blows up with
 * `CompileError: Wasm decoding failed: expected magic word`.
 *
 * Using `?url` imports lets Vite resolve the bundled wasm/mjs files to their
 * real served URLs (via `/@fs/` in dev, hashed assets in build), which the
 * onnxruntime-web runtime then fetches correctly.
 *
 * onnxruntime-web 1.24 uses a single jsep-enabled wasm binary by default.
 * `ort.env.wasm.wasmPaths` accepts an object of `{wasm, mjs}` URLs.
 *
 * This module has top-level side effects on import; importing it once from
 * the worker entry point is sufficient.
 */

import * as ort from 'onnxruntime-web';
// onnxruntime-web's package.json `exports` map exposes the wasm/mjs files
// at the package root (without the `dist/` prefix), so `?url` imports must
// match those entries to satisfy Vite's package resolution.
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';
import mjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url';

ort.env.wasm.wasmPaths = {
  wasm: wasmUrl,
  mjs: mjsUrl,
};
// Single-threaded keeps things simple — we don't need cross-origin isolation
// (COOP/COEP) for the dev server or GitHub Pages, and the model is small
// enough that multi-threading is not a meaningful win.
ort.env.wasm.numThreads = 1;
