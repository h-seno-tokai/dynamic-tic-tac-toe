/**
 * Minimal repro: load `public/model.onnx` via onnxruntime-web in Node mode.
 * If this works fine, the model file itself is OK; the dev-server issue is
 * almost certainly a wasm-path / threading config problem in browser context.
 *
 * Usage: node scripts/test_ort_load.mjs
 */

import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MODEL = path.resolve(__dirname, '..', 'public', 'model.onnx');

const buf = fs.readFileSync(MODEL);
console.log(`model.onnx size = ${buf.length} bytes`);

let ort;
try {
  ort = await import('onnxruntime-web');
} catch (err) {
  console.error('failed to import onnxruntime-web:', err);
  process.exit(1);
}

console.log('ort version:', ort.env?.versions?.common ?? 'unknown');
console.log('attempting InferenceSession.create with [wasm]...');
try {
  const session = await ort.InferenceSession.create(buf, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  console.log('OK — inputs:', session.inputNames, 'outputs:', session.outputNames);
} catch (err) {
  console.error('FAIL:', err);
  if (err && typeof err === 'object') {
    console.error('  message:', err.message);
    console.error('  cause  :', err.cause);
    console.error('  stack  :', err.stack);
  }
  process.exit(2);
}
