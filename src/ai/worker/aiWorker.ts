/**
 * Web Worker entry point for AI inference.
 *
 * Dispatches by `state.rules.boardSize`:
 *   - 3 → strong alpha-beta solver (`Solver3x3`); does not require ONNX init.
 *   - 4 → 4x4 AlphaZero net via `InferenceEngine` + `MCTS`. ONNX is
 *         loaded lazily on the first 4x4 request so a transient model-load
 *         failure does not break the 3x3 path.
 *
 * `init` records the model URL and posts `ready` immediately. The 3x3 solver
 * is self-contained, so the UI can start a 3x3 game without waiting for
 * ONNX. For 4x4 we attempt the ONNX load on demand; if it fails we surface a
 * per-request error so the main thread can use its fallback. We retry on
 * subsequent 4x4 requests (the failure is cached briefly per request, not
 * permanently) so transient network issues self-heal.
 *
 * Handles `abort` so the main thread can cancel in-flight searches when the
 * user undoes / resigns.
 */

/// <reference lib="webworker" />

import { MCTS } from '../mcts';
import { InferenceEngine } from '../inference';
import { getProfile, getProfile3x3 } from '../difficulty';
import type { AiRequest, AiResponse } from '../protocol';
import { deserializeGameState } from '../serialization';
import { Solver3x3 } from '../solver3x3';

declare const self: DedicatedWorkerGlobalScope;

let engine: InferenceEngine | null = null;
let mcts: MCTS | null = null;
let solver3x3: Solver3x3 | null = null;
let currentRequestId: string | null = null;
let currentAbortController: AbortController | null = null;
let modelUrl: string | null = null;
let onnxLoadPromise: Promise<void> | null = null;

function post(msg: AiResponse): void {
  self.postMessage(msg);
}

function handleInit(url: string): void {
  modelUrl = url;
  // 3x3 needs no ONNX, so we signal ready immediately. ONNX is loaded
  // lazily on the first 4x4 request. This keeps 3x3 working even if the
  // model file or onnxruntime-web wasm assets fail to load.
  post({ type: 'ready' });
}

/**
 * Lazily load the ONNX model on first 4x4 request. Memoises the in-flight
 * promise so concurrent requests share one load. If a load fails we clear
 * the cached promise so a subsequent request retries.
 */
function ensureOnnxLoaded(): Promise<void> {
  if (engine?.isReady() && mcts) return Promise.resolve();
  if (onnxLoadPromise) return onnxLoadPromise;
  if (modelUrl === null) {
    return Promise.reject(new Error('AI worker: init not called'));
  }
  const url = modelUrl;
  const loadPromise = (async () => {
    const e = new InferenceEngine();
    await e.load(url);
    engine = e;
    mcts = new MCTS(e);
  })().catch((err: unknown) => {
    onnxLoadPromise = null;
    throw err;
  });
  onnxLoadPromise = loadPromise;
  return loadPromise;
}

function getOrCreateSolver3x3(): Solver3x3 {
  solver3x3 ??= new Solver3x3();
  return solver3x3;
}

async function handleRequest(req: Extract<AiRequest, { type: 'request' }>): Promise<void> {
  currentAbortController?.abort();
  const abortController = new AbortController();
  currentAbortController = abortController;
  currentRequestId = req.requestId;

  const state = deserializeGameState(req.state);
  const t0 = performance.now();

  try {
    let move;
    if (state.rules.boardSize === 3) {
      const profile = getProfile3x3(req.difficulty);
      const solver = getOrCreateSolver3x3();
      const opts: {
        timeBudgetMs: number;
        mistakeRate: number;
        signal: AbortSignal;
      } = {
        timeBudgetMs: req.timeBudgetMs ?? profile.timeBudgetMs,
        mistakeRate: profile.mistakeRate,
        signal: abortController.signal,
      };
      move = await solver.selectMove(state, opts);
    } else {
      try {
        await ensureOnnxLoaded();
      } catch (err) {
        if (abortController.signal.aborted) return;
        post({
          type: 'error',
          requestId: req.requestId,
          error: err instanceof Error ? err.message : String(err),
        });
        return;
      }
      if (!mcts || !engine?.isReady()) {
        post({
          type: 'error',
          requestId: req.requestId,
          error: 'AI worker not initialised',
        });
        return;
      }
      const profile = getProfile(req.difficulty);
      move = await mcts.selectMove(state, profile.simCount, profile.temperature, {
        signal: abortController.signal,
      });
    }
    if (abortController.signal.aborted || currentRequestId !== req.requestId) {
      // Drop the result — a newer request superseded this one.
      return;
    }
    post({
      type: 'response',
      requestId: req.requestId,
      stateHash: req.stateHash,
      move,
      thinkingMs: performance.now() - t0,
    });
  } catch (err) {
    if (abortController.signal.aborted) return;
    post({
      type: 'error',
      requestId: req.requestId,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

self.addEventListener('message', (event: MessageEvent<AiRequest>) => {
  const msg = event.data;
  switch (msg.type) {
    case 'init':
      handleInit(msg.modelUrl);
      break;
    case 'request':
      void handleRequest(msg);
      break;
    case 'abort':
      currentAbortController?.abort();
      break;
  }
});

export {};
