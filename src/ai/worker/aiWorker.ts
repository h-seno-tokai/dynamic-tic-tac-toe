/**
 * Web Worker entry point for AI inference.
 *
 * Dispatches by `state.rules.boardSize`:
 *   - 3 → strong alpha-beta solver (`Solver3x3`); does not require ONNX init.
 *   - 4 → universal-network MCTS via `InferenceEngine` + `MCTS`. Requires
 *         `init` to have completed first; the `ready` event signals that.
 *
 * The `ready` event fires only when the ONNX model has loaded so the UI knows
 * it's safe to start a 4x4 game. 3x3 requests are accepted regardless of
 * whether ONNX init has completed (the solver is self-contained).
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

function post(msg: AiResponse): void {
  self.postMessage(msg);
}

async function handleInit(modelUrl: string): Promise<void> {
  engine = new InferenceEngine();
  await engine.load(modelUrl);
  mcts = new MCTS(engine);
  post({ type: 'ready' });
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
      void handleInit(msg.modelUrl).catch((err: unknown) => {
        post({
          type: 'error',
          error: err instanceof Error ? err.message : String(err),
        });
      });
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
