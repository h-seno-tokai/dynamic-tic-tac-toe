/**
 * Web Worker entry point for AI inference.
 *
 * Loads `InferenceEngine`, runs MCTS at the requested difficulty, and posts
 * results back keyed by `requestId`. Handles `abort` so the main thread can
 * cancel in-flight searches when the user undoes / resigns.
 */

/// <reference lib="webworker" />

import { MCTS } from '../mcts';
import { InferenceEngine } from '../inference';
import { getProfile } from '../difficulty';
import type { AiRequest, AiResponse } from '../protocol';
import { deserializeGameState } from '../serialization';

declare const self: DedicatedWorkerGlobalScope;

let engine: InferenceEngine | null = null;
let mcts: MCTS | null = null;
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

async function handleRequest(req: Extract<AiRequest, { type: 'request' }>): Promise<void> {
  if (!mcts || !engine?.isReady()) {
    post({
      type: 'error',
      requestId: req.requestId,
      error: 'AI worker not initialised',
    });
    return;
  }
  currentAbortController?.abort();
  const abortController = new AbortController();
  currentAbortController = abortController;
  currentRequestId = req.requestId;

  const profile = getProfile(req.difficulty);
  const state = deserializeGameState(req.state);
  const t0 = performance.now();

  try {
    const move = await mcts.selectMove(state, profile.simCount, profile.temperature, {
      signal: abortController.signal,
    });
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
