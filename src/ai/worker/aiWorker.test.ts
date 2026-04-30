/**
 * Integration test for the AI Web Worker entry point.
 *
 * Runs the worker module inside jsdom by hijacking `self.postMessage`. We
 * deliver requests via `self.dispatchEvent(new MessageEvent('message', ...))`
 * and capture every `postMessage` call into a queue. The key contract under
 * test: a failing ONNX model URL must NOT prevent 3x3 requests from being
 * answered (regression test for the "AIモデルを読み込めなかったため、
 * 簡易CPUで対戦を続行します" warning showing on 3x3 games).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { PRESET_3X3, PRESET_4X4_XL, engine, hashState } from '@/domain';

import type { AiRequest, AiResponse } from '../protocol';
import { serializeGameState } from '../serialization';

// Mock the InferenceEngine so we never touch onnxruntime-web in the test.
// Calls to .load() reject, simulating the failing-model scenario.
vi.mock('../inference', () => {
  return {
    InferenceEngine: class {
      load(_url: string): Promise<void> {
        return Promise.reject(new Error('mock onnx load failure'));
      }
      isReady(): boolean {
        return false;
      }
    },
  };
});

interface PostedQueue {
  drain: () => AiResponse[];
}

interface WorkerHarness {
  posted: PostedQueue;
  send: (req: AiRequest) => void;
}

async function loadWorkerWithStubs(): Promise<WorkerHarness> {
  const buffer: AiResponse[] = [];
  (self as unknown as { postMessage: (msg: AiResponse) => void }).postMessage = (
    msg: AiResponse,
  ) => {
    buffer.push(msg);
  };

  // Capture the message listener the worker registers so we can deliver
  // messages directly to the current module's handler. We don't rely on
  // `self.dispatchEvent` because previously registered listeners (from
  // earlier `vi.resetModules()` cycles) would also run and post stale
  // responses into our buffer.
  type Handler = (ev: MessageEvent<AiRequest>) => void;
  const capturedRef: { current: Handler | null } = { current: null };
  const originalAddListener = self.addEventListener.bind(self);
  const restoreAddListener = self.addEventListener;
  (self as unknown as { addEventListener: typeof self.addEventListener }).addEventListener = ((
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: boolean | AddEventListenerOptions,
  ): void => {
    if (type === 'message' && typeof listener === 'function') {
      capturedRef.current = listener;
      return;
    }
    originalAddListener(type, listener, options);
  }) as typeof self.addEventListener;

  vi.resetModules();
  try {
    await import('./aiWorker');
  } finally {
    (self as unknown as { addEventListener: typeof self.addEventListener }).addEventListener =
      restoreAddListener;
  }

  const handler = capturedRef.current;
  if (!handler) throw new Error('worker did not register a message listener');

  return {
    posted: {
      drain: (): AiResponse[] => buffer.splice(0, buffer.length),
    },
    send: (req: AiRequest): void => {
      handler(new MessageEvent('message', { data: req }));
    },
  };
}

describe('aiWorker entry', () => {
  let restorePost: (() => void) | null = null;

  beforeEach(() => {
    const original = (self as unknown as { postMessage: unknown }).postMessage;
    restorePost = () => {
      (self as unknown as { postMessage: unknown }).postMessage = original;
    };
  });

  afterEach(() => {
    restorePost?.();
    restorePost = null;
  });

  it('posts ready immediately on init even when the model URL would fail to load', async () => {
    const { posted, send } = await loadWorkerWithStubs();
    send({ type: 'init', modelUrl: 'mock://will-fail.onnx' });
    // ready is posted synchronously by handleInit — no await needed for ONNX.
    const msgs = posted.drain();
    expect(msgs).toEqual([{ type: 'ready' }]);
  });

  it('answers a 3x3 request with a legal move even if ONNX load would fail', async () => {
    const { posted, send } = await loadWorkerWithStubs();
    send({ type: 'init', modelUrl: 'mock://will-fail.onnx' });
    posted.drain(); // discard the ready event

    const state = engine.initialState(PRESET_3X3);
    const requestId = 'req-3x3-1';
    const stateHash = hashState(state);
    send({
      type: 'request',
      requestId,
      stateHash,
      state: serializeGameState(state),
      difficulty: 1,
      timeBudgetMs: 30,
    });

    // The 3x3 solver yields between iterations; allow enough macrotask /
    // microtask cycles for it to complete its small search budget.
    const deadline = Date.now() + 2000;
    let response: AiResponse | undefined;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 5));
      const msgs = posted.drain();
      response = msgs.find((m) => m.type === 'response' || m.type === 'error');
      if (response) break;
    }

    expect(response, 'worker should have answered the 3x3 request').toBeTruthy();
    if (response?.type !== 'response') {
      throw new Error(
        `expected a response, got ${response?.type ?? 'no message'}: ${JSON.stringify(response)}`,
      );
    }
    expect(response.requestId).toBe(requestId);
    expect(response.stateHash).toBe(stateHash);
    // The chosen move must be one of the legal moves from the start state.
    const legal = engine.legalMoves(state);
    expect(legal.some((m) => JSON.stringify(m) === JSON.stringify(response.move))).toBe(true);
  });

  it('returns an error for a 4x4 request when the ONNX model fails to load (3x3 stays unaffected)', async () => {
    const { posted, send } = await loadWorkerWithStubs();
    send({ type: 'init', modelUrl: 'mock://will-fail.onnx' });
    posted.drain();

    // Request a 4x4 move — the lazy ONNX load should fail and the worker
    // should respond with an error tied to this request, not crash.
    const state4x4 = engine.initialState(PRESET_4X4_XL);
    const requestId = 'req-4x4-1';
    send({
      type: 'request',
      requestId,
      stateHash: hashState(state4x4),
      state: serializeGameState(state4x4),
      difficulty: 1,
      timeBudgetMs: 30,
    });

    // Wait for the lazy-load promise to reject + the error to be posted.
    const deadline = Date.now() + 1000;
    let response: AiResponse | undefined;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 5));
      const msgs = posted.drain();
      response = msgs.find((m) => m.type === 'response' || m.type === 'error');
      if (response) break;
    }

    expect(response?.type).toBe('error');
    if (response?.type === 'error') {
      expect(response.requestId).toBe(requestId);
      expect(response.error).toMatch(/onnx|load/i);
    }

    // 3x3 still works on the same worker after a 4x4 failure.
    const state3x3 = engine.initialState(PRESET_3X3);
    const requestId3 = 'req-3x3-after-failure';
    send({
      type: 'request',
      requestId: requestId3,
      stateHash: hashState(state3x3),
      state: serializeGameState(state3x3),
      difficulty: 1,
      timeBudgetMs: 30,
    });
    const deadline3 = Date.now() + 2000;
    let response3: AiResponse | undefined;
    while (Date.now() < deadline3) {
      await new Promise((r) => setTimeout(r, 5));
      const msgs = posted.drain();
      response3 = msgs.find((m) => m.type === 'response' || m.type === 'error');
      if (response3) break;
    }
    expect(response3?.type).toBe('response');
  });
});
