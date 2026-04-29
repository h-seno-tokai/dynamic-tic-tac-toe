/**
 * Main-thread API for the AI Web Worker.
 *
 * - Generates a fresh `requestId` per request via `crypto.randomUUID`.
 * - Computes a `stateHash` via `hashState` and sends both to the worker.
 * - Discards stale responses (where `response.requestId !== currentLatestId`)
 *   to defeat the "undo race" described in `docs/10_risks.md` §1.3.
 * - `cancel()` rejects the in-flight promise and posts `{type: 'abort'}`.
 */

import type { GameState, Move } from '@/domain';
import { hashState } from '@/domain';
import type { AiRequest, AiResponse } from './protocol';
import { serializeGameState } from './serialization';

export interface AiClientOptions {
  /** Inject a custom Worker (used by tests). */
  workerFactory?: () => Worker;
}

interface PendingRequest {
  requestId: string;
  stateHash: string;
  resolve: (move: Move) => void;
  reject: (err: Error) => void;
}

export class AiClient {
  private readonly worker: Worker;
  private readyPromise: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((err: Error) => void) | null = null;
  private pending: PendingRequest | null = null;

  constructor(modelUrl: string, options: AiClientOptions = {}) {
    if (options.workerFactory) {
      this.worker = options.workerFactory();
    } else {
      this.worker = new Worker(new URL('./worker/aiWorker.ts', import.meta.url), {
        type: 'module',
      });
    }
    this.worker.addEventListener('message', this.onMessage);
    this.worker.addEventListener('error', this.onError);

    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    this.post({ type: 'init', modelUrl });
  }

  /** Resolves once the worker has loaded the ONNX model and signalled `ready`. */
  ready(): Promise<void> {
    return this.readyPromise ?? Promise.resolve();
  }

  /**
   * Request a move from the AI for `state` at the given difficulty (1..10).
   * Only the most recent in-flight request is honoured; older ones are
   * discarded if a new request arrives or `cancel()` is called.
   */
  async requestMove(state: GameState, difficulty: number, timeBudgetMs?: number): Promise<Move> {
    // Reject any prior in-flight promise so it never silently resolves.
    if (this.pending) {
      this.pending.reject(new Error('AiClient: superseded by newer request'));
      this.pending = null;
      this.post({ type: 'abort' });
    }

    const requestId = crypto.randomUUID();
    const stateHash = hashState(state);

    const promise = new Promise<Move>((resolve, reject) => {
      this.pending = { requestId, stateHash, resolve, reject };
    });

    const req: AiRequest = {
      type: 'request',
      requestId,
      stateHash,
      state: serializeGameState(state),
      difficulty,
      ...(timeBudgetMs !== undefined ? { timeBudgetMs } : {}),
    };
    this.post(req);

    return promise;
  }

  /** Abort the in-flight thinking job and reject its promise. */
  cancel(): void {
    if (this.pending) {
      this.pending.reject(new Error('AiClient: cancelled'));
      this.pending = null;
    }
    this.post({ type: 'abort' });
  }

  /** Terminate the underlying worker. After this the client is unusable. */
  dispose(): void {
    if (this.pending) {
      this.pending.reject(new Error('AiClient: disposed'));
      this.pending = null;
    }
    this.worker.removeEventListener('message', this.onMessage);
    this.worker.removeEventListener('error', this.onError);
    this.worker.terminate();
  }

  // ---------- private ----------

  private post(req: AiRequest): void {
    this.worker.postMessage(req);
  }

  private readonly onMessage = (event: MessageEvent<AiResponse>): void => {
    const msg = event.data;
    switch (msg.type) {
      case 'ready':
        this.readyResolve?.();
        this.readyResolve = null;
        this.readyReject = null;
        break;
      case 'response':
        {
          const pending = this.pending;
          if (pending?.requestId === msg.requestId) {
            if (pending.stateHash !== msg.stateHash) {
              pending.reject(new Error('AiClient: state hash mismatch'));
              this.pending = null;
              break;
            }
            pending.resolve(msg.move);
            this.pending = null;
          }
        }
        // Stale response (different requestId) — discard silently.
        break;
      case 'error':
        {
          const pending = this.pending;
          if (msg.requestId && pending?.requestId === msg.requestId) {
            pending.reject(new Error(msg.error));
            this.pending = null;
          } else if (!msg.requestId) {
            // Fatal init error.
            this.readyReject?.(new Error(msg.error));
            this.readyReject = null;
            this.readyResolve = null;
          }
        }
        break;
    }
  };

  private readonly onError = (event: ErrorEvent): void => {
    const err = new Error(event.message || 'AI worker error');
    const pending = this.pending;
    if (pending) {
      pending.reject(err);
      this.pending = null;
    }
    this.readyReject?.(err);
    this.readyReject = null;
    this.readyResolve = null;
  };
}
