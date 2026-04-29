/**
 * Wire protocol between the main thread (`AiClient`) and the AI Web Worker.
 *
 * Each `request` carries a unique `requestId` (UUID) AND a `stateHash`.
 * Responses echo the `requestId` so the main thread can discard stale results
 * (see `docs/10_risks.md` §1.3 — race after "undo / waitta").
 */

import type { Move } from '@/domain';
import type { SerializedGameState } from './serialization';

export type AiRequest =
  | { type: 'init'; modelUrl: string }
  | {
      type: 'request';
      requestId: string;
      stateHash: string;
      state: SerializedGameState;
      difficulty: number;
      timeBudgetMs?: number;
    }
  | { type: 'abort' };

export type AiResponse =
  | { type: 'ready' }
  | {
      type: 'response';
      requestId: string;
      stateHash: string;
      move: Move;
      thinkingMs: number;
    }
  | { type: 'error'; requestId?: string; error: string };
