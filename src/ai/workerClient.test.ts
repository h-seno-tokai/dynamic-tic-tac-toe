import { describe, expect, it } from 'vitest';

import { PRESET_3X3, engine, hashState, type Move } from '@/domain';

import type { AiRequest, AiResponse } from './protocol';
import { AiClient } from './workerClient';

/**
 * Minimal in-test Worker polyfill. Tracks every `postMessage` it receives
 * and lets the test fire synthetic messages back into the client.
 */
class FakeWorker implements Worker {
  onmessage: ((this: Worker, ev: MessageEvent) => unknown) | null = null;
  onmessageerror: ((this: Worker, ev: MessageEvent) => unknown) | null = null;
  onerror: ((this: AbstractWorker, ev: ErrorEvent) => unknown) | null = null;

  private listeners = new Map<string, Set<EventListenerOrEventListenerObject>>();
  posted: AiRequest[] = [];

  postMessage(msg: AiRequest): void {
    this.posted.push(msg);
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(listener);
  }
  removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    this.listeners.get(type)?.delete(listener);
  }
  dispatchEvent(event: Event): boolean {
    const set = this.listeners.get(event.type);
    if (!set) return true;
    for (const l of set) {
      if (typeof l === 'function') l.call(this, event);
      else l.handleEvent(event);
    }
    return true;
  }
  terminate(): void {
    this.listeners.clear();
  }

  /** Test helper: deliver a message to the client. */
  emitMessage(data: AiResponse): void {
    const ev = new MessageEvent('message', { data });
    this.dispatchEvent(ev);
  }
}

function buildClient(): { client: AiClient; worker: FakeWorker } {
  const worker = new FakeWorker();
  const client = new AiClient('mock://model.onnx', { workerFactory: () => worker });
  return { client, worker };
}

describe('AiClient', () => {
  it('posts an init message on construction and resolves ready() on "ready"', async () => {
    const { client, worker } = buildClient();
    expect(worker.posted[0]).toEqual({ type: 'init', modelUrl: 'mock://model.onnx' });

    worker.emitMessage({ type: 'ready' });
    await expect(client.ready()).resolves.toBeUndefined();
  });

  it('discards stale responses (different requestId)', async () => {
    const { client, worker } = buildClient();
    worker.emitMessage({ type: 'ready' });
    await client.ready();

    const state = engine.initialState(PRESET_3X3);
    const promise = client.requestMove(state, 1);

    // Find the request we posted to extract its requestId.
    const reqMsg = worker.posted.find((m) => m.type === 'request');
    expect(reqMsg?.type).toBe('request');
    if (reqMsg?.type !== 'request') throw new Error('expected request');
    const liveId = reqMsg.requestId;

    // Synthesise a stale response with a *different* requestId.
    const staleMove: Move = {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'S',
      to: { row: 0, col: 0 },
    };
    worker.emitMessage({
      type: 'response',
      requestId: 'stale-uuid-zzz',
      stateHash: hashState(state),
      move: staleMove,
      thinkingMs: 1,
    });

    // The pending promise must NOT have resolved yet.
    let settled = false;
    void promise.then(
      () => (settled = true),
      () => (settled = true),
    );
    // Yield once so any erroneous resolution would have surfaced.
    await Promise.resolve();
    expect(settled).toBe(false);

    // Now deliver the legitimate response.
    const realMove: Move = {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'M',
      to: { row: 1, col: 1 },
    };
    worker.emitMessage({
      type: 'response',
      requestId: liveId,
      stateHash: hashState(state),
      move: realMove,
      thinkingMs: 7,
    });

    await expect(promise).resolves.toEqual(realMove);
  });

  it('rejects matching requestId responses with mismatched stateHash', async () => {
    const { client, worker } = buildClient();
    worker.emitMessage({ type: 'ready' });
    await client.ready();

    const state = engine.initialState(PRESET_3X3);
    const promise = client.requestMove(state, 1);
    const reqMsg = worker.posted.find((m) => m.type === 'request');
    if (reqMsg?.type !== 'request') throw new Error('expected request');

    const move: Move = {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'S',
      to: { row: 0, col: 0 },
    };
    worker.emitMessage({
      type: 'response',
      requestId: reqMsg.requestId,
      stateHash: 'not-the-requested-state',
      move,
      thinkingMs: 1,
    });

    await expect(promise).rejects.toThrow(/state hash mismatch/);
  });

  it('posts abort when a new request supersedes an old one', async () => {
    const { client, worker } = buildClient();
    worker.emitMessage({ type: 'ready' });
    await client.ready();

    const state = engine.initialState(PRESET_3X3);
    const oldPromise = client.requestMove(state, 1);
    void client.requestMove(state, 2);

    await expect(oldPromise).rejects.toThrow(/superseded/);
    expect(worker.posted.some((m) => m.type === 'abort')).toBe(true);
  });

  it('cancel() rejects the in-flight promise and posts an abort message', async () => {
    const { client, worker } = buildClient();
    worker.emitMessage({ type: 'ready' });
    await client.ready();

    const state = engine.initialState(PRESET_3X3);
    const promise = client.requestMove(state, 5);
    client.cancel();

    await expect(promise).rejects.toThrow(/cancelled/);
    expect(worker.posted.some((m) => m.type === 'abort')).toBe(true);
  });

  it('a new request supersedes the old one (old promise rejects)', async () => {
    const { client, worker } = buildClient();
    worker.emitMessage({ type: 'ready' });
    await client.ready();

    const state = engine.initialState(PRESET_3X3);
    const oldPromise = client.requestMove(state, 1);
    // Issue a second request before the first resolves.
    const newPromise = client.requestMove(state, 1);

    await expect(oldPromise).rejects.toThrow(/superseded/);

    // Resolve the second one to keep vitest from holding open promises.
    const reqs = worker.posted.filter((m) => m.type === 'request');
    expect(reqs.length).toBe(2);
    const last = reqs[reqs.length - 1];
    if (last?.type !== 'request') throw new Error('expected request');
    const move: Move = {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'S',
      to: { row: 0, col: 0 },
    };
    worker.emitMessage({
      type: 'response',
      requestId: last.requestId,
      stateHash: last.stateHash,
      move,
      thinkingMs: 0,
    });
    await expect(newPromise).resolves.toEqual(move);
  });
});
