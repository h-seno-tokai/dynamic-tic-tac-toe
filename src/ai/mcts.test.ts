import { describe, expect, it } from 'vitest';

import { PRESET_3X3, engine } from '@/domain';

import { TOTAL_ACTIONS } from './encoding';
import type { InferenceEngine, InferenceOutput } from './inference';
import { MCTS } from './mcts';

/**
 * Stub `InferenceEngine` that always returns a uniform policy and value=0.
 * Implements the public surface MCTS depends on.
 */
class StubInferenceEngine {
  async load(_modelUrl: string): Promise<void> {
    /* noop */
  }
  isReady(): boolean {
    return true;
  }
  // eslint-disable-next-line @typescript-eslint/require-await
  async forward(_input: Float32Array): Promise<InferenceOutput> {
    return {
      policy: new Float32Array(TOTAL_ACTIONS), // all zeros -> softmax becomes uniform over legal
      value: 0,
    };
  }
}

describe('MCTS with uniform-stub network', () => {
  it('selects a legal move from the PRESET_3X3 initial state with 10 sims', async () => {
    const stub = new StubInferenceEngine() as unknown as InferenceEngine;
    const mcts = new MCTS(stub);
    const state = engine.initialState(PRESET_3X3);
    const legal = engine.legalMoves(state);

    const move = await mcts.selectMove(state, 10, 1.0);
    expect(move).toBeDefined();
    // The chosen move must be present in the engine's legal-move list.
    const matches = legal.some(
      (m) =>
        m.kind === move.kind &&
        m.player === move.player &&
        JSON.stringify(m) === JSON.stringify(move),
    );
    expect(matches).toBe(true);
  });

  it('search returns a 320-d distribution that sums to ~1 (or 0 if simCount=0)', async () => {
    const stub = new StubInferenceEngine() as unknown as InferenceEngine;
    const mcts = new MCTS(stub);
    const state = engine.initialState(PRESET_3X3);
    const policy = await mcts.search(state, 10);
    expect(policy.length).toBe(TOTAL_ACTIONS);
    let sum = 0;
    for (const v of policy) sum += v;
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it('greedy (temperature=0) selection is deterministic given fixed visits', async () => {
    const stub = new StubInferenceEngine() as unknown as InferenceEngine;
    const mcts = new MCTS(stub);
    const state = engine.initialState(PRESET_3X3);
    const a = await mcts.selectMove(state, 8, 0);
    const b = await mcts.selectMove(state, 8, 0);
    // Same simulation count + deterministic stub -> same action index.
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it('selectAction compares child Q from the parent perspective', () => {
    const stub = new StubInferenceEngine() as unknown as InferenceEngine;
    const mcts = new MCTS(stub, { puctC: 0 });
    const state = engine.initialState(PRESET_3X3);
    const node = {
      state,
      toMoveSign: 1,
      n: 2,
      w: 0,
      children: new Map([
        // Child values are stored from the child side-to-move perspective.
        // The first child is good for the parent, so its child-perspective Q is -1.
        [10, { n: 1, w: -1 }],
        [11, { n: 1, w: 1 }],
      ]),
      priors: new Float32Array(TOTAL_ACTIONS),
      legalActions: [10, 11],
      terminalValue: null,
      expanded: true,
    };

    const selectAction = (
      mcts as unknown as { selectAction: (candidate: typeof node) => number }
    ).selectAction.bind(mcts);

    expect(selectAction(node)).toBe(10);
  });

  it('honors abort signals during search', async () => {
    const stub = new StubInferenceEngine() as unknown as InferenceEngine;
    const mcts = new MCTS(stub);
    const state = engine.initialState(PRESET_3X3);
    const controller = new AbortController();
    controller.abort();

    await expect(mcts.search(state, 10, { signal: controller.signal })).rejects.toThrow(/aborted/i);
  });
});
