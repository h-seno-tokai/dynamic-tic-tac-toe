import { describe, expect, it } from 'vitest';

import { PRESET_3X3, engine } from '@/domain';
import type { GameState, Move } from '@/domain';

import { selectFallbackMove } from './fallbackAi';

function sameMove(a: Move, b: Move): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function apply(state: GameState, move: Move): GameState {
  return engine.applyMove(state, move);
}

describe('selectFallbackMove', () => {
  it('returns a legal move', () => {
    const state = engine.initialState(PRESET_3X3);
    const move = selectFallbackMove(state, 1, () => 0);
    const legal = engine.legalMoves(state);

    expect(legal.some((candidate) => sameMove(candidate, move))).toBe(true);
  });

  it('prioritizes an immediate winning move at high difficulty', () => {
    let state = engine.initialState(PRESET_3X3);
    state = apply(state, {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'S',
      to: { row: 0, col: 0 },
    });
    state = apply(state, {
      kind: 'placeFromReserve',
      player: 'P2',
      sizeId: 'S',
      to: { row: 1, col: 0 },
    });
    state = apply(state, {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'S',
      to: { row: 0, col: 1 },
    });
    state = apply(state, {
      kind: 'placeFromReserve',
      player: 'P2',
      sizeId: 'S',
      to: { row: 1, col: 1 },
    });

    const move = selectFallbackMove(state, 10, () => 0);

    expect(move).toEqual({
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'M',
      to: { row: 0, col: 2 },
    });
  });

  it('prioritizes blocking an opponent immediate win at high difficulty', () => {
    let state = engine.initialState(PRESET_3X3);
    state = apply(state, {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'S',
      to: { row: 2, col: 2 },
    });
    state = apply(state, {
      kind: 'placeFromReserve',
      player: 'P2',
      sizeId: 'S',
      to: { row: 0, col: 0 },
    });
    state = apply(state, {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'S',
      to: { row: 1, col: 0 },
    });
    state = apply(state, {
      kind: 'placeFromReserve',
      player: 'P2',
      sizeId: 'S',
      to: { row: 0, col: 1 },
    });

    const move = selectFallbackMove(state, 10, () => 0);
    const next = engine.applyMove(state, move);
    const opponentWins = engine
      .legalMoves(next)
      .filter((candidate) => engine.outcome(engine.applyMove(next, candidate)) === 'P2');

    expect(move.to.row).toBe(0);
    expect(opponentWins).toHaveLength(0);
  });

  it('throws when no legal moves exist', () => {
    const state: GameState = { ...engine.initialState(PRESET_3X3), outcome: 'draw' };

    expect(() => selectFallbackMove(state, 10, () => 0)).toThrow(/no legal moves/i);
  });
});
