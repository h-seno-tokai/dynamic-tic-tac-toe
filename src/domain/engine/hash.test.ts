import { describe, expect, it } from 'vitest';
import { hashState } from './hash';
import { engine } from './engine';
import { PRESET_3X3 } from '../rules/presets';
import type { Move } from '../types';

describe('hashState', () => {
  it('is deterministic for the same state', () => {
    const s = engine.initialState(PRESET_3X3);
    expect(hashState(s)).toBe(hashState(s));
  });

  it('distinguishes states that differ in side-to-move', () => {
    const a = engine.initialState(PRESET_3X3);
    const b = { ...a, toMove: 'P2' as const };
    expect(hashState(a)).not.toBe(hashState(b));
  });

  it('distinguishes states that differ in board contents', () => {
    const s0 = engine.initialState(PRESET_3X3);
    const place: Move = {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'L',
      to: { row: 0, col: 0 },
    };
    const s1 = engine.applyMove(s0, place);
    expect(hashState(s0)).not.toBe(hashState(s1));
  });

  it('distinguishes states that differ in reserves', () => {
    const a = engine.initialState(PRESET_3X3);
    const b = engine.initialState(PRESET_3X3);
    b.reserves.P1.S = (b.reserves.P1.S ?? 0) - 1;
    expect(hashState(a)).not.toBe(hashState(b));
  });

  it('produces equal hashes for structurally identical positions', () => {
    const a = engine.initialState(PRESET_3X3);
    const b = engine.initialState(PRESET_3X3);
    expect(hashState(a)).toBe(hashState(b));
  });
});
