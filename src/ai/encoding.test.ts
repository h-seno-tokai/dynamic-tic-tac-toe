import { describe, expect, it } from 'vitest';

import { PRESET_3X3, PRESET_4X4_XL, engine, type Move } from '@/domain';

import {
  MAX_BOARD,
  PLACE_ACTIONS,
  TENSOR_LENGTH,
  TOTAL_ACTIONS,
  actionIndexToMove,
  encodeState,
  legalActionMask,
  moveToActionIndex,
} from './encoding';

const PLANE = MAX_BOARD * MAX_BOARD;
const chw = (channel: number, row: number, col: number): number =>
  channel * PLANE + row * MAX_BOARD + col;

describe('encodeState (PRESET_3X3 initial)', () => {
  const state = engine.initialState(PRESET_3X3);
  const tensor = encodeState(state);

  it('produces a Float32Array of length 432', () => {
    expect(tensor).toBeInstanceOf(Float32Array);
    expect(tensor.length).toBe(TENSOR_LENGTH);
    expect(tensor.length).toBe(432);
  });

  it('side-to-move (ch24) is all 1 since P1 starts', () => {
    for (let r = 0; r < MAX_BOARD; r++) {
      for (let c = 0; c < MAX_BOARD; c++) {
        expect(tensor[chw(24, r, c)]).toBe(1);
      }
    }
  });

  it('out-of-board mask (ch25) is 1 outside the 3x3 area, 0 inside', () => {
    for (let r = 0; r < MAX_BOARD; r++) {
      for (let c = 0; c < MAX_BOARD; c++) {
        const expected = r >= 3 || c >= 3 ? 1 : 0;
        expect(tensor[chw(25, r, c)]).toBe(expected);
      }
    }
  });

  it('unused-size mask (ch26) is all 1 (XL slot is unused)', () => {
    for (let r = 0; r < MAX_BOARD; r++) {
      for (let c = 0; c < MAX_BOARD; c++) {
        expect(tensor[chw(26, r, c)]).toBe(1);
      }
    }
  });

  it('reserve channels are normalised (2/3 for both players, S/M/L)', () => {
    for (let i = 0; i < 3; i++) {
      expect(tensor[chw(16 + i, 0, 0)]).toBeCloseTo(2 / 3, 5);
      expect(tensor[chw(20 + i, 0, 0)]).toBeCloseTo(2 / 3, 5);
    }
    // The 4th (XL) slot has no reserve -> 0.
    expect(tensor[chw(19, 0, 0)]).toBe(0);
    expect(tensor[chw(23, 0, 0)]).toBe(0);
  });

  it('top-of-stack and anywhere channels are all zero (empty board)', () => {
    for (let ch = 0; ch < 16; ch++) {
      for (let r = 0; r < MAX_BOARD; r++) {
        for (let c = 0; c < MAX_BOARD; c++) {
          expect(tensor[chw(ch, r, c)]).toBe(0);
        }
      }
    }
  });
});

describe('encodeState (PRESET_4X4_XL initial)', () => {
  const state = engine.initialState(PRESET_4X4_XL);
  const tensor = encodeState(state);

  it('out-of-board mask (ch25) is all 0', () => {
    for (let r = 0; r < MAX_BOARD; r++) {
      for (let c = 0; c < MAX_BOARD; c++) {
        expect(tensor[chw(25, r, c)]).toBe(0);
      }
    }
  });

  it('unused-size mask (ch26) is all 0', () => {
    for (let r = 0; r < MAX_BOARD; r++) {
      for (let c = 0; c < MAX_BOARD; c++) {
        expect(tensor[chw(26, r, c)]).toBe(0);
      }
    }
  });

  it('reserve channels are 1 (3/3) on all four sizes', () => {
    for (let i = 0; i < 4; i++) {
      expect(tensor[chw(16 + i, 1, 1)]).toBeCloseTo(1.0, 5);
      expect(tensor[chw(20 + i, 1, 1)]).toBeCloseTo(1.0, 5);
    }
  });
});

describe('moveToActionIndex / actionIndexToMove round-trip', () => {
  const state = engine.initialState(PRESET_3X3);

  it('round-trips PlaceFromReserve moves', () => {
    const move: Move = {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'M', // index 1
      to: { row: 1, col: 2 },
    };
    const idx = moveToActionIndex(move, state);
    expect(idx).toBe(1 * 16 + (1 * 4 + 2)); // 22
    const decoded = actionIndexToMove(idx, state);
    expect(decoded).toEqual(move);
    expect(idx).toBeLessThan(PLACE_ACTIONS);
  });

  it('round-trips MoveOnBoard moves', () => {
    const move: Move = {
      kind: 'moveOnBoard',
      player: 'P1',
      from: { row: 0, col: 1 }, // from_idx = 1
      to: { row: 2, col: 3 }, // to_idx = 11
    };
    const idx = moveToActionIndex(move, state);
    expect(idx).toBe(64 + 11 * 16 + 1);
    expect(idx).toBeGreaterThanOrEqual(PLACE_ACTIONS);
    expect(idx).toBeLessThan(TOTAL_ACTIONS);
    const decoded = actionIndexToMove(idx, state);
    expect(decoded).toEqual(move);
  });
});

describe('legalActionMask', () => {
  it('PRESET_3X3 initial state has exactly 27 legal actions (3 sizes x 9 cells)', () => {
    const state = engine.initialState(PRESET_3X3);
    const mask = legalActionMask(state);
    let count = 0;
    for (const v of mask) if (v === 1) count++;
    expect(count).toBe(27);
  });

  it('PRESET_4X4_XL initial state has exactly 64 legal actions (4 sizes x 16 cells)', () => {
    const state = engine.initialState(PRESET_4X4_XL);
    const mask = legalActionMask(state);
    let count = 0;
    for (const v of mask) if (v === 1) count++;
    expect(count).toBe(64);
  });
});
