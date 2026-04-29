import { describe, expect, it } from 'vitest';
import { engine } from './engine';
import { hashState } from './hash';
import { PRESET_3X3, PRESET_4X4_XL } from '../rules/presets';
import type { GameRules, GameState, Move } from '../types';

const place = (player: 'P1' | 'P2', sizeId: string, row: number, col: number): Move => ({
  kind: 'placeFromReserve',
  player,
  sizeId,
  to: { row, col },
});

const moveOn = (player: 'P1' | 'P2', fr: number, fc: number, tr: number, tc: number): Move => ({
  kind: 'moveOnBoard',
  player,
  from: { row: fr, col: fc },
  to: { row: tr, col: tc },
});

describe('engine.initialState', () => {
  it('produces an empty 3x3 board with full reserves and P1 to move', () => {
    const s = engine.initialState(PRESET_3X3);
    expect(s.board).toHaveLength(3);
    for (const row of s.board) {
      expect(row).toHaveLength(3);
      for (const cell of row) expect(cell).toEqual([]);
    }
    expect(s.reserves.P1).toEqual({ S: 2, M: 2, L: 2 });
    expect(s.reserves.P2).toEqual({ S: 2, M: 2, L: 2 });
    expect(s.toMove).toBe('P1');
    expect(s.ply).toBe(0);
    expect(s.history).toEqual([]);
    expect(s.outcome).toBeNull();
  });

  it('produces a 4x4 board for the XL preset', () => {
    const s = engine.initialState(PRESET_4X4_XL);
    expect(s.board).toHaveLength(4);
    expect(s.reserves.P1).toEqual({ S: 3, M: 3, L: 3, XL: 3 });
  });
});

describe('engine.legalMoves', () => {
  it('initial 3x3 has pieceSizes.length * boardSize^2 moves', () => {
    const s = engine.initialState(PRESET_3X3);
    const moves = engine.legalMoves(s);
    expect(moves).toHaveLength(PRESET_3X3.pieceSizes.length * PRESET_3X3.boardSize ** 2);
  });

  it('initial 4x4 has pieceSizes.length * boardSize^2 moves', () => {
    const s = engine.initialState(PRESET_4X4_XL);
    const moves = engine.legalMoves(s);
    expect(moves).toHaveLength(PRESET_4X4_XL.pieceSizes.length * PRESET_4X4_XL.boardSize ** 2);
  });

  it('returns an empty list when the game is already decided', () => {
    const s = engine.initialState(PRESET_3X3);
    const decided: GameState = { ...s, outcome: 'P1' };
    expect(engine.legalMoves(decided)).toEqual([]);
  });
});

describe('engine.applyMove (placeFromReserve)', () => {
  it('reduces the reserve, switches player, and increments ply', () => {
    const s0 = engine.initialState(PRESET_3X3);
    const s1 = engine.applyMove(s0, place('P1', 'L', 0, 0));
    expect(s1.reserves.P1.L).toBe(1);
    expect(s1.toMove).toBe('P2');
    expect(s1.ply).toBe(1);
    expect(s1.board[0]?.[0]).toEqual([{ owner: 'P1', sizeId: 'L' }]);
    expect(s1.history).toHaveLength(1);
    // Original state unchanged (immutability).
    expect(s0.reserves.P1.L).toBe(2);
    expect(s0.board[0]?.[0]).toEqual([]);
    expect(s0.ply).toBe(0);
  });

  it('rejects placing when wrong player to move', () => {
    const s0 = engine.initialState(PRESET_3X3);
    expect(() => engine.applyMove(s0, place('P2', 'L', 0, 0))).toThrow();
  });

  it('rejects placing onto an out-of-bounds cell', () => {
    const s0 = engine.initialState(PRESET_3X3);
    expect(() => engine.applyMove(s0, place('P1', 'L', 5, 5))).toThrow();
  });

  it('rejects placing when reserve is empty', () => {
    let s = engine.initialState(PRESET_3X3);
    s = engine.applyMove(s, place('P1', 'L', 0, 0));
    s = engine.applyMove(s, place('P2', 'L', 1, 1));
    s = engine.applyMove(s, place('P1', 'L', 0, 1));
    s = engine.applyMove(s, place('P2', 'L', 2, 2));
    // P1 has used both Ls already.
    expect(s.reserves.P1.L).toBe(0);
    expect(() => engine.applyMove(s, place('P1', 'L', 1, 0))).toThrow();
  });
});

describe('engine.applyMove covering rules', () => {
  it('allows covering smaller pieces with larger', () => {
    let s = engine.initialState(PRESET_3X3);
    s = engine.applyMove(s, place('P1', 'S', 0, 0));
    // P2 covers with M (larger).
    s = engine.applyMove(s, place('P2', 'M', 0, 0));
    expect(s.board[0]?.[0]).toEqual([
      { owner: 'P1', sizeId: 'S' },
      { owner: 'P2', sizeId: 'M' },
    ]);
  });

  it('rejects covering with the same size (allowSameSizeCover=false)', () => {
    let s = engine.initialState(PRESET_3X3);
    s = engine.applyMove(s, place('P1', 'S', 0, 0));
    expect(() => engine.applyMove(s, place('P2', 'S', 0, 0))).toThrow();
  });

  it('rejects covering with a smaller piece', () => {
    let s = engine.initialState(PRESET_3X3);
    s = engine.applyMove(s, place('P1', 'L', 0, 0));
    expect(() => engine.applyMove(s, place('P2', 'S', 0, 0))).toThrow();
  });

  it('allows self-cover (P1 covers own piece with larger of own)', () => {
    let s = engine.initialState(PRESET_3X3);
    s = engine.applyMove(s, place('P1', 'S', 0, 0));
    s = engine.applyMove(s, place('P2', 'S', 1, 1));
    // P1 covers their own S with M.
    s = engine.applyMove(s, place('P1', 'M', 0, 0));
    expect(s.board[0]?.[0]).toEqual([
      { owner: 'P1', sizeId: 'S' },
      { owner: 'P1', sizeId: 'M' },
    ]);
  });

  it('disallows self-cover when allowSelfCover=false', () => {
    const rules: GameRules = { ...PRESET_3X3, allowSelfCover: false };
    let s = engine.initialState(rules);
    s = engine.applyMove(s, place('P1', 'S', 0, 0));
    s = engine.applyMove(s, place('P2', 'S', 1, 1));
    expect(() => engine.applyMove(s, place('P1', 'M', 0, 0))).toThrow();
  });
});

describe('engine.applyMove (moveOnBoard)', () => {
  it('moves a piece across the board (smaller-only cover preserved)', () => {
    let s = engine.initialState(PRESET_3X3);
    s = engine.applyMove(s, place('P1', 'L', 0, 0));
    s = engine.applyMove(s, place('P2', 'S', 1, 1));
    // P1 moves L from (0,0) to (2,2) — empty cell, fine.
    s = engine.applyMove(s, moveOn('P1', 0, 0, 2, 2));
    expect(s.board[0]?.[0]).toEqual([]);
    expect(s.board[2]?.[2]).toEqual([{ owner: 'P1', sizeId: 'L' }]);
  });

  it('rejects moving an opponent piece', () => {
    let s = engine.initialState(PRESET_3X3);
    s = engine.applyMove(s, place('P1', 'L', 0, 0));
    expect(() => engine.applyMove(s, moveOn('P2', 0, 0, 1, 1))).toThrow();
  });
});

describe('engine.isWin', () => {
  it('detects a horizontal line of 3', () => {
    let s = engine.initialState(PRESET_3X3);
    s = engine.applyMove(s, place('P1', 'L', 0, 0));
    s = engine.applyMove(s, place('P2', 'M', 1, 0));
    s = engine.applyMove(s, place('P1', 'L', 0, 1));
    s = engine.applyMove(s, place('P2', 'M', 1, 1));
    s = engine.applyMove(s, place('P1', 'M', 0, 2));
    expect(engine.isWin(s)).toBe('P1');
    expect(engine.outcome(s)).toBe('P1');
  });

  it('detects a vertical line', () => {
    let s = engine.initialState(PRESET_3X3);
    s = engine.applyMove(s, place('P1', 'L', 0, 0));
    s = engine.applyMove(s, place('P2', 'M', 0, 1));
    s = engine.applyMove(s, place('P1', 'L', 1, 0));
    s = engine.applyMove(s, place('P2', 'M', 1, 1));
    s = engine.applyMove(s, place('P1', 'M', 2, 0));
    expect(engine.isWin(s)).toBe('P1');
  });

  it('detects an anti-diagonal line on a 4x4 board', () => {
    let s = engine.initialState(PRESET_4X4_XL);
    s = engine.applyMove(s, place('P1', 'XL', 0, 3));
    s = engine.applyMove(s, place('P2', 'XL', 0, 0));
    s = engine.applyMove(s, place('P1', 'XL', 1, 2));
    s = engine.applyMove(s, place('P2', 'XL', 0, 1));
    s = engine.applyMove(s, place('P1', 'XL', 2, 1));
    s = engine.applyMove(s, place('P2', 'L', 0, 2));
    s = engine.applyMove(s, place('P1', 'L', 3, 0));
    expect(engine.isWin(s)).toBe('P1');
  });

  it('returns null when there is no winner', () => {
    const s = engine.initialState(PRESET_3X3);
    expect(engine.isWin(s)).toBeNull();
  });
});

describe('engine.outcome', () => {
  it('returns "draw" when ply >= maxPly', () => {
    const tinyMaxPly: GameRules = { ...PRESET_3X3, maxPly: 1 };
    let s = engine.initialState(tinyMaxPly);
    s = engine.applyMove(s, place('P1', 'S', 0, 0));
    expect(s.ply).toBe(1);
    expect(engine.outcome(s)).toBe('draw');
  });

  it('returns "draw" on threefold repetition', () => {
    // Manufacture a state whose current-position hash count is already >= 3.
    const s = engine.initialState(PRESET_3X3);
    const h = hashState(s);
    s.repetition.set(h, 3);
    expect(engine.outcome(s)).toBe('draw');
  });

  it('returns null while play is ongoing', () => {
    const s = engine.initialState(PRESET_3X3);
    expect(engine.outcome(s)).toBeNull();
    expect(engine.isTerminal(s)).toBe(false);
  });

  it('isTerminal is true for decided states', () => {
    let s = engine.initialState(PRESET_3X3);
    s = engine.applyMove(s, place('P1', 'L', 0, 0));
    s = engine.applyMove(s, place('P2', 'M', 1, 0));
    s = engine.applyMove(s, place('P1', 'L', 0, 1));
    s = engine.applyMove(s, place('P2', 'M', 1, 1));
    s = engine.applyMove(s, place('P1', 'M', 0, 2));
    expect(engine.isTerminal(s)).toBe(true);
  });
});

describe('engine.undo', () => {
  it('is a no-op on the initial state', () => {
    const s = engine.initialState(PRESET_3X3);
    const back = engine.undo(s);
    expect(back).toBe(s);
  });

  it('reverses placeFromReserve exactly (hash equality)', () => {
    const s0 = engine.initialState(PRESET_3X3);
    const s1 = engine.applyMove(s0, place('P1', 'L', 1, 1));
    const s0back = engine.undo(s1);
    expect(hashState(s0back)).toBe(hashState(s0));
    expect(s0back.ply).toBe(s0.ply);
    expect(s0back.toMove).toBe(s0.toMove);
    expect(s0back.history).toHaveLength(0);
  });

  it('reverses moveOnBoard exactly', () => {
    let s = engine.initialState(PRESET_3X3);
    s = engine.applyMove(s, place('P1', 'L', 0, 0));
    s = engine.applyMove(s, place('P2', 'S', 1, 1));
    const before = s;
    const after = engine.applyMove(s, moveOn('P1', 0, 0, 2, 2));
    const back = engine.undo(after);
    expect(hashState(back)).toBe(hashState(before));
  });
});

describe('engine.validateRules (invalid inputs)', () => {
  it('flags mismatched pieceSizes / piecesPerSize lengths', () => {
    const r = engine.validateRules({
      ...PRESET_3X3,
      piecesPerSize: [2, 2],
    });
    expect(r.ok).toBe(false);
    expect(r.errors.some((e) => e.includes('length'))).toBe(true);
  });

  it('flags non-positive piece counts', () => {
    const r = engine.validateRules({
      ...PRESET_3X3,
      piecesPerSize: [2, 0, 2],
    });
    expect(r.ok).toBe(false);
  });

  it('flags too-small board size', () => {
    const r = engine.validateRules({ ...PRESET_3X3, boardSize: 1 });
    expect(r.ok).toBe(false);
  });

  it('flags duplicate piece-size ids', () => {
    const r = engine.validateRules({
      ...PRESET_3X3,
      pieceSizes: [
        { id: 'S', rank: 0, displayName: { ja: 'a', en: 'a' } },
        { id: 'S', rank: 1, displayName: { ja: 'b', en: 'b' } },
        { id: 'L', rank: 2, displayName: { ja: 'c', en: 'c' } },
      ],
    });
    expect(r.ok).toBe(false);
  });
});
