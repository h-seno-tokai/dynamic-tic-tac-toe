import { beforeEach, describe, expect, it } from 'vitest';
import { engine, PRESET_3X3, type Move } from '@/domain';
import { useGameStore } from '@/stores';
import { computeNextSelection, type Selection } from './PlayPage';

describe('computeNextSelection (piece reselection UX)', () => {
  it('clicking the same reserve pile clears the selection', () => {
    const start: Selection = { kind: 'reserve', sizeId: 'L' };
    const next = computeNextSelection(start, { kind: 'clickReserve', sizeId: 'L' });
    expect(next).toBeNull();
  });

  it('clicking a different reserve pile switches selection', () => {
    const start: Selection = { kind: 'reserve', sizeId: 'L' };
    const next = computeNextSelection(start, { kind: 'clickReserve', sizeId: 'M' });
    expect(next).toEqual({ kind: 'reserve', sizeId: 'M' });
  });

  it('clicking a reserve pile while no selection selects it', () => {
    const next = computeNextSelection(null, { kind: 'clickReserve', sizeId: 'S' });
    expect(next).toEqual({ kind: 'reserve', sizeId: 'S' });
  });

  it('clicking the same own board piece clears the selection', () => {
    const start: Selection = { kind: 'board', from: { row: 1, col: 2 } };
    const next = computeNextSelection(start, {
      kind: 'clickOwnBoardPiece',
      pos: { row: 1, col: 2 },
    });
    expect(next).toBeNull();
  });

  it('clicking a different own board piece switches selection', () => {
    const start: Selection = { kind: 'board', from: { row: 1, col: 2 } };
    const next = computeNextSelection(start, {
      kind: 'clickOwnBoardPiece',
      pos: { row: 0, col: 0 },
    });
    expect(next).toEqual({ kind: 'board', from: { row: 0, col: 0 } });
  });

  it('clicking own board piece while a reserve is selected switches to that board piece', () => {
    const start: Selection = { kind: 'reserve', sizeId: 'L' };
    const next = computeNextSelection(start, {
      kind: 'clickOwnBoardPiece',
      pos: { row: 2, col: 1 },
    });
    expect(next).toEqual({ kind: 'board', from: { row: 2, col: 1 } });
  });

  it('clicking own board piece while no selection selects it', () => {
    const next = computeNextSelection(null, {
      kind: 'clickOwnBoardPiece',
      pos: { row: 0, col: 0 },
    });
    expect(next).toEqual({ kind: 'board', from: { row: 0, col: 0 } });
  });
});

describe('place from reserve still works after a reserve selection', () => {
  beforeEach(() => {
    useGameStore.setState({
      currentGame: null,
      mode: null,
      cpuDifficulty: null,
      humanSide: null,
    });
  });

  it('selecting a reserve piece then dispatching a placeFromReserve to a legal cell succeeds', () => {
    useGameStore.getState().startNewGame(PRESET_3X3, 'local-2p');
    const game = useGameStore.getState().currentGame!;
    // Pick any legal placeFromReserve move from the engine.
    const legal = engine
      .legalMoves(game)
      .find((m): m is Extract<Move, { kind: 'placeFromReserve' }> => m.kind === 'placeFromReserve');
    expect(legal).toBeDefined();
    if (!legal) return;

    // Simulate the UI flow: nothing selected -> click reserve -> selection set,
    // then click target cell -> applyMove fires and is accepted.
    const afterClickReserve = computeNextSelection(null, {
      kind: 'clickReserve',
      sizeId: legal.sizeId,
    });
    expect(afterClickReserve).toEqual({ kind: 'reserve', sizeId: legal.sizeId });

    const accepted = useGameStore.getState().applyMove(legal);
    expect(accepted).toBe(true);
    expect(useGameStore.getState().currentGame?.ply).toBe(1);
  });
});
