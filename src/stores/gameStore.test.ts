import { beforeEach, describe, expect, it, vi } from 'vitest';
import { engine, PRESET_3X3, type Move } from '@/domain';
import { useGameStore } from './gameStore';

function resetStore(): void {
  useGameStore.setState({
    currentGame: null,
    mode: null,
    cpuDifficulty: null,
    humanSide: null,
  });
}

describe('gameStore', () => {
  beforeEach(() => {
    resetStore();
  });

  it('has the expected initial state', () => {
    const s = useGameStore.getState();
    expect(s.currentGame).toBeNull();
    expect(s.mode).toBeNull();
    expect(s.cpuDifficulty).toBeNull();
    expect(s.humanSide).toBeNull();
    expect(s.legalMoves()).toEqual([]);
    expect(s.isCpuTurn()).toBe(false);
    expect(s.isTerminal()).toBe(false);
    expect(s.currentOutcome()).toBeNull();
  });

  it('startNewGame initialises a 2P game using the engine', () => {
    useGameStore.getState().startNewGame(PRESET_3X3, 'local-2p');
    const s = useGameStore.getState();
    expect(s.mode).toBe('local-2p');
    expect(s.cpuDifficulty).toBeNull();
    expect(s.humanSide).toBeNull();
    expect(s.currentGame).not.toBeNull();
    expect(s.currentGame?.toMove).toBe('P1');
    expect(s.currentGame?.ply).toBe(0);
    expect(s.legalMoves().length).toBe(engine.legalMoves(s.currentGame!).length);
  });

  it('startNewGame configures CPU mode with clamped difficulty and humanSide', () => {
    useGameStore.getState().startNewGame(PRESET_3X3, 'cpu', { difficulty: 42, humanSide: 'P2' });
    const s = useGameStore.getState();
    expect(s.mode).toBe('cpu');
    expect(s.cpuDifficulty).toBe(10);
    expect(s.humanSide).toBe('P2');
    // P1 to move; human is P2 -> CPU's turn.
    expect(s.isCpuTurn()).toBe(true);
  });

  it('applyMove delegates to engine and rejects illegal moves', () => {
    useGameStore.getState().startNewGame(PRESET_3X3, 'local-2p');
    const before = useGameStore.getState().currentGame!;
    const legal = engine.legalMoves(before);
    expect(legal.length).toBeGreaterThan(0);

    // Construct an illegal move: wrong player to move.
    const illegalWrongPlayer: Move = {
      kind: 'placeFromReserve',
      player: 'P2',
      sizeId: 'S',
      to: { row: 0, col: 0 },
    };
    expect(useGameStore.getState().applyMove(illegalWrongPlayer)).toBe(false);
    // State unchanged.
    expect(useGameStore.getState().currentGame).toBe(before);

    // Apply a legal move; spy on the engine to verify delegation.
    const applySpy = vi.spyOn(engine, 'applyMove');
    const legalMove = legal[0]!;
    expect(useGameStore.getState().applyMove(legalMove)).toBe(true);
    expect(applySpy).toHaveBeenCalledTimes(1);
    expect(useGameStore.getState().currentGame?.ply).toBe(1);
    expect(useGameStore.getState().currentGame?.toMove).toBe('P2');
    applySpy.mockRestore();

    // An out-of-shape illegal move (cell out of bounds via "fake" move).
    const illegalOOB: Move = {
      kind: 'placeFromReserve',
      player: 'P2',
      sizeId: 'S',
      to: { row: 99, col: 99 },
    };
    expect(useGameStore.getState().applyMove(illegalOOB)).toBe(false);
  });

  it('undo reverts the last move', () => {
    useGameStore.getState().startNewGame(PRESET_3X3, 'local-2p');
    const initial = useGameStore.getState().currentGame!;
    const move = engine.legalMoves(initial)[0]!;
    useGameStore.getState().applyMove(move);
    expect(useGameStore.getState().currentGame?.ply).toBe(1);
    useGameStore.getState().undo();
    const reverted = useGameStore.getState().currentGame!;
    expect(reverted.ply).toBe(0);
    expect(reverted.toMove).toBe('P1');
    expect(reverted.history.length).toBe(0);
  });

  it('surrender ends the game with the opponent winning', () => {
    useGameStore.getState().startNewGame(PRESET_3X3, 'local-2p');
    useGameStore.getState().surrender();
    const s = useGameStore.getState();
    expect(s.currentGame?.outcome).toBe('P2');
    expect(s.isTerminal()).toBe(true);
    expect(s.currentOutcome()).toBe('P2');
  });

  it('endGame clears the store back to its initial state', () => {
    useGameStore.getState().startNewGame(PRESET_3X3, 'cpu', { difficulty: 5, humanSide: 'P1' });
    useGameStore.getState().endGame();
    const s = useGameStore.getState();
    expect(s.currentGame).toBeNull();
    expect(s.mode).toBeNull();
    expect(s.cpuDifficulty).toBeNull();
    expect(s.humanSide).toBeNull();
  });

  it('isCpuTurn is false in 2P mode and after the game ends', () => {
    useGameStore.getState().startNewGame(PRESET_3X3, 'local-2p');
    expect(useGameStore.getState().isCpuTurn()).toBe(false);

    useGameStore.getState().startNewGame(PRESET_3X3, 'cpu', { difficulty: 3, humanSide: 'P1' });
    // Human is P1 and P1 to move -> not CPU's turn yet.
    expect(useGameStore.getState().isCpuTurn()).toBe(false);
    useGameStore.getState().surrender();
    expect(useGameStore.getState().isCpuTurn()).toBe(false);
  });
});
