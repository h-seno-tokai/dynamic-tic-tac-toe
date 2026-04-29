import { create } from 'zustand';
import { engine, type GameRules, type GameState, type Move, type Player } from '@/domain';

export type GameMode = 'local-2p' | 'cpu';

export interface CpuOpts {
  /** CPU difficulty 1..10. Required for 'cpu' mode, ignored otherwise. */
  difficulty?: number;
  /** Which side the human plays. Required for 'cpu' mode, ignored otherwise. */
  humanSide?: Player;
}

export interface GameStoreState {
  // ----- state -----
  currentGame: GameState | null;
  mode: GameMode | null;
  cpuDifficulty: number | null;
  humanSide: Player | null;

  // ----- actions -----
  startNewGame: (rules: GameRules, mode: GameMode, opts?: CpuOpts) => void;
  /** Returns true if the move was applied; false if illegal or no game. */
  applyMove: (move: Move) => boolean;
  undo: () => void;
  surrender: () => void;
  endGame: () => void;

  // ----- selectors (functions, not derived state) -----
  legalMoves: () => Move[];
  isCpuTurn: () => boolean;
  isTerminal: () => boolean;
  currentOutcome: () => Player | 'draw' | null;
}

function clampDifficulty(n: number): number {
  if (!Number.isFinite(n)) return 1;
  const i = Math.round(n);
  if (i < 1) return 1;
  if (i > 10) return 10;
  return i;
}

function isMoveEqual(a: Move, b: Move): boolean {
  if (a.kind !== b.kind) return false;
  if (a.player !== b.player) return false;
  if (a.kind === 'placeFromReserve' && b.kind === 'placeFromReserve') {
    return a.sizeId === b.sizeId && a.to.row === b.to.row && a.to.col === b.to.col;
  }
  if (a.kind === 'moveOnBoard' && b.kind === 'moveOnBoard') {
    return (
      a.from.row === b.from.row &&
      a.from.col === b.from.col &&
      a.to.row === b.to.row &&
      a.to.col === b.to.col
    );
  }
  return false;
}

export const useGameStore = create<GameStoreState>()((set, get) => ({
  currentGame: null,
  mode: null,
  cpuDifficulty: null,
  humanSide: null,

  startNewGame: (rules, mode, opts) => {
    const initial = engine.initialState(rules);
    if (mode === 'cpu') {
      const difficulty = clampDifficulty(opts?.difficulty ?? 1);
      const humanSide: Player = opts?.humanSide ?? 'P1';
      set({
        currentGame: initial,
        mode,
        cpuDifficulty: difficulty,
        humanSide,
      });
    } else {
      set({
        currentGame: initial,
        mode,
        cpuDifficulty: null,
        humanSide: null,
      });
    }
  },

  applyMove: (move) => {
    const { currentGame } = get();
    if (!currentGame) return false;
    if (currentGame.outcome !== null) return false;
    // Validate against legal moves to avoid throwing in normal flow.
    const legal = engine.legalMoves(currentGame);
    if (!legal.some((m) => isMoveEqual(m, move))) return false;
    const next = engine.applyMove(currentGame, move);
    set({ currentGame: next });
    return true;
  },

  undo: () => {
    const { currentGame } = get();
    if (!currentGame) return;
    const prev = engine.undo(currentGame);
    set({ currentGame: prev });
  },

  surrender: () => {
    const { currentGame } = get();
    if (!currentGame) return;
    if (currentGame.outcome !== null) return;
    // The losing side is the one to move (they give up their turn).
    const loser: Player = currentGame.toMove;
    const winner: Player = loser === 'P1' ? 'P2' : 'P1';
    set({ currentGame: { ...currentGame, outcome: winner } });
  },

  endGame: () => {
    set({
      currentGame: null,
      mode: null,
      cpuDifficulty: null,
      humanSide: null,
    });
  },

  legalMoves: () => {
    const { currentGame } = get();
    if (!currentGame) return [];
    return engine.legalMoves(currentGame);
  },

  isCpuTurn: () => {
    const { currentGame, mode, humanSide } = get();
    if (!currentGame) return false;
    if (mode !== 'cpu' || humanSide === null) return false;
    if (currentGame.outcome !== null) return false;
    return currentGame.toMove !== humanSide;
  },

  isTerminal: () => {
    const { currentGame } = get();
    if (!currentGame) return false;
    return engine.isTerminal(currentGame);
  },

  currentOutcome: () => {
    const { currentGame } = get();
    if (!currentGame) return null;
    return engine.outcome(currentGame);
  },
}));
