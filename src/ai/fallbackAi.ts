import type { GameState, Move, Player } from '@/domain';
import { engine } from '@/domain';

export type FallbackRandom = () => number;

function clampDifficulty(difficulty: number): number {
  if (!Number.isFinite(difficulty)) return 1;
  return Math.min(10, Math.max(1, Math.round(difficulty)));
}

function otherPlayer(player: Player): Player {
  return player === 'P1' ? 'P2' : 'P1';
}

function randomItem<T>(items: readonly T[], random: FallbackRandom): T {
  const item = items[Math.floor(random() * items.length)];
  if (item === undefined) {
    throw new Error('selectFallbackMove: no legal moves');
  }
  return item;
}

function winsFor(state: GameState, move: Move, player: Player): boolean {
  const next = engine.applyMove(state, move);
  return engine.outcome(next) === player || engine.isWin(next) === player;
}

function immediateWinningMoves(state: GameState, player: Player): Move[] {
  return engine.legalMoves(state).filter((move) => winsFor(state, move, player));
}

/**
 * Lightweight temporary CPU move picker used while the trained ONNX model is
 * unavailable. It intentionally stays shallow: legal random play with a small
 * one-ply tactic layer for wins and immediate blocks.
 */
export function selectFallbackMove(
  state: GameState,
  difficulty: number,
  random: FallbackRandom = Math.random,
): Move {
  const legalMoves = engine.legalMoves(state);
  if (legalMoves.length === 0) {
    throw new Error('selectFallbackMove: no legal moves');
  }

  const level = clampDifficulty(difficulty);
  const tacticChance = (level - 1) / 9;
  if (random() > tacticChance) {
    return randomItem(legalMoves, random);
  }

  const player = state.toMove;
  const winningMoves = legalMoves.filter((move) => winsFor(state, move, player));
  if (winningMoves.length > 0) {
    return randomItem(winningMoves, random);
  }

  const opponent = otherPlayer(player);
  const opponentTurnState: GameState = { ...state, toMove: opponent };
  if (immediateWinningMoves(opponentTurnState, opponent).length > 0) {
    const blockingMoves = legalMoves.filter((move) => {
      const next = engine.applyMove(state, move);
      return immediateWinningMoves(next, opponent).length === 0;
    });
    if (blockingMoves.length > 0) {
      return randomItem(blockingMoves, random);
    }
  }

  return randomItem(legalMoves, random);
}
