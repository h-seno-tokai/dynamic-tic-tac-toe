/**
 * GameState <-> postMessage-friendly serialised form.
 *
 * `GameState.repetition` is a `Map<string, number>`. While Maps survive the
 * structured-clone algorithm in modern browsers, we explicitly serialise to a
 * tuple-array for two reasons:
 *   1. Older Workers and some test environments stringify postMessage payloads.
 *   2. Explicit shape => stable type at the protocol boundary.
 *
 * We also avoid sending function references inside `rules.winCondition`
 * (custom predicates would not survive), so the serialised state pins the
 * win condition to its `lineOfN` form. The two presets we ship both use
 * `lineOfN`, so this is sufficient.
 */

import type { GameState, Move, Player, WinCondition } from '@/domain';

export interface SerializedGameState {
  rules: SerializedRules;
  board: GameState['board'];
  reserves: GameState['reserves'];
  toMove: Player;
  history: Move[];
  ply: number;
  repetition: [string, number][];
  outcome: GameState['outcome'];
}

interface SerializedRules {
  boardSize: number;
  pieceSizes: GameState['rules']['pieceSizes'];
  piecesPerSize: number[];
  winCondition: WinCondition;
  allowSameSizeCover: boolean;
  allowSelfCover: boolean;
  maxPly: number;
  drawByRepetition: number;
}

export function serializeGameState(state: GameState): SerializedGameState {
  return {
    rules: {
      boardSize: state.rules.boardSize,
      pieceSizes: state.rules.pieceSizes,
      piecesPerSize: state.rules.piecesPerSize,
      winCondition: state.rules.winCondition,
      allowSameSizeCover: state.rules.allowSameSizeCover,
      allowSelfCover: state.rules.allowSelfCover,
      maxPly: state.rules.maxPly,
      drawByRepetition: state.rules.drawByRepetition,
    },
    board: state.board,
    reserves: state.reserves,
    toMove: state.toMove,
    history: state.history,
    ply: state.ply,
    repetition: Array.from(state.repetition.entries()),
    outcome: state.outcome,
  };
}

export function deserializeGameState(data: SerializedGameState): GameState {
  return {
    rules: {
      boardSize: data.rules.boardSize,
      pieceSizes: data.rules.pieceSizes,
      piecesPerSize: data.rules.piecesPerSize,
      winCondition: data.rules.winCondition,
      allowSameSizeCover: data.rules.allowSameSizeCover,
      allowSelfCover: data.rules.allowSelfCover,
      maxPly: data.rules.maxPly,
      drawByRepetition: data.rules.drawByRepetition,
    },
    board: data.board,
    reserves: data.reserves,
    toMove: data.toMove,
    history: data.history,
    ply: data.ply,
    repetition: new Map(data.repetition),
    outcome: data.outcome,
  };
}
