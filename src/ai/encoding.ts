/**
 * TS mirror of `ai-training/src/dttt_train/encoding.py`.
 *
 * Encodes a `GameState` into the universal-network input tensor (27, 4, 4)
 * stored CHW-flat as a `Float32Array` of length 432, and provides legal-action
 * mask + bidirectional move <-> action-index conversion.
 *
 * The numeric layout MUST stay in lock-step with the Python module so that the
 * exported ONNX weights see identical tensors at training and inference time.
 */

import type { GameState, Move, MoveOnBoardMove, PlaceFromReserveMove } from '@/domain';
import { engine } from '@/domain';

export const MAX_BOARD = 4;
export const MAX_PIECE_SIZES = 4;
export const MAX_PIECES_PER_SIZE = 3;
export const NUM_CHANNELS = 27;
export const PLACE_ACTIONS = MAX_PIECE_SIZES * MAX_BOARD * MAX_BOARD; // 64
export const MOVE_ACTIONS = MAX_BOARD * MAX_BOARD * (MAX_BOARD * MAX_BOARD); // 256
export const TOTAL_ACTIONS = PLACE_ACTIONS + MOVE_ACTIONS; // 320
export const TENSOR_LENGTH = NUM_CHANNELS * MAX_BOARD * MAX_BOARD; // 432

const PLANE_SIZE = MAX_BOARD * MAX_BOARD; // 16

/** Flat offset for a (channel, row, col) cell in CHW order. */
function chw(channel: number, row: number, col: number): number {
  return channel * PLANE_SIZE + row * MAX_BOARD + col;
}

/** Build a sizeId -> universal index map (index = position in pieceSizes). */
function sizeIdToIndex(state: GameState): Map<string, number> {
  const m = new Map<string, number>();
  state.rules.pieceSizes.forEach((ps, i) => m.set(ps.id, i));
  return m;
}

/**
 * Encode the state as a CHW-flat Float32Array of length 432.
 * Channel layout (see docs/07 §2.1):
 *   0-3   P1 top-of-stack one-hot per size
 *   4-7   P2 top-of-stack one-hot per size
 *   8-11  P1 anywhere-in-stack
 *   12-15 P2 anywhere-in-stack
 *   16-19 P1 reserve count, normalised, broadcast 4x4
 *   20-23 P2 reserve count, broadcast
 *   24    side-to-move (1 if P1 to move)
 *   25    out-of-board mask (1 outside boardSize)
 *   26    unused-size mask (1 if pieceSizes.length < MAX_PIECE_SIZES)
 */
export function encodeState(state: GameState): Float32Array {
  const out = new Float32Array(TENSOR_LENGTH);
  const { rules, board } = state;
  const bs = rules.boardSize;
  const sizeIdx = sizeIdToIndex(state);

  // Channels 0-7 (top), 8-15 (anywhere)
  for (let r = 0; r < bs; r++) {
    for (let c = 0; c < bs; c++) {
      const cell = board[r]?.[c];
      if (!cell || cell.length === 0) continue;
      const top = cell[cell.length - 1];
      if (!top) continue;
      const topIdx = sizeIdx.get(top.sizeId);
      if (topIdx === undefined) continue;
      const topChannel = topIdx + (top.owner === 'P1' ? 0 : 4);
      out[chw(topChannel, r, c)] = 1;

      // anywhere-in-stack: dedupe by (owner, sizeIdx)
      const seen = new Set<number>();
      for (const piece of cell) {
        const idx = sizeIdx.get(piece.sizeId);
        if (idx === undefined) continue;
        const ownerOffset = piece.owner === 'P1' ? 0 : 4;
        const key = idx * 2 + (ownerOffset === 0 ? 0 : 1);
        if (seen.has(key)) continue;
        seen.add(key);
        out[chw(8 + idx + ownerOffset, r, c)] = 1;
      }
    }
  }

  // Channels 16-23: reserve counts normalised against MAX_PIECES_PER_SIZE.
  const denom = MAX_PIECES_PER_SIZE;
  rules.pieceSizes.forEach((ps, i) => {
    const p1 = (state.reserves.P1[ps.id] ?? 0) / denom;
    const p2 = (state.reserves.P2[ps.id] ?? 0) / denom;
    for (let r = 0; r < MAX_BOARD; r++) {
      for (let c = 0; c < MAX_BOARD; c++) {
        out[chw(16 + i, r, c)] = p1;
        out[chw(20 + i, r, c)] = p2;
      }
    }
  });

  // Channel 24: side-to-move = 1 iff P1 to move
  if (state.toMove === 'P1') {
    for (let r = 0; r < MAX_BOARD; r++) {
      for (let c = 0; c < MAX_BOARD; c++) {
        out[chw(24, r, c)] = 1;
      }
    }
  }

  // Channel 25: out-of-board mask
  if (bs < MAX_BOARD) {
    for (let r = 0; r < MAX_BOARD; r++) {
      for (let c = 0; c < MAX_BOARD; c++) {
        if (r >= bs || c >= bs) {
          out[chw(25, r, c)] = 1;
        }
      }
    }
  }

  // Channel 26: unused-size mask
  if (rules.pieceSizes.length < MAX_PIECE_SIZES) {
    for (let r = 0; r < MAX_BOARD; r++) {
      for (let c = 0; c < MAX_BOARD; c++) {
        out[chw(26, r, c)] = 1;
      }
    }
  }

  return out;
}

/**
 * Map a `Move` to its index in the 320-d action space.
 *  PlaceFromReserve: sizeIdx * 16 + (row*4 + col), 0..63
 *  MoveOnBoard:      64 + to_idx * 16 + from_idx,  64..319
 */
export function moveToActionIndex(move: Move, state: GameState): number {
  const sizeIdx = sizeIdToIndex(state);
  return moveToActionIndexInternal(move, sizeIdx);
}

function moveToActionIndexInternal(move: Move, sizeIdx: Map<string, number>): number {
  if (move.kind === 'placeFromReserve') {
    const idx = sizeIdx.get(move.sizeId);
    if (idx === undefined) {
      throw new Error(`moveToActionIndex: unknown sizeId ${move.sizeId}`);
    }
    return idx * PLANE_SIZE + (move.to.row * MAX_BOARD + move.to.col);
  }
  const fromIdx = move.from.row * MAX_BOARD + move.from.col;
  const toIdx = move.to.row * MAX_BOARD + move.to.col;
  return PLACE_ACTIONS + toIdx * PLANE_SIZE + fromIdx;
}

/**
 * Inverse of `moveToActionIndex`. Returns a candidate move; callers should
 * verify legality via the engine. The decoded player is `state.toMove`.
 */
export function actionIndexToMove(index: number, state: GameState): Move {
  if (!Number.isInteger(index) || index < 0 || index >= TOTAL_ACTIONS) {
    throw new RangeError(`actionIndexToMove: index ${index} out of range`);
  }
  if (index < PLACE_ACTIONS) {
    const sizeIdx = Math.floor(index / PLANE_SIZE);
    const cell = index % PLANE_SIZE;
    const ps = state.rules.pieceSizes[sizeIdx];
    if (!ps) {
      throw new RangeError(
        `actionIndexToMove: size index ${sizeIdx} out of range for current preset`,
      );
    }
    const row = Math.floor(cell / MAX_BOARD);
    const col = cell % MAX_BOARD;
    const placeMove: PlaceFromReserveMove = {
      kind: 'placeFromReserve',
      player: state.toMove,
      sizeId: ps.id,
      to: { row, col },
    };
    return placeMove;
  }
  const moveIdx = index - PLACE_ACTIONS;
  const toIdx = Math.floor(moveIdx / PLANE_SIZE);
  const fromIdx = moveIdx % PLANE_SIZE;
  const fromRow = Math.floor(fromIdx / MAX_BOARD);
  const fromCol = fromIdx % MAX_BOARD;
  const toRow = Math.floor(toIdx / MAX_BOARD);
  const toCol = toIdx % MAX_BOARD;
  const moveMove: MoveOnBoardMove = {
    kind: 'moveOnBoard',
    player: state.toMove,
    from: { row: fromRow, col: fromCol },
    to: { row: toRow, col: toCol },
  };
  return moveMove;
}

/**
 * Build the 320-d legal-action mask (1 = legal, 0 = illegal).
 *
 * IMPORTANT: this mask is applied in TS, never inside the ONNX graph. The
 * graph emits raw logits for the full 320 actions; we add `-Infinity` (or
 * multiply by 0 in MCTS prior code) to the illegal entries before softmax.
 */
export function legalActionMask(state: GameState): Float32Array {
  const mask = new Float32Array(TOTAL_ACTIONS);
  const sizeIdx = sizeIdToIndex(state);
  for (const mv of engine.legalMoves(state)) {
    const idx = moveToActionIndexInternal(mv, sizeIdx);
    mask[idx] = 1;
  }
  return mask;
}
