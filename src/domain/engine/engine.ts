import type {
  Board,
  Cell,
  GameEngine,
  GameRules,
  GameState,
  Move,
  Piece,
  PieceSize,
  Player,
  Position,
  Reserve,
} from '../types';
import { hashState } from './hash';

// ---------- internal helpers ----------

function otherPlayer(p: Player): Player {
  return p === 'P1' ? 'P2' : 'P1';
}

function emptyBoard(size: number): Board {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => [] as Cell));
}

function makeReserve(rules: GameRules): Reserve {
  const r: Reserve = {};
  for (let i = 0; i < rules.pieceSizes.length; i++) {
    const ps = rules.pieceSizes[i];
    const cnt = rules.piecesPerSize[i];
    if (!ps || cnt === undefined) continue;
    r[ps.id] = cnt;
  }
  return r;
}

function cloneBoard(board: Board): Board {
  return board.map((row) => row.map((cell) => cell.slice()));
}

function cloneReserve(r: Reserve): Reserve {
  return { ...r };
}

function cloneState(state: GameState): GameState {
  return {
    rules: state.rules,
    board: cloneBoard(state.board),
    reserves: {
      P1: cloneReserve(state.reserves.P1),
      P2: cloneReserve(state.reserves.P2),
    },
    toMove: state.toMove,
    history: state.history.slice(),
    ply: state.ply,
    repetition: new Map(state.repetition),
    outcome: state.outcome,
  };
}

function rankMap(pieceSizes: readonly PieceSize[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const ps of pieceSizes) m.set(ps.id, ps.rank);
  return m;
}

function topOf(cell: Cell): Piece | undefined {
  return cell.length > 0 ? cell[cell.length - 1] : undefined;
}

function inBounds(rules: GameRules, p: Position): boolean {
  return p.row >= 0 && p.row < rules.boardSize && p.col >= 0 && p.col < rules.boardSize;
}

/**
 * True iff a piece of `incomingRank` belonging to `incomingOwner` may land
 * on `targetCell` under the given rules. Empty cells are always allowed.
 */
function canCover(
  rules: GameRules,
  incomingOwner: Player,
  incomingRank: number,
  targetCell: Cell,
  ranks: Map<string, number>,
): boolean {
  const top = topOf(targetCell);
  if (!top) return true;
  const topRank = ranks.get(top.sizeId);
  if (topRank === undefined) return false;
  if (incomingRank < topRank) return false;
  if (incomingRank === topRank && !rules.allowSameSizeCover) return false;
  if (top.owner === incomingOwner && !rules.allowSelfCover) return false;
  return true;
}

// ---------- public engine ----------

function initialState(rules: GameRules): GameState {
  const state: GameState = {
    rules,
    board: emptyBoard(rules.boardSize),
    reserves: {
      P1: makeReserve(rules),
      P2: makeReserve(rules),
    },
    toMove: 'P1',
    history: [],
    ply: 0,
    repetition: new Map<string, number>(),
    outcome: null,
  };
  // Record the starting position once so repetition tracking is consistent.
  state.repetition.set(hashState(state), 1);
  return state;
}

function legalMoves(state: GameState): Move[] {
  if (state.outcome !== null) return [];
  const { rules, toMove, board } = state;
  const ranks = rankMap(rules.pieceSizes);
  const reserve = state.reserves[toMove];
  const moves: Move[] = [];

  // 1. Place from reserve.
  for (const ps of rules.pieceSizes) {
    const remaining = reserve[ps.id] ?? 0;
    if (remaining <= 0) continue;
    for (let r = 0; r < rules.boardSize; r++) {
      for (let c = 0; c < rules.boardSize; c++) {
        const cell = board[r]?.[c];
        if (!cell) continue;
        if (canCover(rules, toMove, ps.rank, cell, ranks)) {
          moves.push({
            kind: 'placeFromReserve',
            player: toMove,
            sizeId: ps.id,
            to: { row: r, col: c },
          });
        }
      }
    }
  }

  // 2. Move on board: pick up own top piece, drop on a different cell.
  for (let r = 0; r < rules.boardSize; r++) {
    for (let c = 0; c < rules.boardSize; c++) {
      const cell = board[r]?.[c];
      if (!cell) continue;
      const top = topOf(cell);
      if (top?.owner !== toMove) continue;
      const movingRank = ranks.get(top.sizeId);
      if (movingRank === undefined) continue;
      for (let r2 = 0; r2 < rules.boardSize; r2++) {
        for (let c2 = 0; c2 < rules.boardSize; c2++) {
          if (r === r2 && c === c2) continue;
          const target = board[r2]?.[c2];
          if (!target) continue;
          if (canCover(rules, toMove, movingRank, target, ranks)) {
            moves.push({
              kind: 'moveOnBoard',
              player: toMove,
              from: { row: r, col: c },
              to: { row: r2, col: c2 },
            });
          }
        }
      }
    }
  }

  return moves;
}

function applyMove(state: GameState, move: Move): GameState {
  if (state.outcome !== null) {
    throw new Error('applyMove: game has already ended');
  }
  if (move.player !== state.toMove) {
    throw new Error(`applyMove: ${move.player} cannot move; ${state.toMove} to move`);
  }
  const { rules } = state;
  if (!inBounds(rules, move.to)) {
    throw new Error('applyMove: target out of bounds');
  }
  if (move.kind === 'moveOnBoard' && !inBounds(rules, move.from)) {
    throw new Error('applyMove: source out of bounds');
  }

  const next = cloneState(state);
  const ranks = rankMap(rules.pieceSizes);

  if (move.kind === 'placeFromReserve') {
    const remaining = next.reserves[move.player][move.sizeId] ?? 0;
    if (remaining <= 0) {
      throw new Error(`applyMove: no ${move.sizeId} pieces in reserve`);
    }
    const targetCell = next.board[move.to.row]?.[move.to.col];
    if (!targetCell) throw new Error('applyMove: target cell missing');
    const ps = rules.pieceSizes.find((s) => s.id === move.sizeId);
    if (!ps) throw new Error(`applyMove: unknown sizeId ${move.sizeId}`);
    if (!canCover(rules, move.player, ps.rank, targetCell, ranks)) {
      throw new Error('applyMove: illegal cover');
    }
    targetCell.push({ owner: move.player, sizeId: move.sizeId });
    next.reserves[move.player][move.sizeId] = remaining - 1;
  } else {
    const fromCell = next.board[move.from.row]?.[move.from.col];
    const toCell = next.board[move.to.row]?.[move.to.col];
    if (!fromCell || !toCell) throw new Error('applyMove: cell missing');
    const top = topOf(fromCell);
    if (!top) throw new Error('applyMove: source cell empty');
    if (top.owner !== move.player) {
      throw new Error('applyMove: not own piece');
    }
    const movingRank = ranks.get(top.sizeId);
    if (movingRank === undefined) {
      throw new Error('applyMove: unknown size on board');
    }
    if (!canCover(rules, move.player, movingRank, toCell, ranks)) {
      throw new Error('applyMove: illegal cover (move on board)');
    }
    fromCell.pop();
    toCell.push(top);
  }

  next.history.push(move);
  next.ply += 1;
  next.toMove = otherPlayer(state.toMove);

  // Repetition tracking on the resulting position.
  const h = hashState(next);
  next.repetition.set(h, (next.repetition.get(h) ?? 0) + 1);

  // Update outcome eagerly so callers see terminal states.
  next.outcome = computeOutcome(next, h);

  return next;
}

function undo(state: GameState): GameState {
  if (state.history.length === 0) return state;
  const last = state.history[state.history.length - 1];
  if (!last) return state;

  const prev = cloneState(state);
  // Decrement repetition counter for the *current* (post-move) position.
  const currentHash = hashState(prev);
  const cnt = prev.repetition.get(currentHash) ?? 0;
  if (cnt <= 1) prev.repetition.delete(currentHash);
  else prev.repetition.set(currentHash, cnt - 1);

  // Reverse the move.
  if (last.kind === 'placeFromReserve') {
    const cell = prev.board[last.to.row]?.[last.to.col];
    if (!cell || cell.length === 0) {
      throw new Error('undo: target cell empty (corrupt state)');
    }
    const popped = cell.pop();
    if (popped?.owner !== last.player || popped.sizeId !== last.sizeId) {
      throw new Error('undo: top of stack does not match recorded move');
    }
    prev.reserves[last.player][last.sizeId] = (prev.reserves[last.player][last.sizeId] ?? 0) + 1;
  } else {
    const fromCell = prev.board[last.from.row]?.[last.from.col];
    const toCell = prev.board[last.to.row]?.[last.to.col];
    if (!fromCell || !toCell || toCell.length === 0) {
      throw new Error('undo: corrupt state for move-on-board');
    }
    const piece = toCell.pop();
    if (piece?.owner !== last.player) {
      throw new Error('undo: top piece is not the moved piece');
    }
    fromCell.push(piece);
  }

  prev.history.pop();
  prev.ply -= 1;
  prev.toMove = last.player;
  prev.outcome = null;
  return prev;
}

/**
 * Returns the winner if there is an N-in-a-row formed by top-of-stack
 * pieces (rows, columns, both diagonals). N is the board size.
 *
 * If both players somehow have a winning line on the same position
 * (impossible under normal play but defensive), the side that just moved
 * is preferred — but we simply return the first one we find since this
 * scenario does not arise with legal play.
 */
function isWin(state: GameState): Player | null {
  const N = state.rules.boardSize;
  const board = state.board;

  const tops: (Player | null)[][] = Array.from({ length: N }, (_, r) =>
    Array.from({ length: N }, (_, c) => {
      const cell = board[r]?.[c];
      const top = cell ? topOf(cell) : undefined;
      return top ? top.owner : null;
    }),
  );

  const checkLine = (cells: (Player | null)[]): Player | null => {
    if (cells.length === 0) return null;
    const first = cells[0];
    if (first === null || first === undefined) return null;
    for (const v of cells) if (v !== first) return null;
    return first;
  };

  // Rows & columns.
  for (let i = 0; i < N; i++) {
    const row = tops[i];
    if (row) {
      const w = checkLine(row);
      if (w) return w;
    }
    const col: (Player | null)[] = [];
    for (let j = 0; j < N; j++) col.push(tops[j]?.[i] ?? null);
    const wc = checkLine(col);
    if (wc) return wc;
  }

  // Diagonals.
  const diag1: (Player | null)[] = [];
  const diag2: (Player | null)[] = [];
  for (let i = 0; i < N; i++) {
    diag1.push(tops[i]?.[i] ?? null);
    diag2.push(tops[i]?.[N - 1 - i] ?? null);
  }
  const w1 = checkLine(diag1);
  if (w1) return w1;
  const w2 = checkLine(diag2);
  if (w2) return w2;

  return null;
}

/**
 * Compute the outcome for a state, given the (already computed) hash of
 * the current position. Win > maxPly > repetition > legal-moves draw.
 */
function computeOutcome(state: GameState, currentHash: string): Player | 'draw' | null {
  const winner = isWin(state);
  if (winner) return winner;
  if (state.ply >= state.rules.maxPly) return 'draw';
  const reps = state.repetition.get(currentHash) ?? 0;
  if (reps >= state.rules.drawByRepetition) return 'draw';
  // Cheap-to-detect "no legal moves" terminal — Gobblet rarely hits this
  // but the spec calls for it.
  // Use legalMoves directly; we are not in a hot loop yet.
  const stub: GameState = { ...state, outcome: null };
  if (legalMoves(stub).length === 0) return 'draw';
  return null;
}

function outcome(state: GameState): Player | 'draw' | null {
  if (state.outcome !== null) return state.outcome;
  return computeOutcome(state, hashState(state));
}

function isTerminal(state: GameState): boolean {
  return outcome(state) !== null;
}

function validateRules(rules: GameRules): { ok: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!Number.isInteger(rules.boardSize) || rules.boardSize < 2) {
    errors.push('boardSize must be an integer >= 2');
  }
  if (!Array.isArray(rules.pieceSizes) || rules.pieceSizes.length === 0) {
    errors.push('pieceSizes must be a non-empty array');
  }
  if (!Array.isArray(rules.piecesPerSize)) {
    errors.push('piecesPerSize must be an array');
  }
  if (
    Array.isArray(rules.pieceSizes) &&
    Array.isArray(rules.piecesPerSize) &&
    rules.pieceSizes.length !== rules.piecesPerSize.length
  ) {
    errors.push('pieceSizes.length must equal piecesPerSize.length');
  }
  if (Array.isArray(rules.piecesPerSize)) {
    for (let i = 0; i < rules.piecesPerSize.length; i++) {
      const n = rules.piecesPerSize[i];
      if (!Number.isInteger(n) || n! <= 0) {
        errors.push(`piecesPerSize[${i}] must be a positive integer`);
      }
    }
  }
  if (Array.isArray(rules.pieceSizes)) {
    const seenIds = new Set<string>();
    const seenRanks = new Set<number>();
    for (let i = 0; i < rules.pieceSizes.length; i++) {
      const ps = rules.pieceSizes[i];
      if (!ps) continue;
      if (typeof ps.id !== 'string' || ps.id.length === 0) {
        errors.push(`pieceSizes[${i}].id must be a non-empty string`);
      } else if (seenIds.has(ps.id)) {
        errors.push(`pieceSizes[${i}].id duplicates "${ps.id}"`);
      } else {
        seenIds.add(ps.id);
      }
      if (!Number.isInteger(ps.rank) || ps.rank < 0) {
        errors.push(`pieceSizes[${i}].rank must be a non-negative integer`);
      } else if (seenRanks.has(ps.rank)) {
        errors.push(`pieceSizes[${i}].rank duplicates ${ps.rank}`);
      } else {
        seenRanks.add(ps.rank);
      }
    }
  }
  if (rules.winCondition.kind === 'lineOfN') {
    if (!Number.isInteger(rules.winCondition.n) || rules.winCondition.n <= 0) {
      errors.push('winCondition.n must be a positive integer');
    }
  }
  if (!Number.isInteger(rules.maxPly) || rules.maxPly <= 0) {
    errors.push('maxPly must be a positive integer');
  }
  if (!Number.isInteger(rules.drawByRepetition) || rules.drawByRepetition <= 0) {
    errors.push('drawByRepetition must be a positive integer');
  }
  return { ok: errors.length === 0, errors };
}

export const engine: GameEngine = {
  initialState,
  legalMoves,
  applyMove,
  undo,
  isTerminal,
  isWin,
  outcome,
  validateRules,
};
