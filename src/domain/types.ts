// ============================================
// Rule definitions
// ============================================

/** Piece size tier. Ordered: larger rank = larger piece. */
export interface PieceSize {
  /** Identifier (e.g. "S" | "M" | "L" | "XL"; arbitrary string) */
  id: string;
  /** Ordering. 0 is smallest, larger = bigger. */
  rank: number;
  /** Display name (i18n) */
  displayName: { ja: string; en: string };
}

/** Win condition (extensible ADT). */
export type WinCondition =
  | { kind: 'lineOfN'; n: number }
  | { kind: 'custom'; predicate: (board: Board) => Player | null };

/** Game rules. Everything is driven from this object. */
export interface GameRules {
  /** Side length of the board (e.g. 3, 4, ...) */
  boardSize: number;
  /** Piece size definitions, ascending by rank. At least one entry. */
  pieceSizes: PieceSize[];
  /** Per-player counts for each size; same length as `pieceSizes`. */
  piecesPerSize: number[];
  /** Win condition. */
  winCondition: WinCondition;
  /** Allow covering with same-size piece? (default false: only larger.) */
  allowSameSizeCover: boolean;
  /** Allow covering own piece with own piece? (default false.) */
  allowSelfCover: boolean;
  /**
   * Forced draw if ply count exceeds this value.
   * Required to prevent infinite loops in self-play (Gobblet allows
   * indefinite movement, so "no legal moves" alone is insufficient).
   */
  maxPly: number;
  /**
   * Draw if the same position appears this many times (threefold by default).
   */
  drawByRepetition: number;
}

// ============================================
// Players & pieces
// ============================================

export type Player = 'P1' | 'P2';

export interface Piece {
  owner: Player;
  /** References `PieceSize.id`. */
  sizeId: string;
}

// ============================================
// Board
// ============================================

/** A cell holds a stack of pieces; the last entry is on top (visible). */
export type Cell = Piece[];

/** Board: row-major 2D array; length = `GameRules.boardSize`. */
export type Board = Cell[][];

// ============================================
// Reserves
// ============================================

/** Player's off-board pieces, by size id -> remaining count. */
export type Reserve = Record<string, number>;

// ============================================
// Moves
// ============================================

export interface Position {
  row: number;
  col: number;
}

/** Place a piece from the reserve onto the board. */
export interface PlaceFromReserveMove {
  kind: 'placeFromReserve';
  player: Player;
  sizeId: string;
  to: Position;
}

/** Move an on-board piece to another cell (covering allowed under rules). */
export interface MoveOnBoardMove {
  kind: 'moveOnBoard';
  player: Player;
  from: Position;
  to: Position;
}

export type Move = PlaceFromReserveMove | MoveOnBoardMove;

// ============================================
// Game state
// ============================================

export interface GameState {
  rules: GameRules;
  board: Board;
  reserves: { P1: Reserve; P2: Reserve };
  toMove: Player;
  /** Move history (used by undo). Not persisted. */
  history: Move[];
  /** Number of plies elapsed (used for `maxPly`). */
  ply: number;
  /** Hash -> count map of position occurrences (used for repetition draw). */
  repetition: Map<string, number>;
  /** Outcome. `null` = ongoing; `"draw"` = drawn. */
  outcome: Player | 'draw' | null;
}

// ============================================
// Engine (pure functions)
// ============================================

export interface GameEngine {
  initialState(rules: GameRules): GameState;
  legalMoves(state: GameState): Move[];
  applyMove(state: GameState, move: Move): GameState;
  /** No-op when history is empty. */
  undo(state: GameState): GameState;
  isTerminal(state: GameState): boolean;
  isWin(state: GameState): Player | null;
  outcome(state: GameState): Player | 'draw' | null;
  validateRules(rules: GameRules): { ok: boolean; errors: string[] };
}
