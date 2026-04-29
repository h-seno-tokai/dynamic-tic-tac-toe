import type { Move, PieceSize, Position } from '@/domain';

const COL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

export function formatPos(p: Position): string {
  // Columns: letters A, B, C, ... Rows: 1-indexed numbers.
  const col = COL_LETTERS[p.col] ?? `?${p.col}`;
  return `${col}-${p.row + 1}`;
}

export function sizeLabel(sizeId: string, table: PieceSize[] | undefined): string {
  return table?.find((s) => s.id === sizeId)?.displayName.ja ?? sizeId;
}

/**
 * Format a single move into Japanese display text.
 * - place:  "P1: 大 を A-1 に置く"
 * - move:   "P2: A-1 から B-2 へ移動"
 *
 * Note: a `moveOnBoard` move only references the source square, not the
 * piece. We render only positions. Callers wanting size detail can
 * pre-process the history alongside game state to resolve which piece moved.
 */
export function formatMove(move: Move, pieceSizes?: PieceSize[]): string {
  if (move.kind === 'placeFromReserve') {
    return `${move.player}: ${sizeLabel(move.sizeId, pieceSizes)} を ${formatPos(move.to)} に置く`;
  }
  return `${move.player}: ${formatPos(move.from)} から ${formatPos(move.to)} へ移動`;
}
