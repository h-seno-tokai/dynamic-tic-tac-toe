import type { GameState, Piece, PieceSize } from '../types';

/**
 * Encode a single cell's stack as a compact string. Pieces are listed
 * bottom-to-top (matching the array order). Each piece is `<sizeRank><owner>`,
 * pieces separated by `,`, an empty cell becomes `-`.
 *
 * Example: `0P1,2P2` = small P1 piece at the bottom, large P2 piece on top.
 */
function encodeCell(stack: Piece[], sizeRank: Map<string, number>): string {
  if (stack.length === 0) return '-';
  return stack
    .map((p) => {
      const rank = sizeRank.get(p.sizeId);
      // If the size id is unknown (shouldn't happen if rules match the
      // pieces), fall back to the raw id so the hash is still deterministic.
      const r = rank === undefined ? p.sizeId : String(rank);
      return `${r}${p.owner}`;
    })
    .join(',');
}

function rankLookup(pieceSizes: readonly PieceSize[]): Map<string, number> {
  const map = new Map<string, number>();
  for (const ps of pieceSizes) map.set(ps.id, ps.rank);
  return map;
}

/**
 * Canonical hash of a position. Includes:
 *  - board contents (ordered top-of-stack by listing the full stack)
 *  - both reserves (sorted by piece-size rank)
 *  - the side to move
 *
 * The result is stable for equal positions and distinguishes states that
 * differ in any of the above components.
 */
export function hashState(state: GameState): string {
  const sizeRank = rankLookup(state.rules.pieceSizes);
  const N = state.rules.boardSize;

  // Board section: row-major, cells separated by '|', rows by '/'.
  const rows: string[] = [];
  for (let r = 0; r < N; r++) {
    const row = state.board[r];
    const cells: string[] = [];
    for (let c = 0; c < N; c++) {
      // row may be undefined if board malformed, but defaults are already
      // populated in initialState — guard anyway for safety.
      const cell = row?.[c] ?? [];
      cells.push(encodeCell(cell, sizeRank));
    }
    rows.push(cells.join('|'));
  }
  const boardPart = rows.join('/');

  // Reserve section: sort sizes by rank for determinism.
  const orderedSizes = [...state.rules.pieceSizes].sort((a, b) => a.rank - b.rank);
  const reservePart = (['P1', 'P2'] as const)
    .map((player) => {
      const r = state.reserves[player];
      return orderedSizes.map((ps) => `${ps.id}:${r[ps.id] ?? 0}`).join(',');
    })
    .join(';');

  return `${boardPart}#${reservePart}#${state.toMove}`;
}
