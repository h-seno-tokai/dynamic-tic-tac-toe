import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FC,
  type KeyboardEvent,
} from 'react';
import type { GameState, Position } from '@/domain';
import { Cell } from './Cell';

export interface BoardProps {
  state: GameState;
  onCellClick?: (pos: Position) => void;
  highlight?: Position[];
  disabled?: boolean;
  /** Optional aria-label for the grid root. */
  'aria-label'?: string;
}

/**
 * Renders an N×N CSS grid where N = `state.rules.boardSize`.
 *
 * - role="grid" with role="gridcell" children.
 * - Roving tabIndex + arrow-key navigation; Enter / Space activates.
 * - Fully ruleset-agnostic: nothing about the size is hardcoded.
 */
export const Board: FC<BoardProps> = ({
  state,
  onCellClick,
  highlight,
  disabled = false,
  'aria-label': ariaLabel = '盤面',
}) => {
  const N = state.rules.boardSize;
  const pieceSizes = state.rules.pieceSizes;

  // Memoize highlight set as "row,col" strings for O(1) lookup.
  const highlightKey = useMemo(() => {
    if (!highlight || highlight.length === 0) return null;
    const set = new Set<string>();
    for (const h of highlight) set.add(`${h.row},${h.col}`);
    return set;
  }, [highlight]);

  const [focused, setFocused] = useState<Position>({ row: 0, col: 0 });
  const cellRefs = useRef<(HTMLDivElement | null)[]>([]);
  // Resize ref array when N changes.
  if (cellRefs.current.length !== N * N) {
    cellRefs.current = Array.from<HTMLDivElement | null>({ length: N * N }).fill(null);
  }

  const indexOf = useCallback((row: number, col: number) => row * N + col, [N]);

  const focusCell = useCallback(
    (pos: Position) => {
      const next: Position = {
        row: Math.max(0, Math.min(N - 1, pos.row)),
        col: Math.max(0, Math.min(N - 1, pos.col)),
      };
      setFocused(next);
      const el = cellRefs.current[indexOf(next.row, next.col)];
      el?.focus();
    },
    [N, indexOf],
  );

  // Clamp focused position when N changes (e.g. ruleset switches).
  useEffect(() => {
    setFocused((prev) => ({
      row: Math.min(prev.row, N - 1),
      col: Math.min(prev.col, N - 1),
    }));
  }, [N]);

  const handleKeyDown = useCallback(
    (row: number, col: number) => (e: KeyboardEvent<HTMLDivElement>) => {
      switch (e.key) {
        case 'ArrowRight':
          e.preventDefault();
          focusCell({ row, col: col + 1 });
          return;
        case 'ArrowLeft':
          e.preventDefault();
          focusCell({ row, col: col - 1 });
          return;
        case 'ArrowDown':
          e.preventDefault();
          focusCell({ row: row + 1, col });
          return;
        case 'ArrowUp':
          e.preventDefault();
          focusCell({ row: row - 1, col });
          return;
        case 'Home':
          e.preventDefault();
          focusCell({ row, col: 0 });
          return;
        case 'End':
          e.preventDefault();
          focusCell({ row, col: N - 1 });
          return;
        case 'Enter':
        case ' ':
          e.preventDefault();
          if (!disabled) onCellClick?.({ row, col });
          return;
        default:
          return;
      }
    },
    [focusCell, N, disabled, onCellClick],
  );

  const rows = Array.from({ length: N }, (_, r) => r);
  const cols = Array.from({ length: N }, (_, c) => c);

  return (
    <div
      role="grid"
      aria-label={ariaLabel}
      aria-rowcount={N}
      aria-colcount={N}
      data-board-size={N}
      className="grid gap-1 rounded-lg bg-bg p-2"
      style={{
        gridTemplateColumns: `repeat(${N}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${N}, minmax(0, 1fr))`,
      }}
    >
      {rows.map((row) =>
        cols.map((col) => {
          const cell = state.board[row]?.[col] ?? [];
          const isFocused = focused.row === row && focused.col === col;
          const key = `${row}-${col}`;
          const highlighted = highlightKey?.has(`${row},${col}`) ?? false;
          return (
            <Cell
              key={key}
              ref={(el) => {
                cellRefs.current[indexOf(row, col)] = el;
              }}
              pieces={cell}
              coord={{ row, col }}
              pieceSizes={pieceSizes}
              {...(onCellClick ? { onClick: onCellClick } : {})}
              highlighted={highlighted}
              disabled={disabled}
              tabIndex={isFocused ? 0 : -1}
              onKeyDown={handleKeyDown(row, col)}
            />
          );
        }),
      )}
    </div>
  );
};
