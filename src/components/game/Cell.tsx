import { forwardRef, type KeyboardEvent, type MouseEvent } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import type { Piece as PieceType, PieceSize, Position } from '@/domain';
import { Piece, type PieceDisplaySize } from './Piece';

export interface CellProps {
  pieces: PieceType[];
  coord: Position;
  /**
   * Lookup table from `sizeId` → full `PieceSize` (with rank + displayName).
   * Required so the rendered piece has correct visual size and aria-label.
   */
  pieceSizes?: PieceSize[];
  onClick?: (pos: Position) => void;
  highlighted?: boolean;
  disabled?: boolean;
  /** Tab index; used by Board to manage roving focus. Default -1. */
  tabIndex?: number;
  /** Forwarded onKeyDown (used by Board for arrow nav). */
  onKeyDown?: (e: KeyboardEvent<HTMLDivElement>) => void;
  /** Display size for the rendered top piece. */
  pieceDisplaySize?: PieceDisplaySize;
  /** aria-label override (default: "マス {row+1}-{col+1}"). */
  'aria-label'?: string;
}

function resolvePieceSize(sizeId: string, table: PieceSize[] | undefined): PieceSize {
  const found = table?.find((s) => s.id === sizeId);
  if (found) return found;
  // Fallback: rank 0, sizeId-as-name. Keeps Cell renderable in isolation.
  return { id: sizeId, rank: 0, displayName: { ja: sizeId, en: sizeId } };
}

/**
 * A single board cell. Pure presentational — receives a Piece[] stack and
 * renders only the top piece, with a stack indicator when length > 1.
 *
 * The cell is a `role="gridcell"`; the parent <Board> owns `role="grid"`.
 */
export const Cell = forwardRef<HTMLDivElement, CellProps>(function Cell(
  {
    pieces,
    coord,
    pieceSizes,
    onClick,
    highlighted = false,
    disabled = false,
    tabIndex = -1,
    onKeyDown,
    pieceDisplaySize = 'md',
    'aria-label': ariaLabelProp,
  },
  ref,
) {
  const reducedMotion = useReducedMotion();
  const top = pieces.length > 0 ? pieces[pieces.length - 1] : undefined;
  const stacked = pieces.length > 1;

  // (row+col) parity → alternate cell shading. Purely cosmetic, accessibility
  // does not depend on color.
  const isLight = (coord.row + coord.col) % 2 === 0;
  const baseBg = isLight ? 'bg-cellLight' : 'bg-cellDark';

  const ariaLabel = ariaLabelProp ?? `マス ${coord.row + 1}-${coord.col + 1}`;

  const handleClick = (e: MouseEvent<HTMLDivElement>) => {
    if (disabled) return;
    e.preventDefault();
    onClick?.(coord);
  };

  const classes = [
    'relative aspect-square w-full select-none',
    'flex items-center justify-center',
    'rounded-md border border-border',
    baseBg,
    'transition-colors',
    highlighted ? 'ring-2 ring-accent' : '',
    disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer hover:brightness-110',
  ]
    .filter(Boolean)
    .join(' ');

  const resolvedTopSize = top ? resolvePieceSize(top.sizeId, pieceSizes) : undefined;

  return (
    <div
      ref={ref}
      role="gridcell"
      aria-label={ariaLabel}
      aria-disabled={disabled || undefined}
      data-row={coord.row}
      data-col={coord.col}
      tabIndex={tabIndex}
      onClick={handleClick}
      onKeyDown={onKeyDown}
      className={classes}
    >
      {top !== undefined && resolvedTopSize !== undefined && (
        <motion.div
          // Subtle scale-in animation when the top piece changes.
          // Honors reduced motion via Framer's useReducedMotion.
          key={`${top.owner}:${top.sizeId}:${pieces.length}`}
          initial={reducedMotion ? false : { scale: 0.7, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 300, damping: 22 }}
          className="flex items-center justify-center"
        >
          <Piece
            size={resolvedTopSize}
            owner={top.owner}
            displaySize={pieceDisplaySize}
            disabled={disabled}
          />
        </motion.div>
      )}
      {stacked && (
        <span
          aria-hidden="true"
          data-testid="cell-stack-indicator"
          className="bg-fg/60 absolute right-1 top-1 inline-block h-2 w-2 rounded-full"
        />
      )}
    </div>
  );
});
