import type { CSSProperties, FC } from 'react';
import type { PieceSize, Player } from '@/domain';

export type PieceDisplaySize = 'sm' | 'md' | 'lg';

export interface PieceProps {
  size: PieceSize;
  owner: Player;
  displaySize?: PieceDisplaySize;
  selected?: boolean;
  disabled?: boolean;
  className?: string;
  /**
   * Override aria-label. Default uses Japanese display name:
   *   "P1 の {size.displayName.ja}".
   */
  'aria-label'?: string;
}

/**
 * Outer pixel size (width/height of the SVG box) per displaySize.
 * The visual circle radius is scaled by `size.rank` within this box.
 */
const OUTER_PX: Record<PieceDisplaySize, number> = {
  sm: 24,
  md: 40,
  lg: 64,
};

/** Min/max radius ratio relative to outer box (0..0.5). */
const MIN_RADIUS_RATIO = 0.18;
const MAX_RADIUS_RATIO = 0.42;

/** Compute the radius (in svg units, viewBox 100x100) for a given rank. */
function radiusForRank(rank: number, totalRanks: number): number {
  if (totalRanks <= 1) {
    return MAX_RADIUS_RATIO * 100;
  }
  const t = Math.min(Math.max(rank, 0), totalRanks - 1) / (totalRanks - 1);
  const ratio = MIN_RADIUS_RATIO + (MAX_RADIUS_RATIO - MIN_RADIUS_RATIO) * t;
  return ratio * 100;
}

const ownerColorClass: Record<Player, string> = {
  P1: 'text-p1',
  P2: 'text-p2',
};

/**
 * Pure presentational piece: an SVG disc whose radius reflects the piece rank.
 *
 * - Color is driven by Tailwind text-color classes (`text-p1` / `text-p2`)
 *   so the inner SVG can use `currentColor` as its fill.
 * - The component does not assume any specific number of piece-size ranks;
 *   the radius is interpolated from `size.rank` against an internally-known
 *   max (we use `size.rank + 1` if no max is provided — rendering the piece
 *   alone — but consumers usually pass a reasonable `displaySize` and rely on
 *   uniform appearance per rank by passing the same ranks across the board).
 */
export const Piece: FC<PieceProps> = ({
  size,
  owner,
  displaySize = 'md',
  selected = false,
  disabled = false,
  className,
  'aria-label': ariaLabelProp,
}) => {
  const px = OUTER_PX[displaySize];
  // We don't know the total number of ranks here; assume a reasonable upper
  // bound of (rank + 1) clamped to at least 4 so size differences are clearly
  // perceptible regardless of preset. This keeps the component prop-driven:
  // rank 0 is always the smallest visible disc, higher ranks scale up.
  const totalRanks = Math.max(size.rank + 1, 4);
  const r = radiusForRank(size.rank, totalRanks);

  const ariaLabel = ariaLabelProp ?? `${owner} の ${size.displayName.ja}`;

  const wrapperClasses = [
    'inline-flex items-center justify-center rounded-full',
    ownerColorClass[owner],
    selected ? 'ring-2 ring-accent ring-offset-2 ring-offset-bg' : '',
    disabled ? 'opacity-50' : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ');

  const sizeStyle: CSSProperties = { width: px, height: px };

  // Inner concentric rings: rank tells how many rings to draw.
  // Rings divide the interior evenly, making size hierarchy instantly visible.
  const ringCount = size.rank;
  const ringStrokeWidth = Math.max(2.5, r * 0.11);
  const rings = Array.from({ length: ringCount }, (_, i) => (r * (i + 1)) / (ringCount + 1));

  return (
    <span
      role="img"
      aria-label={ariaLabel}
      data-piece-size-id={size.id}
      data-piece-rank={size.rank}
      data-piece-owner={owner}
      className={wrapperClasses}
      style={sizeStyle}
    >
      <svg viewBox="0 0 100 100" width={px} height={px} aria-hidden="true" focusable="false">
        <circle
          cx={50}
          cy={50}
          r={r}
          fill="currentColor"
          stroke="currentColor"
          strokeWidth={2}
          fillOpacity={0.85}
        />
        {rings.map((innerR, i) => (
          <circle
            key={i}
            cx={50}
            cy={50}
            r={innerR}
            fill="none"
            stroke="white"
            strokeWidth={ringStrokeWidth}
            strokeOpacity={0.5}
          />
        ))}
      </svg>
    </span>
  );
};
