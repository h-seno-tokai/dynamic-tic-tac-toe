import type { FC, KeyboardEvent } from 'react';
import type { PieceSize, Player, Reserve } from '@/domain';
import { Piece, type PieceDisplaySize } from './Piece';

export interface ReserveStackProps {
  reserve: Reserve;
  pieceSizes: PieceSize[];
  owner: Player;
  selected?: { sizeId: string } | null;
  onSelect?: (sizeId: string) => void;
  disabled?: boolean;
  /** Display size for individual rendered pieces in each pile. */
  pieceDisplaySize?: PieceDisplaySize;
  /** Optional aria-label for the wrapper. */
  'aria-label'?: string;
}

/**
 * Pure presentational reserve display: one pile per piece size, in
 * rank-DESCENDING order (largest visible first). Each pile shows a count
 * badge and is clickable when count > 0.
 */
export const ReserveStack: FC<ReserveStackProps> = ({
  reserve,
  pieceSizes,
  owner,
  selected,
  onSelect,
  disabled = false,
  pieceDisplaySize = 'md',
  'aria-label': ariaLabel,
}) => {
  // Sort by rank descending (largest first). Spread to avoid mutating prop.
  const ordered = [...pieceSizes].sort((a, b) => b.rank - a.rank);

  const handleSelect = (sizeId: string, count: number) => {
    if (disabled || count <= 0) return;
    onSelect?.(sizeId);
  };

  const handleKeyDown =
    (sizeId: string, count: number) => (e: KeyboardEvent<HTMLButtonElement>) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleSelect(sizeId, count);
      }
    };

  return (
    <div
      role="group"
      aria-label={ariaLabel ?? `${owner} の手駒`}
      data-owner={owner}
      className="flex flex-wrap items-end gap-3"
    >
      {ordered.map((size) => {
        const count = reserve[size.id] ?? 0;
        const isSelected = selected?.sizeId === size.id;
        const isEmpty = count <= 0;
        const isInteractive = !disabled && !isEmpty;

        const pileClasses = [
          'relative inline-flex flex-col items-center justify-end gap-1',
          'rounded-md border border-border bg-bg p-2',
          'transition-colors',
          isSelected ? 'ring-2 ring-accent' : '',
          isEmpty ? 'opacity-40' : '',
          isInteractive ? 'cursor-pointer hover:brightness-110' : 'cursor-not-allowed',
        ]
          .filter(Boolean)
          .join(' ');

        const label = `${owner} の ${size.displayName.ja} 手駒 (残り ${count})`;

        return (
          <button
            key={size.id}
            type="button"
            data-size-id={size.id}
            data-count={count}
            aria-label={label}
            aria-pressed={isSelected || undefined}
            disabled={!isInteractive}
            onClick={() => handleSelect(size.id, count)}
            onKeyDown={handleKeyDown(size.id, count)}
            className={pileClasses}
          >
            <Piece
              size={size}
              owner={owner}
              displaySize={pieceDisplaySize}
              disabled={isEmpty}
              selected={isSelected}
            />
            <span
              data-testid="reserve-count"
              className={[
                'min-w-[1.5rem] rounded-full px-1.5 py-0.5 text-center text-xs font-medium',
                isEmpty ? 'bg-border text-muted' : 'bg-accent text-bg',
              ].join(' ')}
            >
              {count}
            </span>
          </button>
        );
      })}
    </div>
  );
};
