import type { FC } from 'react';
import type { Move, PieceSize } from '@/domain';
import { formatMove } from './formatMove';

export interface MoveHistoryProps {
  history: Move[];
  /**
   * Piece size table (for resolving sizeId → displayName).
   * If omitted, sizeId is shown verbatim.
   */
  pieceSizes?: PieceSize[];
  onJumpTo?: (index: number) => void;
  /** Optional aria-label (default: "棋譜"). */
  'aria-label'?: string;
}

export const MoveHistory: FC<MoveHistoryProps> = ({
  history,
  pieceSizes,
  onJumpTo,
  'aria-label': ariaLabel = '棋譜',
}) => {
  return (
    <div
      role="log"
      aria-label={ariaLabel}
      aria-live="polite"
      className="max-h-64 overflow-y-auto rounded-md border border-border bg-bg p-2"
    >
      {history.length === 0 ? (
        <p className="px-2 py-1 text-sm text-muted">（手なし）</p>
      ) : (
        <ol className="flex flex-col gap-0.5">
          {history.map((move, index) => {
            const text = formatMove(move, pieceSizes);
            const display = `${index + 1}. ${text}`;
            return (
              <li key={index}>
                {onJumpTo !== undefined ? (
                  <button
                    type="button"
                    onClick={() => onJumpTo(index)}
                    data-move-index={index}
                    className="w-full rounded px-2 py-1 text-left text-sm hover:bg-cellLight focus-visible:bg-cellLight"
                  >
                    {display}
                  </button>
                ) : (
                  <span data-move-index={index} className="block px-2 py-1 text-sm">
                    {display}
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
};
