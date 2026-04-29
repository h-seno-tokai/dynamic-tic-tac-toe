import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import type { Move, PieceSize } from '@/domain';
import { MoveHistory } from './MoveHistory';
import { formatMove } from './formatMove';

const pieceSizes: PieceSize[] = [
  { id: 'S', rank: 0, displayName: { ja: '小', en: 'Small' } },
  { id: 'M', rank: 1, displayName: { ja: '中', en: 'Medium' } },
  { id: 'L', rank: 2, displayName: { ja: '大', en: 'Large' } },
];

describe('formatMove', () => {
  it('formats placeFromReserve in Japanese', () => {
    const move: Move = {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'L',
      to: { row: 0, col: 0 },
    };
    expect(formatMove(move, pieceSizes)).toBe('P1: 大 を A-1 に置く');
  });

  it('formats moveOnBoard in Japanese', () => {
    const move: Move = {
      kind: 'moveOnBoard',
      player: 'P2',
      from: { row: 0, col: 0 },
      to: { row: 1, col: 1 },
    };
    expect(formatMove(move, pieceSizes)).toBe('P2: A-1 から B-2 へ移動');
  });

  it('falls back to sizeId when no pieceSizes table is given', () => {
    const move: Move = {
      kind: 'placeFromReserve',
      player: 'P1',
      sizeId: 'M',
      to: { row: 2, col: 1 },
    };
    expect(formatMove(move)).toBe('P1: M を B-3 に置く');
  });
});

describe('MoveHistory', () => {
  it('renders empty state when history is empty', () => {
    render(<MoveHistory history={[]} pieceSizes={pieceSizes} />);
    expect(screen.getByText('（手なし）')).toBeInTheDocument();
  });

  it('renders correct text for place + move-on-board moves', () => {
    const history: Move[] = [
      {
        kind: 'placeFromReserve',
        player: 'P1',
        sizeId: 'L',
        to: { row: 0, col: 0 },
      },
      {
        kind: 'moveOnBoard',
        player: 'P2',
        from: { row: 0, col: 0 },
        to: { row: 1, col: 1 },
      },
    ];
    render(<MoveHistory history={history} pieceSizes={pieceSizes} />);
    expect(screen.getByText('1. P1: 大 を A-1 に置く')).toBeInTheDocument();
    expect(screen.getByText('2. P2: A-1 から B-2 へ移動')).toBeInTheDocument();
  });

  it('fires onJumpTo when an entry is clicked', async () => {
    const user = userEvent.setup();
    const onJumpTo = vi.fn();
    const history: Move[] = [
      {
        kind: 'placeFromReserve',
        player: 'P1',
        sizeId: 'S',
        to: { row: 0, col: 0 },
      },
    ];
    render(<MoveHistory history={history} pieceSizes={pieceSizes} onJumpTo={onJumpTo} />);
    await user.click(screen.getByText('1. P1: 小 を A-1 に置く'));
    expect(onJumpTo).toHaveBeenCalledWith(0);
  });
});
