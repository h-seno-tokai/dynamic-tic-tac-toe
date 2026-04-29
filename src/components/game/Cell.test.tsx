import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import type { Piece, PieceSize } from '@/domain';
import { Cell } from './Cell';

const pieceSizes: PieceSize[] = [
  { id: 'S', rank: 0, displayName: { ja: '小', en: 'Small' } },
  { id: 'M', rank: 1, displayName: { ja: '中', en: 'Medium' } },
  { id: 'L', rank: 2, displayName: { ja: '大', en: 'Large' } },
];

describe('Cell', () => {
  it('renders an empty cell with no piece image', () => {
    render(<Cell pieces={[]} coord={{ row: 0, col: 0 }} pieceSizes={pieceSizes} />);
    const cell = screen.getByRole('gridcell', { name: 'マス 1-1' });
    expect(cell).toBeInTheDocument();
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(screen.queryByTestId('cell-stack-indicator')).not.toBeInTheDocument();
  });

  it('renders a single piece', () => {
    const pieces: Piece[] = [{ owner: 'P1', sizeId: 'M' }];
    render(<Cell pieces={pieces} coord={{ row: 1, col: 2 }} pieceSizes={pieceSizes} />);
    const img = screen.getByRole('img');
    expect(img).toHaveAttribute('aria-label', 'P1 の 中');
    expect(screen.queryByTestId('cell-stack-indicator')).not.toBeInTheDocument();
  });

  it('renders only the top piece when stacked, with stack indicator', () => {
    const pieces: Piece[] = [
      { owner: 'P2', sizeId: 'S' },
      { owner: 'P1', sizeId: 'L' },
    ];
    render(<Cell pieces={pieces} coord={{ row: 0, col: 0 }} pieceSizes={pieceSizes} />);
    const imgs = screen.getAllByRole('img');
    expect(imgs).toHaveLength(1);
    expect(imgs[0]).toHaveAttribute('aria-label', 'P1 の 大');
    expect(screen.getByTestId('cell-stack-indicator')).toBeInTheDocument();
  });

  it('fires onClick with coord', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(
      <Cell pieces={[]} coord={{ row: 2, col: 1 }} onClick={onClick} pieceSizes={pieceSizes} />,
    );
    await user.click(screen.getByRole('gridcell'));
    expect(onClick).toHaveBeenCalledWith({ row: 2, col: 1 });
  });

  it('does not fire onClick when disabled', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(
      <Cell
        pieces={[]}
        coord={{ row: 0, col: 0 }}
        onClick={onClick}
        pieceSizes={pieceSizes}
        disabled
      />,
    );
    await user.click(screen.getByRole('gridcell'));
    expect(onClick).not.toHaveBeenCalled();
  });
});
