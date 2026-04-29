import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import type { PieceSize, Reserve } from '@/domain';
import { ReserveStack } from './ReserveStack';

const pieceSizes: PieceSize[] = [
  { id: 'S', rank: 0, displayName: { ja: '小', en: 'Small' } },
  { id: 'M', rank: 1, displayName: { ja: '中', en: 'Medium' } },
  { id: 'L', rank: 2, displayName: { ja: '大', en: 'Large' } },
];

const reserve: Reserve = { S: 2, M: 0, L: 2 };

describe('ReserveStack', () => {
  it('renders one pile per piece size with counts', () => {
    render(<ReserveStack reserve={reserve} pieceSizes={pieceSizes} owner="P1" />);
    const buttons = screen.getAllByRole('button');
    expect(buttons).toHaveLength(3);
    expect(buttons[0]).toHaveAttribute('data-size-id', 'L'); // largest first
    expect(buttons[2]).toHaveAttribute('data-size-id', 'S');
  });

  it('clicking a pile with count > 0 fires onSelect', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <ReserveStack reserve={reserve} pieceSizes={pieceSizes} owner="P1" onSelect={onSelect} />,
    );
    await user.click(screen.getByLabelText('P1 の 大 手駒 (残り 2)'));
    expect(onSelect).toHaveBeenCalledWith('L');
  });

  it('does not fire onSelect for empty pile', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <ReserveStack reserve={reserve} pieceSizes={pieceSizes} owner="P1" onSelect={onSelect} />,
    );
    const empty = screen.getByLabelText('P1 の 中 手駒 (残り 0)');
    expect(empty).toBeDisabled();
    await user.click(empty);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('disabled prop prevents all clicks', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <ReserveStack
        reserve={reserve}
        pieceSizes={pieceSizes}
        owner="P1"
        onSelect={onSelect}
        disabled
      />,
    );
    await user.click(screen.getByLabelText('P1 の 大 手駒 (残り 2)'));
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('marks selected pile with aria-pressed', () => {
    render(
      <ReserveStack
        reserve={reserve}
        pieceSizes={pieceSizes}
        owner="P2"
        selected={{ sizeId: 'S' }}
      />,
    );
    const selected = screen.getByLabelText('P2 の 小 手駒 (残り 2)');
    expect(selected).toHaveAttribute('aria-pressed', 'true');
  });
});
