import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import type { PieceSize } from '@/domain';
import { Piece } from './Piece';

const small: PieceSize = { id: 'S', rank: 0, displayName: { ja: '小', en: 'Small' } };
const large: PieceSize = { id: 'L', rank: 2, displayName: { ja: '大', en: 'Large' } };

describe('Piece', () => {
  it('renders with P1 color class', () => {
    render(<Piece size={small} owner="P1" />);
    const img = screen.getByRole('img', { name: 'P1 の 小' });
    expect(img.className).toContain('text-p1');
  });

  it('renders with P2 color class', () => {
    render(<Piece size={small} owner="P2" />);
    const img = screen.getByRole('img', { name: 'P2 の 小' });
    expect(img.className).toContain('text-p2');
  });

  it('visual radius scales with rank (larger rank → larger circle)', () => {
    const { container: cSmall, unmount } = render(<Piece size={small} owner="P1" />);
    const rSmall = parseFloat(cSmall.querySelector('circle')?.getAttribute('r') ?? '0');
    unmount();
    const { container: cLarge } = render(<Piece size={large} owner="P1" />);
    const rLarge = parseFloat(cLarge.querySelector('circle')?.getAttribute('r') ?? '0');
    expect(rLarge).toBeGreaterThan(rSmall);
  });

  it('applies selected ring class when selected', () => {
    render(<Piece size={small} owner="P1" selected />);
    expect(screen.getByRole('img').className).toContain('ring-2');
  });

  it('applies reduced opacity when disabled', () => {
    render(<Piece size={small} owner="P1" disabled />);
    expect(screen.getByRole('img').className).toContain('opacity-50');
  });

  it('uses Japanese display name for default aria-label', () => {
    render(<Piece size={large} owner="P2" />);
    expect(screen.getByLabelText('P2 の 大')).toBeInTheDocument();
  });
});
