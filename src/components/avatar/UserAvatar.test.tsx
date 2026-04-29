import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { buildDiceBearAvataaarsUrl, UserAvatar } from './UserAvatar';

describe('UserAvatar', () => {
  it('builds a DiceBear Avataaars URL from a seed', () => {
    expect(buildDiceBearAvataaarsUrl('player one', { size: 64 })).toBe(
      'https://api.dicebear.com/9.x/avataaars/svg?seed=player+one&size=64',
    );
  });

  it('renders an accessible image when label is provided', () => {
    render(<UserAvatar seed="mio" size={72} label="Player Mio" selected />);

    const image = screen.getByRole('img', { name: 'Player Mio' });
    expect(image).toHaveAttribute('src', buildDiceBearAvataaarsUrl('mio', { size: 72 }));
    expect(image).toHaveAttribute('width', '72');
    expect(image).toHaveAttribute('height', '72');
    expect(image).toHaveAttribute('data-selected', 'true');
  });

  it('renders as decorative when no label is provided', () => {
    const { container } = render(<UserAvatar seed="ren" />);
    const image = container.querySelector('img');

    expect(image).toHaveAttribute('alt', '');
    expect(image).toHaveAttribute('aria-hidden', 'true');
  });
});
