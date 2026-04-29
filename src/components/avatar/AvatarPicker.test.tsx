import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { AvatarPicker } from './AvatarPicker';

describe('AvatarPicker', () => {
  it('renders a button grid with pressed state for the selected seed', () => {
    render(<AvatarPicker value="aoi" onChange={vi.fn()} seeds={['haru', 'aoi']} />);

    expect(screen.getByRole('group', { name: 'Choose avatar' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Avatar haru' })).toHaveAttribute(
      'aria-pressed',
      'false',
    );
    expect(screen.getByRole('button', { name: 'Avatar aoi' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
  });

  it('calls onChange with the clicked seed', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<AvatarPicker value="haru" onChange={onChange} seeds={['haru', 'aoi']} />);

    await user.click(screen.getByRole('button', { name: 'Avatar aoi' }));

    expect(onChange).toHaveBeenCalledWith('aoi');
  });

  it('supports custom labels', () => {
    render(
      <AvatarPicker
        value="ren"
        onChange={vi.fn()}
        seeds={['ren']}
        label="Player one avatar"
        getSeedLabel={(seed, index) => `${index + 1}: ${seed}`}
      />,
    );

    expect(screen.getByRole('group', { name: 'Player one avatar' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '1: ren' })).toHaveAttribute('aria-pressed', 'true');
  });
});
