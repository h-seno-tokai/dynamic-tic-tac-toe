import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { Toggle } from './Toggle';

describe('Toggle', () => {
  it('renders as switch with aria-checked', () => {
    render(<Toggle checked={false} onCheckedChange={vi.fn()} aria-label="Sound" />);
    const sw = screen.getByRole('switch', { name: 'Sound' });
    expect(sw).toHaveAttribute('aria-checked', 'false');
  });

  it('reflects checked state', () => {
    render(<Toggle checked={true} onCheckedChange={vi.fn()} aria-label="Sound" />);
    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'true');
  });

  it('toggles on click', async () => {
    const user = userEvent.setup();
    const onCheckedChange = vi.fn();
    render(<Toggle checked={false} onCheckedChange={onCheckedChange} aria-label="x" />);
    await user.click(screen.getByRole('switch'));
    expect(onCheckedChange).toHaveBeenCalledWith(true);
  });

  it('toggles on Space key', async () => {
    const user = userEvent.setup();
    const onCheckedChange = vi.fn();
    render(<Toggle checked={true} onCheckedChange={onCheckedChange} aria-label="x" />);
    const sw = screen.getByRole('switch');
    sw.focus();
    await user.keyboard(' ');
    expect(onCheckedChange).toHaveBeenCalledWith(false);
  });

  it('does not change when disabled', async () => {
    const user = userEvent.setup();
    const onCheckedChange = vi.fn();
    render(<Toggle checked={false} onCheckedChange={onCheckedChange} aria-label="x" disabled />);
    await user.click(screen.getByRole('switch'));
    expect(onCheckedChange).not.toHaveBeenCalled();
  });

  it('renders visible label and links it via aria-labelledby', () => {
    render(<Toggle checked={false} onCheckedChange={vi.fn()} label="BGM" />);
    const sw = screen.getByRole('switch');
    expect(sw).toHaveAccessibleName('BGM');
  });
});
