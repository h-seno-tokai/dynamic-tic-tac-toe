import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Tooltip } from './Tooltip';

describe('Tooltip', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('renders children and tooltip element with role=tooltip', () => {
    render(
      <Tooltip content="Hi there">
        <button>Help</button>
      </Tooltip>,
    );
    expect(screen.getByRole('button', { name: 'Help' })).toBeInTheDocument();
    const tip = screen.getByRole('tooltip', { hidden: true });
    expect(tip).toHaveAttribute('data-state', 'closed');
  });

  it('shows on hover after delay and links via aria-describedby', () => {
    render(
      <Tooltip content="Tip" delay={150}>
        <button>Btn</button>
      </Tooltip>,
    );
    const btn = screen.getByRole('button');
    fireEvent.mouseEnter(btn);
    // Before delay elapsed, still closed.
    expect(screen.getByRole('tooltip', { hidden: true })).toHaveAttribute('data-state', 'closed');
    act(() => {
      vi.advanceTimersByTime(150);
    });
    const tip = screen.getByRole('tooltip');
    expect(tip).toHaveAttribute('data-state', 'open');
    expect(btn.getAttribute('aria-describedby')).toContain(tip.id);
  });

  it('hides on unhover', () => {
    render(
      <Tooltip content="Tip" delay={0}>
        <button>Btn</button>
      </Tooltip>,
    );
    const btn = screen.getByRole('button');
    fireEvent.mouseEnter(btn);
    expect(screen.getByRole('tooltip')).toHaveAttribute('data-state', 'open');
    fireEvent.mouseLeave(btn);
    expect(screen.getByRole('tooltip', { hidden: true })).toHaveAttribute('data-state', 'closed');
  });

  it('shows on focus and hides on blur', () => {
    render(
      <Tooltip content="Tip" delay={0}>
        <button>Btn</button>
      </Tooltip>,
    );
    const btn = screen.getByRole('button');
    act(() => btn.focus());
    expect(screen.getByRole('tooltip')).toHaveAttribute('data-state', 'open');
    act(() => btn.blur());
    expect(screen.getByRole('tooltip', { hidden: true })).toHaveAttribute('data-state', 'closed');
  });
});
