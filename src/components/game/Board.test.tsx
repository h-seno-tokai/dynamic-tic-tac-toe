import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { engine, PRESET_3X3, PRESET_4X4_XL } from '@/domain';
import { Board } from './Board';

describe('Board', () => {
  it('renders 9 cells for PRESET_3X3 initial state', () => {
    const state = engine.initialState(PRESET_3X3);
    render(<Board state={state} />);
    expect(screen.getAllByRole('gridcell')).toHaveLength(9);
    expect(screen.getByRole('grid')).toHaveAttribute('data-board-size', '3');
  });

  it('renders 16 cells for PRESET_4X4_XL initial state', () => {
    const state = engine.initialState(PRESET_4X4_XL);
    render(<Board state={state} />);
    expect(screen.getAllByRole('gridcell')).toHaveLength(16);
    expect(screen.getByRole('grid')).toHaveAttribute('data-board-size', '4');
  });

  it('fires onCellClick with correct coords on click', async () => {
    const user = userEvent.setup();
    const onCellClick = vi.fn();
    const state = engine.initialState(PRESET_3X3);
    render(<Board state={state} onCellClick={onCellClick} />);
    const cell = screen.getByLabelText('マス 2-3');
    await user.click(cell);
    expect(onCellClick).toHaveBeenCalledTimes(1);
    expect(onCellClick).toHaveBeenCalledWith({ row: 1, col: 2 });
  });

  it('fires onCellClick on Enter key for focused cell', async () => {
    const user = userEvent.setup();
    const onCellClick = vi.fn();
    const state = engine.initialState(PRESET_3X3);
    render(<Board state={state} onCellClick={onCellClick} />);
    const firstCell = screen.getByLabelText('マス 1-1');
    firstCell.focus();
    await user.keyboard('{Enter}');
    expect(onCellClick).toHaveBeenCalledWith({ row: 0, col: 0 });
  });

  it('does not fire onCellClick when disabled', async () => {
    const user = userEvent.setup();
    const onCellClick = vi.fn();
    const state = engine.initialState(PRESET_3X3);
    render(<Board state={state} onCellClick={onCellClick} disabled />);
    await user.click(screen.getByLabelText('マス 1-1'));
    expect(onCellClick).not.toHaveBeenCalled();
  });

  it('arrow keys move focus between cells', async () => {
    const user = userEvent.setup();
    const state = engine.initialState(PRESET_3X3);
    render(<Board state={state} />);
    const first = screen.getByLabelText('マス 1-1');
    first.focus();
    await user.keyboard('{ArrowRight}');
    expect(screen.getByLabelText('マス 1-2')).toHaveFocus();
    await user.keyboard('{ArrowDown}');
    expect(screen.getByLabelText('マス 2-2')).toHaveFocus();
  });
});
