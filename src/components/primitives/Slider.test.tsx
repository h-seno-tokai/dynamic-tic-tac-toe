import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Slider } from './Slider';

describe('Slider', () => {
  it('renders with label and value', () => {
    render(<Slider value={50} onChange={vi.fn()} min={0} max={100} label="Volume" />);
    const input = screen.getByRole('slider', { name: 'Volume' });
    expect(input).toHaveAttribute('aria-valuemin', '0');
    expect(input).toHaveAttribute('aria-valuemax', '100');
    expect(input).toHaveAttribute('aria-valuenow', '50');
    expect(screen.getByText('50')).toBeInTheDocument();
  });

  it('formats value via formatValue', () => {
    render(
      <Slider
        value={60}
        onChange={vi.fn()}
        min={0}
        max={100}
        label="Vol"
        formatValue={(v) => `${v}%`}
      />,
    );
    expect(screen.getByText('60%')).toBeInTheDocument();
    expect(screen.getByRole('slider')).toHaveAttribute('aria-valuetext', '60%');
  });

  it('calls onChange when value changes', () => {
    const onChange = vi.fn();
    render(<Slider value={50} onChange={onChange} min={0} max={100} step={1} label="Vol" />);
    const input = screen.getByRole('slider');
    fireEvent.change(input, { target: { value: '75' } });
    expect(onChange).toHaveBeenCalledWith(75);
  });

  it('respects disabled', () => {
    render(<Slider value={10} onChange={vi.fn()} min={0} max={100} label="Vol" disabled />);
    expect(screen.getByRole('slider')).toBeDisabled();
  });
});
