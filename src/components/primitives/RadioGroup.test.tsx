import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { useState } from 'react';
import { RadioGroup } from './RadioGroup';

type Choice = 'a' | 'b' | 'c';

const opts = [
  { value: 'a' as const, label: 'A' },
  { value: 'b' as const, label: 'B' },
  { value: 'c' as const, label: 'C' },
];

const Harness = ({
  initial = 'a',
  onChange,
}: {
  initial?: Choice;
  onChange?: (v: Choice) => void;
}) => {
  const [v, setV] = useState<Choice>(initial);
  return (
    <RadioGroup
      label="Pick one"
      value={v}
      onChange={(next) => {
        setV(next);
        onChange?.(next);
      }}
      options={opts}
    />
  );
};

describe('RadioGroup', () => {
  it('renders radiogroup with options', () => {
    render(<Harness />);
    expect(screen.getByRole('radiogroup', { name: 'Pick one' })).toBeInTheDocument();
    expect(screen.getAllByRole('radio')).toHaveLength(3);
  });

  it('marks the selected option aria-checked=true', () => {
    render(<Harness initial="b" />);
    const radios = screen.getAllByRole('radio');
    expect(radios[0]).toHaveAttribute('aria-checked', 'false');
    expect(radios[1]).toHaveAttribute('aria-checked', 'true');
    expect(radios[2]).toHaveAttribute('aria-checked', 'false');
  });

  it('uses a single tab stop on the selected option', () => {
    render(<Harness initial="b" />);
    const radios = screen.getAllByRole('radio');
    expect(radios[0]).toHaveAttribute('tabindex', '-1');
    expect(radios[1]).toHaveAttribute('tabindex', '0');
    expect(radios[2]).toHaveAttribute('tabindex', '-1');
  });

  it('calls onChange when clicked', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await user.click(screen.getAllByRole('radio')[2]!);
    expect(onChange).toHaveBeenCalledWith('c');
  });

  it('arrow keys move selection and wrap around', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness initial="a" onChange={onChange} />);
    const radios = screen.getAllByRole('radio');
    radios[0]!.focus();
    await user.keyboard('{ArrowDown}');
    expect(onChange).toHaveBeenLastCalledWith('b');
    await user.keyboard('{ArrowDown}');
    expect(onChange).toHaveBeenLastCalledWith('c');
    await user.keyboard('{ArrowDown}');
    expect(onChange).toHaveBeenLastCalledWith('a');
    await user.keyboard('{ArrowUp}');
    expect(onChange).toHaveBeenLastCalledWith('c');
  });

  it('Home/End jump to first/last enabled', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness initial="b" onChange={onChange} />);
    screen.getAllByRole('radio')[1]!.focus();
    await user.keyboard('{End}');
    expect(onChange).toHaveBeenLastCalledWith('c');
    await user.keyboard('{Home}');
    expect(onChange).toHaveBeenLastCalledWith('a');
  });

  it('Space selects the focused option', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Harness initial="a" onChange={onChange} />);
    const radios = screen.getAllByRole('radio');
    radios[2]!.focus();
    await user.keyboard(' ');
    expect(onChange).toHaveBeenCalledWith('c');
  });

  it('respects disabled options', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <RadioGroup
        label="x"
        value="a"
        onChange={onChange}
        options={[
          { value: 'a', label: 'A' },
          { value: 'b', label: 'B', disabled: true },
        ]}
      />,
    );
    const radios = screen.getAllByRole('radio');
    expect(radios[1]).toBeDisabled();
    await user.click(radios[1]!);
    expect(onChange).not.toHaveBeenCalled();
  });
});
