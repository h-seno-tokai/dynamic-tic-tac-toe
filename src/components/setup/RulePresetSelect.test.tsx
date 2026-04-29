import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { RULE_PRESETS } from '../../domain/rules/presets';
import { RulePresetSelect } from './RulePresetSelect';

describe('RulePresetSelect', () => {
  it('renders rule presets using the selected language', () => {
    render(<RulePresetSelect value={RULE_PRESETS[0].id} onChange={vi.fn()} language="en" />);

    expect(screen.getByRole('radiogroup', { name: 'Rules' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /3x3 Classic/i })).toHaveAttribute(
      'aria-checked',
      'true',
    );
    expect(screen.getByRole('radio', { name: /4x4 Huge/i })).toBeInTheDocument();
    expect(screen.getByText(/Line of 3/)).toBeInTheDocument();
  });

  it('calls onChange with the selected preset id', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<RulePresetSelect value={RULE_PRESETS[0].id} onChange={onChange} language="ja" />);

    await user.click(screen.getByRole('radio', { name: /4x4/ }));
    expect(onChange).toHaveBeenCalledWith(RULE_PRESETS[1].id);
  });
});
