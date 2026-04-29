import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { RULE_PRESETS } from '../../domain/rules/presets';
import { CpuSetupForm } from './CpuSetupForm';

describe('CpuSetupForm', () => {
  it('renders CPU setup controls with accessible labels', () => {
    render(
      <CpuSetupForm
        difficulty={4}
        onDifficultyChange={vi.fn()}
        humanSide="P1"
        onHumanSideChange={vi.fn()}
        rulePresetId={RULE_PRESETS[0].id}
        onRulePresetChange={vi.fn()}
        language="en"
      />,
    );

    expect(screen.getByRole('slider', { name: 'CPU difficulty' })).toHaveAttribute(
      'aria-valuetext',
      'Level 4',
    );
    expect(screen.getByRole('radiogroup', { name: 'Your side' })).toBeInTheDocument();
    expect(screen.getByRole('radiogroup', { name: 'Rules' })).toBeInTheDocument();
  });

  it('forwards difficulty, human side, and rule preset changes', async () => {
    const user = userEvent.setup();
    const onDifficultyChange = vi.fn();
    const onHumanSideChange = vi.fn();
    const onRulePresetChange = vi.fn();
    render(
      <CpuSetupForm
        difficulty={4}
        onDifficultyChange={onDifficultyChange}
        humanSide="P1"
        onHumanSideChange={onHumanSideChange}
        rulePresetId={RULE_PRESETS[0].id}
        onRulePresetChange={onRulePresetChange}
        language="en"
      />,
    );

    fireEvent.change(screen.getByRole('slider', { name: 'CPU difficulty' }), {
      target: { value: '8' },
    });
    await user.click(screen.getByRole('radio', { name: 'Second player' }));
    await user.click(screen.getByRole('radio', { name: /4x4 Huge/i }));

    expect(onDifficultyChange).toHaveBeenCalledWith(8);
    expect(onHumanSideChange).toHaveBeenCalledWith('P2');
    expect(onRulePresetChange).toHaveBeenCalledWith(RULE_PRESETS[1].id);
  });
});
