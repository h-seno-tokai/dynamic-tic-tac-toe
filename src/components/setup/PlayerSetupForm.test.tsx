import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { PlayerSetupForm } from './PlayerSetupForm';

describe('PlayerSetupForm', () => {
  it('renders a labelled name input and forwards name changes', async () => {
    const user = userEvent.setup();
    const onNameChange = vi.fn();
    render(
      <PlayerSetupForm
        playerLabel="Player 1"
        name=""
        onNameChange={onNameChange}
        avatarSeed="sun"
        avatarSeeds={['sun', 'moon']}
        onAvatarSeedChange={vi.fn()}
        language="en"
      />,
    );

    expect(screen.getByRole('group', { name: 'Player 1' })).toBeInTheDocument();
    await user.type(screen.getByRole('textbox', { name: 'Name' }), 'Ada');
    expect(onNameChange).toHaveBeenLastCalledWith('a');
  });

  it('marks the current avatar and forwards avatar selections', async () => {
    const user = userEvent.setup();
    const onAvatarSeedChange = vi.fn();
    render(
      <PlayerSetupForm
        playerLabel="プレイヤー1"
        name="A"
        onNameChange={vi.fn()}
        avatarSeed="sun"
        avatarSeeds={['sun', 'moon']}
        onAvatarSeedChange={onAvatarSeedChange}
        language="ja"
      />,
    );

    expect(screen.getByRole('button', { name: 'アバター sun を選択' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    await user.click(screen.getByRole('button', { name: 'アバター moon を選択' }));
    expect(onAvatarSeedChange).toHaveBeenCalledWith('moon');
  });
});
