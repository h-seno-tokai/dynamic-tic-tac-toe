import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { CpuAvatar, getCpuAvatarTier, normalizeCpuDifficulty } from './CpuAvatar';

describe('CpuAvatar', () => {
  it('maps difficulty to tiers', () => {
    expect(getCpuAvatarTier(1)).toBe('bronze');
    expect(getCpuAvatarTier(4)).toBe('silver');
    expect(getCpuAvatarTier(7)).toBe('gold');
    expect(getCpuAvatarTier(10)).toBe('master');
  });

  it('normalizes difficulty into the supported range', () => {
    expect(normalizeCpuDifficulty(0)).toBe(1);
    expect(normalizeCpuDifficulty(11)).toBe(10);
    expect(normalizeCpuDifficulty(6.8)).toBe(6);
    expect(normalizeCpuDifficulty(Number.NaN)).toBe(1);
  });

  it('renders tier and difficulty data attributes with a badge', () => {
    render(<CpuAvatar difficulty={8} label="Strong CPU" selected />);

    const avatar = screen.getByRole('img', { name: 'Strong CPU' });
    expect(avatar).toHaveAttribute('data-difficulty', '8');
    expect(avatar).toHaveAttribute('data-tier', 'gold');
    expect(avatar).toHaveAttribute('data-selected', 'true');
    expect(avatar).toHaveTextContent('LV.8');
    expect(avatar.className).toContain('cpu-avatar-tier-gold');
  });
});
