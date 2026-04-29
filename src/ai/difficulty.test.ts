import { describe, expect, it } from 'vitest';

import { DIFFICULTY_PROFILES, getProfile } from './difficulty';

describe('difficulty profiles', () => {
  it('exposes all 10 levels (1..10)', () => {
    for (let level = 1; level <= 10; level++) {
      expect(DIFFICULTY_PROFILES[level]).toBeDefined();
      expect(DIFFICULTY_PROFILES[level]?.level).toBe(level);
    }
  });

  it('simCount is monotonically non-decreasing across levels', () => {
    for (let level = 2; level <= 10; level++) {
      const prev = DIFFICULTY_PROFILES[level - 1]!.simCount;
      const cur = DIFFICULTY_PROFILES[level]!.simCount;
      expect(cur).toBeGreaterThanOrEqual(prev);
    }
  });

  it('temperature is monotonically non-increasing across levels', () => {
    for (let level = 2; level <= 10; level++) {
      const prev = DIFFICULTY_PROFILES[level - 1]!.temperature;
      const cur = DIFFICULTY_PROFILES[level]!.temperature;
      expect(cur).toBeLessThanOrEqual(prev);
    }
  });

  it('level 10 is greedy (temperature = 0)', () => {
    expect(DIFFICULTY_PROFILES[10]?.temperature).toBe(0);
  });

  it('getProfile clamps out-of-range levels', () => {
    expect(getProfile(0).level).toBe(1);
    expect(getProfile(11).level).toBe(10);
    expect(getProfile(-5).level).toBe(1);
  });

  it('getProfile rounds non-integer levels', () => {
    expect(getProfile(3.4).level).toBe(3);
    expect(getProfile(3.6).level).toBe(4);
  });
});
