import { describe, expect, it } from 'vitest';
import { AI_LIMITS, PRESET_3X3, PRESET_4X4_XL, RULE_PRESETS, isRuleSupportedByAI } from './presets';
import { engine } from '../engine/engine';

describe('rule presets', () => {
  it('PRESET_3X3 passes validateRules', () => {
    const r = engine.validateRules(PRESET_3X3);
    expect(r.ok).toBe(true);
    expect(r.errors).toEqual([]);
  });

  it('PRESET_4X4_XL passes validateRules', () => {
    const r = engine.validateRules(PRESET_4X4_XL);
    expect(r.ok).toBe(true);
    expect(r.errors).toEqual([]);
  });

  it('exposes both presets in RULE_PRESETS in the documented order', () => {
    expect(RULE_PRESETS).toHaveLength(2);
    expect(RULE_PRESETS[0].id).toBe('3x3-classic');
    expect(RULE_PRESETS[1].id).toBe('4x4-huge');
  });

  it('isRuleSupportedByAI accepts both presets', () => {
    expect(isRuleSupportedByAI(PRESET_3X3)).toBe(true);
    expect(isRuleSupportedByAI(PRESET_4X4_XL)).toBe(true);
  });

  it('isRuleSupportedByAI rejects oversized boards', () => {
    expect(
      isRuleSupportedByAI({
        ...PRESET_4X4_XL,
        boardSize: AI_LIMITS.MAX_BOARD + 1,
      }),
    ).toBe(false);
  });

  it('isRuleSupportedByAI rejects too many piece sizes', () => {
    expect(
      isRuleSupportedByAI({
        ...PRESET_3X3,
        pieceSizes: [
          { id: 'A', rank: 0, displayName: { ja: 'a', en: 'a' } },
          { id: 'B', rank: 1, displayName: { ja: 'b', en: 'b' } },
          { id: 'C', rank: 2, displayName: { ja: 'c', en: 'c' } },
          { id: 'D', rank: 3, displayName: { ja: 'd', en: 'd' } },
          { id: 'E', rank: 4, displayName: { ja: 'e', en: 'e' } },
        ],
        piecesPerSize: [1, 1, 1, 1, 1],
      }),
    ).toBe(false);
  });

  it('isRuleSupportedByAI rejects too many pieces per size', () => {
    expect(
      isRuleSupportedByAI({
        ...PRESET_3X3,
        piecesPerSize: [AI_LIMITS.MAX_PIECES_PER_SIZE + 1, 2, 2],
      }),
    ).toBe(false);
  });
});
