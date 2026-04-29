import type { GameRules } from '../types';

/** Preset A: 3x3, S/M/L x 2 each (default). */
export const PRESET_3X3: GameRules = {
  boardSize: 3,
  pieceSizes: [
    { id: 'S', rank: 0, displayName: { ja: '小', en: 'Small' } },
    { id: 'M', rank: 1, displayName: { ja: '中', en: 'Medium' } },
    { id: 'L', rank: 2, displayName: { ja: '大', en: 'Large' } },
  ],
  piecesPerSize: [2, 2, 2],
  winCondition: { kind: 'lineOfN', n: 3 },
  allowSameSizeCover: false,
  allowSelfCover: true,
  maxPly: 60,
  drawByRepetition: 3,
};

/** Preset B: 4x4, S/M/L/XL x 3 each. */
export const PRESET_4X4_XL: GameRules = {
  boardSize: 4,
  pieceSizes: [
    { id: 'S', rank: 0, displayName: { ja: '小', en: 'Small' } },
    { id: 'M', rank: 1, displayName: { ja: '中', en: 'Medium' } },
    { id: 'L', rank: 2, displayName: { ja: '大', en: 'Large' } },
    { id: 'XL', rank: 3, displayName: { ja: '巨大', en: 'Huge' } },
  ],
  piecesPerSize: [3, 3, 3, 3],
  winCondition: { kind: 'lineOfN', n: 4 },
  allowSameSizeCover: false,
  allowSelfCover: true,
  maxPly: 120,
  drawByRepetition: 3,
};

/** User-facing preset list (fixed order). */
export const RULE_PRESETS = [
  {
    id: '3x3-classic',
    label: { ja: '3x3 クラシック', en: '3x3 Classic' },
    rules: PRESET_3X3,
  },
  {
    id: '4x4-huge',
    label: { ja: '4x4 巨大入り', en: '4x4 Huge' },
    rules: PRESET_4X4_XL,
  },
] as const;

export type RulePresetId = (typeof RULE_PRESETS)[number]['id'];

/** Upper limits the universal AI network supports (frozen at training). */
export const AI_LIMITS = {
  MAX_BOARD: 4,
  MAX_PIECE_SIZES: 4,
  MAX_PIECES_PER_SIZE: 3,
} as const;

/** Returns true when the rules fit inside the AI's universal network. */
export function isRuleSupportedByAI(rules: GameRules): boolean {
  return (
    rules.boardSize <= AI_LIMITS.MAX_BOARD &&
    rules.pieceSizes.length <= AI_LIMITS.MAX_PIECE_SIZES &&
    rules.piecesPerSize.every((n) => n <= AI_LIMITS.MAX_PIECES_PER_SIZE)
  );
}
