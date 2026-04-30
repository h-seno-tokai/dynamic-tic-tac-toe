/**
 * Difficulty profiles per `docs/07_ai_design.md` §1.3.
 *
 * 4x4 (universal-network MCTS): one model is shipped; the 10 difficulty
 * levels vary `simCount` (MCTS sims) and `temperature` (move sampling) at
 * inference time. Higher level = more sims and lower temperature.
 *
 * 3x3 (alpha-beta solver): the universal network is unnecessary because the
 * game has been retrograde-solved. Level controls a **time budget** for the
 * iterative-deepening alpha-beta search, plus a **mistake injection rate** —
 * the chance per root move of intentionally picking a non-best legal move.
 * Level 10 is effectively perfect play (≈10 s on the start position).
 */

export interface Difficulty4x4Profile {
  /** 1..10 */
  level: number;
  /** Number of MCTS simulations per move. */
  simCount: number;
  /** Sampling temperature applied to root visit-count distribution. */
  temperature: number;
  /** Localised human-readable description. */
  description: { ja: string; en: string };
}

export interface Difficulty3x3Profile {
  /** 1..10 */
  level: number;
  /** Wall-clock budget for the alpha-beta search, in milliseconds. */
  timeBudgetMs: number;
  /** Probability of intentionally choosing a non-best legal move (0..1). */
  mistakeRate: number;
  /** Localised human-readable description. */
  description: { ja: string; en: string };
}

/**
 * Backwards-compatible alias. Existing callers (4x4 worker path) continue to
 * see the original profile shape via `getProfile`.
 */
export type DifficultyProfile = Difficulty4x4Profile;

export const DIFFICULTY_PROFILES: Record<number, Difficulty4x4Profile> = {
  1: {
    level: 1,
    simCount: 20,
    temperature: 1.5,
    description: {
      ja: 'とてもやさしい',
      en: 'Beginner — very easy.',
    },
  },
  2: {
    level: 2,
    simCount: 50,
    temperature: 1.3,
    description: {
      ja: 'やさしい',
      en: 'Easy.',
    },
  },
  3: {
    level: 3,
    simCount: 100,
    temperature: 1.1,
    description: {
      ja: 'やや弱い',
      en: 'Slightly weak.',
    },
  },
  4: {
    level: 4,
    simCount: 150,
    temperature: 1.0,
    description: {
      ja: '普通(下)',
      en: 'Lower-medium.',
    },
  },
  5: {
    level: 5,
    simCount: 250,
    temperature: 0.85,
    description: {
      ja: '普通',
      en: 'Medium.',
    },
  },
  6: {
    level: 6,
    simCount: 400,
    temperature: 0.7,
    description: {
      ja: '普通(上)',
      en: 'Upper-medium.',
    },
  },
  7: {
    level: 7,
    simCount: 600,
    temperature: 0.5,
    description: {
      ja: 'やや強い',
      en: 'Strong.',
    },
  },
  8: {
    level: 8,
    simCount: 1000,
    temperature: 0.3,
    description: {
      ja: '強い',
      en: 'Very strong.',
    },
  },
  9: {
    level: 9,
    simCount: 1500,
    temperature: 0.15,
    description: {
      ja: 'とても強い',
      en: 'Expert.',
    },
  },
  10: {
    level: 10,
    simCount: 3000,
    temperature: 0,
    description: {
      ja: '最強(貪欲)',
      en: 'Champion (greedy).',
    },
  },
};

export const DIFFICULTY_PROFILES_3X3: Record<number, Difficulty3x3Profile> = {
  1: {
    level: 1,
    timeBudgetMs: 10,
    mistakeRate: 0.7,
    description: {
      ja: 'とてもやさしい',
      en: 'Beginner — very easy.',
    },
  },
  2: {
    level: 2,
    timeBudgetMs: 20,
    mistakeRate: 0.5,
    description: {
      ja: 'やさしい',
      en: 'Easy.',
    },
  },
  3: {
    level: 3,
    timeBudgetMs: 50,
    mistakeRate: 0.3,
    description: {
      ja: 'やや弱い',
      en: 'Slightly weak.',
    },
  },
  4: {
    level: 4,
    timeBudgetMs: 100,
    mistakeRate: 0.2,
    description: {
      ja: '普通(下)',
      en: 'Lower-medium.',
    },
  },
  5: {
    level: 5,
    timeBudgetMs: 200,
    mistakeRate: 0.1,
    description: {
      ja: '普通',
      en: 'Medium.',
    },
  },
  6: {
    level: 6,
    timeBudgetMs: 500,
    mistakeRate: 0.05,
    description: {
      ja: '普通(上)',
      en: 'Upper-medium.',
    },
  },
  7: {
    level: 7,
    timeBudgetMs: 1000,
    mistakeRate: 0.02,
    description: {
      ja: 'やや強い',
      en: 'Strong.',
    },
  },
  8: {
    level: 8,
    timeBudgetMs: 2500,
    mistakeRate: 0,
    description: {
      ja: '強い',
      en: 'Very strong.',
    },
  },
  9: {
    level: 9,
    timeBudgetMs: 5000,
    mistakeRate: 0,
    description: {
      ja: 'とても強い',
      en: 'Expert.',
    },
  },
  10: {
    level: 10,
    timeBudgetMs: 10000,
    mistakeRate: 0,
    description: {
      ja: '最強(完全解析級)',
      en: 'Champion (near-perfect).',
    },
  },
};

const MIN_LEVEL = 1;
const MAX_LEVEL = 10;

function clampLevel(level: number): number {
  return Math.max(MIN_LEVEL, Math.min(MAX_LEVEL, Math.round(level)));
}

/** Returns the 4x4 profile for `level`, clamped into the supported 1..10 range. */
export function getProfile(level: number): Difficulty4x4Profile {
  const clamped = clampLevel(level);
  const profile = DIFFICULTY_PROFILES[clamped];
  if (!profile) {
    throw new Error(`getProfile: unknown level ${level}`);
  }
  return profile;
}

/** Returns the 3x3 (alpha-beta solver) profile for `level`, clamped into 1..10. */
export function getProfile3x3(level: number): Difficulty3x3Profile {
  const clamped = clampLevel(level);
  const profile = DIFFICULTY_PROFILES_3X3[clamped];
  if (!profile) {
    throw new Error(`getProfile3x3: unknown level ${level}`);
  }
  return profile;
}
