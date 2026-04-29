/**
 * Difficulty profiles per `docs/07_ai_design.md` §1.3.
 *
 * One universal network is shipped; the 10 difficulty levels are purely
 * derivative — they vary `simCount` (MCTS sims) and `temperature` (move
 * sampling) at inference time. Higher level = more sims and lower temperature
 * (more deterministic, stronger play).
 */

export interface DifficultyProfile {
  /** 1..10 */
  level: number;
  /** Number of MCTS simulations per move. */
  simCount: number;
  /** Sampling temperature applied to root visit-count distribution. */
  temperature: number;
  /** Localised human-readable description. */
  description: { ja: string; en: string };
}

export const DIFFICULTY_PROFILES: Record<number, DifficultyProfile> = {
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

const MIN_LEVEL = 1;
const MAX_LEVEL = 10;

/** Returns the profile for `level`, clamped into the supported 1..10 range. */
export function getProfile(level: number): DifficultyProfile {
  const clamped = Math.max(MIN_LEVEL, Math.min(MAX_LEVEL, Math.round(level)));
  const profile = DIFFICULTY_PROFILES[clamped];
  if (!profile) {
    // Should never happen since the table is exhaustive 1..10.
    throw new Error(`getProfile: unknown level ${level}`);
  }
  return profile;
}
