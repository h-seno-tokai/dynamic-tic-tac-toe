import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export type GameOutcome = 'win' | 'loss' | 'draw';

export interface DifficultyRecord {
  wins: number;
  losses: number;
  draws: number;
}

export interface StatsState {
  perDifficulty: Record<number, DifficultyRecord>;
  lastPlayedAt: string | null; // ISO 8601
  totalGames: number;

  recordGame: (difficulty: number, outcome: GameOutcome) => void;
  clearStats: () => void;
}

export const STATS_STORAGE_KEY = 'dttt:stats';
export const STATS_SCHEMA_VERSION = 1;

const INITIAL: Pick<StatsState, 'perDifficulty' | 'lastPlayedAt' | 'totalGames'> = {
  perDifficulty: {},
  lastPlayedAt: null,
  totalGames: 0,
};

function emptyRecord(): DifficultyRecord {
  return { wins: 0, losses: 0, draws: 0 };
}

export const useStatsStore = create<StatsState>()(
  persist(
    (set) => ({
      ...INITIAL,

      recordGame: (difficulty, outcome) => {
        set((state) => {
          const prev = state.perDifficulty[difficulty] ?? emptyRecord();
          const updated: DifficultyRecord = {
            wins: prev.wins + (outcome === 'win' ? 1 : 0),
            losses: prev.losses + (outcome === 'loss' ? 1 : 0),
            draws: prev.draws + (outcome === 'draw' ? 1 : 0),
          };
          return {
            perDifficulty: { ...state.perDifficulty, [difficulty]: updated },
            lastPlayedAt: new Date().toISOString(),
            totalGames: state.totalGames + 1,
          };
        });
      },

      clearStats: () => {
        set({ ...INITIAL, perDifficulty: {} });
      },
    }),
    {
      name: STATS_STORAGE_KEY,
      version: STATS_SCHEMA_VERSION,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        perDifficulty: state.perDifficulty,
        lastPlayedAt: state.lastPlayedAt,
        totalGames: state.totalGames,
      }),
    },
  ),
);
