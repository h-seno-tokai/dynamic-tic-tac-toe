import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

interface MockStorage {
  store: Record<string, string>;
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
  removeItem: (key: string) => void;
  clear: () => void;
  key: (i: number) => string | null;
  length: number;
}

function createMockStorage(): MockStorage {
  const data: Record<string, string> = {};
  const api: MockStorage = {
    store: data,
    getItem: (k) => (k in data ? data[k]! : null),
    setItem: (k, v) => {
      data[k] = String(v);
    },
    removeItem: (k) => {
      delete data[k];
    },
    clear: () => {
      for (const k of Object.keys(data)) delete data[k];
    },
    key: (i) => Object.keys(data)[i] ?? null,
    length: 0,
  };
  Object.defineProperty(api, 'length', { get: () => Object.keys(data).length });
  return api;
}

describe('statsStore', () => {
  let mockStorage: MockStorage;

  beforeEach(() => {
    vi.resetModules();
    mockStorage = createMockStorage();
    vi.stubGlobal('localStorage', mockStorage);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('has the expected initial state', async () => {
    const { useStatsStore } = await import('./statsStore');
    const s = useStatsStore.getState();
    expect(s.perDifficulty).toEqual({});
    expect(s.lastPlayedAt).toBeNull();
    expect(s.totalGames).toBe(0);
  });

  it('recordGame accumulates per-difficulty counters and bumps totals', async () => {
    const { useStatsStore } = await import('./statsStore');
    useStatsStore.getState().recordGame(3, 'win');
    useStatsStore.getState().recordGame(3, 'win');
    useStatsStore.getState().recordGame(3, 'loss');
    useStatsStore.getState().recordGame(7, 'draw');

    const s = useStatsStore.getState();
    expect(s.totalGames).toBe(4);
    expect(s.perDifficulty[3]).toEqual({ wins: 2, losses: 1, draws: 0 });
    expect(s.perDifficulty[7]).toEqual({ wins: 0, losses: 0, draws: 1 });
    expect(s.lastPlayedAt).not.toBeNull();
    // ISO 8601 sanity check.
    expect(() => new Date(s.lastPlayedAt!).toISOString()).not.toThrow();
  });

  it('clearStats wipes all stored stats', async () => {
    const { useStatsStore } = await import('./statsStore');
    useStatsStore.getState().recordGame(1, 'win');
    useStatsStore.getState().clearStats();
    const s = useStatsStore.getState();
    expect(s.perDifficulty).toEqual({});
    expect(s.lastPlayedAt).toBeNull();
    expect(s.totalGames).toBe(0);
  });

  it('persists under dttt:stats with schema version 1', async () => {
    const { useStatsStore, STATS_STORAGE_KEY, STATS_SCHEMA_VERSION } = await import('./statsStore');
    expect(STATS_STORAGE_KEY).toBe('dttt:stats');
    useStatsStore.getState().recordGame(5, 'win');

    const raw = mockStorage.getItem(STATS_STORAGE_KEY);
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw!) as {
      state: { perDifficulty: Record<string, unknown>; totalGames: number };
      version: number;
    };
    expect(parsed.version).toBe(STATS_SCHEMA_VERSION);
    expect(parsed.state.totalGames).toBe(1);
    expect(parsed.state.perDifficulty[5]).toEqual({ wins: 1, losses: 0, draws: 0 });
  });

  it('roundtrips persisted state on a fresh module load', async () => {
    mockStorage.setItem(
      'dttt:stats',
      JSON.stringify({
        state: {
          perDifficulty: { 4: { wins: 3, losses: 2, draws: 1 } },
          lastPlayedAt: '2026-01-01T00:00:00.000Z',
          totalGames: 6,
        },
        version: 1,
      }),
    );
    vi.resetModules();
    const { useStatsStore } = await import('./statsStore');
    const s = useStatsStore.getState();
    expect(s.totalGames).toBe(6);
    expect(s.lastPlayedAt).toBe('2026-01-01T00:00:00.000Z');
    expect(s.perDifficulty[4]).toEqual({ wins: 3, losses: 2, draws: 1 });
  });
});
