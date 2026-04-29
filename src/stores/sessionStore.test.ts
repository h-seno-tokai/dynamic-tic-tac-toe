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

describe('sessionStore', () => {
  let mockStorage: MockStorage;

  beforeEach(() => {
    vi.resetModules();
    mockStorage = createMockStorage();
    vi.stubGlobal('localStorage', mockStorage);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('has the expected initial state (all undefined)', async () => {
    const { useSessionStore } = await import('./sessionStore');
    const s = useSessionStore.getState();
    expect(s.lastP1Name).toBeUndefined();
    expect(s.lastP2Name).toBeUndefined();
    expect(s.lastP1AvatarId).toBeUndefined();
    expect(s.lastP2AvatarId).toBeUndefined();
  });

  it('setters update the corresponding fields', async () => {
    const { useSessionStore } = await import('./sessionStore');
    useSessionStore.getState().setLastP1Name('Alice');
    useSessionStore.getState().setLastP2Name('Bob');
    useSessionStore.getState().setLastP1AvatarId('av-1');
    useSessionStore.getState().setLastP2AvatarId('av-2');

    const s = useSessionStore.getState();
    expect(s.lastP1Name).toBe('Alice');
    expect(s.lastP2Name).toBe('Bob');
    expect(s.lastP1AvatarId).toBe('av-1');
    expect(s.lastP2AvatarId).toBe('av-2');
  });

  it('reset() clears all session fields', async () => {
    const { useSessionStore } = await import('./sessionStore');
    useSessionStore.getState().setLastP1Name('Alice');
    useSessionStore.getState().setLastP2AvatarId('av-2');
    useSessionStore.getState().reset();
    const s = useSessionStore.getState();
    expect(s.lastP1Name).toBeUndefined();
    expect(s.lastP2Name).toBeUndefined();
    expect(s.lastP1AvatarId).toBeUndefined();
    expect(s.lastP2AvatarId).toBeUndefined();
  });

  it('persists under dttt:session with schema version 1', async () => {
    const { useSessionStore, SESSION_STORAGE_KEY, SESSION_SCHEMA_VERSION } =
      await import('./sessionStore');
    expect(SESSION_STORAGE_KEY).toBe('dttt:session');
    useSessionStore.getState().setLastP1Name('Alice');
    useSessionStore.getState().setLastP1AvatarId('av-1');

    const raw = mockStorage.getItem(SESSION_STORAGE_KEY);
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw!) as {
      state: { lastP1Name?: string; lastP1AvatarId?: string };
      version: number;
    };
    expect(parsed.version).toBe(SESSION_SCHEMA_VERSION);
    expect(parsed.state.lastP1Name).toBe('Alice');
    expect(parsed.state.lastP1AvatarId).toBe('av-1');
  });

  it('roundtrips persisted state on a fresh module load', async () => {
    mockStorage.setItem(
      'dttt:session',
      JSON.stringify({
        state: {
          lastP1Name: 'Alice',
          lastP2Name: 'Bob',
          lastP1AvatarId: 'av-1',
          lastP2AvatarId: 'av-2',
        },
        version: 1,
      }),
    );
    vi.resetModules();
    const { useSessionStore } = await import('./sessionStore');
    const s = useSessionStore.getState();
    expect(s.lastP1Name).toBe('Alice');
    expect(s.lastP2Name).toBe('Bob');
    expect(s.lastP1AvatarId).toBe('av-1');
    expect(s.lastP2AvatarId).toBe('av-2');
  });
});
