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

describe('settingsStore', () => {
  let mockStorage: MockStorage;

  beforeEach(() => {
    vi.resetModules();
    mockStorage = createMockStorage();
    vi.stubGlobal('localStorage', mockStorage);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('has the expected default initial state', async () => {
    const { useSettingsStore } = await import('./settingsStore');
    const s = useSettingsStore.getState();
    expect(s.language).toBe('ja');
    expect(s.theme).toBe('system');
    expect(s.bgmVolume).toBeGreaterThan(0);
    expect(s.bgmVolume).toBeLessThanOrEqual(1);
    expect(s.bgmEnabled).toBe(true);
    expect(s.sfxEnabled).toBe(true);
  });

  it('setters mutate state correctly and clamp volumes', async () => {
    const { useSettingsStore } = await import('./settingsStore');
    const s = useSettingsStore.getState();
    s.setLanguage('en');
    s.setTheme('dark');
    s.setBgmVolume(2); // clamp -> 1
    s.setSfxVolume(-5); // clamp -> 0
    s.setBgmEnabled(false);
    s.setSfxEnabled(false);
    const after = useSettingsStore.getState();
    expect(after.language).toBe('en');
    expect(after.theme).toBe('dark');
    expect(after.bgmVolume).toBe(1);
    expect(after.sfxVolume).toBe(0);
    expect(after.bgmEnabled).toBe(false);
    expect(after.sfxEnabled).toBe(false);
  });

  it('reset() returns the store to defaults', async () => {
    const { useSettingsStore } = await import('./settingsStore');
    useSettingsStore.getState().setLanguage('en');
    useSettingsStore.getState().setTheme('dark');
    useSettingsStore.getState().reset();
    const s = useSettingsStore.getState();
    expect(s.language).toBe('ja');
    expect(s.theme).toBe('system');
  });

  it('persists changes to localStorage under the dttt:settings key', async () => {
    const { useSettingsStore, SETTINGS_STORAGE_KEY, SETTINGS_SCHEMA_VERSION } =
      await import('./settingsStore');
    expect(SETTINGS_STORAGE_KEY).toBe('dttt:settings');

    useSettingsStore.getState().setLanguage('en');
    useSettingsStore.getState().setTheme('dark');

    const raw = mockStorage.getItem(SETTINGS_STORAGE_KEY);
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw!) as { state: unknown; version: number };
    expect(parsed.version).toBe(SETTINGS_SCHEMA_VERSION);
    const persisted = parsed.state as { language: string; theme: string };
    expect(persisted.language).toBe('en');
    expect(persisted.theme).toBe('dark');
  });

  it('roundtrips persisted state through localStorage on a fresh module load', async () => {
    mockStorage.setItem(
      'dttt:settings',
      JSON.stringify({
        state: {
          language: 'en',
          theme: 'dark',
          bgmVolume: 0.25,
          sfxVolume: 0.1,
          bgmEnabled: false,
          sfxEnabled: false,
        },
        version: 1,
      }),
    );
    vi.resetModules();
    const { useSettingsStore } = await import('./settingsStore');
    const s = useSettingsStore.getState();
    expect(s.language).toBe('en');
    expect(s.theme).toBe('dark');
    expect(s.bgmVolume).toBe(0.25);
    expect(s.sfxVolume).toBe(0.1);
    expect(s.bgmEnabled).toBe(false);
    expect(s.sfxEnabled).toBe(false);
  });
});
