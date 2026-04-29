import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export type Language = 'ja' | 'en';
export type Theme = 'light' | 'dark' | 'system';

export interface SettingsState {
  language: Language;
  theme: Theme;
  bgmVolume: number; // 0..1
  sfxVolume: number; // 0..1
  bgmEnabled: boolean;
  sfxEnabled: boolean;

  setLanguage: (language: Language) => void;
  setTheme: (theme: Theme) => void;
  setBgmVolume: (v: number) => void;
  setSfxVolume: (v: number) => void;
  setBgmEnabled: (enabled: boolean) => void;
  setSfxEnabled: (enabled: boolean) => void;
  reset: () => void;
}

export const SETTINGS_STORAGE_KEY = 'dttt:settings';
export const SETTINGS_SCHEMA_VERSION = 1;

const DEFAULTS = {
  language: 'ja',
  theme: 'system',
  bgmVolume: 0.6,
  sfxVolume: 0.8,
  bgmEnabled: true,
  sfxEnabled: true,
} as const satisfies Pick<
  SettingsState,
  'language' | 'theme' | 'bgmVolume' | 'sfxVolume' | 'bgmEnabled' | 'sfxEnabled'
>;

function clampVolume(v: number): number {
  if (!Number.isFinite(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...DEFAULTS,

      setLanguage: (language) => {
        set({ language });
      },
      setTheme: (theme) => {
        set({ theme });
      },
      setBgmVolume: (v) => {
        set({ bgmVolume: clampVolume(v) });
      },
      setSfxVolume: (v) => {
        set({ sfxVolume: clampVolume(v) });
      },
      setBgmEnabled: (enabled) => {
        set({ bgmEnabled: enabled });
      },
      setSfxEnabled: (enabled) => {
        set({ sfxEnabled: enabled });
      },
      reset: () => {
        set({ ...DEFAULTS });
      },
    }),
    {
      name: SETTINGS_STORAGE_KEY,
      version: SETTINGS_SCHEMA_VERSION,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        language: state.language,
        theme: state.theme,
        bgmVolume: state.bgmVolume,
        sfxVolume: state.sfxVolume,
        bgmEnabled: state.bgmEnabled,
        sfxEnabled: state.sfxEnabled,
      }),
    },
  ),
);
