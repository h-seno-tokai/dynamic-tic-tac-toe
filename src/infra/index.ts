/**
 * Public surface of the infrastructure layer.
 *
 * Per `docs/03_architecture.md` §2.5, this module groups cross-cutting browser
 * concerns: i18n, audio, persistence, and theming.
 */

// i18n
export { i18n, resources, SUPPORTED_LANGUAGES } from './i18n';
export type { SupportedLanguage } from './i18n';

// Audio
export {
  audioService,
  init as initAudio,
  setBgmVolume,
  setSfxVolume,
  setBgmEnabled,
  setSfxEnabled,
  playBgm,
  stopBgm,
  playSfx,
  preload as preloadAudio,
} from './audio/audioService';
export type { AudioConfig } from './audio/audioService';

// Storage
export { safeGet, safeSet, safeRemove, withPrefix } from './storage/localStorage';
export { STORAGE_KEYS, STORAGE_PREFIX } from './storage/keys';
export type { StorageKey } from './storage/keys';

// Theme
export {
  themeService,
  applyTheme,
  getResolvedTheme,
  subscribeSystemTheme,
} from './theme/themeService';
export type { Theme, ResolvedTheme } from './theme/themeService';
