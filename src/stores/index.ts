// Barrel re-exports for the Zustand stores.

export { useGameStore, type GameStoreState, type GameMode, type CpuOpts } from './gameStore';

export {
  useSettingsStore,
  SETTINGS_STORAGE_KEY,
  SETTINGS_SCHEMA_VERSION,
  type SettingsState,
  type Language,
  type Theme,
} from './settingsStore';

export {
  useStatsStore,
  STATS_STORAGE_KEY,
  STATS_SCHEMA_VERSION,
  type StatsState,
  type DifficultyRecord,
  type GameOutcome,
} from './statsStore';

export {
  useSessionStore,
  SESSION_STORAGE_KEY,
  SESSION_SCHEMA_VERSION,
  type SessionState,
} from './sessionStore';
