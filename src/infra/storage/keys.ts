/**
 * Storage keys used across the app. All keys must live under the `dttt:` namespace
 * to avoid clashing with other apps on the same origin.
 */

export const STORAGE_PREFIX = 'dttt:' as const;

/** All known persisted keys, already prefixed. */
export const STORAGE_KEYS = {
  settings: `${STORAGE_PREFIX}settings`,
  stats: `${STORAGE_PREFIX}stats`,
  session: `${STORAGE_PREFIX}session`,
  schemaVersion: `${STORAGE_PREFIX}schemaVersion`,
  i18nLng: `${STORAGE_PREFIX}i18nLng`,
} as const;

export type StorageKey = (typeof STORAGE_KEYS)[keyof typeof STORAGE_KEYS];
