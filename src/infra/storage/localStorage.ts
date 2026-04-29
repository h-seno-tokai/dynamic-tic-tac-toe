/**
 * Typed, defensive wrapper around `window.localStorage`.
 *
 * Goals:
 * - Never throw on missing `localStorage` (e.g. SSR, private mode in some browsers).
 * - Validate persisted JSON via a caller-supplied type guard.
 * - Catch `QuotaExceededError` specifically; surface other errors.
 * - Enforce the `dttt:` namespace prefix.
 */

import { STORAGE_PREFIX } from './keys';

/** Lazy access so the module also imports cleanly when `window` is absent. */
function getStore(): Storage | null {
  try {
    if (typeof window === 'undefined') return null;
    return window.localStorage;
  } catch {
    // Some environments throw on access (Safari private mode historically).
    return null;
  }
}

/** Ensure a key starts with the project namespace. */
export function withPrefix(key: string): string {
  return key.startsWith(STORAGE_PREFIX) ? key : `${STORAGE_PREFIX}${key}`;
}

/** Read + parse + validate a value. Returns null if missing/invalid/parse-failure. */
export function safeGet<T>(key: string, validate: (raw: unknown) => raw is T): T | null {
  const store = getStore();
  if (!store) return null;

  const fullKey = withPrefix(key);
  let raw: string | null;
  try {
    raw = store.getItem(fullKey);
  } catch {
    return null;
  }
  if (raw === null) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }

  return validate(parsed) ? parsed : null;
}

/** Tiny, dependency-free check for the QuotaExceededError DOMException. */
function isQuotaExceeded(err: unknown): boolean {
  if (typeof DOMException !== 'undefined' && err instanceof DOMException) {
    // Standard name + legacy name + numeric code, all checked.
    return (
      err.name === 'QuotaExceededError' ||
      err.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
      err.code === 22
    );
  }
  if (err && typeof err === 'object' && 'name' in err) {
    const name = (err as { name?: unknown }).name;
    return name === 'QuotaExceededError' || name === 'NS_ERROR_DOM_QUOTA_REACHED';
  }
  return false;
}

/**
 * Serialise + write a value.
 *
 * @returns `true` on success, `false` if the write failed because of
 * `QuotaExceededError`. Other errors are re-thrown so they are not silently lost.
 */
export function safeSet<T>(key: string, value: T): boolean {
  const store = getStore();
  if (!store) return false;

  const fullKey = withPrefix(key);
  const serialised = JSON.stringify(value);

  try {
    store.setItem(fullKey, serialised);
    return true;
  } catch (err) {
    if (isQuotaExceeded(err)) return false;
    throw err;
  }
}

/** Remove a key. No-op when storage is unavailable. */
export function safeRemove(key: string): void {
  const store = getStore();
  if (!store) return;
  try {
    store.removeItem(withPrefix(key));
  } catch {
    // Removal failures are non-fatal: ignore.
  }
}
