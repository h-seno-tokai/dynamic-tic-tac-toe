/**
 * Theme service.
 *
 * Manages `document.documentElement[data-theme]` per the architecture in
 * `docs/03_architecture.md` §2.5 (CSS variables + Tailwind, switched via the
 * `data-theme` attribute on the `<html>` root).
 */

export type Theme = 'light' | 'dark' | 'system';
export type ResolvedTheme = 'light' | 'dark';

const DARK_MEDIA_QUERY = '(prefers-color-scheme: dark)';

/** Module-local listener so `applyTheme('system')` cleans up on re-entry. */
let systemMql: MediaQueryList | null = null;
let systemListener: ((e: MediaQueryListEvent) => void) | null = null;

function getMediaQuery(): MediaQueryList | null {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return null;
  }
  return window.matchMedia(DARK_MEDIA_QUERY);
}

function setRootTheme(resolved: ResolvedTheme): void {
  if (typeof document === 'undefined') return;
  document.documentElement.setAttribute('data-theme', resolved);
}

function detachSystemListener(): void {
  if (systemListener) {
    systemMql?.removeEventListener('change', systemListener);
  }
  systemMql = null;
  systemListener = null;
}

/** Resolve the theme that should currently be displayed. */
export function getResolvedTheme(): ResolvedTheme {
  return getMediaQuery()?.matches ? 'dark' : 'light';
}

/** Apply a theme. When `'system'`, also subscribes to OS-level changes. */
export function applyTheme(theme: Theme): void {
  // Always tear down any previous system subscription owned by this module.
  detachSystemListener();

  if (theme === 'light' || theme === 'dark') {
    setRootTheme(theme);
    return;
  }

  // theme === 'system'
  const mql = getMediaQuery();
  const resolved: ResolvedTheme = mql?.matches ? 'dark' : 'light';
  setRootTheme(resolved);

  if (mql) {
    const listener = (e: MediaQueryListEvent): void => {
      setRootTheme(e.matches ? 'dark' : 'light');
    };
    mql.addEventListener('change', listener);
    systemMql = mql;
    systemListener = listener;
  }
}

/**
 * Subscribe to OS-level color-scheme changes independently of `applyTheme`.
 *
 * The callback fires with the new resolved theme whenever the
 * `prefers-color-scheme: dark` media query toggles. Returns an unsubscribe
 * function that removes the listener.
 */
export function subscribeSystemTheme(cb: (resolved: ResolvedTheme) => void): () => void {
  const mql = getMediaQuery();
  if (!mql) return () => undefined;

  const listener = (e: MediaQueryListEvent): void => {
    cb(e.matches ? 'dark' : 'light');
  };
  mql.addEventListener('change', listener);
  return () => mql.removeEventListener('change', listener);
}

/** Test-only escape hatch: drop the module-level system subscription. */
export function __resetForTests(): void {
  detachSystemListener();
}

export const themeService = {
  applyTheme,
  getResolvedTheme,
  subscribeSystemTheme,
};
