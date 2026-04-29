import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  __resetForTests,
  applyTheme,
  getResolvedTheme,
  subscribeSystemTheme,
} from './themeService';

interface FakeMql {
  matches: boolean;
  media: string;
  listeners: Set<(e: MediaQueryListEvent) => void>;
  addEventListener: (type: 'change', listener: (e: MediaQueryListEvent) => void) => void;
  removeEventListener: (type: 'change', listener: (e: MediaQueryListEvent) => void) => void;
  dispatch: (matches: boolean) => void;
}

function makeFakeMql(initialMatches: boolean): FakeMql {
  const mql: FakeMql = {
    matches: initialMatches,
    media: '(prefers-color-scheme: dark)',
    listeners: new Set(),
    addEventListener: (_type, listener) => {
      mql.listeners.add(listener);
    },
    removeEventListener: (_type, listener) => {
      mql.listeners.delete(listener);
    },
    dispatch: (matches: boolean) => {
      mql.matches = matches;
      const event: MediaQueryListEvent = { matches, media: mql.media } as MediaQueryListEvent;
      for (const l of mql.listeners) l(event);
    },
  };
  return mql;
}

describe('theme/themeService', () => {
  let fakeMql: FakeMql;

  beforeEach(() => {
    fakeMql = makeFakeMql(false);
    // jsdom does not implement matchMedia by default — install a stub.
    vi.spyOn(window, 'matchMedia').mockImplementation((query: string) => {
      // Always return our single fake (we only ever query the dark scheme).
      void query;
      return fakeMql as unknown as MediaQueryList;
    });
    document.documentElement.removeAttribute('data-theme');
  });

  afterEach(() => {
    __resetForTests();
    vi.restoreAllMocks();
    document.documentElement.removeAttribute('data-theme');
  });

  it("applyTheme('dark') sets data-theme='dark'", () => {
    applyTheme('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it("applyTheme('light') sets data-theme='light'", () => {
    applyTheme('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it("applyTheme('system') uses the matchMedia result (dark)", () => {
    fakeMql.matches = true;
    applyTheme('system');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it("applyTheme('system') uses the matchMedia result (light)", () => {
    fakeMql.matches = false;
    applyTheme('system');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it("applyTheme('system') updates data-theme when the OS preference changes", () => {
    fakeMql.matches = false;
    applyTheme('system');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');

    fakeMql.dispatch(true);
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');

    fakeMql.dispatch(false);
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('detaches the previous system listener when the theme changes', () => {
    applyTheme('system');
    expect(fakeMql.listeners.size).toBe(1);
    applyTheme('dark');
    expect(fakeMql.listeners.size).toBe(0);
  });

  it('getResolvedTheme reflects the current matchMedia state', () => {
    fakeMql.matches = true;
    expect(getResolvedTheme()).toBe('dark');
    fakeMql.matches = false;
    expect(getResolvedTheme()).toBe('light');
  });

  it('subscribeSystemTheme fires on media-query change and unsubscribe stops it', () => {
    const cb = vi.fn();
    const unsubscribe = subscribeSystemTheme(cb);

    fakeMql.dispatch(true);
    expect(cb).toHaveBeenCalledWith('dark');

    fakeMql.dispatch(false);
    expect(cb).toHaveBeenCalledWith('light');
    expect(cb).toHaveBeenCalledTimes(2);

    unsubscribe();
    fakeMql.dispatch(true);
    expect(cb).toHaveBeenCalledTimes(2);
  });
});
