import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { safeGet, safeRemove, safeSet, withPrefix } from './localStorage';
import { STORAGE_PREFIX } from './keys';

const isString = (v: unknown): v is string => typeof v === 'string';
interface Settings {
  bgmVolume: number;
  language: 'ja' | 'en';
}
const isSettings = (v: unknown): v is Settings =>
  typeof v === 'object' &&
  v !== null &&
  typeof (v as Record<string, unknown>).bgmVolume === 'number' &&
  ((v as Record<string, unknown>).language === 'ja' ||
    (v as Record<string, unknown>).language === 'en');

describe('storage/localStorage', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  describe('withPrefix', () => {
    it('adds the dttt: prefix when missing', () => {
      expect(withPrefix('foo')).toBe(`${STORAGE_PREFIX}foo`);
    });

    it('does not double-prefix when already prefixed', () => {
      expect(withPrefix(`${STORAGE_PREFIX}bar`)).toBe(`${STORAGE_PREFIX}bar`);
    });
  });

  describe('safeGet', () => {
    it('returns null when the key is missing', () => {
      expect(safeGet('missing', isString)).toBeNull();
    });

    it('returns the parsed value when present and valid', () => {
      window.localStorage.setItem(`${STORAGE_PREFIX}greeting`, JSON.stringify('hello'));
      expect(safeGet('greeting', isString)).toBe('hello');
    });

    it('returns null when the validator rejects the value', () => {
      window.localStorage.setItem(`${STORAGE_PREFIX}greeting`, JSON.stringify(42));
      expect(safeGet('greeting', isString)).toBeNull();
    });

    it('returns null on invalid JSON', () => {
      window.localStorage.setItem(`${STORAGE_PREFIX}broken`, '{not valid json');
      expect(safeGet('broken', isString)).toBeNull();
    });

    it('round-trips a structured object via safeSet', () => {
      const settings: Settings = { bgmVolume: 0.4, language: 'ja' };
      expect(safeSet('settings', settings)).toBe(true);
      expect(safeGet('settings', isSettings)).toEqual(settings);
    });
  });

  describe('safeSet', () => {
    it('returns true on success and writes JSON under the prefixed key', () => {
      expect(safeSet('count', 7)).toBe(true);
      expect(window.localStorage.getItem(`${STORAGE_PREFIX}count`)).toBe('7');
    });

    it('returns false on QuotaExceededError without re-throwing', () => {
      const err = new DOMException('quota', 'QuotaExceededError');
      vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        throw err;
      });
      expect(safeSet('big', { x: 1 })).toBe(false);
    });

    it('re-throws non-quota errors', () => {
      vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        throw new Error('boom');
      });
      expect(() => safeSet('foo', 1)).toThrow(/boom/);
    });
  });

  describe('safeRemove', () => {
    it('removes a previously stored key', () => {
      window.localStorage.setItem(`${STORAGE_PREFIX}gone`, '"x"');
      safeRemove('gone');
      expect(window.localStorage.getItem(`${STORAGE_PREFIX}gone`)).toBeNull();
    });
  });

  describe('without window.localStorage', () => {
    it('safeGet/safeSet/safeRemove do not throw and degrade safely', () => {
      // Force the getter to throw, simulating private-mode browsers.
      const spy = vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
        throw new Error('blocked');
      });

      expect(() => safeGet('any', isString)).not.toThrow();
      expect(safeGet('any', isString)).toBeNull();

      expect(() => safeSet('any', 'v')).not.toThrow();
      expect(safeSet('any', 'v')).toBe(false);

      expect(() => safeRemove('any')).not.toThrow();

      spy.mockRestore();
    });
  });
});
