import { describe, expect, it } from 'vitest';
import { i18n, resources, SUPPORTED_LANGUAGES } from './index';

describe('i18n', () => {
  it('initialises with both languages registered', () => {
    expect(i18n.isInitialized).toBe(true);
    expect(SUPPORTED_LANGUAGES).toEqual(['ja', 'en']);
    expect(Object.keys(resources)).toEqual(['ja', 'en']);
  });

  it('exposes Japanese strings under the expected keys', async () => {
    await i18n.changeLanguage('ja');
    expect(i18n.t('title.appName')).toBe('Dynamic Tic-Tac-Toe');
    expect(i18n.t('result.title')).toBe('勝負あり！');
    expect(i18n.t('menu.localPlay')).toBe('2人で対戦');
    expect(i18n.t('preset.p3x3')).toBe('3x3 クラシック');
    expect(i18n.t('settings.themeDark')).toBe('ダーク');
    expect(i18n.t('difficulty.level1')).toMatch(/レベル1/);
  });

  it('switches all keys to English when changeLanguage("en") is called', async () => {
    await i18n.changeLanguage('en');
    expect(i18n.t('title.appName')).toBe('Dynamic Tic-Tac-Toe');
    expect(i18n.t('result.title')).toBe("It's decided!");
    expect(i18n.t('menu.localPlay')).toBe('Local 2-Player');
    expect(i18n.t('preset.p3x3')).toBe('3x3 Classic');
    expect(i18n.t('settings.themeDark')).toBe('Dark');
    expect(i18n.t('difficulty.level10')).toMatch(/Level 10/);
  });

  it('falls back to Japanese for unknown languages', async () => {
    await i18n.changeLanguage('ja');
    expect(i18n.options.fallbackLng).toEqual(['ja']);
  });

  it('exposes the same key shape in both languages', () => {
    const collectKeys = (obj: unknown, prefix = ''): string[] => {
      if (obj === null || typeof obj !== 'object') return [prefix];
      const acc: string[] = [];
      for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
        const next = prefix ? `${prefix}.${k}` : k;
        acc.push(...collectKeys(v, next));
      }
      return acc;
    };
    const jaKeys = collectKeys(resources.ja.translation).sort();
    const enKeys = collectKeys(resources.en.translation).sort();
    expect(enKeys).toEqual(jaKeys);
  });
});
