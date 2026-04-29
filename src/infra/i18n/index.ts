/**
 * i18n configuration for Dynamic Tic-Tac-Toe.
 *
 * - Uses i18next + react-i18next + i18next-browser-languagedetector.
 * - Detection order: localStorage -> navigator (browser language).
 * - The localStorage key follows the project-wide `dttt:` namespace.
 * - Fallback language: Japanese.
 *
 * Importing this module (anywhere) initialises the singleton `i18n` instance.
 */

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

import ja from './resources/ja.json';
import en from './resources/en.json';

import { STORAGE_KEYS } from '../storage/keys';

export const SUPPORTED_LANGUAGES = ['ja', 'en'] as const;
export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

export const resources = {
  ja: { translation: ja },
  en: { translation: en },
} as const;

// Initialise only once, even if this module is imported multiple times
// (e.g. via vitest module-graph or HMR).
if (!i18n.isInitialized) {
  void i18n
    .use(LanguageDetector)
    .use(initReactI18next)
    .init({
      resources,
      fallbackLng: 'ja',
      supportedLngs: [...SUPPORTED_LANGUAGES],
      interpolation: { escapeValue: false },
      detection: {
        // localStorage first, then the browser navigator language.
        order: ['localStorage', 'navigator'],
        lookupLocalStorage: STORAGE_KEYS.i18nLng,
        caches: ['localStorage'],
      },
      returnNull: false,
    });
}

export { i18n };
export default i18n;
