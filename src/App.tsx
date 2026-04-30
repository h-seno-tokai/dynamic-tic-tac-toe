import { useEffect } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { Howler } from 'howler';
import {
  CpuSetupPage,
  LocalSetupPage,
  PlayPage,
  RulesPage,
  SettingsPage,
  StatsPage,
  TitlePage,
} from '@/pages';
import { audioService, themeService, i18n } from '@/infra';
import { useSettingsStore } from '@/stores';

// Ensure i18n singleton is initialised on module load and apply the
// persisted language immediately (before first render).
const _persistedLang = useSettingsStore.getState().language;
if (i18n.language !== _persistedLang) {
  void i18n.changeLanguage(_persistedLang);
}

export const App = () => {
  const theme = useSettingsStore((s) => s.theme);
  const language = useSettingsStore((s) => s.language);
  const bgmEnabled = useSettingsStore((s) => s.bgmEnabled);
  const sfxEnabled = useSettingsStore((s) => s.sfxEnabled);
  const bgmVolume = useSettingsStore((s) => s.bgmVolume);
  const sfxVolume = useSettingsStore((s) => s.sfxVolume);

  useEffect(() => {
    // Prefix every asset URL with Vite's BASE_URL so they resolve correctly
    // when the app is hosted under a sub-path (GitHub Pages: /dynamic-tic-tac-toe/).
    const base = import.meta.env.BASE_URL;
    audioService.init({
      bgm: { game: `${base}audio/bgm/whip.mp3` },
      sfx: {
        pickup: `${base}audio/sfx/pickup.mp3`,
        place: `${base}audio/sfx/place.mp3`,
        undo: `${base}audio/sfx/undo.mp3`,
        start: `${base}audio/sfx/start.mp3`,
        fanfare: `${base}audio/sfx/fanfare.mp3`,
        win: `${base}audio/sfx/win.mp3`,
        lose: `${base}audio/sfx/lose.mp3`,
        button: `${base}audio/sfx/button.mp3`,
        invalid: `${base}audio/sfx/invalid.mp3`,
      },
    });

    // Mobile browsers (especially iOS Safari) start AudioContext in
    // "suspended" state. Resume it on the first user gesture so that
    // SFX (which use Web Audio) play correctly after that point.
    const unlockAudio = () => {
      if (Howler.ctx?.state === 'suspended') {
        void Howler.ctx.resume();
      }
    };
    document.addEventListener('touchstart', unlockAudio, { once: true, passive: true });
    document.addEventListener('click', unlockAudio, { once: true });
    return () => {
      document.removeEventListener('touchstart', unlockAudio);
      document.removeEventListener('click', unlockAudio);
    };
  }, []);

  useEffect(() => {
    themeService.applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    void i18n.changeLanguage(language);
  }, [language]);

  useEffect(() => {
    audioService.setBgmEnabled(bgmEnabled);
  }, [bgmEnabled]);
  useEffect(() => {
    audioService.setSfxEnabled(sfxEnabled);
  }, [sfxEnabled]);
  useEffect(() => {
    audioService.setBgmVolume(bgmVolume);
  }, [bgmVolume]);
  useEffect(() => {
    audioService.setSfxVolume(sfxVolume);
  }, [sfxVolume]);

  return (
    <div className="min-h-screen bg-bg text-fg">
      <Routes>
        <Route path="/" element={<TitlePage />} />
        <Route path="/local/setup" element={<LocalSetupPage />} />
        <Route path="/cpu/setup" element={<CpuSetupPage />} />
        <Route path="/play" element={<PlayPage />} />
        <Route path="/rules" element={<RulesPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/stats" element={<StatsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
};
