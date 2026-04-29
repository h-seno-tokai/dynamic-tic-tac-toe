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

// Ensure i18n singleton is initialised on module load.
void i18n;

export const App = () => {
  const theme = useSettingsStore((s) => s.theme);
  const language = useSettingsStore((s) => s.language);
  const bgmEnabled = useSettingsStore((s) => s.bgmEnabled);
  const sfxEnabled = useSettingsStore((s) => s.sfxEnabled);
  const bgmVolume = useSettingsStore((s) => s.bgmVolume);
  const sfxVolume = useSettingsStore((s) => s.sfxVolume);

  useEffect(() => {
    audioService.init({
      bgm: { game: '/audio/bgm/whip.mp3' },
      sfx: {
        pickup: '/audio/sfx/pickup.mp3',
        place: '/audio/sfx/place.mp3',
        undo: '/audio/sfx/undo.mp3',
        start: '/audio/sfx/start.mp3',
        fanfare: '/audio/sfx/fanfare.mp3',
        win: '/audio/sfx/win.mp3',
        lose: '/audio/sfx/lose.mp3',
        button: '/audio/sfx/button.mp3',
        invalid: '/audio/sfx/invalid.mp3',
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
