import { useEffect } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import {
  CpuSetupPage,
  LocalSetupPage,
  PlayPage,
  ResultPage,
  RulesPage,
  SettingsPage,
  StatsPage,
  TitlePage,
} from '@/pages';
import { audioService, themeService } from '@/infra';
import { useSettingsStore } from '@/stores';

export const App = () => {
  const theme = useSettingsStore((s) => s.theme);
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
  }, []);

  useEffect(() => {
    themeService.applyTheme(theme);
  }, [theme]);
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
        <Route path="/result" element={<ResultPage />} />
        <Route path="/rules" element={<RulesPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/stats" element={<StatsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
};
