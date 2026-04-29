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
import { themeService } from '@/infra';
import { useSettingsStore } from '@/stores';

export const App = () => {
  const theme = useSettingsStore((state) => state.theme);

  useEffect(() => {
    themeService.applyTheme(theme);
  }, [theme]);

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
