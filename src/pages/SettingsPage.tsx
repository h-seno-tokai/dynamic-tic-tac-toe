import { useNavigate } from 'react-router-dom';
import { Button, RadioGroup, Slider, Toggle } from '@/components/primitives';
import { useSettingsStore } from '@/stores';

export const SettingsPage = () => {
  const navigate = useNavigate();
  const settings = useSettingsStore();

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-5 py-6">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">Settings</p>
        <h1 className="mt-2 text-3xl font-bold tracking-normal">設定</h1>
      </header>

      <div className="grid gap-5">
        <RadioGroup
          label="テーマ"
          value={settings.theme}
          onChange={settings.setTheme}
          options={[
            { value: 'system', label: 'システム' },
            { value: 'light', label: 'ライト' },
            { value: 'dark', label: 'ダーク' },
          ]}
        />

        <RadioGroup
          label="言語"
          value={settings.language}
          onChange={settings.setLanguage}
          options={[
            { value: 'ja', label: '日本語' },
            { value: 'en', label: 'English' },
          ]}
        />

        <Toggle
          label="BGM"
          checked={settings.bgmEnabled}
          onCheckedChange={settings.setBgmEnabled}
        />
        <Slider
          label="BGM音量"
          min={0}
          max={1}
          step={0.05}
          value={settings.bgmVolume}
          onChange={settings.setBgmVolume}
        />

        <Toggle
          label="効果音"
          checked={settings.sfxEnabled}
          onCheckedChange={settings.setSfxEnabled}
        />
        <Slider
          label="効果音音量"
          min={0}
          max={1}
          step={0.05}
          value={settings.sfxVolume}
          onChange={settings.setSfxVolume}
        />

        <div className="flex flex-wrap gap-3 pt-2">
          <Button variant="secondary" onClick={settings.reset}>
            初期値に戻す
          </Button>
          <Button variant="ghost" onClick={() => navigate('/')}>
            メニューへ
          </Button>
        </div>
      </div>
    </main>
  );
};
