import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { Button, RadioGroup, Slider, Toggle } from '@/components/primitives';
import { useSettingsStore } from '@/stores';

export const SettingsPage = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const settings = useSettingsStore();

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-5 py-6">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">Settings</p>
        <h1 className="mt-2 text-3xl font-bold tracking-normal">{t('settings.title')}</h1>
      </header>

      <div className="grid gap-5">
        <RadioGroup
          label={t('settings.theme')}
          value={settings.theme}
          onChange={settings.setTheme}
          options={[
            { value: 'system', label: t('settings.themeSystem') },
            { value: 'light', label: t('settings.themeLight') },
            { value: 'dark', label: t('settings.themeDark') },
          ]}
        />

        <RadioGroup
          label={t('settings.language')}
          value={settings.language}
          onChange={settings.setLanguage}
          options={[
            { value: 'ja', label: t('settings.languageJa') },
            { value: 'en', label: t('settings.languageEn') },
          ]}
        />

        <Toggle
          label={t('settings.bgm')}
          checked={settings.bgmEnabled}
          onCheckedChange={settings.setBgmEnabled}
        />
        <Slider
          label={t('settings.bgmVolume')}
          min={0}
          max={1}
          step={0.05}
          value={settings.bgmVolume}
          onChange={settings.setBgmVolume}
        />

        <Toggle
          label={t('settings.sfx')}
          checked={settings.sfxEnabled}
          onCheckedChange={settings.setSfxEnabled}
        />
        <Slider
          label={t('settings.sfxVolume')}
          min={0}
          max={1}
          step={0.05}
          value={settings.sfxVolume}
          onChange={settings.setSfxVolume}
        />

        <div className="flex flex-wrap gap-3 pt-2">
          <Button variant="secondary" onClick={settings.reset}>
            {t('settings.reset')}
          </Button>
          <Button variant="ghost" onClick={() => navigate('/')}>
            {t('common.backToMenu')}
          </Button>
        </div>
      </div>
    </main>
  );
};
