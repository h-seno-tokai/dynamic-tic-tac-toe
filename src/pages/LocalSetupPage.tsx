import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { RULE_PRESETS, type RulePresetId } from '@/domain';
import { AvatarPicker, DEFAULT_AVATAR_SEEDS } from '@/components/avatar';
import { Button, RadioGroup } from '@/components/primitives';
import { useGameStore, useSessionStore } from '@/stores';

export const LocalSetupPage = () => {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();
  const startNewGame = useGameStore((state) => state.startNewGame);
  const session = useSessionStore();

  const [presetId, setPresetId] = useState<RulePresetId>('3x3-classic');
  const [p1Name, setP1Name] = useState(session.lastP1Name ?? 'Player 1');
  const [p2Name, setP2Name] = useState(session.lastP2Name ?? 'Player 2');
  const [p1Avatar, setP1Avatar] = useState(session.lastP1AvatarId ?? DEFAULT_AVATAR_SEEDS[0]);
  const [p2Avatar, setP2Avatar] = useState(session.lastP2AvatarId ?? DEFAULT_AVATAR_SEEDS[1]);

  const lang = i18n.language === 'en' ? 'en' : 'ja';

  const selectedPreset = useMemo(
    () => RULE_PRESETS.find((preset) => preset.id === presetId) ?? RULE_PRESETS[0],
    [presetId],
  );

  const handleStart = () => {
    session.setLastP1Name(p1Name.trim() || 'Player 1');
    session.setLastP2Name(p2Name.trim() || 'Player 2');
    session.setLastP1AvatarId(p1Avatar);
    session.setLastP2AvatarId(p2Avatar);
    startNewGame(selectedPreset.rules, 'local-2p');
    navigate('/play');
  };

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-5 py-6">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">{t('setup.local.subtitle')}</p>
        <h1 className="mt-2 text-3xl font-bold tracking-normal">{t('setup.local.title')}</h1>
        <p className="mt-2 text-sm leading-6 text-muted">{t('setup.local.desc')}</p>
      </header>

      <div className="grid gap-6">
        <fieldset className="grid gap-3">
          <legend className="text-sm font-semibold">{t('setup.p1Label')}</legend>
          <label className="grid gap-1.5">
            <span className="text-xs text-muted">{t('setup.nameLabel')}</span>
            <input
              value={p1Name}
              onChange={(e) => setP1Name(e.target.value)}
              className="h-10 rounded-md border border-border bg-bg px-3 text-base"
            />
          </label>
          <div className="grid gap-1.5">
            <span className="text-xs text-muted">{t('setup.avatarLabel')}</span>
            <AvatarPicker
              value={p1Avatar}
              onChange={setP1Avatar}
              label={t('setup.p1AvatarLabel')}
              getSeedLabel={(seed) => t('setup.avatarSeedLabel', { seed })}
            />
          </div>
        </fieldset>

        <fieldset className="grid gap-3">
          <legend className="text-sm font-semibold">{t('setup.p2Label')}</legend>
          <label className="grid gap-1.5">
            <span className="text-xs text-muted">{t('setup.nameLabel')}</span>
            <input
              value={p2Name}
              onChange={(e) => setP2Name(e.target.value)}
              className="h-10 rounded-md border border-border bg-bg px-3 text-base"
            />
          </label>
          <div className="grid gap-1.5">
            <span className="text-xs text-muted">{t('setup.avatarLabel')}</span>
            <AvatarPicker
              value={p2Avatar}
              onChange={setP2Avatar}
              label={t('setup.p2AvatarLabel')}
              getSeedLabel={(seed) => t('setup.avatarSeedLabel', { seed })}
            />
          </div>
        </fieldset>

        <RadioGroup
          label={t('setup.presetLabel')}
          value={presetId}
          onChange={setPresetId}
          options={RULE_PRESETS.map((preset) => ({
            value: preset.id,
            label: preset.label[lang],
            description: `${preset.rules.boardSize}x${preset.rules.boardSize} / ${preset.rules.pieceSizes
              .map((size) => size.displayName[lang])
              .join(' · ')}`,
          }))}
        />

        <div className="flex flex-wrap gap-3 pt-2">
          <Button onClick={handleStart}>{t('setup.start')}</Button>
          <Button variant="secondary" onClick={() => navigate('/')}>
            {t('common.backToMenu')}
          </Button>
        </div>
      </div>
    </main>
  );
};
