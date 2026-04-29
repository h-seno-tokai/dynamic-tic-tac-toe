import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { RULE_PRESETS, type Player, type RulePresetId } from '@/domain';
import { AvatarPicker, DEFAULT_AVATAR_SEEDS } from '@/components/avatar';
import { Button, RadioGroup, Slider } from '@/components/primitives';
import { useGameStore, useSessionStore } from '@/stores';

export const CpuSetupPage = () => {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();
  const startNewGame = useGameStore((state) => state.startNewGame);
  const session = useSessionStore();

  const [presetId, setPresetId] = useState<RulePresetId>('3x3-classic');
  const [difficulty, setDifficulty] = useState(3);
  const [humanSide, setHumanSide] = useState<Player>('P1');
  const [humanName, setHumanName] = useState(session.lastP1Name ?? 'Player 1');
  const [avatar, setAvatar] = useState(session.lastP1AvatarId ?? DEFAULT_AVATAR_SEEDS[0]);

  const lang = i18n.language === 'en' ? 'en' : 'ja';

  // When humanSide changes, restore the previously saved name/avatar for that slot.
  useEffect(() => {
    if (humanSide === 'P1') {
      setHumanName(session.lastP1Name ?? 'Player 1');
      setAvatar(session.lastP1AvatarId ?? DEFAULT_AVATAR_SEEDS[0]);
    } else {
      setHumanName(session.lastP2Name ?? 'Player 2');
      setAvatar(session.lastP2AvatarId ?? DEFAULT_AVATAR_SEEDS[1]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [humanSide]);

  const selectedPreset = useMemo(
    () => RULE_PRESETS.find((preset) => preset.id === presetId) ?? RULE_PRESETS[0],
    [presetId],
  );

  const handleStart = () => {
    const trimmed = humanName.trim() || (humanSide === 'P1' ? 'Player 1' : 'Player 2');
    if (humanSide === 'P1') {
      session.setLastP1Name(trimmed);
      session.setLastP1AvatarId(avatar);
    } else {
      session.setLastP2Name(trimmed);
      session.setLastP2AvatarId(avatar);
    }
    startNewGame(selectedPreset.rules, 'cpu', { difficulty, humanSide });
    navigate('/play');
  };

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-5 py-6">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">{t('setup.cpu.subtitle')}</p>
        <h1 className="mt-2 text-3xl font-bold tracking-normal">{t('setup.cpu.title')}</h1>
        <p className="mt-2 text-sm leading-6 text-muted">{t('setup.cpu.desc')}</p>
      </header>

      <div className="grid gap-6">
        <fieldset className="grid gap-3">
          <legend className="text-sm font-semibold">{t('setup.profileLabel')}</legend>
          <label className="grid gap-1.5">
            <span className="text-xs text-muted">{t('setup.nameLabel')}</span>
            <input
              value={humanName}
              onChange={(e) => setHumanName(e.target.value)}
              className="h-10 rounded-md border border-border bg-bg px-3 text-base"
            />
          </label>
          <div className="grid gap-1.5">
            <span className="text-xs text-muted">{t('setup.avatarLabel')}</span>
            <AvatarPicker
              value={avatar}
              onChange={setAvatar}
              label={t('setup.profileLabel')}
              getSeedLabel={(seed) => t('setup.avatarSeedLabel', { seed })}
            />
          </div>
        </fieldset>

        <Slider
          label={t('setup.difficultyLabel', { level: difficulty })}
          min={1}
          max={10}
          step={1}
          value={difficulty}
          onChange={setDifficulty}
        />

        <RadioGroup
          label={t('setup.humanSideLabel')}
          value={humanSide}
          onChange={setHumanSide}
          options={[
            { value: 'P1', label: t('setup.sideP1') },
            { value: 'P2', label: t('setup.sideP2') },
          ]}
        />

        <RadioGroup
          label={t('setup.presetLabel')}
          value={presetId}
          onChange={setPresetId}
          options={RULE_PRESETS.map((preset) => ({
            value: preset.id,
            label: preset.label[lang],
            description: `${preset.rules.boardSize}x${preset.rules.boardSize}`,
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
