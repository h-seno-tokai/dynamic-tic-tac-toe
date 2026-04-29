import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { RULE_PRESETS, type Player, type RulePresetId } from '@/domain';
import { AvatarPicker, DEFAULT_AVATAR_SEEDS } from '@/components/avatar';
import { Button, RadioGroup, Slider } from '@/components/primitives';
import { useGameStore, useSessionStore } from '@/stores';

export const CpuSetupPage = () => {
  const navigate = useNavigate();
  const startNewGame = useGameStore((state) => state.startNewGame);
  const session = useSessionStore();

  const [presetId, setPresetId] = useState<RulePresetId>('3x3-classic');
  const [difficulty, setDifficulty] = useState(3);
  const [humanSide, setHumanSide] = useState<Player>('P1');
  const [avatar, setAvatar] = useState(session.lastP1AvatarId ?? DEFAULT_AVATAR_SEEDS[0]);

  const selectedPreset = useMemo(
    () => RULE_PRESETS.find((preset) => preset.id === presetId) ?? RULE_PRESETS[0],
    [presetId],
  );

  const handleStart = () => {
    if (humanSide === 'P1') {
      session.setLastP1AvatarId(avatar);
    } else {
      session.setLastP2AvatarId(avatar);
    }
    startNewGame(selectedPreset.rules, 'cpu', { difficulty, humanSide });
    navigate('/play');
  };

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-5 py-6">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">CPU match</p>
        <h1 className="mt-2 text-3xl font-bold tracking-normal">CPU対戦</h1>
        <p className="mt-2 text-sm leading-6 text-muted">
          学習済みONNXモデルを接続するまでは、合法手ベースの暫定CPUで動かします。
        </p>
      </header>

      <div className="grid gap-6">
        <div className="grid gap-3">
          <span className="text-sm font-semibold">あなたのアバター</span>
          <AvatarPicker
            value={avatar}
            onChange={setAvatar}
            label="プレイヤーアバター"
            getSeedLabel={(seed) => `アバター ${seed}`}
          />
        </div>

        <Slider
          label={`難易度 ${difficulty}`}
          min={1}
          max={10}
          step={1}
          value={difficulty}
          onChange={setDifficulty}
        />

        <RadioGroup
          label="あなたの手番"
          value={humanSide}
          onChange={setHumanSide}
          options={[
            { value: 'P1', label: '先手' },
            { value: 'P2', label: '後手' },
          ]}
        />

        <RadioGroup
          label="ルールプリセット"
          value={presetId}
          onChange={setPresetId}
          options={RULE_PRESETS.map((preset) => ({
            value: preset.id,
            label: preset.label.ja,
            description: `${preset.rules.boardSize}x${preset.rules.boardSize}`,
          }))}
        />

        <div className="flex flex-wrap gap-3 pt-2">
          <Button onClick={handleStart}>対局開始</Button>
          <Button variant="secondary" onClick={() => navigate('/')}>
            メニューへ
          </Button>
        </div>
      </div>
    </main>
  );
};
