import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { RULE_PRESETS, type RulePresetId } from '@/domain';
import { Button, RadioGroup } from '@/components/primitives';
import { useGameStore, useSessionStore } from '@/stores';

export const LocalSetupPage = () => {
  const navigate = useNavigate();
  const startNewGame = useGameStore((state) => state.startNewGame);
  const session = useSessionStore();

  const [presetId, setPresetId] = useState<RulePresetId>('3x3-classic');
  const [p1Name, setP1Name] = useState(session.lastP1Name ?? 'Player 1');
  const [p2Name, setP2Name] = useState(session.lastP2Name ?? 'Player 2');

  const selectedPreset = useMemo(
    () => RULE_PRESETS.find((preset) => preset.id === presetId) ?? RULE_PRESETS[0],
    [presetId],
  );

  const handleStart = () => {
    session.setLastP1Name(p1Name.trim() || 'Player 1');
    session.setLastP2Name(p2Name.trim() || 'Player 2');
    startNewGame(selectedPreset.rules, 'local-2p');
    navigate('/play');
  };

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-5 py-6">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">Local match</p>
        <h1 className="mt-2 text-3xl font-bold tracking-normal">ローカル2人対戦</h1>
        <p className="mt-2 text-sm leading-6 text-muted">
          プレイヤー名とルールを決めて、同じ画面で交互に操作します。
        </p>
      </header>

      <div className="grid gap-5">
        <label className="grid gap-2">
          <span className="text-sm font-medium">先手プレイヤー</span>
          <input
            value={p1Name}
            onChange={(event) => setP1Name(event.target.value)}
            className="h-11 rounded-md border border-border bg-bg px-3 text-base"
          />
        </label>

        <label className="grid gap-2">
          <span className="text-sm font-medium">後手プレイヤー</span>
          <input
            value={p2Name}
            onChange={(event) => setP2Name(event.target.value)}
            className="h-11 rounded-md border border-border bg-bg px-3 text-base"
          />
        </label>

        <RadioGroup
          label="ルールプリセット"
          value={presetId}
          onChange={setPresetId}
          options={RULE_PRESETS.map((preset) => ({
            value: preset.id,
            label: preset.label.ja,
            description: `${preset.rules.boardSize}x${preset.rules.boardSize} / ${preset.rules.pieceSizes
              .map((size) => size.displayName.ja)
              .join('・')}`,
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
