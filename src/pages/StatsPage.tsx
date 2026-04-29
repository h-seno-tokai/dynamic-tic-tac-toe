import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/primitives';
import { useStatsStore } from '@/stores';

export const StatsPage = () => {
  const navigate = useNavigate();
  const { perDifficulty, totalGames, lastPlayedAt, clearStats } = useStatsStore();
  const levels = Array.from({ length: 10 }, (_, index) => index + 1);

  return (
    <main className="mx-auto min-h-screen w-full max-w-4xl px-5 py-6">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">Stats</p>
        <h1 className="mt-2 text-3xl font-bold tracking-normal">戦績</h1>
        <p className="mt-2 text-sm text-muted">
          合計 {totalGames} 局 / 最終対局 {lastPlayedAt ?? 'なし'}
        </p>
      </header>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full min-w-[520px] border-collapse text-sm">
          <thead className="bg-[color:var(--color-cell-light)] text-left">
            <tr>
              <th className="px-3 py-2">難易度</th>
              <th className="px-3 py-2">勝ち</th>
              <th className="px-3 py-2">負け</th>
              <th className="px-3 py-2">引き分け</th>
            </tr>
          </thead>
          <tbody>
            {levels.map((level) => {
              const record = perDifficulty[level] ?? { wins: 0, losses: 0, draws: 0 };
              return (
                <tr key={level} className="border-t border-border">
                  <td className="px-3 py-2">Lv.{level}</td>
                  <td className="px-3 py-2">{record.wins}</td>
                  <td className="px-3 py-2">{record.losses}</td>
                  <td className="px-3 py-2">{record.draws}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <Button variant="danger" onClick={clearStats}>
          戦績クリア
        </Button>
        <Button variant="secondary" onClick={() => navigate('/')}>
          メニューへ
        </Button>
      </div>
    </main>
  );
};
