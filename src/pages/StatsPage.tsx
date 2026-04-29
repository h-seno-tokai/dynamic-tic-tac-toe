import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/primitives';
import { useStatsStore } from '@/stores';

export const StatsPage = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { perDifficulty, totalGames, lastPlayedAt, clearStats } = useStatsStore();
  const levels = Array.from({ length: 10 }, (_, index) => index + 1);

  return (
    <main className="mx-auto min-h-screen w-full max-w-4xl px-5 py-6">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">{t('stats.subtitle')}</p>
        <h1 className="mt-2 text-3xl font-bold tracking-normal">{t('stats.title')}</h1>
        <p className="mt-2 text-sm text-muted">
          {t('stats.summary', {
            total: totalGames,
            lastPlayed: lastPlayedAt ?? t('stats.noGames'),
          })}
        </p>
      </header>

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full min-w-[520px] border-collapse text-sm">
          <thead className="bg-[color:var(--color-cell-light)] text-left">
            <tr>
              <th className="px-3 py-2">{t('stats.colDifficulty')}</th>
              <th className="px-3 py-2">{t('stats.colWins')}</th>
              <th className="px-3 py-2">{t('stats.colLosses')}</th>
              <th className="px-3 py-2">{t('stats.colDraws')}</th>
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
          {t('stats.clearButton')}
        </Button>
        <Button variant="secondary" onClick={() => navigate('/')}>
          {t('common.backToMenu')}
        </Button>
      </div>
    </main>
  );
};
