import { Navigate, useNavigate } from 'react-router-dom';
import { Button } from '@/components/primitives';
import { useGameStore } from '@/stores';

export const ResultPage = () => {
  const navigate = useNavigate();
  const { currentGame, mode, cpuDifficulty, humanSide, startNewGame, endGame } = useGameStore();

  if (!currentGame) return <Navigate to="/" replace />;

  const outcome = currentGame.outcome;
  const title = outcome === 'draw' ? '引き分け' : outcome ? `${outcome} の勝ち` : '対局中';

  const handleRematch = () => {
    if (mode === 'cpu') {
      startNewGame(currentGame.rules, 'cpu', {
        difficulty: cpuDifficulty ?? 1,
        humanSide: humanSide ?? 'P1',
      });
    } else {
      startNewGame(currentGame.rules, 'local-2p');
    }
    navigate('/play');
  };

  const handleMenu = () => {
    endGame();
    navigate('/');
  };

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col justify-center px-5 py-6">
      <p className="text-sm font-medium text-accent">Result</p>
      <h1 className="mt-2 text-4xl font-bold tracking-normal">勝負あり！</h1>
      <p className="mt-4 text-2xl font-semibold">{title}</p>

      <div className="mt-8 flex flex-wrap gap-3">
        <Button onClick={handleRematch}>再戦</Button>
        <Button variant="secondary" onClick={handleMenu}>
          メニューへ
        </Button>
        <Button variant="ghost" onClick={() => navigate('/play')}>
          盤面を見る
        </Button>
      </div>
    </main>
  );
};
