import { useEffect } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Button } from '@/components/primitives';
import { playSfx } from '@/infra';
import { useGameStore } from '@/stores';

export const ResultPage = () => {
  const navigate = useNavigate();
  const { currentGame, mode, cpuDifficulty, humanSide, startNewGame, endGame } = useGameStore();

  useEffect(() => {
    playSfx('fanfare');
  }, []);

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
      <motion.h1
        className="mt-2 text-4xl font-bold tracking-normal"
        initial={{ scale: 0.5, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 200, damping: 15 }}
      >
        勝負あり！
      </motion.h1>
      <motion.p
        className="mt-4 text-2xl font-semibold"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25, duration: 0.35 }}
      >
        {title}
      </motion.p>

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
