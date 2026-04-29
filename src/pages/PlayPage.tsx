import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, Navigate, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Flag, RotateCcw } from 'lucide-react';
import { Board, MoveHistory, ReserveStack } from '@/components/game';
import { Button } from '@/components/primitives';
import { AiClient, selectFallbackMove } from '@/ai';
import { UserAvatar } from '@/components/avatar';
import { engine, type GameState, type Move, type Position } from '@/domain';
import { playBgm, playSfx, stopBgm } from '@/infra';
import { useGameStore, useSessionStore } from '@/stores';

const MODEL_URL = '/model.onnx';

type Selection = { kind: 'reserve'; sizeId: string } | { kind: 'board'; from: Position } | null;

function samePosition(a: Position, b: Position): boolean {
  return a.row === b.row && a.col === b.col;
}

function topOwnerAt(state: GameState, pos: Position) {
  const cell = state.board[pos.row]?.[pos.col];
  const top = cell?.[cell.length - 1];
  return top?.owner;
}

export const PlayPage = () => {
  const navigate = useNavigate();
  const { currentGame, mode, humanSide, cpuDifficulty, applyMove, undo, surrender } =
    useGameStore();
  const session = useSessionStore();
  const [selection, setSelection] = useState<Selection>(null);
  const [message, setMessage] = useState('手駒か盤上の自分の駒を選んでください。');
  const [invalidFlash, setInvalidFlash] = useState(false);
  const [aiReady, setAiReady] = useState(false);
  const prevOutcomeRef = useRef(currentGame?.outcome);
  const aiClientRef = useRef<AiClient | null>(null);

  useEffect(() => {
    playBgm('game');
    return () => stopBgm();
  }, []);

  // Initialize AI worker (CPU mode only).
  useEffect(() => {
    if (mode !== 'cpu') return;
    const client = new AiClient(MODEL_URL);
    aiClientRef.current = client;
    client
      .ready()
      .then(() => setAiReady(true))
      .catch(() => setAiReady(false));
    return () => {
      client.dispose();
      aiClientRef.current = null;
    };
  }, [mode]);

  useEffect(() => {
    const prev = prevOutcomeRef.current;
    const curr = currentGame?.outcome;
    if ((prev === null || prev === undefined) && curr != null) {
      navigate('/result');
    }
    prevOutcomeRef.current = curr;
  }, [currentGame?.outcome, navigate]);

  const isCpuTurn = mode === 'cpu' && humanSide !== null && currentGame?.toMove !== humanSide;

  useEffect(() => {
    if (!isCpuTurn || currentGame?.outcome !== null) return;
    let cancelled = false;

    const runCpuTurn = async () => {
      const client = aiClientRef.current;
      let move: Move;
      if (client && aiReady) {
        try {
          move = await client.requestMove(currentGame, cpuDifficulty ?? 5);
        } catch {
          move = selectFallbackMove(currentGame, cpuDifficulty ?? 5);
        }
      } else {
        await new Promise((r) => setTimeout(r, 700));
        move = selectFallbackMove(currentGame, cpuDifficulty ?? 5);
      }
      if (!cancelled) {
        applyMove(move);
        playSfx('place');
      }
    };

    void runCpuTurn();
    return () => {
      cancelled = true;
      aiClientRef.current?.cancel();
    };
  }, [isCpuTurn, currentGame, cpuDifficulty, applyMove, aiReady]);

  const legalMoves = useMemo(
    () => (currentGame ? engine.legalMoves(currentGame) : []),
    [currentGame],
  );

  if (!currentGame) return <Navigate to="/" replace />;

  const currentName =
    currentGame.toMove === 'P1'
      ? (session.lastP1Name ?? 'Player 1')
      : (session.lastP2Name ?? 'Player 2');
  const currentAvatarSeed =
    currentGame.toMove === 'P1'
      ? (session.lastP1AvatarId ?? 'haru')
      : (session.lastP2AvatarId ?? 'aoi');

  const highlight = legalMoves
    .filter((move) => {
      if (selection?.kind === 'reserve') {
        return move.kind === 'placeFromReserve' && move.sizeId === selection.sizeId;
      }
      if (selection?.kind === 'board') {
        return move.kind === 'moveOnBoard' && samePosition(move.from, selection.from);
      }
      return false;
    })
    .map((move) => move.to);

  const tryApply = (move: Move) => {
    if (applyMove(move)) {
      playSfx('place');
      setSelection(null);
      setMessage('手を進めました。');
      return;
    }
    playSfx('invalid');
    setMessage('その手は無理です！');
    setInvalidFlash(true);
    setTimeout(() => setInvalidFlash(false), 600);
  };

  const handleCellClick = (pos: Position) => {
    if (isCpuTurn) return;

    if (selection?.kind === 'reserve') {
      tryApply({
        kind: 'placeFromReserve',
        player: currentGame.toMove,
        sizeId: selection.sizeId,
        to: pos,
      });
      return;
    }

    if (selection?.kind === 'board') {
      if (samePosition(selection.from, pos)) {
        setSelection(null);
        setMessage('選択を解除しました。');
        return;
      }
      tryApply({
        kind: 'moveOnBoard',
        player: currentGame.toMove,
        from: selection.from,
        to: pos,
      });
      return;
    }

    if (topOwnerAt(currentGame, pos) === currentGame.toMove) {
      playSfx('pickup');
      setSelection({ kind: 'board', from: pos });
      setMessage('移動先を選んでください。');
    } else {
      setMessage('まず手駒、または見えている自分の駒を選んでください。');
    }
  };

  const handleReserveSelect = (sizeId: string) => {
    if (isCpuTurn) return;
    playSfx('pickup');
    setSelection({ kind: 'reserve', sizeId });
    setMessage('配置先を選んでください。');
  };

  return (
    <main className="mx-auto min-h-screen w-full max-w-6xl px-4 py-4">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <UserAvatar seed={currentAvatarSeed} size={48} label={currentName} />
          <div>
            <p className="text-sm font-medium text-accent">
              {mode === 'cpu' ? 'CPU対戦' : 'ローカル対戦'}
            </p>
            <h1 className="text-2xl font-bold tracking-normal">{currentName} の手番</h1>
            <p role="status" className="mt-1 text-sm text-muted">
              {isCpuTurn ? (aiReady ? 'AI が考えています…' : 'AI モデル読み込み中…') : message}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="ghost" size="sm" iconLeft={<ArrowLeft className="h-4 w-4" />}>
            <Link to="/">メニュー</Link>
          </Button>
          <Button
            variant="secondary"
            size="sm"
            iconLeft={<RotateCcw className="h-4 w-4" />}
            onClick={() => {
              playSfx('undo');
              undo();
              setSelection(null);
            }}
          >
            待った
          </Button>
          <Button
            variant="danger"
            size="sm"
            iconLeft={<Flag className="h-4 w-4" />}
            onClick={surrender}
          >
            投了
          </Button>
        </div>
      </header>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <section className="grid gap-4">
          <div className="grid gap-3 md:grid-cols-2">
            <ReserveStack
              owner="P1"
              reserve={currentGame.reserves.P1}
              pieceSizes={currentGame.rules.pieceSizes}
              selected={
                currentGame.toMove === 'P1' && selection?.kind === 'reserve' ? selection : null
              }
              {...(currentGame.toMove === 'P1' ? { onSelect: handleReserveSelect } : {})}
              disabled={isCpuTurn || currentGame.toMove !== 'P1'}
              aria-label="P1 の手駒"
            />
            <ReserveStack
              owner="P2"
              reserve={currentGame.reserves.P2}
              pieceSizes={currentGame.rules.pieceSizes}
              selected={
                currentGame.toMove === 'P2' && selection?.kind === 'reserve' ? selection : null
              }
              {...(currentGame.toMove === 'P2' ? { onSelect: handleReserveSelect } : {})}
              disabled={isCpuTurn || currentGame.toMove !== 'P2'}
              aria-label="P2 の手駒"
            />
          </div>

          <motion.div
            className="relative mx-auto aspect-square w-full max-w-[min(78vh,38rem)]"
            animate={invalidFlash ? { x: [-6, 6, -6, 6, -3, 3, 0] } : { x: 0 }}
            transition={{ duration: 0.4, ease: 'easeInOut' }}
          >
            <Board
              state={currentGame}
              onCellClick={handleCellClick}
              highlight={highlight}
              disabled={isCpuTurn}
              aria-label="対局盤"
            />
            <AnimatePresence>
              {invalidFlash && (
                <motion.div
                  key="invalid-overlay"
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.85 }}
                  transition={{ duration: 0.15 }}
                  className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-lg"
                >
                  <span className="rounded-full bg-red-600/90 px-5 py-2 text-lg font-bold text-white shadow-lg">
                    その手は無理です！
                  </span>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </section>

        <aside className="grid content-start gap-3">
          <div className="rounded-lg border border-border p-3">
            <p className="text-sm font-medium">局面</p>
            <dl className="mt-2 grid grid-cols-2 gap-2 text-sm text-muted">
              <dt>手数</dt>
              <dd>{currentGame.ply}</dd>
              <dt>盤面</dt>
              <dd>
                {currentGame.rules.boardSize}x{currentGame.rules.boardSize}
              </dd>
            </dl>
          </div>
          <MoveHistory history={currentGame.history} pieceSizes={currentGame.rules.pieceSizes} />
        </aside>
      </div>
    </main>
  );
};
