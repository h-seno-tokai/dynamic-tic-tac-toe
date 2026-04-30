import { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, Navigate, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Flag, RotateCcw } from 'lucide-react';
import { Board, MoveHistory, ReserveStack } from '@/components/game';
import { Button } from '@/components/primitives';
import { UserAvatar } from '@/components/avatar';
import { AiClient, selectFallbackMove } from '@/ai';
import { engine, type GameState, type Move, type Position } from '@/domain';
import { playBgm, playSfx, stopBgm } from '@/infra';
import { useGameStore, useSessionStore, useStatsStore } from '@/stores';

const MODEL_URL = `${import.meta.env.BASE_URL}model.onnx`;

export type Selection =
  | { kind: 'reserve'; sizeId: string }
  | { kind: 'board'; from: Position }
  | null;

/**
 * Possible UI events that can change selection.
 * - 'clickReserve': user clicked a reserve pile of their own.
 * - 'clickOwnBoardPiece': user clicked a visible piece of their own on the board.
 */
export type SelectionInput =
  | { kind: 'clickReserve'; sizeId: string }
  | { kind: 'clickOwnBoardPiece'; pos: Position };

/**
 * Pure reducer used for piece-(re)selection UX. Given the current selection
 * and the user's click on a *selectable* piece (reserve pile or own board
 * piece), compute the next selection.
 *
 * Semantics:
 * - Clicking the same target that is already selected clears the selection.
 * - Clicking a different selectable target replaces the selection.
 *
 * Note: clicks on empty/enemy cells (i.e. potential placement/movement
 * targets) are NOT handled here — they are dispatched as moves by the page
 * directly. This helper only decides selection changes.
 */
export function computeNextSelection(current: Selection, input: SelectionInput): Selection {
  if (input.kind === 'clickReserve') {
    if (current?.kind === 'reserve' && current.sizeId === input.sizeId) {
      return null; // toggle off
    }
    return { kind: 'reserve', sizeId: input.sizeId };
  }
  // clickOwnBoardPiece
  if (current?.kind === 'board' && samePosition(current.from, input.pos)) {
    return null; // toggle off
  }
  return { kind: 'board', from: input.pos };
}

function samePosition(a: Position, b: Position): boolean {
  return a.row === b.row && a.col === b.col;
}

function topOwnerAt(state: GameState, pos: Position) {
  const cell = state.board[pos.row]?.[pos.col];
  const top = cell?.[cell.length - 1];
  return top?.owner;
}

export const PlayPage = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const {
    currentGame,
    mode,
    humanSide,
    cpuDifficulty,
    applyMove,
    undo,
    surrender,
    startNewGame,
    endGame,
  } = useGameStore();
  const session = useSessionStore();
  const recordGame = useStatsStore((s) => s.recordGame);
  const [selection, setSelection] = useState<Selection>(null);
  const [message, setMessage] = useState('');
  const [invalidFlash, setInvalidFlash] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [aiStatus, setAiStatus] = useState<'loading' | 'ready' | 'failed'>('loading');
  const prevOutcomeRef = useRef(currentGame?.outcome);
  const aiClientRef = useRef<AiClient | null>(null);

  useEffect(() => {
    playBgm('game');
    return () => stopBgm();
  }, []);

  useEffect(() => {
    if (mode !== 'cpu') return;
    setAiStatus('loading');
    const client = new AiClient(MODEL_URL);
    aiClientRef.current = client;
    client
      .ready()
      .then(() => setAiStatus('ready'))
      .catch(() => setAiStatus('failed'));
    return () => {
      client.dispose();
      aiClientRef.current = null;
    };
  }, [mode]);

  useEffect(() => {
    const prev = prevOutcomeRef.current;
    const curr = currentGame?.outcome;
    if ((prev === null || prev === undefined) && curr != null) {
      setShowResult(true);
      if (mode === 'cpu' && humanSide != null) {
        const outcome = curr === 'draw' ? 'draw' : curr === humanSide ? 'win' : 'loss';
        // CPU mode SFX: win → win, loss → lose, draw → fanfare (neutral).
        if (outcome === 'win') playSfx('win');
        else if (outcome === 'loss') playSfx('lose');
        else playSfx('fanfare');
        if (cpuDifficulty != null) {
          recordGame(cpuDifficulty, outcome);
        }
      } else {
        // Local 2P: keep existing fanfare on every outcome.
        playSfx('fanfare');
      }
    }
    prevOutcomeRef.current = curr;
  }, [currentGame?.outcome]);

  const isCpuTurn = mode === 'cpu' && humanSide !== null && currentGame?.toMove !== humanSide;

  useEffect(() => {
    if (!isCpuTurn || currentGame?.outcome !== null) return;
    let cancelled = false;

    const runCpuTurn = async () => {
      const client = aiClientRef.current;
      let move: Move;
      if (client && aiStatus === 'ready') {
        try {
          move = await client.requestMove(currentGame, cpuDifficulty ?? 5);
        } catch {
          // Worker rejected the request. For 3x3 this never depends on
          // ONNX (alpha-beta solver), but for 4x4 a model-load failure
          // surfaces here. Flip to 'failed' so the banner explains the
          // fallback the user is now seeing.
          if (currentGame.rules.boardSize > 3) {
            setAiStatus('failed');
          }
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
  }, [isCpuTurn, currentGame, cpuDifficulty, applyMove, aiStatus]);

  const legalMoves = useMemo(
    () => (currentGame ? engine.legalMoves(currentGame) : []),
    [currentGame],
  );

  if (!currentGame) return <Navigate to="/" replace />;

  const p1Name = session.lastP1Name ?? 'Player 1';
  const p2Name = session.lastP2Name ?? 'Player 2';
  const p1AvatarSeed = session.lastP1AvatarId ?? 'haru';
  const p2AvatarSeed = session.lastP2AvatarId ?? 'aoi';

  // In CPU mode the CPU side always shows as "AI", never reads its session slot.
  const cpuSide = mode === 'cpu' && humanSide != null ? (humanSide === 'P1' ? 'P2' : 'P1') : null;

  const resolveName = (player: 'P1' | 'P2') =>
    cpuSide === player ? 'AI' : player === 'P1' ? p1Name : p2Name;
  const resolveAvatar = (player: 'P1' | 'P2') =>
    cpuSide === player ? 'cpu-robot' : player === 'P1' ? p1AvatarSeed : p2AvatarSeed;

  const currentName = resolveName(currentGame.toMove);
  const currentAvatarSeed = resolveAvatar(currentGame.toMove);

  const outcome = currentGame.outcome;
  const winnerName = outcome === 'draw' || outcome == null ? null : resolveName(outcome);
  const winnerAvatarSeed = outcome === 'draw' || outcome == null ? null : resolveAvatar(outcome);

  // For CPU mode we show a different overlay. Compute the user-perspective
  // result here so the JSX below stays simple.
  const cpuResult: 'win' | 'loss' | 'draw' | null =
    mode === 'cpu' && humanSide != null && outcome != null
      ? outcome === 'draw'
        ? 'draw'
        : outcome === humanSide
          ? 'win'
          : 'loss'
      : null;

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
      setMessage(t('play.moved'));
      return;
    }
    playSfx('invalid');
    setMessage(t('play.invalid'));
    setInvalidFlash(true);
    setTimeout(() => setInvalidFlash(false), 600);
  };

  const handleCellClick = (pos: Position) => {
    if (isCpuTurn) return;

    const isOwnPiece = topOwnerAt(currentGame, pos) === currentGame.toMove;

    // Clicking a selectable own board piece: toggle/switch selection.
    // This applies regardless of current selection (reserve / board / none).
    if (isOwnPiece) {
      const next = computeNextSelection(selection, { kind: 'clickOwnBoardPiece', pos });
      setSelection(next);
      if (next === null) {
        playSfx('undo');
        setMessage(t('play.deselect'));
      } else {
        playSfx('pickup');
        setMessage(t('play.moveMsg'));
      }
      return;
    }

    // Otherwise, the click is treated as a move/place target.
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
      tryApply({
        kind: 'moveOnBoard',
        player: currentGame.toMove,
        from: selection.from,
        to: pos,
      });
      return;
    }

    setMessage(t('play.selectFirst'));
  };

  const handleReserveSelect = (sizeId: string) => {
    if (isCpuTurn) return;
    const next = computeNextSelection(selection, { kind: 'clickReserve', sizeId });
    setSelection(next);
    if (next === null) {
      playSfx('undo');
      setMessage(t('play.deselect'));
    } else {
      playSfx('pickup');
      setMessage(t('play.placeMsg'));
    }
  };

  const handleRematch = () => {
    if (mode === 'cpu') {
      startNewGame(currentGame.rules, 'cpu', {
        difficulty: cpuDifficulty ?? 1,
        humanSide: humanSide ?? 'P1',
      });
    } else {
      startNewGame(currentGame.rules, 'local-2p');
    }
    setShowResult(false);
    setSelection(null);
    setMessage(t('play.chooseMsg'));
  };

  const handleMenu = () => {
    endGame();
    navigate('/');
  };

  return (
    <main className="mx-auto min-h-screen w-full max-w-6xl px-4 py-4">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <UserAvatar seed={currentAvatarSeed} size={48} label={currentName} />
          <div>
            <p className="flex flex-wrap items-center gap-x-2 text-sm font-medium text-accent">
              <span>{mode === 'cpu' ? t('play.cpuMode') : t('play.localMode')}</span>
              {mode === 'cpu' && cpuDifficulty != null && (
                <span className="text-muted">· {t(`difficulty.level${cpuDifficulty}`)}</span>
              )}
              {mode === 'cpu' && (
                <span
                  className={
                    'inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ' +
                    (aiStatus === 'ready'
                      ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
                      : aiStatus === 'failed'
                        ? 'bg-red-500/15 text-red-700 dark:text-red-300'
                        : 'bg-amber-500/15 text-amber-700 dark:text-amber-300')
                  }
                  aria-live="polite"
                >
                  <span
                    aria-hidden="true"
                    className={
                      'h-2 w-2 rounded-full ' +
                      (aiStatus === 'ready'
                        ? 'bg-emerald-500'
                        : aiStatus === 'failed'
                          ? 'bg-red-500'
                          : 'animate-pulse bg-amber-500')
                    }
                  />
                  {aiStatus === 'ready'
                    ? t('play.aiStatusReady')
                    : aiStatus === 'failed'
                      ? t('play.aiStatusFailed')
                      : t('play.aiStatusLoading')}
                </span>
              )}
            </p>
            <h1 className="text-2xl font-bold tracking-normal">
              {t('play.turn', { name: currentName })}
            </h1>
            <p role="status" className="mt-1 text-sm text-muted">
              {isCpuTurn
                ? aiStatus === 'ready'
                  ? t('play.aiThinking')
                  : aiStatus === 'failed'
                    ? t('play.aiFallbackThinking')
                    : t('play.aiLoading')
                : message || t('play.chooseMsg')}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="ghost" size="sm" iconLeft={<ArrowLeft className="h-4 w-4" />}>
            <Link to="/">{t('play.menu')}</Link>
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
            {t('play.undo')}
          </Button>
          <Button
            variant="danger"
            size="sm"
            iconLeft={<Flag className="h-4 w-4" />}
            onClick={surrender}
          >
            {t('play.surrender')}
          </Button>
        </div>
      </header>

      {mode === 'cpu' && aiStatus === 'failed' && (
        <div
          role="alert"
          className="mb-4 rounded-md border border-amber-500/60 bg-amber-500/10 px-3 py-2 text-sm text-amber-700 dark:text-amber-200"
        >
          {t('play.aiFallbackBanner')}
        </div>
      )}

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
              aria-label={t('play.p1reserve')}
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
              aria-label={t('play.p2reserve')}
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
            <p className="text-sm font-medium">{t('play.position')}</p>
            <dl className="mt-2 grid grid-cols-2 gap-2 text-sm text-muted">
              <dt>{t('play.ply')}</dt>
              <dd>{currentGame.ply}</dd>
              <dt>{t('play.board')}</dt>
              <dd>
                {currentGame.rules.boardSize}x{currentGame.rules.boardSize}
              </dd>
            </dl>
          </div>
          <MoveHistory history={currentGame.history} pieceSizes={currentGame.rules.pieceSizes} />
        </aside>
      </div>

      {/* Result modal overlay */}
      <AnimatePresence>
        {showResult && outcome != null && (
          <motion.div
            key="result-backdrop"
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <motion.div
              className="mx-4 w-full max-w-sm rounded-2xl border border-border bg-bg p-8 shadow-2xl"
              initial={{ scale: 0.8, opacity: 0, y: 24 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.8, opacity: 0, y: 24 }}
              transition={{ type: 'spring', stiffness: 260, damping: 22 }}
            >
              <p className="text-sm font-medium text-accent">{t('result.label')}</p>
              <motion.h2
                className="mt-1 text-4xl font-bold tracking-normal"
                initial={{ scale: 0.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: 'spring', stiffness: 200, damping: 15, delay: 0.1 }}
              >
                {cpuResult === 'win'
                  ? t('result.cpuWinTitle')
                  : cpuResult === 'loss'
                    ? t('result.cpuLossTitle')
                    : cpuResult === 'draw'
                      ? t('result.cpuDrawTitle')
                      : t('result.title')}
              </motion.h2>

              <motion.div
                className="mt-5"
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3, duration: 0.3 }}
              >
                {cpuResult === 'win' ? (
                  <p className="text-lg text-muted">{t('result.cpuWinSubtitle')}</p>
                ) : cpuResult === 'loss' ? (
                  <p className="text-lg text-muted">{t('result.cpuLossSubtitle')}</p>
                ) : cpuResult === 'draw' ? (
                  <p className="text-lg text-muted">{t('result.cpuDrawSubtitle')}</p>
                ) : outcome === 'draw' ? (
                  <p className="text-2xl font-semibold">{t('result.draw')}</p>
                ) : (
                  <div className="flex items-center gap-4">
                    {winnerAvatarSeed && winnerName !== 'AI' && (
                      <UserAvatar seed={winnerAvatarSeed} size={64} label={winnerName ?? ''} />
                    )}
                    <div>
                      <p className="text-2xl font-bold">
                        {winnerName === 'AI' ? t('result.aiWins') : winnerName}
                      </p>
                      {winnerName !== 'AI' && (
                        <p className="text-sm text-muted">{t('result.wins')}</p>
                      )}
                    </div>
                  </div>
                )}
              </motion.div>

              <motion.div
                className="mt-8 flex flex-wrap gap-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.45 }}
              >
                <Button onClick={handleRematch}>{t('result.rematch')}</Button>
                <Button variant="secondary" onClick={handleMenu}>
                  {t('result.backToMenu')}
                </Button>
                <Button variant="ghost" onClick={() => setShowResult(false)}>
                  {t('result.viewBoard')}
                </Button>
              </motion.div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  );
};
