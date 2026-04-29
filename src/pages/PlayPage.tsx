import { useEffect, useMemo, useState } from 'react';
import { Link, Navigate, useNavigate } from 'react-router-dom';
import { ArrowLeft, Flag, RotateCcw } from 'lucide-react';
import { Board, MoveHistory, ReserveStack } from '@/components/game';
import { Button } from '@/components/primitives';
import { engine, type GameState, type Move, type Position } from '@/domain';
import { useGameStore, useSessionStore } from '@/stores';

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
  const { currentGame, mode, humanSide, applyMove, undo, surrender } = useGameStore();
  const session = useSessionStore();
  const [selection, setSelection] = useState<Selection>(null);
  const [message, setMessage] = useState('手駒か盤上の自分の駒を選んでください。');

  useEffect(() => {
    if (currentGame?.outcome !== null && currentGame?.outcome !== undefined) {
      navigate('/result');
    }
  }, [currentGame?.outcome, navigate]);

  const legalMoves = useMemo(
    () => (currentGame ? engine.legalMoves(currentGame) : []),
    [currentGame],
  );

  if (!currentGame) return <Navigate to="/" replace />;

  const isCpuTurn = mode === 'cpu' && humanSide !== null && currentGame.toMove !== humanSide;
  const currentName =
    currentGame.toMove === 'P1'
      ? (session.lastP1Name ?? 'Player 1')
      : (session.lastP2Name ?? 'Player 2');

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
      setSelection(null);
      setMessage('手を進めました。');
      return;
    }
    setMessage('その手は合法手ではありません。');
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
      setSelection({ kind: 'board', from: pos });
      setMessage('移動先を選んでください。');
    } else {
      setMessage('まず手駒、または見えている自分の駒を選んでください。');
    }
  };

  const handleReserveSelect = (sizeId: string) => {
    if (isCpuTurn) return;
    setSelection({ kind: 'reserve', sizeId });
    setMessage('配置先を選んでください。');
  };

  return (
    <main className="mx-auto min-h-screen w-full max-w-6xl px-4 py-4">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-accent">
            {mode === 'cpu' ? 'CPU対戦' : 'ローカル対戦'}
          </p>
          <h1 className="text-2xl font-bold tracking-normal">{currentName} の手番</h1>
          <p role="status" className="mt-1 text-sm text-muted">
            {isCpuTurn ? 'CPU接続待ちです。暫定CPUを接続中です。' : message}
          </p>
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

          <div className="mx-auto aspect-square w-full max-w-[min(78vh,38rem)]">
            <Board
              state={currentGame}
              onCellClick={handleCellClick}
              highlight={highlight}
              disabled={isCpuTurn}
              aria-label="対局盤"
            />
          </div>
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
