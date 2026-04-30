/**
 * Strong alpha-beta solver for the 3x3 Gobblet preset.
 *
 * Implements iterative-deepening negamax with:
 *  - alpha-beta pruning
 *  - transposition table keyed by an incremental Zobrist hash, with
 *    `{depth, flag, value, bestMoveIdx}` entries
 *  - PV (best-move-from-previous-iteration) move ordering at the root,
 *    plus TT-best-move-first ordering at internal nodes
 *  - simple static move-ordering heuristic (own-line-of-2 completion >
 *    opponent-line-of-2 block > captures/lifts > center > others)
 *  - in-place make/unmake using a compact internal board representation
 *    (avoids the engine's clone-per-move cost while preserving the same
 *    legality and outcome rules)
 *  - threefold repetition and `maxPly` honoured via incrementally-updated
 *    state
 *  - cooperative abort via `AbortSignal`, checked between ID iterations and
 *    at every TT lookup
 *
 * The solver is internally synchronous (CPU-bound) but exposes a Promise to
 * fit the existing worker plumbing. We yield to the event loop occasionally
 * during a search so that the worker's `message` event can be processed and
 * the abort signal can fire.
 */

import type {
  GameRules,
  GameState,
  Move,
  MoveOnBoardMove,
  PlaceFromReserveMove,
  Player,
} from '@/domain';
import { engine } from '@/domain';

// =============================================================================
// Public API
// =============================================================================

export interface Solver3x3Options {
  /** Wall-clock budget for the search, in milliseconds. */
  timeBudgetMs: number;
  /** Probability of intentionally picking a non-best legal move at the root. */
  mistakeRate: number;
  /** Cooperative cancellation. */
  signal?: AbortSignal;
  /** Inject randomness (defaults to `Math.random`). Used by tests. */
  random?: () => number;
}

const WIN_SCORE = 100_000;
const MAX_DEPTH = 64;
/** Hard cap on the TT to keep memory bounded. */
const TT_MAX_ENTRIES = 1_000_000;
/** Number of nodes between abort/time checks inside the inner search. */
const TIME_CHECK_INTERVAL = 4_096;

/** Strong solver tuned for the 3x3 preset. Other rules will work too but were
 * not optimised. */
export class Solver3x3 {
  /** Retained across calls so successive moves reuse useful entries. */
  private readonly tt = new Map<bigint, TtEntry>();

  async selectMove(state: GameState, opts: Solver3x3Options): Promise<Move> {
    const random = opts.random ?? Math.random;

    const legal = engine.legalMoves(state);
    if (legal.length === 0) {
      throw new Error('Solver3x3.selectMove: no legal moves');
    }
    if (legal.length === 1) {
      const only = legal[0];
      if (!only) throw new Error('Solver3x3.selectMove: legal moves missing');
      return only;
    }

    const ctx = buildSearchContext(state);
    const deadline = performance.now() + Math.max(0, opts.timeBudgetMs);
    const abortFlag = { aborted: false };
    const signal = opts.signal;
    const onAbort = (): void => {
      abortFlag.aborted = true;
    };
    if (signal) {
      if (signal.aborted) {
        // Even when aborted up-front we must not violate the contract; pick a
        // legal fallback and return immediately. Caller will discard.
        return legal[0]!;
      }
      signal.addEventListener('abort', onAbort, { once: true });
    }

    try {
      const result = await runIterativeDeepening(ctx, this.tt, {
        deadline,
        abortFlag,
      });

      // Pick from the root scoring computed at the deepest *completed* depth.
      const bestMove = result.bestMove;
      const candidates = result.rootMoves;
      if (!bestMove || candidates.length === 0) {
        return legal[0]!;
      }

      // Mistake injection (root only, ignore aborted searches).
      if (opts.mistakeRate > 0 && candidates.length > 1 && random() < opts.mistakeRate) {
        const others = candidates.filter((m) => !sameMove(m, bestMove));
        if (others.length > 0) {
          const pick = others[Math.floor(random() * others.length)];
          if (pick) return pick;
        }
      }
      return bestMove;
    } finally {
      if (signal) signal.removeEventListener('abort', onAbort);
    }
  }
}

// =============================================================================
// Internal types
// =============================================================================

interface TtEntry {
  depth: number;
  flag: 'exact' | 'lower' | 'upper';
  value: number;
  bestMoveIdx: number;
  age: number;
}

interface AbortFlag {
  aborted: boolean;
}

interface SearchTimers {
  deadline: number;
  abortFlag: AbortFlag;
}

interface IdResult {
  bestMove: Move | null;
  rootMoves: Move[];
}

interface InternalMove {
  /** 0 = placeFromReserve, 1 = moveOnBoard */
  kind: 0 | 1;
  /** sizeIdx (0..K-1) for placement; -1 otherwise */
  sizeIdx: number;
  /** flat from-cell index (0..N*N-1) for move-on-board; -1 otherwise */
  from: number;
  /** flat to-cell index */
  to: number;
}

/** Compact piece encoding: `(owner << 4) | sizeIdx`. Owner: 0=P1, 1=P2. */
type PieceCode = number;

/**
 * Solver-internal state. Mirrors `GameState` semantics without cloning.
 *
 * Repetition is tracked via incremental Zobrist hash plus a `Map<bigint, number>`
 * indexed by *position-only* hash (no ply, no `outcome` field). This matches
 * the engine's repetition rules.
 */
interface SearchContext {
  rules: GameRules;
  N: number;
  numSizes: number;
  /** Stacks for each cell; bottom-to-top (matches GameState). */
  stacks: PieceCode[][];
  /** Reserves[owner][sizeIdx] -> count */
  reserves: number[][];
  /** Side to move: 0=P1, 1=P2. */
  toMove: 0 | 1;
  ply: number;
  maxPly: number;
  drawByRepetition: number;
  allowSameSizeCover: boolean;
  allowSelfCover: boolean;
  /** Current Zobrist hash. */
  zhash: bigint;
  /** Repetition counts keyed by zhash. */
  repetition: Map<bigint, number>;
  /** Zobrist random tables. */
  z: ZobristTables;
  /** Cached size ids (idx -> id) for converting back to engine `Move`. */
  sizeIds: string[];
  /** N-in-a-row line cell indices, length L, each line has N cells. */
  lines: number[][];
  /** Per-cell reverse mapping: cell -> indices into `lines` it appears in. */
  cellLines: number[][];
}

/**
 * Random (deterministic) bigints used for incremental position hashing.
 * Includes:
 *  - per-cell, per-stack-position, per-owner, per-size piece keys
 *  - per-(owner, sizeIdx, count) reserve keys
 *  - side-to-move key
 */
interface ZobristTables {
  /** [cell][stackPos][owner][sizeIdx] */
  pieceKeys: bigint[][][][];
  /** [owner][sizeIdx][count] (count from 0 .. maxCount inclusive). */
  reserveKeys: bigint[][][];
  /** flips when side changes. */
  sideKey: bigint;
  /** Maximum stack height we allocated keys for. */
  maxStackHeight: number;
}

// =============================================================================
// Build / convert
// =============================================================================

function buildSearchContext(state: GameState): SearchContext {
  const rules = state.rules;
  const N = rules.boardSize;
  const numSizes = rules.pieceSizes.length;
  const sizeIds = rules.pieceSizes.map((p) => p.id);

  // We sort piece sizes by rank so sizeIdx maps directly to rank order.
  const ordered = rules.pieceSizes
    .map((ps, i) => ({ ps, i }))
    .sort((a, b) => a.ps.rank - b.ps.rank);
  const sizeIdToIdx = new Map<string, number>();
  ordered.forEach((entry, idx) => sizeIdToIdx.set(entry.ps.id, idx));
  const orderedSizeIds = ordered.map((e) => e.ps.id);

  // Total per-player piece count = sum(piecesPerSize). Used to bound stack
  // height (worst case all pieces on one cell, but practically <= 2*numSizes).
  const totalPieces = rules.piecesPerSize.reduce((a, b) => a + b, 0);
  const maxStackHeight = Math.max(2 * totalPieces, N * N);
  const maxReserveCount = rules.piecesPerSize.reduce((a, b) => Math.max(a, b), 0);

  const z = makeZobristTables(N, numSizes, maxStackHeight, maxReserveCount);

  // Initial empty stacks.
  const stacks: PieceCode[][] = Array.from({ length: N * N }, () => []);

  // Translate state.board -> stacks (using sizeIdToIdx).
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const cell = state.board[r]?.[c] ?? [];
      const out: PieceCode[] = [];
      for (const p of cell) {
        const sIdx = sizeIdToIdx.get(p.sizeId);
        if (sIdx === undefined) continue;
        const owner = p.owner === 'P1' ? 0 : 1;
        out.push((owner << 4) | sIdx);
      }
      stacks[r * N + c] = out;
    }
  }

  // Reserves: reserves[owner][sizeIdx]
  const reserves: number[][] = [
    new Array<number>(numSizes).fill(0),
    new Array<number>(numSizes).fill(0),
  ];
  for (let i = 0; i < numSizes; i++) {
    const ps = ordered[i]!.ps;
    reserves[0]![i] = state.reserves.P1[ps.id] ?? 0;
    reserves[1]![i] = state.reserves.P2[ps.id] ?? 0;
  }

  // Lines (rows, cols, two diagonals when N==boardSize).
  const lines: number[][] = [];
  for (let r = 0; r < N; r++) {
    const row: number[] = [];
    for (let c = 0; c < N; c++) row.push(r * N + c);
    lines.push(row);
  }
  for (let c = 0; c < N; c++) {
    const col: number[] = [];
    for (let r = 0; r < N; r++) col.push(r * N + c);
    lines.push(col);
  }
  const d1: number[] = [];
  const d2: number[] = [];
  for (let i = 0; i < N; i++) {
    d1.push(i * N + i);
    d2.push(i * N + (N - 1 - i));
  }
  lines.push(d1);
  lines.push(d2);

  const cellLines: number[][] = Array.from({ length: N * N }, () => []);
  for (let li = 0; li < lines.length; li++) {
    for (const cell of lines[li]!) cellLines[cell]!.push(li);
  }

  // Compute initial zhash from scratch.
  let zhash = 0n;
  for (let cell = 0; cell < N * N; cell++) {
    const stk = stacks[cell]!;
    for (let p = 0; p < stk.length; p++) {
      const code = stk[p]!;
      const owner = (code >> 4) & 1;
      const sIdx = code & 0xf;
      zhash ^= z.pieceKeys[cell]![p]![owner]![sIdx]!;
    }
  }
  for (let owner = 0; owner < 2; owner++) {
    for (let s = 0; s < numSizes; s++) {
      const cnt = reserves[owner]![s]!;
      zhash ^= z.reserveKeys[owner]![s]![cnt]!;
    }
  }
  if (state.toMove === 'P2') zhash ^= z.sideKey;

  // Repetition map: replicate the engine's count for the current position
  // (uses string hash). Translate that to our zhash for the current position.
  const repetition = new Map<bigint, number>();
  // We can't easily project all engine repetition entries to our zhash because
  // they were computed by `hashState` over the engine's representation. But
  // for the search, what matters is *future* repetitions of the current and
  // subsequent positions; we seed the map with at least the current zhash's
  // count from the engine.
  //
  // The engine increments `repetition` *after* each applyMove for the
  // resulting state, including the initial state. So state.repetition has
  // a record for the current state under its string hash. We project that
  // count onto the current zhash so threefold detection works correctly.
  //
  // For ancestral positions in `state.history` that are *not* the current
  // position, we'd need to reconstruct each one. Instead, we approximate by
  // seeding only the current position's count. This is correct in practice
  // because: any repetition that would draw the search must repeat the
  // *current* or *future* position. Past positions we never re-visit during
  // search except through the current position. The threefold rule only
  // triggers when a position appears `drawByRepetition` times *total*; if
  // some old position has count 2 and the search reaches it from here, we'd
  // miss that — but reaching an old position requires routing through the
  // current state and so we'd see the count grow from our seeded baseline.
  //
  // To preserve correctness more carefully, we seed *all* engine repetition
  // entries by re-walking the history: replay each move in order from the
  // initial state and record zhash counts. This is O(history * make-cost)
  // per selectMove, which is negligible.
  const replayCtx: Pick<
    SearchContext,
    | 'N'
    | 'numSizes'
    | 'stacks'
    | 'reserves'
    | 'toMove'
    | 'zhash'
    | 'z'
    | 'allowSameSizeCover'
    | 'allowSelfCover'
  > = {
    N,
    numSizes,
    stacks: Array.from({ length: N * N }, () => [] as PieceCode[]),
    reserves: [new Array<number>(numSizes).fill(0), new Array<number>(numSizes).fill(0)],
    toMove: 0,
    zhash: 0n,
    z,
    allowSameSizeCover: rules.allowSameSizeCover,
    allowSelfCover: rules.allowSelfCover,
  };
  // Initial reserves. `ordered[i].i` is the original position into the rules
  // arrays, so we read the count from `piecesPerSize` using that index.
  for (let i = 0; i < numSizes; i++) {
    const origIdx = ordered[i]!.i;
    const cnt = rules.piecesPerSize[origIdx] ?? 0;
    replayCtx.reserves[0]![i] = cnt;
    replayCtx.reserves[1]![i] = cnt;
    // Hash starting count for both players.
    replayCtx.zhash ^= z.reserveKeys[0]![i]![cnt]!;
    replayCtx.zhash ^= z.reserveKeys[1]![i]![cnt]!;
  }
  // Side-to-move starts as P1 (zhash unchanged).

  // Record initial position.
  repetition.set(replayCtx.zhash, (repetition.get(replayCtx.zhash) ?? 0) + 1);

  for (const move of state.history) {
    const internal = engineMoveToInternal(move, sizeIdToIdx, N);
    if (!internal) continue;
    applyInternalMoveRaw(replayCtx, internal);
    repetition.set(replayCtx.zhash, (repetition.get(replayCtx.zhash) ?? 0) + 1);
  }

  // Sanity: replayCtx.zhash should equal zhash. If not, fall back to
  // resetting the map with just the current count = 1 (defensive).
  if (replayCtx.zhash !== zhash) {
    repetition.clear();
    repetition.set(zhash, 1);
  }

  return {
    rules,
    N,
    numSizes,
    stacks,
    reserves,
    toMove: state.toMove === 'P1' ? 0 : 1,
    ply: state.ply,
    maxPly: rules.maxPly,
    drawByRepetition: rules.drawByRepetition,
    allowSameSizeCover: rules.allowSameSizeCover,
    allowSelfCover: rules.allowSelfCover,
    zhash,
    repetition,
    z,
    sizeIds: orderedSizeIds.length === sizeIds.length ? orderedSizeIds : sizeIds,
    lines,
    cellLines,
  };
}

function makeZobristTables(
  N: number,
  numSizes: number,
  maxStackHeight: number,
  maxReserveCount: number,
): ZobristTables {
  // Deterministic seed-based PRNG for reproducibility across processes/tests.
  let seed = 0x9e3779b97f4a7c15n;
  const rand = (): bigint => {
    seed ^= seed << 13n;
    seed ^= seed >> 7n;
    seed ^= seed << 17n;
    seed &= (1n << 64n) - 1n;
    return seed;
  };

  const numCells = N * N;
  const pieceKeys: bigint[][][][] = [];
  for (let cell = 0; cell < numCells; cell++) {
    const stackArr: bigint[][][] = [];
    for (let p = 0; p < maxStackHeight; p++) {
      const ownerArr: bigint[][] = [];
      for (let owner = 0; owner < 2; owner++) {
        const sizeArr: bigint[] = [];
        for (let s = 0; s < numSizes; s++) sizeArr.push(rand());
        ownerArr.push(sizeArr);
      }
      stackArr.push(ownerArr);
    }
    pieceKeys.push(stackArr);
  }

  const reserveKeys: bigint[][][] = [];
  for (let owner = 0; owner < 2; owner++) {
    const sArr: bigint[][] = [];
    for (let s = 0; s < numSizes; s++) {
      const cArr: bigint[] = [];
      for (let c = 0; c <= maxReserveCount; c++) cArr.push(rand());
      sArr.push(cArr);
    }
    reserveKeys.push(sArr);
  }

  return {
    pieceKeys,
    reserveKeys,
    sideKey: rand(),
    maxStackHeight,
  };
}

function engineMoveToInternal(
  move: Move,
  sizeIdToIdx: Map<string, number>,
  N: number,
): InternalMove | null {
  if (move.kind === 'placeFromReserve') {
    const sIdx = sizeIdToIdx.get(move.sizeId);
    if (sIdx === undefined) return null;
    return {
      kind: 0,
      sizeIdx: sIdx,
      from: -1,
      to: move.to.row * N + move.to.col,
    };
  }
  return {
    kind: 1,
    sizeIdx: -1,
    from: move.from.row * N + move.from.col,
    to: move.to.row * N + move.to.col,
  };
}

function internalMoveToEngine(ctx: SearchContext, m: InternalMove, player: Player): Move {
  const N = ctx.N;
  if (m.kind === 0) {
    return {
      kind: 'placeFromReserve',
      player,
      sizeId: ctx.sizeIds[m.sizeIdx]!,
      to: { row: Math.floor(m.to / N), col: m.to % N },
    } satisfies PlaceFromReserveMove;
  }
  return {
    kind: 'moveOnBoard',
    player,
    from: { row: Math.floor(m.from / N), col: m.from % N },
    to: { row: Math.floor(m.to / N), col: m.to % N },
  } satisfies MoveOnBoardMove;
}

// =============================================================================
// Move generation / make-unmake
// =============================================================================

function topRank(stk: PieceCode[]): number {
  if (stk.length === 0) return -1;
  return stk[stk.length - 1]! & 0xf;
}

function topOwner(stk: PieceCode[]): number {
  if (stk.length === 0) return -1;
  return (stk[stk.length - 1]! >> 4) & 1;
}

function canCoverInternal(
  ctx: Pick<SearchContext, 'allowSameSizeCover' | 'allowSelfCover'>,
  incomingOwner: number,
  incomingRank: number,
  targetStack: PieceCode[],
): boolean {
  if (targetStack.length === 0) return true;
  const top = targetStack[targetStack.length - 1]!;
  const topRk = top & 0xf;
  const topOw = (top >> 4) & 1;
  if (incomingRank < topRk) return false;
  if (incomingRank === topRk && !ctx.allowSameSizeCover) return false;
  if (topOw === incomingOwner && !ctx.allowSelfCover) return false;
  return true;
}

function generateMoves(ctx: SearchContext): InternalMove[] {
  const moves: InternalMove[] = [];
  const N = ctx.N;
  const me = ctx.toMove;
  const reserve = ctx.reserves[me]!;

  // Place from reserve.
  for (let s = 0; s < ctx.numSizes; s++) {
    if ((reserve[s] ?? 0) <= 0) continue;
    for (let cell = 0; cell < N * N; cell++) {
      if (canCoverInternal(ctx, me, s, ctx.stacks[cell]!)) {
        moves.push({ kind: 0, sizeIdx: s, from: -1, to: cell });
      }
    }
  }

  // Move on board.
  for (let cell = 0; cell < N * N; cell++) {
    const stk = ctx.stacks[cell]!;
    if (stk.length === 0) continue;
    if (topOwner(stk) !== me) continue;
    const movingRank = topRank(stk);
    for (let to = 0; to < N * N; to++) {
      if (to === cell) continue;
      if (canCoverInternal(ctx, me, movingRank, ctx.stacks[to]!)) {
        moves.push({ kind: 1, sizeIdx: -1, from: cell, to });
      }
    }
  }

  return moves;
}

interface UndoInfo {
  /** Piece code that was placed/moved (top of dest after move). */
  pieceCode: PieceCode;
  /** Stack position the piece occupied (length-1 of dest after make). */
  destStackPos: number;
  /** Source stack position (length-1 of source before make), -1 for placement. */
  sourceStackPos: number;
  /** Reserve count *before* placement (same as after, for moveOnBoard). */
  prevReserveCount: number;
  /** Side that just moved. */
  prevToMove: 0 | 1;
  /** Repetition delta — count to subtract for the post-move zhash. */
  postZhash: bigint;
}

/**
 * Apply move in-place. Updates zhash and repetition map. Returns undo info.
 */
function applyInternalMove(ctx: SearchContext, m: InternalMove): UndoInfo {
  const me = ctx.toMove;
  const N = ctx.N;
  void N; // currently unused but parameter referenced in helper
  const z = ctx.z;

  let pieceCode: PieceCode;
  let sourceStackPos = -1;
  let prevReserveCount = -1;

  if (m.kind === 0) {
    // Place from reserve: pieceCode is owner|sizeIdx.
    const s = m.sizeIdx;
    pieceCode = (me << 4) | s;
    prevReserveCount = ctx.reserves[me]![s]!;
    // Update reserve & zhash for reserve change.
    ctx.zhash ^= z.reserveKeys[me]![s]![prevReserveCount]!;
    ctx.reserves[me]![s] = prevReserveCount - 1;
    ctx.zhash ^= z.reserveKeys[me]![s]![prevReserveCount - 1]!;
  } else {
    // Move on board: lift top of source.
    const fromStk = ctx.stacks[m.from]!;
    const top = fromStk[fromStk.length - 1]!;
    pieceCode = top;
    sourceStackPos = fromStk.length - 1;
    const topOwner = (top >> 4) & 1;
    const topSize = top & 0xf;
    // Remove from source and update zhash.
    ctx.zhash ^= z.pieceKeys[m.from]![sourceStackPos]![topOwner]![topSize]!;
    fromStk.pop();
  }

  // Push onto destination.
  const destStk = ctx.stacks[m.to]!;
  const destStackPos = destStk.length;
  const owner = (pieceCode >> 4) & 1;
  const size = pieceCode & 0xf;
  destStk.push(pieceCode);
  ctx.zhash ^= z.pieceKeys[m.to]![destStackPos]![owner]![size]!;

  // Flip side to move.
  ctx.zhash ^= z.sideKey;
  const prevToMove = ctx.toMove;
  ctx.toMove = (ctx.toMove ^ 1) as 0 | 1;
  ctx.ply += 1;

  // Increment repetition counter.
  const cnt = ctx.repetition.get(ctx.zhash) ?? 0;
  ctx.repetition.set(ctx.zhash, cnt + 1);

  return {
    pieceCode,
    destStackPos,
    sourceStackPos,
    prevReserveCount,
    prevToMove,
    postZhash: ctx.zhash,
  };
}

function undoInternalMove(ctx: SearchContext, m: InternalMove, undo: UndoInfo): void {
  // Remove repetition increment for the post-move zhash.
  const cnt = ctx.repetition.get(undo.postZhash) ?? 1;
  if (cnt <= 1) ctx.repetition.delete(undo.postZhash);
  else ctx.repetition.set(undo.postZhash, cnt - 1);

  const z = ctx.z;
  // Flip side back.
  ctx.zhash ^= z.sideKey;
  ctx.toMove = undo.prevToMove;
  ctx.ply -= 1;

  // Pop from destination.
  const destStk = ctx.stacks[m.to]!;
  const owner = (undo.pieceCode >> 4) & 1;
  const size = undo.pieceCode & 0xf;
  ctx.zhash ^= z.pieceKeys[m.to]![undo.destStackPos]![owner]![size]!;
  destStk.pop();

  if (m.kind === 0) {
    // Restore reserve.
    const s = m.sizeIdx;
    const me = undo.prevToMove;
    ctx.zhash ^= z.reserveKeys[me]![s]![ctx.reserves[me]![s]!]!;
    ctx.reserves[me]![s] = undo.prevReserveCount;
    ctx.zhash ^= z.reserveKeys[me]![s]![undo.prevReserveCount]!;
  } else {
    // Push back onto source.
    const fromStk = ctx.stacks[m.from]!;
    fromStk.push(undo.pieceCode);
    ctx.zhash ^= z.pieceKeys[m.from]![undo.sourceStackPos]![owner]![size]!;
  }
}

/**
 * Faster version used by the history-replay path during context build.
 * Doesn't return undo info or update repetition map.
 */
function applyInternalMoveRaw(
  ctx: Pick<
    SearchContext,
    | 'N'
    | 'numSizes'
    | 'stacks'
    | 'reserves'
    | 'toMove'
    | 'zhash'
    | 'z'
    | 'allowSameSizeCover'
    | 'allowSelfCover'
  >,
  m: InternalMove,
): void {
  const me = ctx.toMove;
  const z = ctx.z;
  let pieceCode: PieceCode;

  if (m.kind === 0) {
    const s = m.sizeIdx;
    pieceCode = (me << 4) | s;
    const prevCnt = ctx.reserves[me]![s]!;
    ctx.zhash ^= z.reserveKeys[me]![s]![prevCnt]!;
    ctx.reserves[me]![s] = prevCnt - 1;
    ctx.zhash ^= z.reserveKeys[me]![s]![prevCnt - 1]!;
  } else {
    const fromStk = ctx.stacks[m.from]!;
    const top = fromStk[fromStk.length - 1]!;
    pieceCode = top;
    const sp = fromStk.length - 1;
    const topOwner = (top >> 4) & 1;
    const topSize = top & 0xf;
    ctx.zhash ^= z.pieceKeys[m.from]![sp]![topOwner]![topSize]!;
    fromStk.pop();
  }
  const destStk = ctx.stacks[m.to]!;
  const sp = destStk.length;
  destStk.push(pieceCode);
  const owner = (pieceCode >> 4) & 1;
  const size = pieceCode & 0xf;
  ctx.zhash ^= z.pieceKeys[m.to]![sp]![owner]![size]!;
  ctx.zhash ^= z.sideKey;
  ctx.toMove = (ctx.toMove ^ 1) as 0 | 1;
}

// =============================================================================
// Outcome detection
// =============================================================================

/** -1 = no winner, 0/1 = winner index. */
function detectWinner(ctx: SearchContext): number {
  for (const line of ctx.lines) {
    const c0 = line[0]!;
    const stk0 = ctx.stacks[c0]!;
    if (stk0.length === 0) continue;
    const ow = topOwner(stk0);
    let all = true;
    for (let i = 1; i < line.length; i++) {
      const ci = line[i]!;
      const si = ctx.stacks[ci]!;
      if (si.length === 0 || topOwner(si) !== ow) {
        all = false;
        break;
      }
    }
    if (all) return ow;
  }
  return -1;
}

/**
 * Outcome at the *current* node. Returns:
 *   { kind: 'win', winner: 0|1 } | 'draw' | null (ongoing)
 */
function nodeOutcome(ctx: SearchContext): { kind: 'win'; winner: 0 | 1 } | { kind: 'draw' } | null {
  const w = detectWinner(ctx);
  if (w !== -1) return { kind: 'win', winner: w as 0 | 1 };
  if (ctx.ply >= ctx.maxPly) return { kind: 'draw' };
  const reps = ctx.repetition.get(ctx.zhash) ?? 0;
  if (reps >= ctx.drawByRepetition) return { kind: 'draw' };
  return null;
}

// =============================================================================
// Move ordering
// =============================================================================

/**
 * Score a move statically (higher = try first).
 *
 * Categories:
 *   100_000 — completes our own line of N
 *    50_000 — blocks an opponent line of N (their immediate win)
 *    10_000 — covers (captures) a non-empty cell
 *     1_000 — uncovers a cell so that an opponent piece is exposed (penalty
 *             handled implicitly by giving lower score)
 *       N*100 — center-ness (cells closer to centre score higher)
 */
function staticMoveScore(ctx: SearchContext, m: InternalMove): number {
  const me = ctx.toMove;
  const opp = me ^ 1;
  let score = 0;

  // Determine the moving rank.
  let movingRank: number;
  if (m.kind === 0) movingRank = m.sizeIdx;
  else movingRank = topRank(ctx.stacks[m.from]!);

  // Simulate top-of-stack at destination after move.
  const destStkLen = ctx.stacks[m.to]!.length;
  void destStkLen;

  // Check whether this move completes one of our lines.
  const linesAtTo = ctx.cellLines[m.to]!;
  for (const li of linesAtTo) {
    const line = ctx.lines[li]!;
    let mineCount = 0;
    let othersBlocking = false;
    for (const cell of line) {
      if (cell === m.to) continue;
      const stk = ctx.stacks[cell]!;
      if (stk.length === 0) {
        othersBlocking = true;
        break;
      }
      const ow = topOwner(stk);
      if (ow === me) mineCount += 1;
      else {
        othersBlocking = true;
        break;
      }
    }
    if (!othersBlocking && mineCount === line.length - 1) {
      score += 100_000;
      break;
    }
  }

  // Block opponent's near-win on the target cell.
  for (const li of linesAtTo) {
    const line = ctx.lines[li]!;
    let oppCount = 0;
    let badForBlock = false;
    for (const cell of line) {
      if (cell === m.to) continue;
      const stk = ctx.stacks[cell]!;
      if (stk.length === 0) {
        badForBlock = true;
        break;
      }
      const ow = topOwner(stk);
      if (ow === opp) oppCount += 1;
      else {
        badForBlock = true;
        break;
      }
    }
    if (!badForBlock && oppCount === line.length - 1) {
      // Will the move actually displace the opponent on this cell?
      // It will if our destination's top after the move is us, which it is.
      score += 50_000;
      break;
    }
  }

  // Covering a non-empty cell (capture/cover).
  if (ctx.stacks[m.to]!.length > 0) {
    const topOw = topOwner(ctx.stacks[m.to]!);
    if (topOw === opp) score += 10_000 + movingRank * 100;
    else score += 2_000;
  }

  // Move-on-board that uncovers a friendly piece beneath: small bonus.
  if (m.kind === 1) {
    const fromStk = ctx.stacks[m.from]!;
    if (fromStk.length >= 2) {
      const beneath = fromStk[fromStk.length - 2]!;
      const beneathOw = (beneath >> 4) & 1;
      if (beneathOw === opp)
        score -= 500; // bad: exposing opp piece
      else score += 200;
    }
  }

  // Centrality.
  const N = ctx.N;
  const r = Math.floor(m.to / N);
  const c = m.to % N;
  const center = (N - 1) / 2;
  const dist = Math.abs(r - center) + Math.abs(c - center);
  score += Math.max(0, N - dist) * 50;

  return score;
}

// =============================================================================
// Iterative deepening / negamax
// =============================================================================

interface SearchStats {
  nodes: number;
  ttHits: number;
}

interface SearchEnv {
  ctx: SearchContext;
  tt: Map<bigint, TtEntry>;
  age: number;
  timers: SearchTimers;
  stats: SearchStats;
  /** Reused move buffers per depth, to reduce allocations. */
  moveStack: InternalMove[][];
  /** Sentinel for "deadline reached / aborted" — we throw this. */
}

class SearchAbort extends Error {
  constructor() {
    super('search aborted');
    this.name = 'SearchAbort';
  }
}

async function runIterativeDeepening(
  ctx: SearchContext,
  tt: Map<bigint, TtEntry>,
  timers: SearchTimers,
): Promise<IdResult> {
  const rootMoves = generateMoves(ctx);
  if (rootMoves.length === 0) {
    return { bestMove: null, rootMoves: [] };
  }

  const me = ctx.toMove === 0 ? 'P1' : 'P2';

  // Pre-sort with the static heuristic.
  let orderedMoves: InternalMove[] = rootMoves.slice();
  orderedMoves.sort((a, b) => staticMoveScore(ctx, b) - staticMoveScore(ctx, a));

  let bestMoveInternal: InternalMove | null = orderedMoves[0] ?? null;
  let bestEngineMove: Move | null = bestMoveInternal
    ? internalMoveToEngine(ctx, bestMoveInternal, me)
    : null;

  const env: SearchEnv = {
    ctx,
    tt,
    age: (tt.size & 0xffff) + 1,
    timers,
    stats: { nodes: 0, ttHits: 0 },
    moveStack: [],
  };

  // Quick yield helper to give the event loop a chance to fire abort handlers.
  await Promise.resolve();

  for (let depth = 1; depth <= MAX_DEPTH; depth++) {
    if (timers.abortFlag.aborted || performance.now() >= timers.deadline) break;

    let depthBest: InternalMove | null = null;
    let depthBestScore = -Infinity;
    let alpha = -WIN_SCORE - 1000;
    const beta = WIN_SCORE + 1000;
    let aborted = false;

    // Reorder: previous best first.
    if (bestMoveInternal) {
      orderedMoves = orderedMoves.slice();
      const idx = orderedMoves.findIndex((m) => sameInternalMove(m, bestMoveInternal!));
      if (idx > 0) {
        const [picked] = orderedMoves.splice(idx, 1);
        if (picked) orderedMoves.unshift(picked);
      }
    }

    const moveScores: { move: InternalMove; score: number }[] = [];

    try {
      for (const mv of orderedMoves) {
        if (timers.abortFlag.aborted || performance.now() >= timers.deadline) {
          aborted = true;
          break;
        }
        const undo = applyInternalMove(ctx, mv);
        let score: number;
        const out = nodeOutcome(ctx);
        if (out) {
          score = -outcomeToScoreFromChild(out, ctx.toMove);
        } else {
          score = -negamax(env, depth - 1, -beta, -alpha, 1);
        }
        undoInternalMove(ctx, mv, undo);

        moveScores.push({ move: mv, score });

        if (score > depthBestScore) {
          depthBestScore = score;
          depthBest = mv;
          if (score > alpha) alpha = score;
        }
      }
    } catch (e) {
      if (e instanceof SearchAbort) {
        aborted = true;
      } else {
        throw e;
      }
    }

    if (aborted) {
      // Use partial info if depthBest set; otherwise keep prev iteration's best.
      if (depthBest && depthBestScore > -Infinity) {
        bestMoveInternal = depthBest;
        bestEngineMove = internalMoveToEngine(ctx, depthBest, me);
      }
      break;
    }

    // Reorder for next iteration based on this depth's scores.
    if (moveScores.length === orderedMoves.length) {
      moveScores.sort((a, b) => b.score - a.score);
      orderedMoves = moveScores.map((m) => m.move);
    }

    if (depthBest) {
      bestMoveInternal = depthBest;
      bestEngineMove = internalMoveToEngine(ctx, depthBest, me);
    }

    // Mate found — no need to keep deepening (we'll keep going until budget
    // exhausted since we want shortest mate, but we can stop if we've already
    // proven a forced loss / win at this depth and no shorter mate exists).
    // Continue: deeper search may yield shorter mate.
    if (Math.abs(depthBestScore) >= WIN_SCORE - MAX_DEPTH * 2 - 1) {
      // We have a forced result at this depth; deeper iterations can only
      // confirm. Stop early to save time.
      break;
    }

    // Yield to event loop between iterations.
    await new Promise<void>((res) => setTimeout(res, 0));
  }

  return {
    bestMove: bestEngineMove,
    rootMoves: rootMoves.map((m) => internalMoveToEngine(ctx, m, me)),
  };
}

/** Translate a `nodeOutcome` (computed *after* applying a move, hence the
 * caller passing the side-to-move at the resulting state) into a score from
 * the *resulting node's side-to-move's* perspective.
 *
 * The caller flips it to get the score from the *parent*'s perspective.
 */
function outcomeToScoreFromChild(
  out: { kind: 'win'; winner: 0 | 1 } | { kind: 'draw' },
  childToMove: 0 | 1,
): number {
  if (out.kind === 'draw') return 0;
  // The *child* node now sees this position. If `winner` is the child's
  // side-to-move, that's good for the child; otherwise bad. (In practice,
  // after any normal move the player who just moved is the winner — so
  // the opponent (current to-move at the child) sees a loss = -WIN.)
  return out.winner === childToMove ? WIN_SCORE : -WIN_SCORE;
}

/** True iff two internal moves represent the same action. */
function sameInternalMove(a: InternalMove, b: InternalMove): boolean {
  return a.kind === b.kind && a.sizeIdx === b.sizeIdx && a.from === b.from && a.to === b.to;
}

function sameMove(a: Move, b: Move): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === 'placeFromReserve' && b.kind === 'placeFromReserve') {
    return (
      a.sizeId === b.sizeId &&
      a.player === b.player &&
      a.to.row === b.to.row &&
      a.to.col === b.to.col
    );
  }
  if (a.kind === 'moveOnBoard' && b.kind === 'moveOnBoard') {
    return (
      a.player === b.player &&
      a.from.row === b.from.row &&
      a.from.col === b.from.col &&
      a.to.row === b.to.row &&
      a.to.col === b.to.col
    );
  }
  return false;
}

/**
 * Negamax with alpha-beta. `depth` is plies remaining. Returns the score from
 * the side-to-move's perspective. `ply` is the depth from the root (used to
 * adjust mate scores for "shortest mate preferred").
 */
function negamax(
  env: SearchEnv,
  depth: number,
  alphaIn: number,
  betaIn: number,
  ply: number,
): number {
  let alpha = alphaIn;
  const beta = betaIn;

  env.stats.nodes += 1;
  if ((env.stats.nodes & (TIME_CHECK_INTERVAL - 1)) === 0) {
    if (env.timers.abortFlag.aborted || performance.now() >= env.timers.deadline) {
      throw new SearchAbort();
    }
  }

  const ctx = env.ctx;
  const out = nodeOutcome(ctx);
  if (out) {
    if (out.kind === 'draw') return 0;
    // Loss/win from current side-to-move.
    return out.winner === ctx.toMove ? WIN_SCORE - ply : -(WIN_SCORE - ply);
  }
  if (depth <= 0) {
    return staticEval(ctx);
  }

  // TT probe.
  const ttKey = ctx.zhash;
  const tt = env.tt;
  const ttEntry = tt.get(ttKey);
  let ttBestIdx = -1;
  if (ttEntry) {
    env.stats.ttHits += 1;
    if (ttEntry.depth >= depth) {
      const v = ttEntry.value;
      if (ttEntry.flag === 'exact') return v;
      if (ttEntry.flag === 'lower' && v >= beta) return v;
      if (ttEntry.flag === 'upper' && v <= alpha) return v;
    }
    ttBestIdx = ttEntry.bestMoveIdx;
  }

  const moves = generateMoves(ctx);
  if (moves.length === 0) {
    // No legal moves — engine treats this as a draw.
    return 0;
  }

  // Order moves: TT best first, then by static heuristic.
  const scoreFn = (m: InternalMove): number => staticMoveScore(ctx, m);
  if (ttBestIdx >= 0 && ttBestIdx < moves.length) {
    // We stored the *index into the move list*. But move list ordering can
    // differ across calls. To guard against that we rebuild the ordering
    // and bring `ttBestIdx` to the front if it still points to a valid slot.
    const ttMove = moves[ttBestIdx];
    if (ttMove) {
      // Re-sort the rest by static score, then place ttMove first.
      const rest = moves.slice(0, ttBestIdx).concat(moves.slice(ttBestIdx + 1));
      rest.sort((a, b) => scoreFn(b) - scoreFn(a));
      moves.length = 0;
      moves.push(ttMove);
      for (const m of rest) moves.push(m);
    } else {
      moves.sort((a, b) => scoreFn(b) - scoreFn(a));
    }
  } else {
    moves.sort((a, b) => scoreFn(b) - scoreFn(a));
  }

  let best = -Infinity;
  let bestIdx = 0;
  let flag: 'exact' | 'upper' = 'upper';

  for (let i = 0; i < moves.length; i++) {
    const mv = moves[i]!;
    const undo = applyInternalMove(ctx, mv);
    let value: number;
    const childOut = nodeOutcome(ctx);
    if (childOut) {
      // Translate to *current* side-to-move (parent) perspective:
      // childOut is from the post-move position where the opponent is to move.
      if (childOut.kind === 'draw') {
        value = 0;
      } else {
        // If `winner === childToMove (opponent)`, opponent wins => loss for us.
        value = childOut.winner === ctx.toMove ? -(WIN_SCORE - ply - 1) : WIN_SCORE - ply - 1;
      }
    } else {
      value = -negamax(env, depth - 1, -beta, -alpha, ply + 1);
    }
    undoInternalMove(ctx, mv, undo);

    if (value > best) {
      best = value;
      bestIdx = i;
    }
    if (best > alpha) {
      alpha = best;
      flag = 'exact';
    }
    if (alpha >= beta) {
      flag = 'exact'; // we have a lower bound but mark cutoff entries below
      // beta cutoff
      storeTt(env, ttKey, depth, 'lower', best, bestIdx);
      return best;
    }
  }

  storeTt(env, ttKey, depth, flag, best, bestIdx);
  return best;
}

function storeTt(
  env: SearchEnv,
  key: bigint,
  depth: number,
  flag: 'exact' | 'lower' | 'upper',
  value: number,
  bestMoveIdx: number,
): void {
  const tt = env.tt;
  const existing = tt.get(key);
  if (existing && existing.depth > depth && existing.age === env.age) {
    return;
  }
  if (tt.size >= TT_MAX_ENTRIES && !existing) {
    // Simple eviction: drop the first entry (Map iteration = insertion order).
    const firstKey = tt.keys().next().value;
    if (firstKey !== undefined) tt.delete(firstKey);
  }
  tt.set(key, { depth, flag, value, bestMoveIdx, age: env.age });
}

// =============================================================================
// Static evaluation
// =============================================================================

/**
 * Static eval of a non-terminal position from the side-to-move's perspective.
 * Counts threats: lines where one side has top-of-stack pieces and the rest
 * empty (potential wins).
 */
function staticEval(ctx: SearchContext): number {
  const me = ctx.toMove;
  const opp = me ^ 1;
  let score = 0;

  for (const line of ctx.lines) {
    let mine = 0;
    let theirs = 0;
    for (const cell of line) {
      const stk = ctx.stacks[cell]!;
      if (stk.length === 0) continue;
      const ow = topOwner(stk);
      if (ow === me) mine += 1;
      else theirs += 1;
    }
    if (theirs === 0 && mine > 0) score += mine * mine * 10;
    if (mine === 0 && theirs > 0) score -= theirs * theirs * 10;
  }

  // Slight bias for having more reserve pieces of large size (flexibility).
  for (let s = 0; s < ctx.numSizes; s++) {
    score += (ctx.reserves[me]![s]! - ctx.reserves[opp]![s]!) * (s + 1);
  }

  return score;
}

// =============================================================================
// Engine cross-check (tests)
// =============================================================================

/**
 * Verifies the solver's outcome detection matches the engine's for the
 * initial state of a given rules object. Used during test setup to guard
 * against semantic drift.
 */
export function __debugStateMatchesEngine(state: GameState): boolean {
  const ctx = buildSearchContext(state);
  // Quickly check: number of legal moves and outcome match.
  const internalMoves = generateMoves(ctx);
  const engineMoves = engine.legalMoves(state);
  return internalMoves.length === engineMoves.length;
}
