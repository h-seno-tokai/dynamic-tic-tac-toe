"""Fast NumPy-backed rule engine specialised to ``PRESET_4X4_XL``.

Internal representation
-----------------------
* ``board``    : ``int8[4, 4, 4]`` — 3rd axis is the stack, *bottom-up*.
                 ``board[r, c, h] = 0`` means empty slot; otherwise it stores
                 ``+(rank + 1)`` for P1 and ``-(rank + 1)`` for P2.
                 Stack height is bounded by 4 because no-same-size-cover means
                 each rank can appear at most once per cell.
* ``heights`` : ``int8[4, 4]`` — number of occupied slots in each cell.
* ``top``     : ``int8[4, 4]`` — signed value of the top piece (0 if empty).
                 Maintained incrementally in ``apply_move``.
* ``reserves``: ``int8[2, 4]`` — rows = player (0=P1, 1=P2), cols = rank.
* ``to_move`` : ``int`` (0 = P1, 1 = P2).
* ``ply``     : ``int``.
* ``repetition``: ``dict[bytes, int]``  — keyed by canonical hash bytes.
* ``outcome`` : ``None``, the string ``"draw"``, or ``Player.P1`` / ``Player.P2``.

Cover legality table
--------------------
Computed once at import time as ``_CAN_COVER[top_rank_plus_one, mover_rank]``
of shape ``(5, 4)``.  Entry ``[0, m]`` corresponds to "empty cell" and is
always True.  Otherwise (rank in 0..3 + 1 = 1..4) the rule is:

* mover_rank > top_rank  ⇒ ok (allow_self_cover = True)
* mover_rank == top_rank ⇒ ok only if allow_same_size_cover (False here)
* mover_rank <  top_rank ⇒ never

The ``_CAN_COVER_OWN`` table additionally enforces ``allow_self_cover``
(currently True for the preset, so it equals ``_CAN_COVER``).
"""

from __future__ import annotations

import copy as _stdlib_copy
from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np

from dttt_train.rules import (
    MoveOnBoardMove,
    PlaceFromReserveMove,
    Player,
)

from .rules import (
    ALLOW_SAME_SIZE_COVER,
    ALLOW_SELF_COVER,
    BOARD_SIZE,
    DRAW_BY_REPETITION,
    MAX_PLY,
    MAX_STACK,
    NUM_SIZES,
    OUTCOME_DRAW,
    PIECES_PER_SIZE,
    SIZE_IDS,
    SIZE_RANK,
)

Outcome = Union[Player, Literal["draw"], None]
Move = Union[PlaceFromReserveMove, MoveOnBoardMove]


# ---------------------------------------------------------------------------
# Static cover-legality lookup
# ---------------------------------------------------------------------------

def _build_can_cover() -> np.ndarray:
    """Return ``(5, 4)`` bool array; row 0 is the empty-cell case."""
    out = np.zeros((5, NUM_SIZES), dtype=bool)
    # Empty cell — anything legal.
    out[0, :] = True
    for top_rank in range(NUM_SIZES):
        for mover_rank in range(NUM_SIZES):
            if mover_rank > top_rank:
                out[top_rank + 1, mover_rank] = True
            elif mover_rank == top_rank:
                out[top_rank + 1, mover_rank] = ALLOW_SAME_SIZE_COVER
            else:
                out[top_rank + 1, mover_rank] = False
    return out


_CAN_COVER: np.ndarray = _build_can_cover()


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class GameState:
    """Compact mutable game-state.

    Engine functions still return *new* states (no in-place mutation of inputs)
    so MCTS / self-play reasoning remains straightforward, but each instance
    is internally a small handful of NumPy buffers — far cheaper to copy than
    the old list-of-lists representation.
    """

    board: np.ndarray            # int8 (4, 4, 4)
    heights: np.ndarray          # int8 (4, 4)
    top: np.ndarray              # int8 (4, 4) — signed top piece per cell
    reserves: np.ndarray         # int8 (2, 4)
    to_move: int                 # 0 = P1, 1 = P2
    ply: int = 0
    history: list[Move] = field(default_factory=list)
    repetition: dict[bytes, int] = field(default_factory=dict)
    outcome: Outcome = None
    # Cached canonical preset reference (lets state_to_tensor read .rules.piece_sizes).
    rules: object = None  # type: ignore[assignment]

    # ---- compatibility shim for legacy callers -----------------------------
    @property
    def to_move_player(self) -> Player:
        return Player.P1 if self.to_move == 0 else Player.P2


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def initial_state(rules: object | None = None) -> GameState:
    """Return the empty ``PRESET_4X4_XL`` start state.

    The optional ``rules`` argument is accepted for API symmetry with
    ``dttt_train.rules.initial_state``; if ``None`` we use ``PRESET_4X4_XL``.
    """
    from .rules import PRESET_4X4_XL
    rules = rules if rules is not None else PRESET_4X4_XL

    board = np.zeros((BOARD_SIZE, BOARD_SIZE, MAX_STACK), dtype=np.int8)
    heights = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    top = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    reserves = np.full((2, NUM_SIZES), PIECES_PER_SIZE, dtype=np.int8)

    return GameState(
        board=board,
        heights=heights,
        top=top,
        reserves=reserves,
        to_move=0,
        ply=0,
        history=[],
        repetition={},
        outcome=None,
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Adapters (compat with old dttt_train.GameState)
# ---------------------------------------------------------------------------


def from_legacy_state(legacy: object) -> GameState:
    """Build a fast :class:`GameState` from a legacy ``dttt_train`` state.

    Used by the parity test.  Only the 4x4 XL preset is supported.
    """
    from dttt_train.rules import GameState as LegacyState  # local import
    if not isinstance(legacy, LegacyState):  # pragma: no cover — defensive
        raise TypeError("expected a dttt_train GameState")

    s = initial_state()
    bs = legacy.rules.board_size
    if bs != BOARD_SIZE:
        raise ValueError("from_legacy_state only supports the 4x4 preset")

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = legacy.board[r][c]
            for h, piece in enumerate(cell):
                rank = SIZE_RANK[piece.size_id]
                signed = (rank + 1) if piece.owner is Player.P1 else -(rank + 1)
                s.board[r, c, h] = signed
            s.heights[r, c] = len(cell)
            if cell:
                rank = SIZE_RANK[cell[-1].size_id]
                signed = (rank + 1) if cell[-1].owner is Player.P1 else -(rank + 1)
                s.top[r, c] = signed
    for sid, idx in SIZE_RANK.items():
        s.reserves[0, idx] = legacy.reserves[Player.P1].get(sid, 0)
        s.reserves[1, idx] = legacy.reserves[Player.P2].get(sid, 0)
    s.to_move = 0 if legacy.to_move is Player.P1 else 1
    s.ply = legacy.ply
    s.outcome = legacy.outcome
    return s


# ---------------------------------------------------------------------------
# Hashing for repetition detection
# ---------------------------------------------------------------------------


def _state_hash(state: GameState) -> bytes:
    """Canonical hash bytes (board + reserves + side-to-move).

    ``state.board`` already encodes everything we need: signed top-down stack
    contents, with deterministic ordering.  We omit history & ply.
    """
    return state.board.tobytes() + state.reserves.tobytes() + bytes([state.to_move])


# ---------------------------------------------------------------------------
# Win detection (vectorised)
# ---------------------------------------------------------------------------

# Pre-computed line indices (10 lines × 4 cells) as flat 0..15 indices.
def _build_lines() -> np.ndarray:
    n = BOARD_SIZE
    lines: list[list[int]] = []
    # rows
    for r in range(n):
        lines.append([r * n + c for c in range(n)])
    # cols
    for c in range(n):
        lines.append([r * n + c for r in range(n)])
    # diagonal \\
    lines.append([i * n + i for i in range(n)])
    # diagonal /
    lines.append([i * n + (n - 1 - i) for i in range(n)])
    return np.asarray(lines, dtype=np.int64)


_LINES: np.ndarray = _build_lines()  # shape (10, 4)
# Pure-Python tuple of (i0, i1, i2, i3) per line — much faster for ``_check_winner``
# than fancy-indexing on a tiny 4x4 array (numpy fixed overhead dominates).
_LINES_PY: tuple[tuple[int, int, int, int], ...] = tuple(
    tuple(int(x) for x in line) for line in _LINES.tolist()
)


def _check_winner(top: np.ndarray) -> int:
    """Return +1 if P1 wins, -1 if P2 wins, 0 otherwise.

    ``top`` is an ``int8[4, 4]`` array of signed top-piece values.
    Implemented as a pure-Python loop over 10 lines because numpy fixed
    overhead is 5x worse on this size.
    """
    flat = top.reshape(-1).tolist()
    for line in _LINES_PY:
        a = flat[line[0]]
        if a == 0:
            continue
        b = flat[line[1]]
        if b == 0:
            continue
        c = flat[line[2]]
        if c == 0:
            continue
        d = flat[line[3]]
        if d == 0:
            continue
        if a > 0 and b > 0 and c > 0 and d > 0:
            return 1
        if a < 0 and b < 0 and c < 0 and d < 0:
            return -1
    return 0


# ---------------------------------------------------------------------------
# Move enumeration & application
# ---------------------------------------------------------------------------


# --- precomputed (row, col) cell coords in flat order, used by legal_moves ---
_CELLS: tuple[tuple[int, int], ...] = tuple(
    (r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
)
# Flat mover-LUT keyed by ``(top_signed + 4) * 4 + mover_rank`` — domain is
# top_signed in {-4..-1, 0, 1..4} (9 values) and mover_rank in {0..3} (4
# values). We pre-bake the *full* legality including self-cover so the
# inner loops do a single dict-style lookup.
_FLAT_COVER: list[bool] = [False] * (9 * 4)
for _t in range(-NUM_SIZES, NUM_SIZES + 1):
    _key_base = (_t + NUM_SIZES) * NUM_SIZES
    for _m in range(NUM_SIZES):
        if _t == 0:
            _FLAT_COVER[_key_base + _m] = True
            continue
        _top_rank = abs(_t) - 1
        if not _CAN_COVER[_top_rank + 1, _m]:
            _FLAT_COVER[_key_base + _m] = False
            continue
        # self-cover gate: top P1 (>0) covered by P1-owner mover (rank+1>0
        # signed by mover sign — handled at the call site by selecting the
        # appropriate (top_signed) entries).
        _FLAT_COVER[_key_base + _m] = True


def legal_moves(state: GameState) -> list[Move]:
    """Enumerate legal moves for the side to move.

    Hot path. Read the 16-element top vector into a Python tuple once, then
    iterate with builtin int operations only — no per-cell ``int8 -> int``
    casts in the inner loop.
    """
    if state.outcome is not None:
        return []

    me = state.to_move                  # 0 / 1
    me_player = Player.P1 if me == 0 else Player.P2

    # Snapshot top + reserves as plain Python lists once.  ``ndarray.tolist()``
    # is ~5x faster than ``tuple(int(x) for x in arr)`` for these sizes.
    top_flat = state.top.reshape(-1).tolist()             # list of 16 ints
    reserves_py = state.reserves[me].tolist()             # list of 4 ints

    # Pre-compute cover-legality for each cell against each rank: a (16, 4)
    # bool flat list. We use the flat _FLAT_COVER LUT keyed by top_signed.
    # However self-cover currently always allowed in this preset, so the
    # LUT is independent of mover-owner — we collapse it here.
    NS = NUM_SIZES                     # 4
    flat_cover = _FLAT_COVER

    moves: list[Move] = []
    cells = _CELLS
    SIDS = SIZE_IDS                     # local alias for speed

    # ------ Place-from-reserve ------
    for rank in range(NS):
        if reserves_py[rank] <= 0:
            continue
        sid = SIDS[rank]
        for cell_idx in range(16):
            t = top_flat[cell_idx]
            # _FLAT_COVER index: (t + 4) * 4 + rank
            if not flat_cover[(t + NS) * NS + rank]:
                continue
            r, c = cells[cell_idx]
            moves.append(PlaceFromReserveMove(
                player=me_player, size_id=sid, to_row=r, to_col=c,
            ))

    # ------ Move-on-board ------
    # Only a top piece owned by ``me`` may be lifted. me=0 -> sign>0; me=1 -> sign<0.
    if me == 0:
        own_iter = [(i, t) for i, t in enumerate(top_flat) if t > 0]
    else:
        own_iter = [(i, t) for i, t in enumerate(top_flat) if t < 0]

    for from_idx, t in own_iter:
        mover_rank = abs(t) - 1
        fr, fc = cells[from_idx]
        for to_idx in range(16):
            if to_idx == from_idx:
                continue
            target = top_flat[to_idx]
            if not flat_cover[(target + NS) * NS + mover_rank]:
                continue
            tr, tc = cells[to_idx]
            moves.append(MoveOnBoardMove(
                player=me_player,
                from_row=fr, from_col=fc,
                to_row=tr, to_col=tc,
            ))
    return moves


def _clone_state(state: GameState) -> GameState:
    """Shallow clone with copied numpy buffers and a copied repetition dict."""
    return GameState(
        board=state.board.copy(),
        heights=state.heights.copy(),
        top=state.top.copy(),
        reserves=state.reserves.copy(),
        to_move=state.to_move,
        ply=state.ply,
        history=list(state.history),
        repetition=_stdlib_copy.copy(state.repetition),
        outcome=state.outcome,
        rules=state.rules,
    )


def apply_move(state: GameState, move: Move) -> GameState:
    """Return a new state with ``move`` applied. Pure (no input mutation)."""
    if state.outcome is not None:
        raise ValueError("cannot apply a move to a terminal state")

    s = _clone_state(state)
    me = s.to_move
    me_sign = 1 if me == 0 else -1

    reserves = s.reserves
    heights = s.heights
    board = s.board
    top = s.top

    if isinstance(move, PlaceFromReserveMove):
        rank = SIZE_RANK[move.size_id]
        cur = int(reserves[me, rank])
        if cur <= 0:
            raise ValueError("no reserve piece of that size")
        reserves[me, rank] = cur - 1
        r, c = move.to_row, move.to_col
        h = int(heights[r, c])
        if h >= MAX_STACK:
            raise ValueError("cell stack overflow")
        signed = me_sign * (rank + 1)
        board[r, c, h] = signed
        heights[r, c] = h + 1
        top[r, c] = signed
    else:
        fr, fc = move.from_row, move.from_col
        tr, tc = move.to_row, move.to_col
        h_from = int(heights[fr, fc])
        if h_from <= 0:
            raise ValueError("source cell is empty")
        signed = int(board[fr, fc, h_from - 1])
        owner_p1 = signed > 0
        if (me == 0) != owner_p1:
            raise ValueError("can only move own piece")
        # Pop from source.
        board[fr, fc, h_from - 1] = 0
        heights[fr, fc] = h_from - 1
        if h_from - 1 > 0:
            top[fr, fc] = board[fr, fc, h_from - 2]
        else:
            top[fr, fc] = 0
        # Push at destination.
        h_to = int(heights[tr, tc])
        if h_to >= MAX_STACK:
            raise ValueError("cell stack overflow")
        board[tr, tc, h_to] = signed
        heights[tr, tc] = h_to + 1
        top[tr, tc] = signed

    s.history.append(move)
    s.ply += 1
    s.to_move = 1 - me

    # Repetition / terminal detection
    h_key = _state_hash(s)
    s.repetition[h_key] = s.repetition.get(h_key, 0) + 1

    winner = _check_winner(s.top)
    if winner == 1:
        s.outcome = Player.P1
    elif winner == -1:
        s.outcome = Player.P2
    elif s.repetition[h_key] >= DRAW_BY_REPETITION:
        s.outcome = OUTCOME_DRAW
    elif s.ply >= MAX_PLY:
        s.outcome = OUTCOME_DRAW
    else:
        # Stalemate detection (extremely rare in 4x4 Gobblet under
        # max_ply=120, draw_by_repetition=3). Skip the expensive probe
        # entirely unless the *side to move* has no reserves AND no own
        # piece on the board — only then is stalemate even possible.
        next_me = s.to_move
        next_reserves = s.reserves[next_me].tolist()
        if next_reserves[0] + next_reserves[1] + next_reserves[2] + next_reserves[3] == 0:
            # Probe: any own piece on the board?
            sign_pos = next_me == 0
            top_list = s.top.reshape(-1).tolist()
            has_own = False
            for t in top_list:
                if t == 0:
                    continue
                if (t > 0) == sign_pos:
                    has_own = True
                    break
            if not has_own:
                s.outcome = OUTCOME_DRAW
            elif not _has_any_legal_move(s):
                s.outcome = OUTCOME_DRAW

    return s


# ---------------------------------------------------------------------------
# Cheap "any legal move?" probe (early-exit)
# ---------------------------------------------------------------------------


def _has_any_legal_move(state: GameState) -> bool:
    """Return True iff the side-to-move has at least one legal move (fast probe)."""
    me = state.to_move
    top_flat = state.top.reshape(-1).tolist()
    reserves_py = state.reserves[me].tolist()
    NS = NUM_SIZES
    flat_cover = _FLAT_COVER

    for rank in range(NS):
        if reserves_py[rank] <= 0:
            continue
        for cell_idx in range(16):
            t = top_flat[cell_idx]
            if flat_cover[(t + NS) * NS + rank]:
                return True

    if me == 0:
        own_iter = [(i, t) for i, t in enumerate(top_flat) if t > 0]
    else:
        own_iter = [(i, t) for i, t in enumerate(top_flat) if t < 0]
    for from_idx, t in own_iter:
        mover_rank = abs(t) - 1
        for to_idx in range(16):
            if to_idx == from_idx:
                continue
            target = top_flat[to_idx]
            if flat_cover[(target + NS) * NS + mover_rank]:
                return True
    return False


def is_terminal(state: GameState) -> bool:
    return state.outcome is not None


def outcome(state: GameState) -> Outcome:
    return state.outcome


__all__ = [
    "GameState",
    "Move",
    "Outcome",
    "initial_state",
    "from_legacy_state",
    "legal_moves",
    "apply_move",
    "is_terminal",
    "outcome",
    "_state_hash",
    "_check_winner",
]
