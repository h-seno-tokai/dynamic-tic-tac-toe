"""State -> tensor encoding and 320-d action helpers (4x4 fast engine).

For drop-in compatibility with the existing ``dttt_train.augment`` and the
network input layer we keep the same ``(27, 4, 4)`` channel layout that the
old engine produces.  The two trailing channels (25 = out-of-board mask, 26
= unused-size mask) are *vestigial* in the 4x4-only world — both are zero
for every preset 4x4 state — but we keep the channel count fixed so the
existing trained networks remain loadable and the augmentation logic
unchanged.

Channel layout (mirrors :mod:`dttt_train.encoding`)::

  0-3   P1 top-of-stack one-hot per rank
  4-7   P2 top-of-stack
  8-11  P1 anywhere-in-stack
  12-15 P2 anywhere-in-stack
  16-19 P1 reserve count, normalised, broadcast 4x4
  20-23 P2 reserve count
  24    side-to-move (1 if P1)
  25    out-of-board mask                (always 0 for 4x4)
  26    unused-size mask                 (always 0 for 4x4)

Action layout (320 = 64 + 256), unchanged::
  [0,  64) PlaceFromReserve: idx = rank * 16 + (row * 4 + col)
  [64, 320) MoveOnBoard:     idx = 64 + to_idx * 16 + from_idx
"""

from __future__ import annotations

import numpy as np

from dttt_train.rules import (  # re-use the *same* dataclass types
    MoveOnBoardMove,
    PlaceFromReserveMove,
    Player,
)

from .engine import GameState, legal_moves
from .rules import (
    BOARD_SIZE,
    MAX_BOARD,
    MAX_PIECE_SIZES,
    MAX_PIECES_PER_SIZE,
    NUM_SIZES,
    PIECES_PER_SIZE,
    PLACE_ACTIONS,
    SIZE_IDS,
    SIZE_RANK,
    TOTAL_ACTIONS,
)

NUM_CHANNELS: int = 27


# ---------------------------------------------------------------------------
# state_to_tensor
# ---------------------------------------------------------------------------


_DENOM_INV: float = 1.0 / float(MAX_PIECES_PER_SIZE)


def state_to_tensor(state: GameState) -> np.ndarray:
    """Return a ``(27, 4, 4)`` float32 tensor.

    Hot path. We dispatch most of the per-cell work in pure Python (cheap
    when only a handful of cells are non-empty) and finish with a single
    numpy assignment for the broadcast-scalar planes.
    """
    tensor = np.zeros((NUM_CHANNELS, MAX_BOARD, MAX_BOARD), dtype=np.float32)
    flat_view = tensor.reshape(NUM_CHANNELS, 16)            # zero-copy view

    # ---- channels 0-15 : top-of-stack + anywhere-in-stack --------------
    # Single pass over the (cell, level) tuples in board_flat. We track
    # already-set "anywhere" channels per cell with a bitmask; the top piece
    # is the highest level and additionally sets channels 0..7.
    h_flat = state.heights.reshape(-1).tolist()
    if any(h_flat):
        board_flat = state.board.reshape(-1).tolist()
        stack = state.board.shape[2]   # always 4 for the preset
        for cell_idx in range(16):
            h = h_flat[cell_idx]
            if h == 0:
                continue
            base = cell_idx * stack
            seen = 0
            top_level = h - 1
            for level in range(h):
                v = board_flat[base + level]
                if v == 0:
                    continue
                ch_top_idx = (v - 1) if v > 0 else (-v + 3)   # 0..7
                ch_anywhere = ch_top_idx + 8                  # 8..15
                bit = 1 << ch_anywhere
                if not (seen & bit):
                    seen |= bit
                    flat_view[ch_anywhere, cell_idx] = 1.0
                if level == top_level:
                    flat_view[ch_top_idx, cell_idx] = 1.0

    # ---- channels 16-23 : reserve counts (broadcast in one shot) -------
    # ``state.reserves`` is (2, 4) int8 — flatten to (8,) of float32, then
    # broadcast to the (8, 4, 4) slice in a single numpy assignment.
    tensor[16:24] = (state.reserves.astype(np.float32) * _DENOM_INV).reshape(8, 1, 1)

    # ---- channel 24 : side-to-move = 1 iff P1 --------------------------
    if state.to_move == 0:
        tensor[24, :, :] = 1.0

    # Channels 25 (out-of-board) and 26 (unused-size) are always zero for
    # the 4x4-XL preset — left at zero by the initialisation.
    return tensor


# ---------------------------------------------------------------------------
# Action <-> Move conversion
# ---------------------------------------------------------------------------


def _cell_idx(row: int, col: int) -> int:
    return row * MAX_BOARD + col


def _move_to_action_index_inner(move) -> int:
    if isinstance(move, PlaceFromReserveMove):
        rank = SIZE_RANK[move.size_id]
        cell = _cell_idx(move.to_row, move.to_col)
        return rank * (MAX_BOARD * MAX_BOARD) + cell
    if isinstance(move, MoveOnBoardMove):
        from_idx = _cell_idx(move.from_row, move.from_col)
        to_idx = _cell_idx(move.to_row, move.to_col)
        return PLACE_ACTIONS + to_idx * (MAX_BOARD * MAX_BOARD) + from_idx
    raise TypeError(f"unknown move kind: {move!r}")


def move_to_action_index(state: GameState, move) -> int:
    """Map a Move to its index in the 320-d action space."""
    return _move_to_action_index_inner(move)


def action_index_to_move(state: GameState, index: int):
    """Inverse of ``move_to_action_index``.

    Returns a *candidate* move; legality is up to the caller.
    """
    if index < 0 or index >= TOTAL_ACTIONS:
        raise IndexError(f"action index {index} out of range")
    me_player = Player.P1 if state.to_move == 0 else Player.P2
    if index < PLACE_ACTIONS:
        rank, cell = divmod(index, MAX_BOARD * MAX_BOARD)
        if rank >= NUM_SIZES:
            raise IndexError("size index out of range for this preset")
        row, col = divmod(cell, MAX_BOARD)
        return PlaceFromReserveMove(
            player=me_player,
            size_id=SIZE_IDS[rank],
            to_row=row,
            to_col=col,
        )
    move_idx = index - PLACE_ACTIONS
    to_idx, from_idx = divmod(move_idx, MAX_BOARD * MAX_BOARD)
    tr, tc = divmod(to_idx, MAX_BOARD)
    fr, fc = divmod(from_idx, MAX_BOARD)
    return MoveOnBoardMove(
        player=me_player,
        from_row=fr, from_col=fc,
        to_row=tr, to_col=tc,
    )


# ---------------------------------------------------------------------------
# Legal-action mask
# ---------------------------------------------------------------------------


from .engine import _FLAT_COVER as _FLAT_COVER_FAST
from .engine import _CELLS as _ENGINE_CELLS  # noqa: F401  (kept for symmetry)


# Per-mover-rank "cover row": for each mover_rank in 0..3, a 9-byte string
# where ``row[t + 4]`` is 1 iff a piece of that rank can land on top-signed t.
_COVER_ROWS: tuple[bytes, ...] = tuple(
    bytes(
        1 if _FLAT_COVER_FAST[(t + NUM_SIZES) * NUM_SIZES + mover_rank] else 0
        for t in range(-NUM_SIZES, NUM_SIZES + 1)
    )
    for mover_rank in range(NUM_SIZES)
)


def legal_action_mask(state: GameState) -> np.ndarray:
    """Return a ``(320,)`` bool array; True for legal actions.

    Does *not* materialise per-move dataclass instances (which dominate
    the cost of :func:`legal_moves` for ~86 mid-game moves). The work is
    O(64 + 256) integer comparisons.
    """
    if state.outcome is not None:
        return np.zeros((TOTAL_ACTIONS,), dtype=bool)

    # Use a bytearray (cheap per-element writes) and convert once at the end.
    out = bytearray(TOTAL_ACTIONS)

    me = state.to_move
    top_flat = state.top.reshape(-1).tolist()
    reserves_py = state.reserves[me].tolist()

    # ---- Place actions: rank * 16 + cell --------------------------------
    for rank in range(NUM_SIZES):
        if reserves_py[rank] <= 0:
            continue
        base = rank * 16
        row = _COVER_ROWS[rank]
        for cell_idx in range(16):
            if row[top_flat[cell_idx] + NUM_SIZES]:
                out[base + cell_idx] = 1

    # ---- Move-on-board actions: 64 + to_idx * 16 + from_idx -------------
    if me == 0:
        own_iter = [(i, t) for i, t in enumerate(top_flat) if t > 0]
    else:
        own_iter = [(i, t) for i, t in enumerate(top_flat) if t < 0]
    for from_idx, t in own_iter:
        row = _COVER_ROWS[abs(t) - 1]
        # Pre-compute the destination index base for this from_idx.
        for to_idx in range(16):
            if to_idx == from_idx:
                continue
            if row[top_flat[to_idx] + NUM_SIZES]:
                out[PLACE_ACTIONS + to_idx * 16 + from_idx] = 1

    # Convert bytearray (0/1) to a (320,) bool ndarray. We construct a new
    # bool array (writable) — callers occasionally mutate the returned mask.
    arr = np.empty(TOTAL_ACTIONS, dtype=bool)
    arr[:] = np.frombuffer(out, dtype=np.uint8).astype(bool)
    return arr


__all__ = [
    "NUM_CHANNELS",
    "TOTAL_ACTIONS",
    "PLACE_ACTIONS",
    "state_to_tensor",
    "legal_action_mask",
    "move_to_action_index",
    "action_index_to_move",
]
