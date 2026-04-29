"""State -> tensor encoding (27 channels) and 320-d legal-action mask.

Channel layout (see ``docs/07_ai_design.md`` 2.1):

  0-3   P1 top-of-stack one-hot per size (size index = piece_sizes order)
  4-7   P2 top-of-stack
  8-11  P1 anywhere-in-stack (covered + visible)
  12-15 P2 anywhere-in-stack
  16-19 P1 reserve count, normalised 0..1, broadcast 4x4
  20-23 P2 reserve count, broadcast
  24    side-to-move (1 if P1)
  25    out-of-board mask (1 on cells outside the current board_size)
  26    unused-size mask (1 on size channels not present in current rules)

Action layout (320 = 64 + 256):
  [0      , 64 )  PlaceFromReserve: index = size_idx * 16 + (row*4 + col)
  [64     , 320)  MoveOnBoard:      index = 64 + to_idx * 16 + from_idx
                                    where idx = row * 4 + col
"""

from __future__ import annotations

import numpy as np

from .rules import (
    MAX_BOARD,
    MAX_PIECE_SIZES,
    MAX_PIECES_PER_SIZE,
    MOVE_ACTIONS,
    PLACE_ACTIONS,
    TOTAL_ACTIONS,
    GameState,
    Move,
    MoveOnBoardMove,
    PlaceFromReserveMove,
    Player,
)
from .engine import legal_moves

NUM_CHANNELS: int = 27


def _cell_idx(row: int, col: int) -> int:
    return row * MAX_BOARD + col


def state_to_tensor(state: GameState) -> np.ndarray:
    """Return a ``(27, 4, 4)`` float32 tensor for ``state``.

    The tensor is in NCHW order (channels first) so it can be fed directly
    into a PyTorch ``Conv2d`` after adding a batch dimension.
    """
    rules = state.rules
    bs = rules.board_size
    tensor = np.zeros((NUM_CHANNELS, MAX_BOARD, MAX_BOARD), dtype=np.float32)

    # Map size_id -> universal-size index (0..MAX_PIECE_SIZES-1).
    # We assume rules.piece_sizes is ordered by rank ascending.
    size_id_to_idx: dict[str, int] = {ps.id: i for i, ps in enumerate(rules.piece_sizes)}

    # Channels 0-7 (top), 8-15 (anywhere)
    for r in range(bs):
        for c in range(bs):
            cell = state.board[r][c]
            if not cell:
                continue
            top = cell[-1]
            top_idx = size_id_to_idx[top.size_id]
            top_channel = top_idx + (0 if top.owner is Player.P1 else 4)
            tensor[top_channel, r, c] = 1.0

            seen_sizes: dict[tuple[Player, int], bool] = {}
            for piece in cell:
                idx = size_id_to_idx[piece.size_id]
                key = (piece.owner, idx)
                if key in seen_sizes:
                    continue
                seen_sizes[key] = True
                anywhere_channel = 8 + idx + (0 if piece.owner is Player.P1 else 4)
                tensor[anywhere_channel, r, c] = 1.0

    # Channels 16-23: reserve counts, normalised to [0, 1].
    denom = float(MAX_PIECES_PER_SIZE)
    for ps in rules.piece_sizes:
        idx = size_id_to_idx[ps.id]
        p1 = state.reserves[Player.P1].get(ps.id, 0) / denom
        p2 = state.reserves[Player.P2].get(ps.id, 0) / denom
        tensor[16 + idx, :, :] = p1
        tensor[20 + idx, :, :] = p2

    # Channel 24: side-to-move = 1 iff P1
    if state.to_move is Player.P1:
        tensor[24, :, :] = 1.0

    # Channel 25: out-of-board mask (cells outside [0, bs))
    if bs < MAX_BOARD:
        tensor[25, bs:, :] = 1.0
        tensor[25, :, bs:] = 1.0

    # Channel 26: unused-size mask (size indices >= len(piece_sizes))
    used_sizes = len(rules.piece_sizes)
    if used_sizes < MAX_PIECE_SIZES:
        # Mark the entire 4x4 plane: the network sees "this size doesn't
        # exist for this preset". Using a uniform plane is simplest.
        tensor[26, :, :] = 1.0  # default-on; we zero it below if all sizes used
    # If all sizes used, the channel stays zero. Otherwise we keep the plane
    # at 1.0 to indicate "preset has unused size slots". (The network learns
    # which size channels are zeroed out via the encoded data itself.)

    return tensor


def _move_to_action_index(move: Move, size_id_to_idx: dict[str, int]) -> int:
    if isinstance(move, PlaceFromReserveMove):
        sidx = size_id_to_idx[move.size_id]
        cell = _cell_idx(move.to_row, move.to_col)
        return sidx * (MAX_BOARD * MAX_BOARD) + cell
    if isinstance(move, MoveOnBoardMove):
        from_idx = _cell_idx(move.from_row, move.from_col)
        to_idx = _cell_idx(move.to_row, move.to_col)
        return PLACE_ACTIONS + to_idx * (MAX_BOARD * MAX_BOARD) + from_idx
    raise TypeError(f"unknown move kind: {move!r}")


def move_to_action_index(state: GameState, move: Move) -> int:
    """Map a Move to its index in the 320-d action space."""
    size_id_to_idx = {ps.id: i for i, ps in enumerate(state.rules.piece_sizes)}
    return _move_to_action_index(move, size_id_to_idx)


def action_index_to_move(state: GameState, index: int) -> Move:
    """Inverse of ``move_to_action_index``.

    Note: this returns a *candidate* move; callers should verify legality
    via the engine. Used by MCTS for action lookup.
    """
    rules = state.rules
    if index < 0 or index >= TOTAL_ACTIONS:
        raise IndexError(f"action index {index} out of range")
    if index < PLACE_ACTIONS:
        sidx, cell = divmod(index, MAX_BOARD * MAX_BOARD)
        if sidx >= len(rules.piece_sizes):
            raise IndexError("size index out of range for this preset")
        row, col = divmod(cell, MAX_BOARD)
        return PlaceFromReserveMove(
            player=state.to_move,
            size_id=rules.piece_sizes[sidx].id,
            to_row=row,
            to_col=col,
        )
    move_idx = index - PLACE_ACTIONS
    to_idx, from_idx = divmod(move_idx, MAX_BOARD * MAX_BOARD)
    tr, tc = divmod(to_idx, MAX_BOARD)
    fr, fc = divmod(from_idx, MAX_BOARD)
    return MoveOnBoardMove(
        player=state.to_move,
        from_row=fr,
        from_col=fc,
        to_row=tr,
        to_col=tc,
    )


def legal_action_mask(state: GameState) -> np.ndarray:
    """Return a ``(320,)`` bool array; True for legal actions."""
    mask = np.zeros((TOTAL_ACTIONS,), dtype=bool)
    size_id_to_idx = {ps.id: i for i, ps in enumerate(state.rules.piece_sizes)}
    for mv in legal_moves(state):
        idx = _move_to_action_index(mv, size_id_to_idx)
        mask[idx] = True
    return mask


__all__ = [
    "NUM_CHANNELS",
    "TOTAL_ACTIONS",
    "PLACE_ACTIONS",
    "MOVE_ACTIONS",
    "state_to_tensor",
    "legal_action_mask",
    "move_to_action_index",
    "action_index_to_move",
]
