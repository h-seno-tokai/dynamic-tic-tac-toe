"""Pure-Python rule engine, mirror of the TypeScript engine.

Implements:

* ``legal_moves(state)``           - enumerate legal moves for ``state.to_move``
* ``apply_move(state, move)``      - return a new state after applying ``move``
* ``is_terminal(state)`` / ``outcome(state)``
* threefold-repetition + ``maxPly`` draw rules
* covering rules (``allow_same_size_cover`` / ``allow_self_cover``)
"""

from __future__ import annotations

import copy
from typing import Literal

from .rules import (
    Board,
    GameRules,
    GameState,
    MoveOnBoardMove,
    Piece,
    PlaceFromReserveMove,
    Player,
    initial_state,
)

Outcome = Player | Literal["draw"] | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _top(cell: list[Piece]) -> Piece | None:
    return cell[-1] if cell else None


def _can_cover(rules: GameRules, top: Piece | None, mover: Piece) -> bool:
    """Can ``mover`` legally land on ``top``?"""
    if top is None:
        return True
    mover_rank = rules.size_id_to_rank(mover.size_id)
    top_rank = rules.size_id_to_rank(top.size_id)
    if mover_rank > top_rank:
        # Larger always covers (subject to self-cover rule).
        if top.owner == mover.owner and not rules.allow_self_cover:
            return False
        return True
    if mover_rank == top_rank:
        if not rules.allow_same_size_cover:
            return False
        if top.owner == mover.owner and not rules.allow_self_cover:
            return False
        return True
    # Smaller never covers larger.
    return False


def _state_hash(state: GameState) -> str:
    """Canonical hash used for threefold-repetition detection.

    Encodes board stacks, reserves, and side-to-move; history is intentionally
    excluded.
    """
    parts: list[str] = []
    for row in state.board:
        for cell in row:
            parts.append("/".join(f"{p.owner.value}:{p.size_id}" for p in cell))
            parts.append("|")
        parts.append(";")
    parts.append("R1=")
    for ps in state.rules.piece_sizes:
        parts.append(f"{ps.id}{state.reserves[Player.P1].get(ps.id, 0)}")
    parts.append(",R2=")
    for ps in state.rules.piece_sizes:
        parts.append(f"{ps.id}{state.reserves[Player.P2].get(ps.id, 0)}")
    parts.append(f",T={state.to_move.value}")
    return "".join(parts)


def _check_winner(rules: GameRules, board: Board) -> Player | None:
    """Apply the current ``WinCondition`` (only ``lineOfN`` is implemented)."""
    n = rules.win_condition.n
    bs = rules.board_size

    def line_owner(cells: list[list[Piece]]) -> Player | None:
        if any(not c for c in cells):
            return None
        owners = {c[-1].owner for c in cells}
        return next(iter(owners)) if len(owners) == 1 else None

    # Rows
    for r in range(bs):
        for c0 in range(bs - n + 1):
            owner = line_owner([board[r][c0 + k] for k in range(n)])
            if owner is not None:
                return owner
    # Columns
    for c in range(bs):
        for r0 in range(bs - n + 1):
            owner = line_owner([board[r0 + k][c] for k in range(n)])
            if owner is not None:
                return owner
    # Diagonals \
    for r0 in range(bs - n + 1):
        for c0 in range(bs - n + 1):
            owner = line_owner([board[r0 + k][c0 + k] for k in range(n)])
            if owner is not None:
                return owner
    # Diagonals /
    for r0 in range(bs - n + 1):
        for c0 in range(n - 1, bs):
            owner = line_owner([board[r0 + k][c0 - k] for k in range(n)])
            if owner is not None:
                return owner
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def legal_moves(state: GameState) -> list[PlaceFromReserveMove | MoveOnBoardMove]:
    if state.outcome is not None:
        return []

    rules = state.rules
    bs = rules.board_size
    me = state.to_move
    moves: list[PlaceFromReserveMove | MoveOnBoardMove] = []

    # Place-from-reserve moves
    for ps in rules.piece_sizes:
        if state.reserves[me].get(ps.id, 0) <= 0:
            continue
        mover = Piece(owner=me, size_id=ps.id)
        for r in range(bs):
            for c in range(bs):
                top = _top(state.board[r][c])
                if _can_cover(rules, top, mover):
                    moves.append(
                        PlaceFromReserveMove(player=me, size_id=ps.id, to_row=r, to_col=c)
                    )

    # Move-on-board: only the player's own top piece may be lifted.
    for fr in range(bs):
        for fc in range(bs):
            top = _top(state.board[fr][fc])
            if top is None or top.owner != me:
                continue
            for tr in range(bs):
                for tc in range(bs):
                    if (fr, fc) == (tr, tc):
                        continue
                    target_top = _top(state.board[tr][tc])
                    if _can_cover(rules, target_top, top):
                        moves.append(
                            MoveOnBoardMove(
                                player=me,
                                from_row=fr,
                                from_col=fc,
                                to_row=tr,
                                to_col=tc,
                            )
                        )
    return moves


def apply_move(state: GameState, move: PlaceFromReserveMove | MoveOnBoardMove) -> GameState:
    """Return a new state with ``move`` applied. Pure (no mutation of input)."""
    if state.outcome is not None:
        raise ValueError("cannot apply a move to a terminal state")

    new_board: Board = [[list(cell) for cell in row] for row in state.board]
    new_reserves = {p: dict(r) for p, r in state.reserves.items()}

    if isinstance(move, PlaceFromReserveMove):
        if new_reserves[move.player].get(move.size_id, 0) <= 0:
            raise ValueError("no reserve piece of that size")
        new_reserves[move.player][move.size_id] -= 1
        new_board[move.to_row][move.to_col].append(
            Piece(owner=move.player, size_id=move.size_id)
        )
    else:
        src = new_board[move.from_row][move.from_col]
        if not src:
            raise ValueError("source cell is empty")
        piece = src.pop()
        if piece.owner != move.player:
            raise ValueError("can only move own piece")
        new_board[move.to_row][move.to_col].append(piece)

    new_history = list(state.history) + [move]
    new_ply = state.ply + 1
    new_to_move = state.to_move.opponent()

    new_state = GameState(
        rules=state.rules,
        board=new_board,
        reserves=new_reserves,
        to_move=new_to_move,
        history=new_history,
        ply=new_ply,
        repetition=copy.copy(state.repetition),
        outcome=None,
    )

    # Repetition / terminal detection
    h = _state_hash(new_state)
    new_state.repetition[h] = new_state.repetition.get(h, 0) + 1

    winner = _check_winner(state.rules, new_state.board)
    if winner is not None:
        new_state.outcome = winner
    elif new_state.repetition[h] >= state.rules.draw_by_repetition:
        new_state.outcome = "draw"
    elif new_state.ply >= state.rules.max_ply:
        new_state.outcome = "draw"
    elif not legal_moves(new_state):
        # Rare in Gobblet but defined for completeness.
        new_state.outcome = "draw"

    return new_state


def is_terminal(state: GameState) -> bool:
    return state.outcome is not None


def outcome(state: GameState) -> Outcome:
    return state.outcome


__all__ = [
    "initial_state",
    "legal_moves",
    "apply_move",
    "is_terminal",
    "outcome",
]
