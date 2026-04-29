"""Engine parity tests against the TS rules."""

from __future__ import annotations

import pytest

from dttt_train.engine import apply_move, is_terminal, legal_moves
from dttt_train.rules import (
    PRESET_3X3,
    PRESET_4X4_XL,
    GameRules,
    MoveOnBoardMove,
    PieceSize,
    PlaceFromReserveMove,
    Player,
    WinConditionLineOfN,
    initial_state,
)


def test_initial_state_legal_move_count_3x3() -> None:
    state = initial_state(PRESET_3X3)
    moves = legal_moves(state)
    # No on-board pieces yet, so all legal moves are place-from-reserve:
    # 3 sizes x 9 cells = 27.
    assert len(moves) == 27
    assert all(isinstance(m, PlaceFromReserveMove) for m in moves)


def test_initial_state_legal_move_count_4x4() -> None:
    state = initial_state(PRESET_4X4_XL)
    moves = legal_moves(state)
    # 4 sizes x 16 cells = 64.
    assert len(moves) == 64


def test_apply_place_decrements_reserve_and_flips_turn() -> None:
    state = initial_state(PRESET_3X3)
    move = PlaceFromReserveMove(player=Player.P1, size_id="L", to_row=1, to_col=1)
    new_state = apply_move(state, move)
    assert new_state.reserves[Player.P1]["L"] == 1
    assert new_state.reserves[Player.P2]["L"] == 2
    assert new_state.to_move is Player.P2
    assert new_state.ply == 1
    assert len(new_state.board[1][1]) == 1


def test_lineof3_win_horizontal() -> None:
    state = initial_state(PRESET_3X3)
    moves: list = [
        PlaceFromReserveMove(player=Player.P1, size_id="L", to_row=0, to_col=0),
        PlaceFromReserveMove(player=Player.P2, size_id="S", to_row=2, to_col=0),
        PlaceFromReserveMove(player=Player.P1, size_id="L", to_row=0, to_col=1),
        PlaceFromReserveMove(player=Player.P2, size_id="S", to_row=2, to_col=1),
        PlaceFromReserveMove(player=Player.P1, size_id="M", to_row=0, to_col=2),
    ]
    for m in moves:
        state = apply_move(state, m)
    assert is_terminal(state)
    assert state.outcome == Player.P1


def test_cover_smaller_piece_legality() -> None:
    state = initial_state(PRESET_3X3)
    state = apply_move(
        state, PlaceFromReserveMove(player=Player.P1, size_id="S", to_row=0, to_col=0)
    )
    # P2 covers P1's small with a Large piece -> legal.
    legal = legal_moves(state)
    cover = PlaceFromReserveMove(player=Player.P2, size_id="L", to_row=0, to_col=0)
    assert cover in legal


def test_cant_cover_with_smaller() -> None:
    state = initial_state(PRESET_3X3)
    state = apply_move(
        state, PlaceFromReserveMove(player=Player.P1, size_id="L", to_row=0, to_col=0)
    )
    legal = legal_moves(state)
    too_small = PlaceFromReserveMove(player=Player.P2, size_id="S", to_row=0, to_col=0)
    assert too_small not in legal


def test_max_ply_draw() -> None:
    rules = GameRules(
        board_size=3,
        piece_sizes=(PieceSize(id="L", rank=2),),
        pieces_per_size=(2,),
        win_condition=WinConditionLineOfN(n=3),
        max_ply=2,
        draw_by_repetition=99,
    )
    state = initial_state(rules)
    state = apply_move(
        state, PlaceFromReserveMove(player=Player.P1, size_id="L", to_row=0, to_col=0)
    )
    state = apply_move(
        state, PlaceFromReserveMove(player=Player.P2, size_id="L", to_row=2, to_col=2)
    )
    assert is_terminal(state)
    assert state.outcome == "draw"


def test_threefold_repetition_draw() -> None:
    """Move a single piece back and forth between two cells to trigger
    threefold repetition. The minimal preset has 1 size, 1 piece each, on a
    3x3 board, with maxPly large enough that repetition triggers first."""
    rules = GameRules(
        board_size=3,
        piece_sizes=(PieceSize(id="S", rank=0),),
        pieces_per_size=(1,),
        win_condition=WinConditionLineOfN(n=99),  # never wins
        max_ply=999,
        draw_by_repetition=3,
    )
    state = initial_state(rules)
    # Both players place a piece
    state = apply_move(
        state, PlaceFromReserveMove(player=Player.P1, size_id="S", to_row=0, to_col=0)
    )
    state = apply_move(
        state, PlaceFromReserveMove(player=Player.P2, size_id="S", to_row=2, to_col=2)
    )
    # Now the position after the next 4-ply cycle should repeat. Each cycle:
    # P1 moves, P2 moves, P1 moves back, P2 moves back -> same canonical state.
    cycles = 0
    while not is_terminal(state) and cycles < 10:
        # P1: 0,0 <-> 0,1
        p1_top = state.board[0][0]
        if p1_top:
            state = apply_move(
                state,
                MoveOnBoardMove(player=Player.P1, from_row=0, from_col=0, to_row=0, to_col=1),
            )
        else:
            state = apply_move(
                state,
                MoveOnBoardMove(player=Player.P1, from_row=0, from_col=1, to_row=0, to_col=0),
            )
        if is_terminal(state):
            break
        # P2: 2,2 <-> 2,1
        p2_top = state.board[2][2]
        if p2_top:
            state = apply_move(
                state,
                MoveOnBoardMove(player=Player.P2, from_row=2, from_col=2, to_row=2, to_col=1),
            )
        else:
            state = apply_move(
                state,
                MoveOnBoardMove(player=Player.P2, from_row=2, from_col=1, to_row=2, to_col=2),
            )
        cycles += 1
    assert is_terminal(state)
    assert state.outcome == "draw"


def test_outcome_none_for_fresh_state() -> None:
    state = initial_state(PRESET_3X3)
    assert not is_terminal(state)
    assert state.outcome is None


def test_apply_move_does_not_mutate_input() -> None:
    state = initial_state(PRESET_3X3)
    move = PlaceFromReserveMove(player=Player.P1, size_id="M", to_row=0, to_col=0)
    new_state = apply_move(state, move)
    assert state.board[0][0] == []
    assert new_state.board[0][0] != []
    assert state.reserves[Player.P1]["M"] == 2
    assert new_state.reserves[Player.P1]["M"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
