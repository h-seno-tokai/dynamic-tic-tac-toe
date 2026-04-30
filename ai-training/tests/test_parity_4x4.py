"""Parity tests: fast 4x4 engine (``dttt_train_4x4``) vs old ``dttt_train``.

Plays 50 deterministic random games on the 4x4 XL preset and asserts:

* both engines produce the same set of legal moves at every ply
* both engines reach the same outcome (winner / draw / None)
* both engines reach the same ply count
"""

from __future__ import annotations

import random

import pytest

from dttt_train.engine import (
    apply_move as legacy_apply,
    is_terminal as legacy_is_terminal,
    legal_moves as legacy_legal_moves,
)
from dttt_train.rules import (
    PRESET_4X4_XL,
    MoveOnBoardMove as LegacyMoveOnBoard,
    PlaceFromReserveMove as LegacyPlace,
    Player,
    initial_state as legacy_initial_state,
)

from dttt_train_4x4.engine import (
    apply_move as fast_apply,
    initial_state as fast_initial_state,
    is_terminal as fast_is_terminal,
    legal_moves as fast_legal_moves,
)


def _move_key(m: object) -> tuple:
    """Hashable key used to compare legal-move sets between engines."""
    if isinstance(m, LegacyPlace):
        return ("P", m.player.value, m.size_id, m.to_row, m.to_col)
    if isinstance(m, LegacyMoveOnBoard):
        return ("M", m.player.value, m.from_row, m.from_col, m.to_row, m.to_col)
    raise TypeError(f"unknown move kind: {m!r}")


@pytest.mark.parametrize("seed", list(range(50)))
def test_engine_parity_4x4_random_game(seed: int) -> None:
    rng = random.Random(seed)

    legacy_state = legacy_initial_state(PRESET_4X4_XL)
    fast_state = fast_initial_state()

    while True:
        legacy_term = legacy_is_terminal(legacy_state)
        fast_term = fast_is_terminal(fast_state)
        assert legacy_term == fast_term, (
            f"terminal mismatch at ply={legacy_state.ply} seed={seed}"
        )
        if legacy_term:
            break

        legacy_moves = legacy_legal_moves(legacy_state)
        fast_moves = fast_legal_moves(fast_state)

        legacy_keys = {_move_key(m) for m in legacy_moves}
        fast_keys = {_move_key(m) for m in fast_moves}
        assert legacy_keys == fast_keys, (
            f"legal_moves mismatch at ply={legacy_state.ply} seed={seed}\n"
            f" only-legacy={legacy_keys - fast_keys}\n"
            f" only-fast  ={fast_keys - legacy_keys}"
        )

        if not legacy_moves:
            break

        # Pick a deterministic move via sorted move-keys + rng.
        sorted_keys = sorted(legacy_keys)
        chosen_key = rng.choice(sorted_keys)
        legacy_move = next(m for m in legacy_moves if _move_key(m) == chosen_key)
        fast_move = next(m for m in fast_moves if _move_key(m) == chosen_key)

        legacy_state = legacy_apply(legacy_state, legacy_move)
        fast_state = fast_apply(fast_state, fast_move)

        # Outcome and ply must agree at every step.
        assert legacy_state.ply == fast_state.ply, (
            f"ply mismatch at seed={seed}: legacy={legacy_state.ply} fast={fast_state.ply}"
        )
        # Side-to-move must agree.
        legacy_to_move = legacy_state.to_move
        fast_to_move = Player.P1 if fast_state.to_move == 0 else Player.P2
        assert legacy_to_move == fast_to_move, (
            f"to_move mismatch at ply={legacy_state.ply} seed={seed}"
        )

    # Final outcome equality.
    legacy_outcome = legacy_state.outcome
    fast_outcome = fast_state.outcome
    assert legacy_outcome == fast_outcome, (
        f"outcome mismatch seed={seed}: legacy={legacy_outcome} fast={fast_outcome}"
    )
    assert legacy_state.ply == fast_state.ply, (
        f"final ply mismatch seed={seed}: legacy={legacy_state.ply} fast={fast_state.ply}"
    )
