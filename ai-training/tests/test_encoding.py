"""Encoding tests: tensor shape, channel semantics, action mask shape."""

from __future__ import annotations

import numpy as np

from dttt_train.encoding import (
    NUM_CHANNELS,
    TOTAL_ACTIONS,
    action_index_to_move,
    legal_action_mask,
    move_to_action_index,
    state_to_tensor,
)
from dttt_train.engine import apply_move
from dttt_train.rules import (
    PRESET_3X3,
    PRESET_4X4_XL,
    PlaceFromReserveMove,
    Player,
    initial_state,
)


def test_initial_3x3_tensor_shape() -> None:
    state = initial_state(PRESET_3X3)
    t = state_to_tensor(state)
    assert t.shape == (NUM_CHANNELS, 4, 4)
    assert t.dtype == np.float32


def test_initial_3x3_channel_values() -> None:
    state = initial_state(PRESET_3X3)
    t = state_to_tensor(state)

    # No pieces on board, so top-of-stack and anywhere channels are zero.
    assert np.all(t[0:8] == 0)
    assert np.all(t[8:16] == 0)

    # Reserve channels: 3x3 preset uses sizes S/M/L (indices 0,1,2) with 2 each.
    # Values are normalised by MAX_PIECES_PER_SIZE = 3 -> 2/3.
    assert np.allclose(t[16, :, :], 2.0 / 3.0)
    assert np.allclose(t[17, :, :], 2.0 / 3.0)
    assert np.allclose(t[18, :, :], 2.0 / 3.0)
    assert np.all(t[19, :, :] == 0)  # XL slot unused
    assert np.allclose(t[20, :, :], 2.0 / 3.0)
    assert np.all(t[23, :, :] == 0)

    # Side-to-move = P1 -> channel 24 all ones.
    assert np.all(t[24, :, :] == 1.0)

    # Out-of-board mask: row/col >= 3 are out-of-board for the 3x3 preset.
    assert np.all(t[25, 3:, :] == 1.0)
    assert np.all(t[25, :, 3:] == 1.0)
    assert np.all(t[25, :3, :3] == 0.0)

    # Unused-size mask: 3x3 preset omits XL -> channel 26 should be on.
    assert np.all(t[26, :, :] == 1.0)


def test_initial_4x4_unused_size_channel_zero() -> None:
    state = initial_state(PRESET_4X4_XL)
    t = state_to_tensor(state)
    # All four sizes are used; out-of-board mask is zero everywhere.
    assert np.all(t[25, :, :] == 0.0)


def test_top_channel_after_place() -> None:
    state = initial_state(PRESET_3X3)
    state = apply_move(
        state, PlaceFromReserveMove(player=Player.P1, size_id="L", to_row=1, to_col=1)
    )
    t = state_to_tensor(state)
    # P1 + size L (index 2) => top channel 2 is set at (1,1).
    assert t[2, 1, 1] == 1.0
    # Anywhere channel for the same is also set.
    assert t[8 + 2, 1, 1] == 1.0
    # Side-to-move flipped to P2 -> channel 24 all zeros.
    assert np.all(t[24, :, :] == 0.0)


def test_legal_action_mask_shape_and_count() -> None:
    state = initial_state(PRESET_3X3)
    mask = legal_action_mask(state)
    assert mask.shape == (TOTAL_ACTIONS,)
    assert mask.dtype == bool
    # 3 sizes x 9 cells = 27 legal place actions in the empty 3x3 state.
    assert int(mask.sum()) == 27


def test_action_round_trip() -> None:
    state = initial_state(PRESET_3X3)
    move = PlaceFromReserveMove(player=Player.P1, size_id="M", to_row=2, to_col=1)
    idx = move_to_action_index(state, move)
    decoded = action_index_to_move(state, idx)
    assert decoded == move
