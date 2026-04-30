"""Rule constants and Move dataclasses for the fast 4x4-XL engine.

We deliberately re-export the *same* ``Player``, ``PlaceFromReserveMove`` and
``MoveOnBoardMove`` dataclasses as :mod:`dttt_train.rules` so that the existing
augmentation / encoding / parity-test helpers can use either engine
interchangeably without dataclass-identity surprises.

The engine is hard-coded to ``PRESET_4X4_XL``:

* ``board_size = 4``
* 4 sizes (S=rank 0 .. XL=rank 3), 3 pieces of each per player
* ``allow_self_cover = True``, ``allow_same_size_cover = False``
* ``win_condition = lineOfN(n=4)``
* ``max_ply = 120``, ``draw_by_repetition = 3``

Stack-height bound: with no-same-size-cover, each cell can hold at most one
piece of each rank (because covering needs a strictly larger rank), so the
maximum stack height is ``MAX_PIECE_SIZES = 4``.  We therefore use a fixed
``(4, 4, 4) int8`` board buffer in the engine.
"""

from __future__ import annotations

# Re-export from dttt_train.rules so callers and tests share dataclasses.
from dttt_train.rules import (  # noqa: F401
    MAX_BOARD,
    MAX_PIECE_SIZES,
    MAX_PIECES_PER_SIZE,
    MOVE_ACTIONS,
    PLACE_ACTIONS,
    TOTAL_ACTIONS,
    GameRules,
    Move,
    MoveOnBoardMove,
    PieceSize,
    Piece,
    PlaceFromReserveMove,
    Player,
    PRESET_4X4_XL,
    WinConditionLineOfN,
)

# ---------------------------------------------------------------------------
# Hard-coded preset constants for the fast engine
# ---------------------------------------------------------------------------

BOARD_SIZE: int = 4
NUM_CELLS: int = 16
NUM_SIZES: int = 4              # S, M, L, XL  (rank 0..3)
PIECES_PER_SIZE: int = 3
MAX_STACK: int = NUM_SIZES      # 4 — tight bound under no-same-size-cover
WIN_N: int = 4
MAX_PLY: int = 120
DRAW_BY_REPETITION: int = 3
ALLOW_SELF_COVER: bool = True
ALLOW_SAME_SIZE_COVER: bool = False

# Size-id <-> rank tables (index = rank, value = id, and inverse).
SIZE_IDS: tuple[str, ...] = ("S", "M", "L", "XL")
SIZE_RANK: dict[str, int] = {sid: i for i, sid in enumerate(SIZE_IDS)}

# Outcome string constants (mirror of ``GameState.outcome`` typing).
OUTCOME_DRAW: str = "draw"


__all__ = [
    "BOARD_SIZE", "NUM_CELLS", "NUM_SIZES", "PIECES_PER_SIZE", "MAX_STACK",
    "WIN_N", "MAX_PLY", "DRAW_BY_REPETITION",
    "ALLOW_SELF_COVER", "ALLOW_SAME_SIZE_COVER",
    "SIZE_IDS", "SIZE_RANK", "OUTCOME_DRAW",
    # Re-exports (for compat with old callers + tests)
    "MAX_BOARD", "MAX_PIECE_SIZES", "MAX_PIECES_PER_SIZE",
    "PLACE_ACTIONS", "MOVE_ACTIONS", "TOTAL_ACTIONS",
    "GameRules", "Move", "MoveOnBoardMove", "PieceSize", "Piece",
    "PlaceFromReserveMove", "Player", "PRESET_4X4_XL", "WinConditionLineOfN",
]
