"""Python mirror of the TypeScript domain types in ``docs/06_data_model.md``.

The Python engine is a parity implementation; the canonical rules live in TS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Union

# AI universal-network limits (mirror of AI_LIMITS in the TS code).
MAX_BOARD: int = 4
MAX_PIECE_SIZES: int = 4
MAX_PIECES_PER_SIZE: int = 3

# Policy action-space layout. Place section: MAX_PIECE_SIZES * MAX_BOARD^2.
# Move section: MAX_BOARD^2 * MAX_BOARD^2 (to-cell x from-cell).
PLACE_ACTIONS: int = MAX_PIECE_SIZES * MAX_BOARD * MAX_BOARD  # 64
MOVE_ACTIONS: int = (MAX_BOARD * MAX_BOARD) * (MAX_BOARD * MAX_BOARD)  # 256
TOTAL_ACTIONS: int = PLACE_ACTIONS + MOVE_ACTIONS  # 320


class Player(str, Enum):
    """Player identifier; string-valued so it serialises trivially."""

    P1 = "P1"
    P2 = "P2"

    def opponent(self) -> "Player":
        return Player.P2 if self is Player.P1 else Player.P1


@dataclass(frozen=True)
class PieceSize:
    """Size tier. Higher ``rank`` means physically larger."""

    id: str
    rank: int
    display_name_ja: str = ""
    display_name_en: str = ""


@dataclass(frozen=True)
class Piece:
    owner: Player
    size_id: str


@dataclass(frozen=True)
class WinConditionLineOfN:
    n: int
    kind: Literal["lineOfN"] = "lineOfN"


WinCondition = WinConditionLineOfN


@dataclass(frozen=True)
class GameRules:
    """Canonical rule object - all rule-driven behaviour reads from here."""

    board_size: int
    piece_sizes: tuple[PieceSize, ...]
    pieces_per_size: tuple[int, ...]
    win_condition: WinCondition
    allow_same_size_cover: bool = False
    allow_self_cover: bool = True
    max_ply: int = 60
    draw_by_repetition: int = 3

    def __post_init__(self) -> None:
        if len(self.piece_sizes) != len(self.pieces_per_size):
            raise ValueError("piece_sizes and pieces_per_size must have equal length")
        if self.board_size < 1 or self.board_size > MAX_BOARD:
            raise ValueError(f"board_size out of supported range (<= {MAX_BOARD})")
        if len(self.piece_sizes) > MAX_PIECE_SIZES:
            raise ValueError(f"too many piece sizes (>{MAX_PIECE_SIZES})")
        for n in self.pieces_per_size:
            if n > MAX_PIECES_PER_SIZE:
                raise ValueError(f"pieces_per_size entry {n} > {MAX_PIECES_PER_SIZE}")

    def size_id_to_rank(self, size_id: str) -> int:
        for ps in self.piece_sizes:
            if ps.id == size_id:
                return ps.rank
        raise KeyError(f"unknown size id: {size_id}")

    def size_id_to_index(self, size_id: str) -> int:
        for i, ps in enumerate(self.piece_sizes):
            if ps.id == size_id:
                return i
        raise KeyError(f"unknown size id: {size_id}")


@dataclass(frozen=True)
class PlaceFromReserveMove:
    player: Player
    size_id: str
    to_row: int
    to_col: int
    kind: Literal["placeFromReserve"] = "placeFromReserve"


@dataclass(frozen=True)
class MoveOnBoardMove:
    player: Player
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    kind: Literal["moveOnBoard"] = "moveOnBoard"


Move = Union[PlaceFromReserveMove, MoveOnBoardMove]


# A cell is a stack of pieces; the last element is the visible top.
Cell = list[Piece]
Board = list[list[Cell]]
Reserve = dict[str, int]


@dataclass
class GameState:
    """Mutable-style game state. Engine functions return new instances; we do
    not mutate in place to keep MCTS / selfplay reasoning straightforward."""

    rules: GameRules
    board: Board
    reserves: dict[Player, Reserve]
    to_move: Player
    history: list[Move] = field(default_factory=list)
    ply: int = 0
    repetition: dict[str, int] = field(default_factory=dict)
    outcome: Player | Literal["draw"] | None = None


# ---------------------------------------------------------------------------
# Presets - mirror of TS PRESET_3X3 / PRESET_4X4_XL
# ---------------------------------------------------------------------------

PRESET_3X3: GameRules = GameRules(
    board_size=3,
    piece_sizes=(
        PieceSize(id="S", rank=0, display_name_ja="小", display_name_en="Small"),
        PieceSize(id="M", rank=1, display_name_ja="中", display_name_en="Medium"),
        PieceSize(id="L", rank=2, display_name_ja="大", display_name_en="Large"),
    ),
    pieces_per_size=(2, 2, 2),
    win_condition=WinConditionLineOfN(n=3),
    allow_same_size_cover=False,
    allow_self_cover=True,
    max_ply=60,
    draw_by_repetition=3,
)

PRESET_4X4_XL: GameRules = GameRules(
    board_size=4,
    piece_sizes=(
        PieceSize(id="S", rank=0, display_name_ja="小", display_name_en="Small"),
        PieceSize(id="M", rank=1, display_name_ja="中", display_name_en="Medium"),
        PieceSize(id="L", rank=2, display_name_ja="大", display_name_en="Large"),
        PieceSize(id="XL", rank=3, display_name_ja="巨大", display_name_en="Huge"),
    ),
    pieces_per_size=(3, 3, 3, 3),
    win_condition=WinConditionLineOfN(n=4),
    allow_same_size_cover=False,
    allow_self_cover=True,
    max_ply=120,
    draw_by_repetition=3,
)


def initial_state(rules: GameRules) -> GameState:
    """Build the empty starting state for ``rules``."""
    board: Board = [[[] for _ in range(rules.board_size)] for _ in range(rules.board_size)]
    reserves: dict[Player, Reserve] = {
        Player.P1: {ps.id: n for ps, n in zip(rules.piece_sizes, rules.pieces_per_size)},
        Player.P2: {ps.id: n for ps, n in zip(rules.piece_sizes, rules.pieces_per_size)},
    }
    state = GameState(
        rules=rules,
        board=board,
        reserves=reserves,
        to_move=Player.P1,
        history=[],
        ply=0,
        repetition={},
        outcome=None,
    )
    return state
