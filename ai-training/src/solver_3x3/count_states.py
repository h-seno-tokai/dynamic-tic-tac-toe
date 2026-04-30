"""Forward BFS over PRESET_3X3 to count unique reachable positions.

Uses the existing engine (``apply_move`` / ``legal_moves``) and a canonical
*position* hash (board stacks + reserves + side-to-move).  The threefold-
repetition counter is intentionally *excluded* from the position key: we are
counting position-equivalence, not game-equivalence.

A position is considered terminal if any of these hold *based on the position
itself* (not on history):
    - a 3-in-a-row of visible top pieces (someone just won), or
    - the side-to-move has zero legal moves (rare).

Because we drop the repetition counter, threefold-draw cannot be detected
from the position alone -- that is fine for *counting* and will be handled
by the actual solver later.

Run from the repo root, e.g.:

    PYTHONPATH=ai-training/src python -m solver_3x3.count_states

or:

    PYTHONPATH=ai-training/src python ai-training/src/solver_3x3/count_states.py
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from typing import Iterable

from dttt_train.engine import apply_move, legal_moves
from dttt_train.rules import (
    GameRules,
    GameState,
    PRESET_3X3,
    Player,
    initial_state,
)


# ---------------------------------------------------------------------------
# Canonical *position* hash (no repetition counter, no history, no ply)
# ---------------------------------------------------------------------------


def position_key(state: GameState) -> bytes:
    """Compact bytes key for board + reserves + side-to-move.

    The encoding does NOT include the ply counter, the move history, or the
    threefold-repetition dictionary, so equal-key states are the same
    *position* even if reached on different game paths.
    """
    rules = state.rules
    parts: list[bytes] = []
    # Board stacks. Each cell is a stack of pieces; encode as
    # b"<owner><sizeIdx>..."  with owners in {1,2} and sizeIdx in 0..255.
    for row in state.board:
        for cell in row:
            for piece in cell:
                owner_byte = 1 if piece.owner is Player.P1 else 2
                size_idx = rules.size_id_to_index(piece.size_id)
                parts.append(bytes((owner_byte, size_idx)))
            parts.append(b"|")  # cell separator
        parts.append(b";")  # row separator
    # Reserves (per player, in rules-defined size order).
    for player in (Player.P1, Player.P2):
        parts.append(b"R")
        parts.append(bytes((1 if player is Player.P1 else 2,)))
        for ps in rules.piece_sizes:
            parts.append(bytes((state.reserves[player].get(ps.id, 0),)))
    # Side to move.
    parts.append(b"T")
    parts.append(bytes((1 if state.to_move is Player.P1 else 2,)))
    return b"".join(parts)


def is_position_terminal(state: GameState) -> bool:
    """Position-only terminal test.

    ``state.outcome`` is set by ``apply_move`` based on win-line, threefold,
    max-ply or zero-legal-moves.  Win-line and zero-legal-moves are pure
    functions of the position; threefold and max-ply are *path-dependent*.
    For a position-equivalence count we treat only win-line + zero-moves as
    terminal.  (The actual solver will add threefold handling.)
    """
    return state.outcome is not None and state.outcome != "draw" or (
        state.outcome == "draw" and len(legal_moves_no_terminal(state)) == 0
    )


def legal_moves_no_terminal(state: GameState):
    # Helper: legal_moves returns [] for terminal states; we want the raw list
    # for a fresh, non-terminal-marked state.  Construct a shallow clone with
    # outcome=None just for the legality check.
    if state.outcome is None:
        return legal_moves(state)
    clone = GameState(
        rules=state.rules,
        board=state.board,
        reserves=state.reserves,
        to_move=state.to_move,
        history=state.history,
        ply=state.ply,
        repetition=state.repetition,
        outcome=None,
    )
    return legal_moves(clone)


# ---------------------------------------------------------------------------
# BFS counter
# ---------------------------------------------------------------------------


def bfs_count(
    rules: GameRules,
    *,
    bail_at: int | None = 50_000_000,
    progress_every: int = 250_000,
) -> dict:
    """Forward BFS from ``initial_state(rules)``.

    Returns a dict with totals and per-ply distributions.  If ``bail_at`` is
    set and the visited set exceeds it, BFS stops early and the result dict
    has ``partial=True``.
    """
    start = initial_state(rules)
    start_key = position_key(start)

    # depth[key] = ply at which we first reached this position.
    depth: dict[bytes, int] = {start_key: 0}
    # is_terminal[key] = True iff the position is a position-terminal state.
    terminal_keys: set[bytes] = set()
    # Per-ply count (using first-discovery ply).
    by_ply: dict[int, int] = {0: 1}

    # Frontier as (state, key). We need the GameState to expand; the key is
    # cached so we don't rehash twice.
    frontier: deque[tuple[GameState, bytes]] = deque()
    frontier.append((start, start_key))
    peak_frontier = 1
    expansions = 0

    t0 = time.perf_counter()
    partial = False

    while frontier:
        state, key = frontier.popleft()
        d = depth[key]

        # If state is terminal at the position level, no children.
        if state.outcome is not None:
            # apply_move flagged this terminal (win, threefold, max-ply, or
            # zero-moves).  Win and zero-moves are position-terminal; the
            # others are path-dependent.  For *counting*, treat any flagged
            # terminal as terminal (the position will not have children
            # discovered along this path; if it is reachable as non-terminal
            # via a different path, BFS will explore it then).
            terminal_keys.add(key)
            continue

        moves = legal_moves(state)
        if not moves:
            terminal_keys.add(key)
            continue

        for mv in moves:
            child = apply_move(state, mv)
            ckey = position_key(child)
            if ckey in depth:
                continue
            depth[ckey] = d + 1
            by_ply[d + 1] = by_ply.get(d + 1, 0) + 1
            frontier.append((child, ckey))

        if len(frontier) > peak_frontier:
            peak_frontier = len(frontier)

        expansions += 1
        if expansions % progress_every == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [bfs] expanded={expansions:,} visited={len(depth):,} "
                f"frontier={len(frontier):,} max_ply={max(by_ply):,} "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )

        if bail_at is not None and len(depth) > bail_at:
            partial = True
            print(
                f"  [bfs] BAIL: visited > {bail_at:,}; stopping early.",
                file=sys.stderr,
                flush=True,
            )
            break

    elapsed = time.perf_counter() - t0
    return {
        "rules_repr": f"board={rules.board_size}x{rules.board_size}, "
        f"sizes={[p.id for p in rules.piece_sizes]}, "
        f"per_size={list(rules.pieces_per_size)}, "
        f"self_cover={rules.allow_self_cover}, "
        f"same_size_cover={rules.allow_same_size_cover}",
        "total_positions": len(depth),
        "by_ply": dict(sorted(by_ply.items())),
        "terminals": len(terminal_keys),
        "non_terminals": len(depth) - len(terminal_keys),
        "peak_frontier": peak_frontier,
        "elapsed_sec": elapsed,
        "partial": partial,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(result: dict) -> str:
    bytes_per_state = 2  # 2 bits W/D/L + ~9 bits best-move => ~11 bits ~= 2 B
    n = result["total_positions"]
    tablebase_bytes = n * bytes_per_state
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PRESET_3X3 forward-BFS position count")
    lines.append("=" * 72)
    lines.append(f"rules:            {result['rules_repr']}")
    lines.append(f"partial:          {result['partial']}")
    lines.append(f"total positions:  {n:,}")
    lines.append(f"  terminals:      {result['terminals']:,}")
    lines.append(f"  non-terminals:  {result['non_terminals']:,}")
    lines.append(f"peak frontier:    {result['peak_frontier']:,}")
    lines.append(f"elapsed:          {result['elapsed_sec']:.2f} s")
    lines.append("")
    lines.append("Distribution by ply (first-discovery depth):")
    for ply, c in result["by_ply"].items():
        lines.append(f"  ply {ply:>3}:  {c:,}")
    lines.append("")
    lines.append("Tablebase size estimate (~2 bytes / state: 2-bit W/D/L + 9-bit move):")
    lines.append(
        f"  {tablebase_bytes:,} bytes  "
        f"(~{tablebase_bytes / 1024 / 1024:.1f} MiB,  "
        f"~{tablebase_bytes / 1024 / 1024 / 1024:.2f} GiB)"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bail-at",
        type=int,
        default=50_000_000,
        help="abort BFS once visited exceeds this (default 50M; use 0 to disable)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=250_000,
        help="print progress every N expansions",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    bail_at = args.bail_at if args.bail_at > 0 else None
    result = bfs_count(
        PRESET_3X3, bail_at=bail_at, progress_every=args.progress_every
    )
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
