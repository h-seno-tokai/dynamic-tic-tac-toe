"""Profile the pure-Python game engine to identify AlphaZero self-play bottlenecks.

Run with PYTHONPATH=ai-training/src:
    python ai-training/scripts/profile_engine.py
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import random
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path even if PYTHONPATH was not set.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dttt_train.engine import (  # noqa: E402
    apply_move,
    legal_moves,
    _state_hash,
)
from dttt_train.encoding import state_to_tensor  # noqa: E402
from dttt_train.rules import (  # noqa: E402
    PRESET_3X3,
    PRESET_4X4_XL,
    GameRules,
    GameState,
    initial_state,
)


# ---------------------------------------------------------------------------
# Random self-play game (deterministic with seed)
# ---------------------------------------------------------------------------

def play_random_game(rules: GameRules, rng: random.Random) -> int:
    """Play one random game; return number of plies played."""
    state = initial_state(rules)
    plies = 0
    while state.outcome is None:
        moves = legal_moves(state)
        if not moves:
            break
        mv = rng.choice(moves)
        state = apply_move(state, mv)
        plies += 1
    return plies


def play_n_games(rules: GameRules, n: int, base_seed: int) -> tuple[int, int]:
    """Play n games sequentially; return (total_plies, n_games)."""
    total_plies = 0
    for i in range(n):
        rng = random.Random(base_seed + i)
        total_plies += play_random_game(rules, rng)
    return total_plies, n


# ---------------------------------------------------------------------------
# Mid-game state capture
# ---------------------------------------------------------------------------

def midgame_state(rules: GameRules, seed: int, target_ply: int) -> GameState:
    """Play a random game until ply >= target_ply (or terminal); return state."""
    rng = random.Random(seed)
    state = initial_state(rules)
    while state.outcome is None and state.ply < target_ply:
        moves = legal_moves(state)
        if not moves:
            break
        mv = rng.choice(moves)
        state = apply_move(state, mv)
    return state


# ---------------------------------------------------------------------------
# cProfile run
# ---------------------------------------------------------------------------

def run_cprofile(rules: GameRules, n_games: int, base_seed: int, label: str) -> float:
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    total_plies, _ = play_n_games(rules, n_games, base_seed)
    pr.disable()
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 78)
    print(f"cProfile: {label} - {n_games} random self-play games")
    print(f"  total plies = {total_plies}, wall = {elapsed*1000:.1f} ms"
          f" ({elapsed/total_plies*1e6:.1f} us/ply)")
    print("=" * 78)

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    # Top 20 by cumulative time, restricted to dttt_train (skip stdlib noise)
    ps.print_stats(40)
    text = s.getvalue()
    # Print first ~60 lines (header + top entries)
    for line in text.splitlines()[:60]:
        print(line)

    # Also dump a "tottime" view (self time excluding subcalls)
    print("\n-- sorted by tottime (self time) --")
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats(pstats.SortKey.TIME)
    ps2.print_stats(20)
    for line in s2.getvalue().splitlines()[:32]:
        print(line)
    return elapsed


# ---------------------------------------------------------------------------
# Micro-benchmarks
# ---------------------------------------------------------------------------

def bench(label: str, fn, n: int, wall_time_per_game: float | None = None) -> None:
    # Warm-up
    for _ in range(min(50, n // 10)):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - t0
    per = elapsed / n * 1e6
    extra = ""
    if wall_time_per_game is not None and wall_time_per_game > 0:
        extra = f"  (~{per/1e6 / wall_time_per_game * 100:.2f}% of one game wall)"
    print(f"  {label:<48} n={n:>6}  total={elapsed*1000:8.2f} ms"
          f"  per-call={per:8.2f} us{extra}")


def run_microbench(rules: GameRules, label: str, wall_per_game: float | None) -> None:
    print()
    print("-" * 78)
    print(f"Micro-bench: {label}")
    print("-" * 78)

    s0 = initial_state(rules)
    target = 8 if rules is PRESET_4X4_XL else 4
    s_mid = midgame_state(rules, seed=1234, target_ply=target)
    moves_mid = legal_moves(s_mid)
    # If we landed in a terminal/empty-move state, walk back to a non-terminal ancestor
    if not moves_mid:
        for seed_try in range(2, 50):
            s_mid = midgame_state(rules, seed=seed_try, target_ply=target)
            moves_mid = legal_moves(s_mid)
            if moves_mid:
                break
    first_legal = moves_mid[0]
    n_legal_mid = len(moves_mid)
    print(f"  initial branching = {len(legal_moves(s0))}, "
          f"mid-game branching = {n_legal_mid} (ply={s_mid.ply})")

    bench("legal_moves(initial_state)", lambda: legal_moves(s0), 1000, wall_per_game)
    bench("legal_moves(mid-game)",       lambda: legal_moves(s_mid), 1000, wall_per_game)
    bench("apply_move(mid-game, first)", lambda: apply_move(s_mid, first_legal), 1000, wall_per_game)
    bench("state_to_tensor(mid-game)",   lambda: state_to_tensor(s_mid), 1000, wall_per_game)
    bench("_state_hash(mid-game)",       lambda: _state_hash(s_mid), 1000, wall_per_game)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Python: {sys.version.split()[0]}  exe={sys.executable}")
    print(f"cwd: {os.getcwd()}")

    # 1) cProfile a full random game on each preset + 10-game batch
    play_n_games(PRESET_4X4_XL, 1, 42)  # JIT-style warm
    play_n_games(PRESET_3X3, 1, 42)

    elapsed_4x4 = run_cprofile(PRESET_4X4_XL, n_games=10, base_seed=100,
                               label="PRESET_4X4_XL")
    elapsed_3x3 = run_cprofile(PRESET_3X3, n_games=10, base_seed=200,
                               label="PRESET_3X3")

    wall_per_game_4x4 = elapsed_4x4 / 10.0
    wall_per_game_3x3 = elapsed_3x3 / 10.0

    # 2) Micro-benchmarks
    run_microbench(PRESET_4X4_XL, "PRESET_4X4_XL", wall_per_game_4x4)
    run_microbench(PRESET_3X3,    "PRESET_3X3",    wall_per_game_3x3)

    # 3) Summary line for quick eyeballing
    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"  4x4 XL: 10 games in {elapsed_4x4*1000:.1f} ms "
          f"(~{wall_per_game_4x4*1000:.1f} ms/game)")
    print(f"  3x3   : 10 games in {elapsed_3x3*1000:.1f} ms "
          f"(~{wall_per_game_3x3*1000:.1f} ms/game)")


if __name__ == "__main__":
    main()
