"""Profile the *fast* NumPy engine (``dttt_train_4x4``) vs the old ``dttt_train``.

Mirrors the micro-benchmarks in ``profile_engine.py`` (legal_moves /
apply_move / state_to_tensor / state_hash) and prints per-call us values
plus the speedup factor over the old engine for the 4x4-XL preset.

Run with PYTHONPATH=ai-training/src::

    PYTHONPATH=ai-training/src python ai-training/scripts/profile_engine_v2.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

# Make src/ importable even without PYTHONPATH set.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dttt_train.encoding import (  # noqa: E402
    legal_action_mask as legacy_mask,
    state_to_tensor as legacy_tensor,
)
from dttt_train.engine import (  # noqa: E402
    _state_hash as legacy_hash,
    apply_move as legacy_apply,
    legal_moves as legacy_legal_moves,
)
from dttt_train.rules import PRESET_4X4_XL, initial_state as legacy_init  # noqa: E402

from dttt_train_4x4.encoding import (  # noqa: E402
    legal_action_mask as fast_mask,
    state_to_tensor as fast_tensor,
)
from dttt_train_4x4.engine import (  # noqa: E402
    _state_hash as fast_hash,
    apply_move as fast_apply,
    from_legacy_state,
    initial_state as fast_init,
    legal_moves as fast_legal_moves,
)


# ---------------------------------------------------------------------------
# Random self-play helpers
# ---------------------------------------------------------------------------


def _legacy_play_random(rules, rng):
    state = legacy_init(rules)
    plies = 0
    while state.outcome is None:
        moves = legacy_legal_moves(state)
        if not moves:
            break
        state = legacy_apply(state, rng.choice(moves))
        plies += 1
    return plies


def _fast_play_random(rng):
    state = fast_init()
    plies = 0
    while state.outcome is None:
        moves = fast_legal_moves(state)
        if not moves:
            break
        state = fast_apply(state, rng.choice(moves))
        plies += 1
    return plies


# Self-play mimicking the actual MCTS loop (uses legal_action_mask, not legal_moves).
import numpy as _np  # noqa: E402


def _legacy_play_mcts_like(rules, rng):
    from dttt_train.encoding import action_index_to_move as legacy_a2m
    state = legacy_init(rules)
    plies = 0
    while state.outcome is None:
        mask = legacy_mask(state)
        legal_idx = _np.flatnonzero(mask)
        if legal_idx.size == 0:
            break
        a = int(legal_idx[rng.randrange(legal_idx.size)])
        state = legacy_apply(state, legacy_a2m(state, a))
        # In MCTS this is also where state_to_tensor would be called.
        legacy_tensor(state)
        plies += 1
    return plies


def _fast_play_mcts_like(rng):
    from dttt_train_4x4.encoding import action_index_to_move as fast_a2m
    state = fast_init()
    plies = 0
    while state.outcome is None:
        mask = fast_mask(state)
        legal_idx = _np.flatnonzero(mask)
        if legal_idx.size == 0:
            break
        a = int(legal_idx[rng.randrange(legal_idx.size)])
        state = fast_apply(state, fast_a2m(state, a))
        fast_tensor(state)
        plies += 1
    return plies


def _legacy_midgame(rules, seed: int, target_ply: int):
    rng = random.Random(seed)
    state = legacy_init(rules)
    while state.outcome is None and state.ply < target_ply:
        moves = legacy_legal_moves(state)
        if not moves:
            break
        state = legacy_apply(state, rng.choice(moves))
    return state


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


def _bench(label: str, fn, n: int) -> float:
    # warm
    for _ in range(min(50, n // 10)):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - t0
    per = elapsed / n * 1e6
    print(f"  {label:<48} n={n:>6}  total={elapsed*1000:8.2f} ms  per-call={per:8.2f} us")
    return per


def _bench_full_game(label: str, play_fn, n_games: int = 10) -> float:
    # warm
    for _ in range(2):
        play_fn(random.Random(1))
    t0 = time.perf_counter()
    total_plies = 0
    for i in range(n_games):
        rng = random.Random(100 + i)
        total_plies += play_fn(rng)
    elapsed = time.perf_counter() - t0
    per_ply = elapsed / max(1, total_plies) * 1e6
    per_game = elapsed / n_games * 1000
    print(f"  {label:<48} {n_games} games "
          f"({total_plies} plies)  wall={elapsed*1000:.1f} ms  "
          f"per-ply={per_ply:.1f} us  per-game={per_game:.2f} ms")
    return per_ply


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Python: {sys.version.split()[0]}  exe={sys.executable}")
    print(f"cwd: {os.getcwd()}")

    target_ply = 8
    legacy_mid = _legacy_midgame(PRESET_4X4_XL, seed=1234, target_ply=target_ply)
    if not legacy_legal_moves(legacy_mid):
        for s in range(2, 50):
            legacy_mid = _legacy_midgame(PRESET_4X4_XL, seed=s, target_ply=target_ply)
            if legacy_legal_moves(legacy_mid):
                break
    fast_mid = from_legacy_state(legacy_mid)

    legacy_first = legacy_legal_moves(legacy_mid)[0]
    fast_first = fast_legal_moves(fast_mid)[0]

    n_legal_legacy = len(legacy_legal_moves(legacy_mid))
    n_legal_fast = len(fast_legal_moves(fast_mid))
    print(f"\nMid-game branching: legacy={n_legal_legacy} fast={n_legal_fast} "
          f"(ply={legacy_mid.ply})")

    legacy_initial = legacy_init(PRESET_4X4_XL)
    fast_initial = fast_init()

    print()
    print("-" * 78)
    print("Legacy engine (dttt_train, 4x4-XL)")
    print("-" * 78)
    n = 1000
    leg_legal_init = _bench("legal_moves(initial_state)",
                            lambda: legacy_legal_moves(legacy_initial), n)
    leg_legal_mid = _bench("legal_moves(mid-game)",
                           lambda: legacy_legal_moves(legacy_mid), n)
    leg_mask = _bench("legal_action_mask(mid-game)",
                      lambda: legacy_mask(legacy_mid), n)
    leg_apply = _bench("apply_move(mid-game, first)",
                       lambda: legacy_apply(legacy_mid, legacy_first), n)
    leg_tensor = _bench("state_to_tensor(mid-game)",
                        lambda: legacy_tensor(legacy_mid), n)
    leg_hash = _bench("_state_hash(mid-game)",
                      lambda: legacy_hash(legacy_mid), n)
    leg_per_ply = _bench_full_game(
        "self-play (random) full games",
        lambda rng: _legacy_play_random(PRESET_4X4_XL, rng),
        n_games=10,
    )
    leg_per_ply_mcts = _bench_full_game(
        "self-play (MCTS-like, mask+tensor)",
        lambda rng: _legacy_play_mcts_like(PRESET_4X4_XL, rng),
        n_games=10,
    )

    print()
    print("-" * 78)
    print("Fast engine (dttt_train_4x4, 4x4-XL)")
    print("-" * 78)
    fast_legal_init = _bench("legal_moves(initial_state)",
                             lambda: fast_legal_moves(fast_initial), n)
    fast_legal_mid = _bench("legal_moves(mid-game)",
                            lambda: fast_legal_moves(fast_mid), n)
    fast_mask_us = _bench("legal_action_mask(mid-game)",
                          lambda: fast_mask(fast_mid), n)
    fast_apply_us = _bench("apply_move(mid-game, first)",
                           lambda: fast_apply(fast_mid, fast_first), n)
    fast_tensor_us = _bench("state_to_tensor(mid-game)",
                            lambda: fast_tensor(fast_mid), n)
    fast_hash_us = _bench("_state_hash(mid-game)",
                          lambda: fast_hash(fast_mid), n)
    fast_per_ply = _bench_full_game(
        "self-play (random) full games",
        lambda rng: _fast_play_random(rng),
        n_games=10,
    )
    fast_per_ply_mcts = _bench_full_game(
        "self-play (MCTS-like, mask+tensor)",
        lambda rng: _fast_play_mcts_like(rng),
        n_games=10,
    )

    print()
    print("=" * 78)
    print("Speedup (legacy / fast)")
    print("=" * 78)
    def _r(a: float, b: float) -> str:
        return f"{a/b:.1f}x" if b > 0 else "n/a"
    print(f"  legal_moves(initial)        {_r(leg_legal_init, fast_legal_init)}")
    print(f"  legal_moves(mid-game)       {_r(leg_legal_mid,  fast_legal_mid)}")
    print(f"  legal_action_mask(mid-game) {_r(leg_mask,       fast_mask_us)}")
    print(f"  apply_move(mid-game)        {_r(leg_apply,      fast_apply_us)}")
    print(f"  state_to_tensor(mid-game)   {_r(leg_tensor,     fast_tensor_us)}")
    print(f"  _state_hash(mid-game)       {_r(leg_hash,       fast_hash_us)}")
    print(f"  full-game per-ply (random)  {_r(leg_per_ply,      fast_per_ply)}")
    print(f"  full-game per-ply (MCTS-like)  {_r(leg_per_ply_mcts, fast_per_ply_mcts)}")


if __name__ == "__main__":
    main()
