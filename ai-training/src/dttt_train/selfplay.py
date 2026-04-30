"""Self-play game generator (single-game variant).

``play_game`` runs a single MCTS-driven self-play episode and returns a list
of training samples. Used by the multi-process worker path; the parallel
batched path is in ``parallel_selfplay``.

Sample type: ``(state_tensor, policy_target, value_z, q_target)``
  * ``state_tensor`` : float32 (27, 4, 4)
  * ``policy_target``: float32 (320,) — MCTS visit distribution
  * ``value_z``      : final game outcome from this state's POV (-1, 0, +1)
  * ``q_target``     : root Q at this state (visit-weighted, same POV)

The trainer mixes ``z`` and ``q_target`` to form the value-loss target.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .encoding import state_to_tensor
from .engine import apply_move, is_terminal
from .mcts import MCTS, make_torch_evaluator
from .rules import GameRules, Player, initial_state


class _NetworkLike(Protocol):
    def __call__(self, x: object) -> tuple[object, object]: ...
    def parameters(self) -> object: ...
    def eval(self) -> object: ...


_TEMPERATURE_THRESHOLD_PLY: int = 10
_EARLY_TEMPERATURE: float = 1.0
_LATE_TEMPERATURE: float = 0.1


def _temperature_for_ply(ply: int) -> float:
    return _EARLY_TEMPERATURE if ply < _TEMPERATURE_THRESHOLD_PLY else _LATE_TEMPERATURE


def play_game(
    network: _NetworkLike,
    rules: GameRules,
    num_sims: int = 200,
    c_puct: float = 1.5,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Play a single self-play game and return training tuples.

    Returns
    -------
    list of (state_tensor, policy_target, value_z, q_target)
    """
    if seed is not None:
        np.random.seed(seed)

    evaluator = make_torch_evaluator(network)
    mcts = MCTS(evaluator=evaluator, c_puct=c_puct)

    state = initial_state(rules)
    history: list[tuple[np.ndarray, np.ndarray, Player, float]] = []

    while not is_terminal(state):
        policy = mcts.search(state, num_sims=num_sims)
        # Capture root Q (visit-weighted) for value-target mixing.
        root = mcts.root
        q = 0.0
        if root is not None and root.visit_count > 0:
            q = float(root.value_sum) / float(root.visit_count)
        history.append((state_to_tensor(state), policy.copy(), state.to_move, q))
        action = mcts.select_move(state, temperature=_temperature_for_ply(state.ply))
        from .encoding import action_index_to_move
        move = action_index_to_move(state, action)
        state = apply_move(state, move)

    final = state.outcome
    samples: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for tensor, policy, mover, q in history:
        if final == "draw" or final is None:
            v = 0.0
        else:
            v = 1.0 if final == mover else -1.0
        samples.append((tensor, policy, v, q))
    return samples


__all__ = ["play_game"]
