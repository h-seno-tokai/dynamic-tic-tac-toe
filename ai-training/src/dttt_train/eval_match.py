"""Best-network gating tournament.

Runs ``num_games`` evaluation games between a *candidate* and *best* network,
alternating colours, with low-temperature, no-Dirichlet MCTS. Returns the
candidate's score (wins + 0.5 * draws) over total games. The training loop
uses this to decide whether to promote the candidate to ``best.pt``.

This module deliberately *re-uses* :mod:`parallel_selfplay`'s MCTS internals
but takes two evaluators (one per network) and routes leaf evaluations to
the network whose turn it is to move at each leaf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .encoding import (
    TOTAL_ACTIONS,
    action_index_to_move,
    legal_action_mask,
    state_to_tensor,
)
from .engine import apply_move, is_terminal
from .rules import GameRules, GameState, Player, initial_state

_C_PUCT_INIT: float = 1.25
_C_PUCT_BASE: float = 19652.0
_FPU: float = 0.25
_VIRTUAL_LOSS: float = 1.0


@dataclass
class _Node:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    virtual_loss: int = 0
    children: dict[int, "_Node"] = field(default_factory=dict)
    is_expanded: bool = False
    to_move: Player | None = None


def _q(node: _Node) -> float:
    n = node.visit_count + node.virtual_loss
    if n == 0:
        return 0.0
    return (node.value_sum - node.virtual_loss * _VIRTUAL_LOSS) / n


def _c_puct(parent_n: int) -> float:
    return _C_PUCT_INIT + math.log((parent_n + _C_PUCT_BASE + 1.0) / _C_PUCT_BASE)


def _fpu_q(parent: _Node) -> float:
    visited_pi = 0.0
    for c in parent.children.values():
        if c.visit_count > 0 or c.virtual_loss > 0:
            visited_pi += c.prior
    return _q(parent) - _FPU * math.sqrt(max(0.0, visited_pi))


def _best_child(parent: _Node) -> tuple[int, _Node]:
    n_parent = max(1, parent.visit_count + parent.virtual_loss)
    cpuct = _c_puct(n_parent)
    fpu = _fpu_q(parent)
    sqrt_n = math.sqrt(n_parent)
    best_a, best_node, best_s = -1, None, -math.inf
    for a, child in parent.children.items():
        q_est = fpu if (child.visit_count == 0 and child.virtual_loss == 0) else _q(child)
        u = cpuct * child.prior * sqrt_n / (1 + child.visit_count + child.virtual_loss)
        s = q_est + u
        if s > best_s:
            best_s, best_a, best_node = s, a, child
    assert best_node is not None
    return best_a, best_node


def _expand(node: _Node, state: GameState, priors: np.ndarray) -> None:
    mask = legal_action_mask(state)
    masked = priors * mask.astype(np.float32)
    total = masked.sum()
    if total > 0:
        masked /= total
    elif mask.any():
        masked = mask.astype(np.float32) / float(mask.sum())
    node.to_move = state.to_move
    for a in np.flatnonzero(mask):
        node.children[int(a)] = _Node(prior=float(masked[a]))
    node.is_expanded = True


def _backup(path: list[_Node], leaf_value: float, leaf_player: Player) -> None:
    for node in reversed(path):
        node.visit_count += 1
        sign = 1.0 if node.to_move == leaf_player else -1.0
        node.value_sum += sign * leaf_value
        if node.virtual_loss > 0:
            node.virtual_loss -= 1


def _select_to_leaf(root: _Node, root_state: GameState) -> tuple[list[_Node], GameState]:
    node, state = root, root_state
    path = [node]
    while node.is_expanded and node.children and not is_terminal(state):
        a, child = _best_child(node)
        child.virtual_loss += 1
        state = apply_move(state, action_index_to_move(state, a))
        node = child
        path.append(node)
    return path, state


def _argmax_visits(root: _Node) -> int:
    best_a, best_n = -1, -1
    for a, child in root.children.items():
        if child.visit_count > best_n:
            best_n, best_a = child.visit_count, a
    return best_a


def _make_torch_evaluator(network: object):
    import torch
    import torch.nn.functional as F

    device = next(network.parameters()).device  # type: ignore[attr-defined]
    network.eval()  # type: ignore[attr-defined]

    @torch.no_grad()
    def evaluator(states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        if not states:
            return np.empty((0, TOTAL_ACTIONS), np.float32), np.empty((0,), np.float32)
        batch = np.stack([state_to_tensor(s) for s in states])
        x = torch.from_numpy(batch).to(device)
        logits, wdl_logits = network(x)  # type: ignore[operator]
        priors = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        wdl = F.softmax(wdl_logits, dim=1).cpu().numpy().astype(np.float32)
        v = (wdl[:, 0] - wdl[:, 2]).astype(np.float32)
        return priors, v

    return evaluator


def _play_one_game(
    rules: GameRules,
    eval_p1,  # evaluator for P1
    eval_p2,  # evaluator for P2
    num_sims: int,
) -> str:
    """Run one game between two evaluators. Returns 'P1', 'P2', or 'draw'."""
    state = initial_state(rules)
    root = _Node()
    # Initial expansion uses the side-to-move's evaluator.
    side_eval = eval_p1 if state.to_move is Player.P1 else eval_p2
    priors_np, _ = side_eval([state])
    _expand(root, state, priors_np[0])

    while not is_terminal(state):
        # Run num_sims simulations. Each leaf eval routed to the player at that leaf.
        for _ in range(num_sims):
            path, leaf = _select_to_leaf(root, state)
            if is_terminal(leaf):
                o = leaf.outcome
                v = 0.0 if (o == "draw" or o is None) else (
                    1.0 if o == leaf.to_move else -1.0
                )
                _backup(path, v, leaf.to_move)
                continue
            leaf_node = path[-1]
            if leaf_node.is_expanded:
                # Already expanded by a prior selection; just back-up its current Q.
                _backup(path, _q(leaf_node), leaf.to_move)
                continue
            leaf_eval = eval_p1 if leaf.to_move is Player.P1 else eval_p2
            ep, ev = leaf_eval([leaf])
            _expand(leaf_node, leaf, ep[0])
            _backup(path, float(ev[0]), leaf.to_move)

        action = _argmax_visits(root)
        if action < 0:
            # No expanded children - pick a random legal action and continue.
            mask = legal_action_mask(state)
            action = int(np.random.choice(np.flatnonzero(mask)))
        state = apply_move(state, action_index_to_move(state, action))

        # Tree reuse for the chosen child if present.
        child = root.children.get(action)
        if child is not None and child.is_expanded:
            child.virtual_loss = 0
            root = child
        else:
            root = _Node()
            if not is_terminal(state):
                side_eval = eval_p1 if state.to_move is Player.P1 else eval_p2
                priors_np, _ = side_eval([state])
                _expand(root, state, priors_np[0])

    o = state.outcome
    if o == "draw" or o is None:
        return "draw"
    return "P1" if o == Player.P1 else "P2"


def evaluate_match(
    candidate: object,
    best: object,
    rules_list: list[GameRules],
    num_games: int = 40,
    num_sims: int = 100,
) -> tuple[int, int, int]:
    """Play ``num_games`` between ``candidate`` and ``best`` (alternating colours).

    Returns ``(wins, draws, losses)`` from the candidate's perspective.
    ``rules_list`` is cycled when ``num_games > len(rules_list)``.
    """
    cand_eval = _make_torch_evaluator(candidate)
    best_eval = _make_torch_evaluator(best)

    wins = draws = losses = 0
    for g in range(num_games):
        rules = rules_list[g % len(rules_list)]
        cand_is_p1 = (g % 2 == 0)
        e_p1 = cand_eval if cand_is_p1 else best_eval
        e_p2 = best_eval if cand_is_p1 else cand_eval
        result = _play_one_game(rules, e_p1, e_p2, num_sims=num_sims)
        if result == "draw":
            draws += 1
        elif (result == "P1" and cand_is_p1) or (result == "P2" and not cand_is_p1):
            wins += 1
        else:
            losses += 1
    return wins, draws, losses


__all__ = ["evaluate_match"]
