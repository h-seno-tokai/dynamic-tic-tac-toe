"""Parallel self-play with batched GPU leaf evaluation.

Runs N games simultaneously and batches every leaf-node neural-network
evaluation into a single GPU forward pass per simulation step.
This keeps the GPU busy and achieves near-full utilisation.

                   game-0   game-1  ...  game-N
                     |        |            |
sim step 1:  select leaf, select leaf, ..., select leaf
                         ↓ batch eval (1 GPU call)
                   expand+backup for all N games
sim step 2:  ...
...
sim step S:  done

After S simulations, select moves for all N games, advance states,
and repeat until all games terminate.

Public API
----------
play_games_parallel(network, rules_list, num_sims, ...)
    Returns list[list[sample]] — one sample-list per game.
    Each sample: (state_tensor float32 (27,4,4), policy float32 (320,), value float32)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .encoding import (
    TOTAL_ACTIONS,
    action_index_to_move,
    legal_action_mask,
    state_to_tensor,
)
from .engine import apply_move, is_terminal
from .rules import GameRules, GameState, Player, initial_state

if TYPE_CHECKING:
    pass

DEFAULT_C_PUCT: float = 1.5
# Dirichlet alpha: AlphaZero uses 10 / (avg legal moves).
# 3×3 has ~27 legal moves initially → α≈0.3; 4×4 has ~64 → α≈0.15.
# We use 0.25 as a reasonable middle ground.
DEFAULT_DIRICHLET_ALPHA: float = 0.25
DEFAULT_DIRICHLET_EPS: float = 0.25

_TEMP_THRESHOLD_PLY: int = 10
_EARLY_TEMP: float = 1.0
_LATE_TEMP: float = 0.05


# ---------------------------------------------------------------------------
# Lightweight MCTS node
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "_Node"] = field(default_factory=dict)
    is_expanded: bool = False
    to_move: Player | None = None


def _q(node: _Node) -> float:
    return node.value_sum / node.visit_count if node.visit_count > 0 else 0.0


def _puct(parent_n: int, child: _Node, c: float) -> float:
    return _q(child) + c * child.prior * math.sqrt(parent_n) / (1 + child.visit_count)


def _best_child(parent: _Node, c: float) -> tuple[int, _Node]:
    n = max(1, parent.visit_count)
    best_a, best_node, best_s = -1, None, -math.inf
    for a, child in parent.children.items():
        s = _puct(n, child, c)
        if s > best_s:
            best_s, best_a, best_node = s, a, child
    assert best_node is not None
    return best_a, best_node


def _expand_node(node: _Node, state: GameState, priors: np.ndarray) -> None:
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


def _add_dirichlet(node: _Node, alpha: float, eps: float) -> None:
    """Mix Dirichlet noise into root priors for exploration."""
    n = len(node.children)
    if n == 0:
        return
    noise = np.random.dirichlet([alpha] * n).astype(np.float32)
    for child, eta in zip(node.children.values(), noise):
        child.prior = (1.0 - eps) * child.prior + eps * float(eta)


def _backup(path: list[_Node], leaf_value: float, leaf_player: Player) -> None:
    """Propagate value up the path. Flip sign when perspective changes."""
    for node in reversed(path):
        node.visit_count += 1
        sign = 1.0 if node.to_move == leaf_player else -1.0
        node.value_sum += sign * leaf_value


def _select_to_leaf(
    root: _Node, root_state: GameState, c: float
) -> tuple[list[_Node], GameState]:
    """Walk tree (PUCT) to an unexpanded node or terminal state."""
    node, state = root, root_state
    path = [node]
    while node.is_expanded and node.children and not is_terminal(state):
        a, child = _best_child(node, c)
        state = apply_move(state, action_index_to_move(state, a))
        node = child
        path.append(node)
    return path, state


def _temperature_for_ply(ply: int) -> float:
    return _EARLY_TEMP if ply < _TEMP_THRESHOLD_PLY else _LATE_TEMP


def _visit_dist(root: _Node) -> np.ndarray:
    dist = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
    total = 0
    for a, child in root.children.items():
        dist[a] = child.visit_count
        total += child.visit_count
    if total > 0:
        dist /= total
    return dist


def _sample_action(root: _Node, state: GameState, temperature: float) -> int:
    visits = np.zeros(TOTAL_ACTIONS, dtype=np.float64)
    for a, child in root.children.items():
        visits[a] = child.visit_count
    if visits.sum() == 0:
        legal = legal_action_mask(state)
        return int(np.random.choice(np.flatnonzero(legal)))
    if temperature < 1e-6:
        return int(np.argmax(visits))
    scaled = visits ** (1.0 / temperature)
    scaled /= scaled.sum()
    return int(np.random.choice(TOTAL_ACTIONS, p=scaled))


# ---------------------------------------------------------------------------
# Per-game context
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    idx: int
    state: GameState
    root: _Node = field(default_factory=_Node)
    history: list[tuple[np.ndarray, np.ndarray, Player]] = field(default_factory=list)
    done: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def play_games_parallel(
    network: object,
    rules_list: list[GameRules],
    num_sims: int = 200,
    c_puct: float = DEFAULT_C_PUCT,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_eps: float = DEFAULT_DIRICHLET_EPS,
) -> list[list[tuple[np.ndarray, np.ndarray, float]]]:
    """Run ``len(rules_list)`` games in parallel with batched GPU evaluation.

    Returns a list (one per game) of sample-lists.
    Each sample is ``(state_tensor, policy_target, value_target)``.
    """
    import torch
    import torch.nn.functional as F

    net = network  # type: ignore[assignment]
    device = next(net.parameters()).device  # type: ignore[attr-defined]
    net.eval()  # type: ignore[attr-defined]

    def _batch_eval(states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        if not states:
            return np.empty((0, TOTAL_ACTIONS), np.float32), np.empty((0,), np.float32)
        batch = np.stack([state_to_tensor(s) for s in states])
        x = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            logits, values = net(x)  # type: ignore[operator]
        priors = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        vals = values.squeeze(-1).cpu().numpy().astype(np.float32)
        return priors, vals

    # ---------- initialise game contexts & expand roots ----------
    ctxs = [_Ctx(i, initial_state(r)) for i, r in enumerate(rules_list)]
    priors_np, _ = _batch_eval([c.state for c in ctxs])
    for ctx, p in zip(ctxs, priors_np):
        _expand_node(ctx.root, ctx.state, p)
        _add_dirichlet(ctx.root, dirichlet_alpha, dirichlet_eps)

    # ---------- main loop: advance until all games done ----------
    while any(not c.done for c in ctxs):
        active = [c for c in ctxs if not c.done]

        # -- MCTS: num_sims simulation steps, all active games in lockstep --
        for _ in range(num_sims):
            paths: list[list[_Node]] = []
            leaf_states: list[GameState] = []

            # Selection for all active games
            for ctx in active:
                path, leaf = _select_to_leaf(ctx.root, ctx.state, c_puct)
                paths.append(path)
                leaf_states.append(leaf)

            # Split into "needs eval" and "terminal"
            eval_idx = [
                i for i, (ls, p) in enumerate(zip(leaf_states, paths))
                if not is_terminal(ls) and not p[-1].is_expanded
            ]
            term_idx = [i for i, ls in enumerate(leaf_states) if is_terminal(ls)]

            # Batched network evaluation
            if eval_idx:
                ep, ev = _batch_eval([leaf_states[i] for i in eval_idx])
                for j, i in enumerate(eval_idx):
                    leaf_node = paths[i][-1]
                    _expand_node(leaf_node, leaf_states[i], ep[j])
                    _backup(paths[i], float(ev[j]), leaf_states[i].to_move)

            # Terminal backups (no network call)
            for i in term_idx:
                ls = leaf_states[i]
                o = ls.outcome
                v = 0.0 if (o == "draw" or o is None) else (
                    1.0 if o == ls.to_move else -1.0
                )
                _backup(paths[i], v, ls.to_move)

        # -- Move selection & state advancement --
        new_root_ctxs: list[_Ctx] = []
        for ctx in active:
            tensor = state_to_tensor(ctx.state)
            policy = _visit_dist(ctx.root)
            ctx.history.append((tensor, policy, ctx.state.to_move))

            temp = _temperature_for_ply(ctx.state.ply)
            action = _sample_action(ctx.root, ctx.state, temp)
            ctx.state = apply_move(ctx.state, action_index_to_move(ctx.state, action))

            if is_terminal(ctx.state):
                ctx.done = True
            else:
                ctx.root = _Node()
                new_root_ctxs.append(ctx)

        # Expand new roots (batch)
        if new_root_ctxs:
            priors_np, _ = _batch_eval([c.state for c in new_root_ctxs])
            for ctx, p in zip(new_root_ctxs, priors_np):
                _expand_node(ctx.root, ctx.state, p)
                _add_dirichlet(ctx.root, dirichlet_alpha, dirichlet_eps)

    # ---------- assign game outcomes as value targets ----------
    results: list[list[tuple[np.ndarray, np.ndarray, float]]] = []
    for ctx in ctxs:
        final = ctx.state.outcome
        samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        for tensor, policy, mover in ctx.history:
            if final == "draw" or final is None:
                v = 0.0
            else:
                v = 1.0 if final == mover else -1.0
            samples.append((tensor, policy, v))
        results.append(samples)
    return results


__all__ = ["play_games_parallel", "DEFAULT_DIRICHLET_ALPHA", "DEFAULT_DIRICHLET_EPS"]
