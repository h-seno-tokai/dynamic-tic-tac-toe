"""Parallel self-play with batched GPU leaf evaluation.

Runs N games simultaneously and batches every leaf-node neural-network
evaluation into a single GPU forward pass per simulation step.

Improvements over the AlphaZero scaffold:
  * **Tree reuse** between moves   (subtree of chosen action becomes new root)
  * **Playout Cap Randomization** (KataGo)   — only ``capped_frac`` of the
    moves use full sims and are stored as training samples; the rest use
    ``reduced_sims`` and contribute only to game flow.
  * **FPU reduction** — un-visited children inherit ``parent_q - fpu_red``
  * **Dynamic c_puct** (Lc0 schedule)
  * **Virtual loss** correctly accounted for in Q
  * **Q-target capture** — root Q saved per stored sample for value mixing.

Public API
----------
play_games_parallel(network, rules_list, num_sims, ...)
    Returns ``list[list[Sample]]`` — one sample-list per game.
    Sample = ``(state_tensor, policy_target, value_target, q_target)``.
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

DEFAULT_C_PUCT_INIT: float = 1.25
DEFAULT_C_PUCT_BASE: float = 19652.0
# FPU reduction (Lc0 default ~0.2): un-visited children inherit
# ``parent_q - FPU_REDUCTION * sqrt(visited_policy_mass)``.
DEFAULT_FPU_REDUCTION: float = 0.25
# Virtual loss size per outstanding leaf eval.
VIRTUAL_LOSS: float = 1.0
# Dirichlet alpha: AlphaZero uses 10 / (avg legal moves).
# 3x3 has ~27 legal moves initially, 4x4 has ~64 → a middle ground.
DEFAULT_DIRICHLET_ALPHA: float = 0.25
DEFAULT_DIRICHLET_EPS: float = 0.25

# Playout-cap randomization (KataGo): see https://arxiv.org/abs/1902.10565
DEFAULT_PCR_PROB: float = 0.25         # probability a move uses *full* sims
DEFAULT_REDUCED_SIM_FRAC: float = 0.25  # reduced sims = max(8, full * this)

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
    virtual_loss: int = 0
    children: dict[int, "_Node"] = field(default_factory=dict)
    is_expanded: bool = False
    to_move: Player | None = None


def _q(node: _Node) -> float:
    """Mean value adjusted for outstanding virtual losses."""
    n = node.visit_count + node.virtual_loss
    if n == 0:
        return 0.0
    return (node.value_sum - node.virtual_loss * VIRTUAL_LOSS) / n


def _c_puct(parent_n: int, c_init: float, c_base: float) -> float:
    """Lc0 dynamic c_puct schedule."""
    return c_init + math.log((parent_n + c_base + 1.0) / c_base)


def _fpu_q(parent: _Node, fpu_red: float) -> float:
    """Estimated Q for a not-yet-visited child = parent_q - fpu_red * sqrt(visited_pi).

    Visited prior mass is summed across already-visited children. This is the
    standard Lc0 / KataGo treatment.
    """
    visited_pi = 0.0
    for c in parent.children.values():
        if c.visit_count > 0 or c.virtual_loss > 0:
            visited_pi += c.prior
    return _q(parent) - fpu_red * math.sqrt(max(0.0, visited_pi))


def _best_child(
    parent: _Node,
    c_init: float,
    c_base: float,
    fpu_red: float,
) -> tuple[int, _Node]:
    n_parent = max(1, parent.visit_count + parent.virtual_loss)
    cpuct = _c_puct(n_parent, c_init, c_base)
    fpu = _fpu_q(parent, fpu_red)
    sqrt_n = math.sqrt(n_parent)
    best_a, best_node, best_s = -1, None, -math.inf
    for a, child in parent.children.items():
        if child.visit_count == 0 and child.virtual_loss == 0:
            q_est = fpu
        else:
            q_est = _q(child)
        u = cpuct * child.prior * sqrt_n / (1 + child.visit_count + child.virtual_loss)
        s = q_est + u
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
    for node in reversed(path):
        node.visit_count += 1
        sign = 1.0 if node.to_move == leaf_player else -1.0
        node.value_sum += sign * leaf_value
        if node.virtual_loss > 0:
            node.virtual_loss -= 1


def _select_to_leaf(
    root: _Node,
    root_state: GameState,
    c_init: float,
    c_base: float,
    fpu_red: float,
) -> tuple[list[_Node], GameState]:
    node, state = root, root_state
    path = [node]
    while node.is_expanded and node.children and not is_terminal(state):
        a, child = _best_child(node, c_init, c_base, fpu_red)
        # Apply virtual loss on the *selected* edge so that other concurrent
        # roll-outs in this batch tend to pick a different child.
        child.virtual_loss += 1
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


def _root_q(root: _Node) -> float:
    """Visit-weighted Q at the root (from root's POV)."""
    if root.visit_count <= 0:
        return 0.0
    return root.value_sum / root.visit_count


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
    # history: per stored sample (tensor, policy, mover, root_q)
    history: list[tuple[np.ndarray, np.ndarray, Player, float]] = field(default_factory=list)
    done: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Sample type: (state_tensor, policy, z_value, q_value)
Sample = tuple[np.ndarray, np.ndarray, float, float]


def play_games_parallel(
    network: object,
    rules_list: list[GameRules],
    num_sims: int = 200,
    c_puct_init: float = DEFAULT_C_PUCT_INIT,
    c_puct_base: float = DEFAULT_C_PUCT_BASE,
    fpu_reduction: float = DEFAULT_FPU_REDUCTION,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_eps: float = DEFAULT_DIRICHLET_EPS,
    pcr_prob: float = DEFAULT_PCR_PROB,
    reduced_sim_frac: float = DEFAULT_REDUCED_SIM_FRAC,
    add_root_noise: bool = True,
) -> list[list[Sample]]:
    """Run ``len(rules_list)`` games in parallel with batched GPU evaluation.

    Implements MCTS tree reuse, FPU reduction, dynamic c_puct, virtual-loss-
    aware Q, and Playout Cap Randomization (KataGo). Each stored sample also
    captures the root visit-weighted Q so the trainer can mix Q with z.

    Returns ``list[list[Sample]]``; each sample is
    ``(state_tensor, policy_target, value_z, root_q)``.
    """
    import torch
    import torch.nn.functional as F

    net = network  # type: ignore[assignment]
    device = next(net.parameters()).device  # type: ignore[attr-defined]
    net.eval()  # type: ignore[attr-defined]

    reduced_sims = max(8, int(num_sims * reduced_sim_frac))

    @torch.no_grad()
    def _batch_eval(states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        if not states:
            return np.empty((0, TOTAL_ACTIONS), np.float32), np.empty((0,), np.float32)
        batch = np.stack([state_to_tensor(s) for s in states])
        x = torch.from_numpy(batch).to(device)
        logits, wdl_logits = net(x)  # type: ignore[operator]
        priors = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        wdl_p = F.softmax(wdl_logits, dim=1).cpu().numpy().astype(np.float32)
        # scalar value Q = P(win) - P(loss)
        vals = (wdl_p[:, 0] - wdl_p[:, 2]).astype(np.float32)
        return priors, vals

    # ---------- initialise game contexts & expand roots ----------
    ctxs = [_Ctx(i, initial_state(r)) for i, r in enumerate(rules_list)]
    priors_np, _ = _batch_eval([c.state for c in ctxs])
    for ctx, p in zip(ctxs, priors_np):
        _expand_node(ctx.root, ctx.state, p)
        if add_root_noise:
            _add_dirichlet(ctx.root, dirichlet_alpha, dirichlet_eps)

    # ---------- main loop: advance until all games done ----------
    while any(not c.done for c in ctxs):
        active = [c for c in ctxs if not c.done]

        # Decide per-game whether this move is full-sim (stored) or reduced.
        is_full: dict[int, bool] = {}
        sims_for_ctx: dict[int, int] = {}
        max_sims = num_sims
        for ctx in active:
            full = (np.random.random() < pcr_prob)
            is_full[ctx.idx] = full
            sims_for_ctx[ctx.idx] = num_sims if full else reduced_sims
            if sims_for_ctx[ctx.idx] > max_sims:
                max_sims = sims_for_ctx[ctx.idx]

        # All games iterate the same number of total sim slots; games whose
        # quota is reached early simply skip selection. This keeps batching
        # simple while still respecting per-game sim budgets.
        for sim_step in range(max_sims):
            paths: list[list[_Node]] = []
            leaf_states: list[GameState] = []
            paths_owner: list[_Ctx] = []

            for ctx in active:
                if sim_step >= sims_for_ctx[ctx.idx]:
                    continue
                path, leaf = _select_to_leaf(
                    ctx.root, ctx.state,
                    c_puct_init, c_puct_base, fpu_reduction,
                )
                paths.append(path)
                leaf_states.append(leaf)
                paths_owner.append(ctx)

            if not paths:
                continue

            eval_idx = [
                i for i, (ls, p) in enumerate(zip(leaf_states, paths))
                if not is_terminal(ls) and not p[-1].is_expanded
            ]
            term_idx = [i for i, ls in enumerate(leaf_states) if is_terminal(ls)]
            # Already-expanded non-terminal leaves (rare: race when a child of
            # the root was already expanded by a previous selection).
            other_idx = [
                i for i in range(len(leaf_states))
                if i not in eval_idx and i not in term_idx
            ]

            if eval_idx:
                ep, ev = _batch_eval([leaf_states[i] for i in eval_idx])
                for j, i in enumerate(eval_idx):
                    leaf_node = paths[i][-1]
                    _expand_node(leaf_node, leaf_states[i], ep[j])
                    _backup(paths[i], float(ev[j]), leaf_states[i].to_move)

            for i in term_idx:
                ls = leaf_states[i]
                o = ls.outcome
                v = 0.0 if (o == "draw" or o is None) else (
                    1.0 if o == ls.to_move else -1.0
                )
                _backup(paths[i], v, ls.to_move)

            # Free virtual loss for the rare "already expanded" path: back-up
            # using the node's current Q estimate.
            for i in other_idx:
                node = paths[i][-1]
                _backup(paths[i], _q(node), leaf_states[i].to_move)

        # -- Move selection & state advancement (with tree reuse) --
        new_root_ctxs: list[_Ctx] = []
        for ctx in active:
            store_sample = is_full[ctx.idx]
            if store_sample:
                tensor = state_to_tensor(ctx.state)
                policy = _visit_dist(ctx.root)
                rq = _root_q(ctx.root)
                ctx.history.append((tensor, policy, ctx.state.to_move, rq))

            temp = _temperature_for_ply(ctx.state.ply)
            action = _sample_action(ctx.root, ctx.state, temp)
            ctx.state = apply_move(ctx.state, action_index_to_move(ctx.state, action))

            if is_terminal(ctx.state):
                ctx.done = True
                continue

            # Tree reuse: chosen child becomes the new root.
            child = ctx.root.children.get(action)
            if child is not None and child.is_expanded:
                # Detach: clear virtual losses (none should remain after backup).
                child.virtual_loss = 0
                ctx.root = child
            else:
                ctx.root = _Node()
                new_root_ctxs.append(ctx)

        # Expand new roots that did not have a usable subtree.
        if new_root_ctxs:
            priors_np, _ = _batch_eval([c.state for c in new_root_ctxs])
            for ctx, p in zip(new_root_ctxs, priors_np):
                _expand_node(ctx.root, ctx.state, p)

        # Re-noise the root for *every* active game on each move so that
        # exploration stays alive. (This is what AlphaZero does.)
        if add_root_noise:
            for ctx in active:
                if not ctx.done:
                    _add_dirichlet(ctx.root, dirichlet_alpha, dirichlet_eps)

    # ---------- assign game outcomes as value targets ----------
    results: list[list[Sample]] = []
    for ctx in ctxs:
        final = ctx.state.outcome
        samples: list[Sample] = []
        for tensor, policy, mover, rq in ctx.history:
            if final == "draw" or final is None:
                v = 0.0
            else:
                v = 1.0 if final == mover else -1.0
            samples.append((tensor, policy, v, rq))
        results.append(samples)
    return results


__all__ = [
    "play_games_parallel",
    "Sample",
    "DEFAULT_C_PUCT_INIT",
    "DEFAULT_C_PUCT_BASE",
    "DEFAULT_FPU_REDUCTION",
    "DEFAULT_DIRICHLET_ALPHA",
    "DEFAULT_DIRICHLET_EPS",
    "DEFAULT_PCR_PROB",
]
