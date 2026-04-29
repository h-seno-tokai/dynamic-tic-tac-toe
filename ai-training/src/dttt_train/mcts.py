"""Leaf-batched MCTS skeleton with virtual loss.

The legal-action mask is applied **here in Python** - it deliberately does
not appear in the network forward graph so that the ONNX export stays free
of ``-inf + softmax`` patterns (see ``docs/10_risks.md`` 1.2).

This is a scaffold: the search() method drives a single-leaf rollout per
simulation. A real run would batch leaf evaluations across multiple
trees / workers (the virtual-loss machinery is in place).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .encoding import (
    TOTAL_ACTIONS,
    action_index_to_move,
    legal_action_mask,
    move_to_action_index,
    state_to_tensor,
)
from .engine import apply_move, is_terminal, legal_moves
from .rules import GameState, Player

# Type alias: an evaluator takes a list of states and returns
# (policy_priors[batch, 320], values[batch]). policy_priors must already be
# softmaxed AND legal-masked by the caller of the evaluator (we mask here).
Evaluator = Callable[[list[GameState]], tuple[np.ndarray, np.ndarray]]

# PUCT exploration constant (AlphaZero default).
DEFAULT_C_PUCT: float = 1.5
# Virtual loss applied to a node while a worker is evaluating from it.
VIRTUAL_LOSS: float = 1.0


@dataclass
class _Node:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    virtual_loss: int = 0
    children: dict[int, "_Node"] = field(default_factory=dict)
    is_expanded: bool = False
    # cached legal actions / state pointer (filled on expand)
    legal: np.ndarray | None = None  # (320,) bool
    state: GameState | None = None
    to_move: Player | None = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss * VIRTUAL_LOSS) / max(
            1, self.visit_count + self.virtual_loss
        )


def _puct_score(parent_visits: int, child: _Node, c_puct: float) -> float:
    u = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count + child.virtual_loss)
    return child.q_value + u


def _select_child(parent: _Node, c_puct: float) -> tuple[int, _Node]:
    parent_visits = max(1, parent.visit_count)
    best_action = -1
    best_score = -math.inf
    best_child: _Node | None = None
    for action, child in parent.children.items():
        score = _puct_score(parent_visits, child, c_puct)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    assert best_child is not None
    return best_action, best_child


def _expand(node: _Node, state: GameState, priors: np.ndarray) -> None:
    """Populate ``node`` with children for every legal action in ``state``."""
    legal = legal_action_mask(state)
    masked = priors * legal.astype(priors.dtype)
    s = masked.sum()
    if s > 0:
        masked = masked / s
    else:
        # Fall back to uniform over legal actions (network gave zero mass).
        if legal.any():
            masked = legal.astype(np.float32) / float(legal.sum())
        else:
            masked = np.zeros_like(priors)

    node.legal = legal
    node.state = state
    node.to_move = state.to_move
    node.is_expanded = True
    for action_idx in np.flatnonzero(legal):
        node.children[int(action_idx)] = _Node(prior=float(masked[action_idx]))


def _backup(path: list[_Node], leaf_value: float, leaf_player: Player) -> None:
    """Propagate ``leaf_value`` (from leaf player's POV) up the path.

    AlphaZero convention: each node stores Q from the perspective of the
    *player whose turn it is to move at that node*. We flip the sign at
    every level whose to_move differs from the leaf player.
    """
    for node in reversed(path):
        node.visit_count += 1
        sign = 1.0 if node.to_move == leaf_player else -1.0
        node.value_sum += sign * leaf_value
        # Remove any virtual loss this thread applied during selection.
        if node.virtual_loss > 0:
            node.virtual_loss -= 1


class MCTS:
    """Single-tree leaf-batched MCTS.

    Parameters
    ----------
    evaluator
        Callable taking a list of GameState and returning
        ``(priors[batch, 320] float32, values[batch] float32 in [-1, 1])``.
    c_puct
        Exploration constant in the PUCT formula.
    """

    def __init__(self, evaluator: Evaluator, c_puct: float = DEFAULT_C_PUCT) -> None:
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.root: _Node | None = None
        self.root_state: GameState | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, state: GameState, num_sims: int) -> np.ndarray:
        """Run ``num_sims`` simulations from ``state`` and return a 320-d
        visit-count distribution (sums to 1 over legal actions)."""
        self.root = _Node()
        self.root_state = state
        priors, _ = self.evaluator([state])
        _expand(self.root, state, priors[0])

        for _ in range(num_sims):
            self._simulate_once(state, self.root)

        # Compose the policy target from root visit counts.
        policy = np.zeros((TOTAL_ACTIONS,), dtype=np.float32)
        for action_idx, child in self.root.children.items():
            policy[action_idx] = float(child.visit_count)
        total = policy.sum()
        if total > 0:
            policy /= total
        return policy

    def select_move(self, state: GameState, temperature: float = 1.0) -> int:
        """After ``search``, sample an action index from the root.

        ``temperature == 0`` => argmax visit count.
        """
        if self.root is None:
            raise RuntimeError("call search() before select_move()")
        visits = np.zeros((TOTAL_ACTIONS,), dtype=np.float64)
        for action_idx, child in self.root.children.items():
            visits[action_idx] = child.visit_count
        if visits.sum() == 0:
            # Should be unreachable post-search, but fall back to uniform-legal.
            mask = legal_action_mask(state)
            return int(np.random.choice(np.flatnonzero(mask)))
        if temperature <= 1e-6:
            return int(np.argmax(visits))
        scaled = visits ** (1.0 / temperature)
        scaled /= scaled.sum()
        return int(np.random.choice(TOTAL_ACTIONS, p=scaled))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _simulate_once(self, root_state: GameState, root: _Node) -> None:
        node = root
        state = root_state
        path: list[_Node] = [node]
        action_path: list[int] = []

        # Selection
        while node.is_expanded and node.children:
            action, child = _select_child(node, self.c_puct)
            child.virtual_loss += 1
            action_path.append(action)
            move = action_index_to_move(state, action)
            state = apply_move(state, move)
            node = child
            path.append(node)
            if is_terminal(state):
                break

        # Evaluation
        if is_terminal(state):
            outcome = state.outcome
            if outcome == "draw" or outcome is None:
                leaf_value = 0.0
            else:
                # value from the perspective of the side just moved (i.e. the
                # opponent of state.to_move). We back up from the *to_move*
                # POV for consistency with the network output convention.
                leaf_value = 1.0 if outcome == state.to_move else -1.0
        else:
            priors, values = self.evaluator([state])
            _expand(node, state, priors[0])
            leaf_value = float(values[0])

        leaf_player = state.to_move
        _backup(path, leaf_value, leaf_player)


# ---------------------------------------------------------------------------
# Convenience: build an evaluator from a torch network
# ---------------------------------------------------------------------------


def make_torch_evaluator(network: object) -> Evaluator:
    """Build an Evaluator that runs ``network`` (a ``DTTTNet``) on CPU/GPU.

    Imports torch lazily so the rest of the package can be imported without
    a torch install (e.g. in some test contexts).
    """
    import torch
    import torch.nn.functional as F

    @torch.no_grad()
    def evaluator(states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        batch = np.stack([state_to_tensor(s) for s in states], axis=0)
        x = torch.from_numpy(batch)
        net = network  # type: ignore[assignment]
        net.eval()  # type: ignore[attr-defined]
        device = next(net.parameters()).device  # type: ignore[attr-defined]
        x = x.to(device)
        logits, value = net(x)  # type: ignore[operator]
        # NOTE: mask is applied in MCTS, not here. We softmax the raw logits
        # and let MCTS multiply by the legal mask.
        priors = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        v = value.squeeze(-1).cpu().numpy().astype(np.float32)
        return priors, v

    return evaluator


__all__ = [
    "MCTS",
    "Evaluator",
    "make_torch_evaluator",
    "VIRTUAL_LOSS",
    "DEFAULT_C_PUCT",
]


# Quiet unused-import lint (legal_moves is part of the public engine surface
# but referenced indirectly via legal_action_mask).
_ = legal_moves
_ = move_to_action_index
