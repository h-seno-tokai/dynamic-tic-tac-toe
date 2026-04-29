"""Training-loop skeleton.

This is intentionally a scaffold:

* It wires up the network, optimiser, replay buffer, and a multiprocessing
  self-play worker pool.
* It does **not** run a full AlphaZero loop - the body of the optimiser step
  is real but minimal, and the worker is a placeholder that uses an
  untrained network. A production run will need:
    - a separate inference server / shared weights so workers don't each
      hold their own GPU
    - target-network refresh
    - evaluation arena to gate checkpoint promotion
    - real logging (TensorBoard / wandb)

Hyperparameters follow ``docs/07_ai_design.md`` 2.4.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .network import DTTTNet
from .replay_buffer import ReplayBuffer
from .rules import PRESET_3X3, PRESET_4X4_XL, GameRules
from .selfplay import play_game

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults (mirror docs/07_ai_design.md 2.4)
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE: int = 1024
DEFAULT_LR: float = 1e-3
DEFAULT_WD: float = 1e-4
DEFAULT_REPLAY_CAPACITY: int = 50_000  # ~games; close enough for samples
DEFAULT_MCTS_SIMS_TRAIN: int = 200
DEFAULT_PRESET_RATIO_3X3: float = 0.5  # 50/50


def _sample_preset(rng: random.Random, ratio_3x3: float) -> GameRules:
    return PRESET_3X3 if rng.random() < ratio_3x3 else PRESET_4X4_XL


# ---------------------------------------------------------------------------
# Self-play worker (multiprocessing target)
# ---------------------------------------------------------------------------


def _selfplay_worker(
    worker_id: int,
    games_per_worker: int,
    num_sims: int,
    ratio_3x3: float,
    seed: int,
    out_queue: "mp.Queue[list[tuple[np.ndarray, np.ndarray, float]]]",
) -> None:
    """A single self-play worker process.

    NOTE (scaffold): this constructs a fresh untrained network locally. A
    real run would receive a checkpoint path or shared weights via a Manager.
    """
    rng = random.Random(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    net = DTTTNet()
    net.eval()
    for _ in range(games_per_worker):
        rules = _sample_preset(rng, ratio_3x3)
        samples = play_game(net, rules, num_sims=num_sims)
        out_queue.put(samples)


# ---------------------------------------------------------------------------
# Optimiser step
# ---------------------------------------------------------------------------


def _train_step(
    net: DTTTNet,
    optim: torch.optim.Optimizer,
    states: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
) -> tuple[float, float, float]:
    """One AlphaZero update step. Returns ``(total, policy, value)`` losses."""
    net.train()
    optim.zero_grad(set_to_none=True)
    logits, v_pred = net(states)
    # Policy CE against MCTS visit distribution: -sum(pi * log_softmax(logits))
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(policies * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(v_pred.squeeze(-1), values)
    total = policy_loss + value_loss
    total.backward()
    optim.step()
    return float(total.item()), float(policy_loss.item()), float(value_loss.item())


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DTTT AlphaZero training (scaffold)")
    parser.add_argument("--total-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--games-per-worker", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WD)
    parser.add_argument("--num-sims", type=int, default=DEFAULT_MCTS_SIMS_TRAIN)
    parser.add_argument("--ratio-3x3", type=float, default=DEFAULT_PRESET_RATIO_3X3)
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY)
    parser.add_argument("--ckpt-dir", type=str, default="runs")
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    net = DTTTNet().to(device)
    optim = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=max(1, args.total_steps))
    buffer = ReplayBuffer(capacity=args.replay_capacity)

    # Spawn self-play workers.
    ctx = mp.get_context("spawn")
    queue: "mp.Queue[list[tuple[np.ndarray, np.ndarray, float]]]" = ctx.Queue()
    workers = []
    for wid in range(args.num_workers):
        p = ctx.Process(
            target=_selfplay_worker,
            args=(wid, args.games_per_worker, args.num_sims, args.ratio_3x3, args.seed, queue),
            daemon=True,
        )
        p.start()
        workers.append(p)

    rng = np.random.default_rng(args.seed)

    for step in range(1, args.total_steps + 1):
        # Drain whatever self-play games are ready (non-blocking).
        drained = 0
        while not queue.empty() and drained < 32:
            samples = queue.get()
            buffer.extend(samples)
            drained += 1

        if len(buffer) < args.batch_size:
            logger.info("step %d: buffer fill %d/%d (waiting)", step, len(buffer), args.batch_size)
            continue

        states_np, policies_np, values_np = buffer.sample(args.batch_size, rng=rng)
        states = torch.from_numpy(states_np).to(device)
        policies = torch.from_numpy(policies_np).to(device)
        values = torch.from_numpy(values_np).to(device)

        total, pol, val = _train_step(net, optim, states, policies, values)
        scheduler.step()
        logger.info(
            "step %d  total=%.4f  policy=%.4f  value=%.4f  lr=%.2e",
            step,
            total,
            pol,
            val,
            scheduler.get_last_lr()[0],
        )

        if step % args.ckpt_every == 0 or step == args.total_steps:
            path = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save({"step": step, "model_state": net.state_dict()}, path)
            latest = ckpt_dir / "latest.pt"
            torch.save({"step": step, "model_state": net.state_dict()}, latest)
            logger.info("saved checkpoint %s", path)

    for p in workers:
        if p.is_alive():
            p.terminate()
        p.join(timeout=5)


if __name__ == "__main__":
    # On Windows multiprocessing requires the spawn-protected entry-point.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
