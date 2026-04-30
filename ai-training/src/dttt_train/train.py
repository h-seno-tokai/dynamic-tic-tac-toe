"""AlphaZero training loop for DTTT.

Single-process mode (default, --num-workers 0):
  Interleaves self-play and gradient updates on the same GPU.
  Loop: generate 1 game -> if buffer >= batch_size: do 1 training step.
  This gives roughly 1:1 games-to-steps ratio after warmup.

Multi-process mode (--num-workers N, N>=1):
  N worker processes do self-play on --worker-device (default cpu).
  Main process does gradient updates on --device (default cuda).

Usage
-----
  # Recommended: single-process GPU (RTX 2080, ~5-10 sec/game)
  python -m dttt_train.train --total-steps 500 --num-sims 200 --batch-size 256

  # After training, export ONNX:
  python -m dttt_train.export_onnx --ckpt runs/latest.pt --out public/model.onnx
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

from .augment import augment_sample
from .network import DTTTNet
from .parallel_selfplay import play_games_parallel
from .replay_buffer import ReplayBuffer
from .rules import PRESET_3X3, PRESET_4X4_XL, GameRules
from .selfplay import play_game

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE: int = 256
DEFAULT_LR: float = 1e-3
DEFAULT_WD: float = 1e-4
DEFAULT_REPLAY_CAPACITY: int = 20_000
DEFAULT_MCTS_SIMS: int = 200
DEFAULT_PRESET_RATIO_3X3: float = 0.5


def _sample_preset(rng: random.Random, ratio_3x3: float) -> GameRules:
    return PRESET_3X3 if rng.random() < ratio_3x3 else PRESET_4X4_XL


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
    net.train()
    optim.zero_grad(set_to_none=True)
    logits, v_pred = net(states)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(policies * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(v_pred.squeeze(-1), values)
    total = policy_loss + value_loss
    total.backward()
    optim.step()
    net.eval()
    return float(total.item()), float(policy_loss.item()), float(value_loss.item())


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------


def _save_checkpoint(
    ckpt_dir: Path,
    net: DTTTNet,
    optim: torch.optim.Optimizer,
    step: int,
) -> None:
    state = {"step": step, "model_state": net.state_dict(), "optim_state": optim.state_dict()}
    path = ckpt_dir / f"ckpt_{step:06d}.pt"
    torch.save(state, path)
    torch.save(state, ckpt_dir / "latest.pt")
    logger.info("checkpoint saved -> %s", path)


def _load_checkpoint(
    ckpt_path: Path,
    net: DTTTNet,
    optim: torch.optim.Optimizer,
) -> int:
    if not ckpt_path.exists():
        return 0
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model_state"])
    if "optim_state" in state:
        optim.load_state_dict(state["optim_state"])
    step = int(state["step"])
    logger.info("resumed from %s  (step %d)", ckpt_path, step)
    return step


# ---------------------------------------------------------------------------
# Single-process loop  (1 game -> 1 training step after warmup)
# ---------------------------------------------------------------------------


def _run_single_process(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    num_parallel = max(1, args.num_parallel)
    logger.info(
        "single-process  device=%s  parallel=%d  sims=%d  batch=%d  steps=%d",
        device, num_parallel, args.num_sims, args.batch_size, args.total_steps,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    net = DTTTNet().to(device)
    net.eval()
    optim = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=max(1, args.total_steps), eta_min=1e-5)
    buffer = ReplayBuffer(capacity=args.replay_capacity)

    resume_path = Path(args.resume_from) if args.resume_from else (ckpt_dir / "latest.pt" if args.resume else None)
    start_step = _load_checkpoint(resume_path, net, optim) if resume_path else 0
    if start_step > 0:
        for _ in range(start_step):
            scheduler.step()
        logger.info("scheduler fast-forwarded to step %d", start_step)

    rng_preset = random.Random(args.seed + start_step)
    rng_buf = np.random.default_rng(args.seed + start_step)
    games_played = start_step * num_parallel
    train_step = start_step

    while train_step < args.total_steps:
        # --- generate num_parallel games in one batched GPU call ---
        rules_list = [_sample_preset(rng_preset, args.ratio_3x3) for _ in range(num_parallel)]
        all_game_samples = play_games_parallel(
            net,
            rules_list,
            num_sims=args.num_sims,
        )
        total_moves = 0
        for game_samples in all_game_samples:
            for s in game_samples:
                aug = augment_sample(s[0], s[1], s[2])
                buffer.extend(aug)
            total_moves += len(game_samples)
        games_played += num_parallel

        if len(buffer) < args.batch_size:
            logger.info(
                "warmup  games=%d  moves_this_batch=%d  buffer=%d/%d",
                games_played, total_moves, len(buffer), args.batch_size,
            )
            continue

        # --- num_parallel * grad_mult gradient steps per batch ---
        grad_steps = num_parallel * args.grad_mult
        for _ in range(grad_steps):
            if train_step >= args.total_steps:
                break
            states_np, policies_np, values_np = buffer.sample(args.batch_size, rng=rng_buf)
            states = torch.from_numpy(states_np).to(device)
            policies = torch.from_numpy(policies_np).to(device)
            values = torch.from_numpy(values_np).to(device)

            total, pol, val = _train_step(net, optim, states, policies, values)
            scheduler.step()
            train_step += 1

            if train_step % 100 == 0 or train_step == args.total_steps:
                logger.info(
                    "step %d/%d  total=%.4f  policy=%.4f  value=%.4f  lr=%.2e  games=%d",
                    train_step, args.total_steps, total, pol, val,
                    scheduler.get_last_lr()[0], games_played,
                )
            if train_step % args.ckpt_every == 0 or train_step == args.total_steps:
                _save_checkpoint(ckpt_dir, net, optim, train_step)


# ---------------------------------------------------------------------------
# Multi-process worker
# ---------------------------------------------------------------------------


def _selfplay_worker(
    worker_id: int,
    num_sims: int,
    ratio_3x3: float,
    seed: int,
    out_queue: "mp.Queue[list[tuple[np.ndarray, np.ndarray, float]]]",
    stop_event: "mp.Event",  # type: ignore[type-arg]
    device_str: str,
) -> None:
    rng = random.Random(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

    device = torch.device(device_str)
    net = DTTTNet().to(device)
    net.eval()

    game_idx = 0
    while not stop_event.is_set():
        rules = _sample_preset(rng, ratio_3x3)
        samples = play_game(net, rules, num_sims=num_sims, seed=seed + worker_id + game_idx)
        out_queue.put(samples)
        game_idx += 1


# ---------------------------------------------------------------------------
# Multi-process loop  (1 game received -> 1 training step)
# ---------------------------------------------------------------------------


def _run_multi_process(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    logger.info(
        "multi-process  device=%s  worker-device=%s  workers=%d  sims=%d  batch=%d  steps=%d",
        device, args.worker_device, args.num_workers,
        args.num_sims, args.batch_size, args.total_steps,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    net = DTTTNet().to(device)
    net.eval()
    optim = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=max(1, args.total_steps))
    buffer = ReplayBuffer(capacity=args.replay_capacity)

    ctx = mp.get_context("spawn")
    queue: "mp.Queue[list[tuple[np.ndarray, np.ndarray, float]]]" = ctx.Queue(maxsize=256)
    stop_event = ctx.Event()

    workers = []
    for wid in range(args.num_workers):
        p = ctx.Process(
            target=_selfplay_worker,
            args=(wid, args.num_sims, args.ratio_3x3, args.seed, queue, stop_event, args.worker_device),
            daemon=True,
        )
        p.start()
        workers.append(p)

    logger.info("%d workers started...", args.num_workers)
    rng = np.random.default_rng(args.seed)
    games_received = 0
    train_step = 0

    try:
        while train_step < args.total_steps:
            # block until a game arrives
            samples = queue.get()
            buffer.extend(samples)
            games_received += 1

            if len(buffer) < args.batch_size:
                logger.info(
                    "warmup  game=%d  moves=%d  buffer=%d/%d",
                    games_received, len(samples), len(buffer), args.batch_size,
                )
                continue

            states_np, policies_np, values_np = buffer.sample(args.batch_size, rng=rng)
            states = torch.from_numpy(states_np).to(device)
            policies = torch.from_numpy(policies_np).to(device)
            values = torch.from_numpy(values_np).to(device)

            total, pol, val = _train_step(net, optim, states, policies, values)
            scheduler.step()
            train_step += 1

            logger.info(
                "step %d/%d  total=%.4f  policy=%.4f  value=%.4f  lr=%.2e  games=%d",
                train_step, args.total_steps, total, pol, val,
                scheduler.get_last_lr()[0], games_received,
            )

            if train_step % args.ckpt_every == 0 or train_step == args.total_steps:
                _save_checkpoint(ckpt_dir, net, train_step)
    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        logger.info("done – %d steps, %d games", train_step, games_received)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DTTT AlphaZero training")
    parser.add_argument("--total-steps", type=int, default=1500,
                        help="number of gradient steps (approx = number of game batches after warmup)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from latest.pt in --ckpt-dir if it exists")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="path to a specific checkpoint to resume from")
    parser.add_argument("--grad-mult", type=int, default=8,
                        help="gradient steps per game = num_parallel * grad_mult (default 8 matches 8x augmentation)")
    parser.add_argument("--num-parallel", type=int, default=16,
                        help="number of games to run simultaneously per step (batched GPU eval)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 = single-process GPU mode (recommended)")
    parser.add_argument("--num-sims", type=int, default=DEFAULT_MCTS_SIMS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--warmup-games", type=int, default=0,
                        help="games to collect before training starts (0 = auto = batch_size // avg_game_len)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WD)
    parser.add_argument("--ratio-3x3", type=float, default=DEFAULT_PRESET_RATIO_3X3)
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY)
    parser.add_argument("--ckpt-dir", type=str, default="runs")
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--worker-device", type=str, default="cpu")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.num_workers == 0:
        _run_single_process(args)
    else:
        _run_multi_process(args)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
