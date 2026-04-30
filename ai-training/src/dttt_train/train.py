"""AlphaZero training loop for DTTT.

Single-process mode (default, --num-workers 0):
  Interleaves self-play and gradient updates on the same GPU.
  Loop: generate ``num_parallel`` games -> ``num_parallel * grad_mult``
  gradient steps. The data path now also captures root-Q per stored
  position so the value loss target can mix Q with the final outcome z.

Multi-process mode (--num-workers N, N>=1):
  N worker processes do self-play on --worker-device (default cpu).
  Main process does gradient updates on --device (default cuda).

Improvements over the AlphaZero scaffold:
  * **WDL value head** (Win / Draw / Loss categorical cross-entropy)
  * **Q-value target mixing** (z, root_Q -> WDL distribution target)
  * **Mixed precision (AMP)** on the optimisation step
  * **LR warmup** (linear warmup -> cosine annealing)
  * **EMA model**: a slow moving average of weights, used by self-play
  * **Best-network gating**: candidate vs best tournament every N steps
  * **Larger, recency-biased replay buffer**

Usage
-----
  python -m dttt_train.train --total-steps 50000 --num-sims 400

  # After training, export ONNX:
  python -m dttt_train.export_onnx --ckpt runs/latest.pt --out public/model.onnx
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .augment import augment_sample
from .network import DTTTNet
from .parallel_selfplay import play_games_parallel
from .replay_buffer import ReplayBuffer
from .rules import PRESET_3X3, PRESET_4X4_XL, GameRules
from .selfplay import play_game

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE: int = 512
DEFAULT_LR: float = 2e-3
DEFAULT_WD: float = 1e-4
DEFAULT_REPLAY_CAPACITY: int = 500_000
DEFAULT_MCTS_SIMS: int = 200
DEFAULT_PRESET_RATIO_3X3: float = 0.5
# z / Q mixing for the value target: target_scalar = (1-Q_MIX)*z + Q_MIX*q
# (KataGo uses ~0.25; dlshogi uses 0.3). We pick 0.25 as a robust default.
DEFAULT_Q_MIX: float = 0.25
# Linear warmup steps (cosine kicks in afterwards).
DEFAULT_WARMUP_STEPS: int = 500
# EMA decay for the slow eval model used in self-play.
DEFAULT_EMA_DECAY: float = 0.999
# Recency bias for replay sampling (0 -> uniform).
DEFAULT_RECENCY_ALPHA: float = 1.0


# ---------------------------------------------------------------------------
# Sim scheduler: increase num_sims when loss plateaus
# ---------------------------------------------------------------------------


class _SimScheduler:
    def __init__(
        self,
        init_sims: int,
        max_sims: int,
        factor: float = 2.0,
        patience: int = 5,
        threshold: float = 0.02,
    ) -> None:
        self.sims = init_sims
        self.max_sims = max_sims
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self._best = float("inf")
        self._stale = 0

    def update(self, avg_loss: float) -> bool:
        if avg_loss < self._best - self.threshold:
            self._best = avg_loss
            self._stale = 0
        else:
            self._stale += 1
        if self._stale >= self.patience and self.sims < self.max_sims:
            self.sims = min(int(self.sims * self.factor), self.max_sims)
            self._stale = 0
            self._best = float("inf")
            return True
        return False


def _sample_preset(rng: random.Random, ratio_3x3: float) -> GameRules:
    return PRESET_3X3 if rng.random() < ratio_3x3 else PRESET_4X4_XL


# ---------------------------------------------------------------------------
# WDL helpers
# ---------------------------------------------------------------------------


def _wdl_target_from_z_q(
    values_z: torch.Tensor,
    q_values: torch.Tensor,
    q_mix: float,
) -> torch.Tensor:
    """Build a soft (W, D, L) probability target.

    Strategy:
      * scalar = (1 - q_mix) * z + q_mix * q   (in [-1, 1])
      * Map scalar -> WDL probabilities:
            P(win)  = max(0,  scalar)
            P(loss) = max(0, -scalar)
            P(draw) = 1 - P(win) - P(loss)
        This is monotone, exact at z in {-1, 0, +1}, and consistent with
        the "scalar = P(win) - P(loss)" identity used at inference.
    """
    scalar = (1.0 - q_mix) * values_z + q_mix * q_values
    scalar = torch.clamp(scalar, -1.0, 1.0)
    p_win = torch.clamp(scalar, min=0.0)
    p_loss = torch.clamp(-scalar, min=0.0)
    p_draw = (1.0 - p_win - p_loss).clamp(min=0.0)
    target = torch.stack([p_win, p_draw, p_loss], dim=-1)
    return target


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------


class _EMA:
    """Simple parameter EMA (no buffers/BN running stats — those track
    naturally since BN running stats are buffers updated only in train mode).

    We copy the *full* state_dict (params + buffers) so the EMA model is a
    drop-in replacement for inference / self-play."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        ssd = self.shadow.state_dict()
        for k in ssd.keys():
            if not torch.is_floating_point(ssd[k]):
                ssd[k].copy_(msd[k])
                continue
            ssd[k].mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)


# ---------------------------------------------------------------------------
# Optimiser step (AMP-aware)
# ---------------------------------------------------------------------------


def _train_step(
    net: DTTTNet,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    states: torch.Tensor,
    policies: torch.Tensor,
    values_z: torch.Tensor,
    q_values: torch.Tensor,
    q_mix: float,
    value_loss_weight: float,
    use_amp: bool,
    device_type: str,
) -> tuple[float, float, float]:
    net.train()
    optim.zero_grad(set_to_none=True)

    wdl_target = _wdl_target_from_z_q(values_z, q_values, q_mix)

    if use_amp:
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
            logits, wdl_logits = net(states)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(policies * log_probs).sum(dim=1).mean()
            log_wdl = F.log_softmax(wdl_logits, dim=1)
            value_loss = -(wdl_target * log_wdl).sum(dim=1).mean()
            total = policy_loss + value_loss_weight * value_loss
        assert scaler is not None
        scaler.scale(total).backward()
        # Gradient clipping (light): scaler.unscale_ before clipping.
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        scaler.step(optim)
        scaler.update()
    else:
        logits, wdl_logits = net(states)
        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -(policies * log_probs).sum(dim=1).mean()
        log_wdl = F.log_softmax(wdl_logits, dim=1)
        value_loss = -(wdl_target * log_wdl).sum(dim=1).mean()
        total = policy_loss + value_loss_weight * value_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optim.step()

    net.eval()
    return float(total.item()), float(policy_loss.item()), float(value_loss.item())


# ---------------------------------------------------------------------------
# LR schedule (linear warmup -> cosine)
# ---------------------------------------------------------------------------


def _lr_factor(step: int, warmup_steps: int, total_steps: int, eta_min_ratio: float = 0.05) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return eta_min_ratio + (1.0 - eta_min_ratio) * cosine


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------


def _save_checkpoint(
    ckpt_dir: Path,
    net: DTTTNet,
    optim: torch.optim.Optimizer,
    step: int,
    ema: _EMA | None = None,
) -> None:
    state = {
        "step": step,
        "model_state": net.state_dict(),
        "optim_state": optim.state_dict(),
    }
    if ema is not None:
        state["ema_state"] = ema.shadow.state_dict()
    path = ckpt_dir / f"ckpt_{step:06d}.pt"
    torch.save(state, path)
    torch.save(state, ckpt_dir / "latest.pt")
    logger.info("checkpoint saved -> %s", path)


def _load_checkpoint(
    ckpt_path: Path,
    net: DTTTNet,
    optim: torch.optim.Optimizer,
    ema: _EMA | None = None,
) -> int:
    if not ckpt_path.exists():
        return 0
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    try:
        net.load_state_dict(state["model_state"])
    except RuntimeError as exc:
        # WDL head changes shape vs scalar value head -> can't resume from
        # an old checkpoint cleanly. Bail loudly.
        logger.error(
            "failed to load %s: %s. The value head changed (scalar -> WDL); "
            "do NOT resume from pre-WDL checkpoints.",
            ckpt_path, exc,
        )
        raise
    if "optim_state" in state:
        try:
            optim.load_state_dict(state["optim_state"])
        except (RuntimeError, ValueError) as exc:
            logger.warning("optimizer state ignored (%s) - starting AdamW fresh", exc)
    if ema is not None and "ema_state" in state:
        ema.shadow.load_state_dict(state["ema_state"])
    step = int(state["step"])
    logger.info("resumed from %s  (step %d)", ckpt_path, step)
    return step


# ---------------------------------------------------------------------------
# Best-network gating
# ---------------------------------------------------------------------------


def _maybe_promote_best(
    net: DTTTNet,
    best: DTTTNet,
    rules_list: list[GameRules],
    args: argparse.Namespace,
    train_step: int,
    ckpt_dir: Path,
) -> bool:
    """Run a tournament; if candidate beats best, copy weights and dump best.pt.

    Returns True if the candidate was promoted.
    """
    from .eval_match import evaluate_match

    logger.info(
        "*** gating tournament @ step %d : %d games, %d sims ***",
        train_step, args.gating_games, args.gating_sims,
    )
    wins, draws, losses = evaluate_match(
        candidate=net,
        best=best,
        rules_list=rules_list,
        num_games=args.gating_games,
        num_sims=args.gating_sims,
    )
    score = (wins + 0.5 * draws) / max(1, wins + draws + losses)
    logger.info(
        "gating result: W=%d D=%d L=%d  score=%.3f  threshold=%.3f",
        wins, draws, losses, score, args.gating_threshold,
    )
    if score >= args.gating_threshold:
        best.load_state_dict(net.state_dict())
        torch.save({"step": train_step, "model_state": best.state_dict()},
                   ckpt_dir / "best.pt")
        logger.info("*** candidate PROMOTED to best.pt @ step %d ***", train_step)
        return True
    return False


# ---------------------------------------------------------------------------
# Single-process loop
# ---------------------------------------------------------------------------


def _run_single_process(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    device_type = device.type
    use_amp = args.amp and device_type == "cuda"
    num_parallel = max(1, args.num_parallel)
    logger.info(
        "single-process  device=%s  amp=%s  parallel=%d  sims=%d  batch=%d  steps=%d",
        device, use_amp, num_parallel, args.num_sims, args.batch_size, args.total_steps,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    net = DTTTNet().to(device)
    net.eval()
    optim = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device_type) if use_amp else None
    buffer = ReplayBuffer(capacity=args.replay_capacity)
    ema = _EMA(net, decay=args.ema_decay) if args.ema else None

    # Best network used for self-play if gating is enabled. Until the first
    # promotion, best == initial network.
    best = DTTTNet().to(device)
    best.eval()
    best.load_state_dict(net.state_dict())

    resume_path = (
        Path(args.resume_from) if args.resume_from
        else (ckpt_dir / "latest.pt" if args.resume else None)
    )
    start_step = _load_checkpoint(resume_path, net, optim, ema) if resume_path else 0
    if start_step > 0:
        # If a best.pt exists, load it; otherwise initialise best from net.
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            blob = torch.load(best_path, map_location="cpu", weights_only=False)
            best.load_state_dict(blob["model_state"])
            logger.info("loaded best.pt from %s", best_path)
        else:
            best.load_state_dict(net.state_dict())

    # Apply LR factor for the resumed step so the schedule is consistent.
    for g in optim.param_groups:
        g["lr"] = args.lr * _lr_factor(start_step, args.warmup_steps, args.total_steps)

    rng_preset = random.Random(args.seed + start_step)
    rng_buf = np.random.default_rng(args.seed + start_step)
    games_played = start_step * num_parallel
    train_step = start_step

    sim_sched = _SimScheduler(
        init_sims=args.num_sims,
        max_sims=args.max_sims,
        factor=args.sim_factor,
        patience=args.sim_patience,
        threshold=args.sim_threshold,
    )

    # Self-play uses the BEST network when gating is on; otherwise the
    # current candidate. EMA, when enabled and gating is off, is also a fine
    # choice (we use it for self-play if --ema is on and --gating is off).
    def _selfplay_net() -> DTTTNet:
        if args.gating:
            return best
        if ema is not None:
            return ema.shadow
        return net

    while train_step < args.total_steps:
        rules_list = [_sample_preset(rng_preset, args.ratio_3x3) for _ in range(num_parallel)]
        all_game_samples = play_games_parallel(
            _selfplay_net(),
            rules_list,
            num_sims=sim_sched.sims,
            pcr_prob=args.pcr_prob,
            reduced_sim_frac=args.reduced_sim_frac,
        )
        total_moves = 0
        for game_samples in all_game_samples:
            for s in game_samples:
                # s = (tensor, policy, z, q)
                aug = augment_sample(s[0], s[1], s[2], s[3])
                buffer.extend(aug)
            total_moves += len(game_samples)
        games_played += num_parallel

        if len(buffer) < args.batch_size:
            logger.info(
                "warmup  games=%d  moves_this_batch=%d  buffer=%d/%d",
                games_played, total_moves, len(buffer), args.batch_size,
            )
            continue

        grad_steps = num_parallel * args.grad_mult
        batch_losses: list[float] = []
        for _ in range(grad_steps):
            if train_step >= args.total_steps:
                break

            states_np, policies_np, values_np, q_np = buffer.sample(
                args.batch_size, rng=rng_buf, recency_alpha=args.recency_alpha,
            )
            states = torch.from_numpy(states_np).to(device, non_blocking=True)
            policies = torch.from_numpy(policies_np).to(device, non_blocking=True)
            values = torch.from_numpy(values_np).to(device, non_blocking=True)
            q_targets = torch.from_numpy(q_np).to(device, non_blocking=True)

            # Apply LR for this step.
            lr = args.lr * _lr_factor(train_step, args.warmup_steps, args.total_steps)
            for g in optim.param_groups:
                g["lr"] = lr

            total, pol, val = _train_step(
                net, optim, scaler, states, policies, values, q_targets,
                q_mix=args.q_mix,
                value_loss_weight=args.value_loss_weight,
                use_amp=use_amp,
                device_type=device_type,
            )
            if ema is not None:
                ema.update(net)
            train_step += 1
            batch_losses.append(total)

            if train_step % 100 == 0 or train_step == args.total_steps:
                logger.info(
                    "step %d/%d  total=%.4f  policy=%.4f  value=%.4f  lr=%.2e  sims=%d  games=%d  buf=%d",
                    train_step, args.total_steps, total, pol, val,
                    lr, sim_sched.sims, games_played, len(buffer),
                )
            if train_step % args.ckpt_every == 0 or train_step == args.total_steps:
                _save_checkpoint(ckpt_dir, net, optim, train_step, ema)

            # Best-network gating tournament.
            if args.gating and train_step > 0 and train_step % args.gating_every == 0:
                _maybe_promote_best(
                    net=ema.shadow if ema is not None else net,
                    best=best,
                    rules_list=[PRESET_3X3, PRESET_4X4_XL],
                    args=args,
                    train_step=train_step,
                    ckpt_dir=ckpt_dir,
                )

        if batch_losses:
            avg_loss = sum(batch_losses) / len(batch_losses)
            if sim_sched.update(avg_loss):
                logger.info(
                    "*** sim count increased to %d (loss plateaued at %.4f) ***",
                    sim_sched.sims, avg_loss,
                )


# ---------------------------------------------------------------------------
# Multi-process worker (kept simple, no gating / no EMA)
# ---------------------------------------------------------------------------


def _selfplay_worker(
    worker_id: int,
    num_sims: int,
    ratio_3x3: float,
    seed: int,
    out_queue: "mp.Queue[list[tuple[np.ndarray, np.ndarray, float, float]]]",
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
    buffer = ReplayBuffer(capacity=args.replay_capacity)

    ctx = mp.get_context("spawn")
    queue: "mp.Queue[list[tuple[np.ndarray, np.ndarray, float, float]]]" = ctx.Queue(maxsize=256)
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
            samples = queue.get()
            buffer.extend(samples)
            games_received += 1

            if len(buffer) < args.batch_size:
                continue

            states_np, policies_np, values_np, q_np = buffer.sample(args.batch_size, rng=rng)
            states = torch.from_numpy(states_np).to(device)
            policies = torch.from_numpy(policies_np).to(device)
            values = torch.from_numpy(values_np).to(device)
            q_targets = torch.from_numpy(q_np).to(device)

            total, pol, val = _train_step(
                net, optim, None, states, policies, values, q_targets,
                q_mix=args.q_mix,
                value_loss_weight=args.value_loss_weight,
                use_amp=False,
                device_type=device.type,
            )
            train_step += 1

            logger.info(
                "step %d/%d  total=%.4f  policy=%.4f  value=%.4f  games=%d",
                train_step, args.total_steps, total, pol, val, games_received,
            )

            if train_step % args.ckpt_every == 0 or train_step == args.total_steps:
                _save_checkpoint(ckpt_dir, net, optim, train_step)
    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        logger.info("done - %d steps, %d games", train_step, games_received)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DTTT AlphaZero training")
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--grad-mult", type=int, default=8)
    parser.add_argument("--num-parallel", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    # MCTS
    parser.add_argument("--num-sims", type=int, default=DEFAULT_MCTS_SIMS)
    parser.add_argument("--max-sims", type=int, default=3200)
    parser.add_argument("--sim-patience", type=int, default=2)
    parser.add_argument("--sim-threshold", type=float, default=0.02)
    parser.add_argument("--sim-factor", type=float, default=2.0)
    parser.add_argument("--pcr-prob", type=float, default=0.25,
                        help="probability that a move uses full sims (KataGo PCR)")
    parser.add_argument("--reduced-sim-frac", type=float, default=0.25)

    # Optimisation
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WD)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--q-mix", type=float, default=DEFAULT_Q_MIX,
                        help="value target = (1-q_mix)*z + q_mix*root_q")
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="use mixed precision (CUDA only)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")

    # EMA / gating
    parser.add_argument("--ema", action="store_true", default=True,
                        help="track an EMA copy and use it for self-play / gating")
    parser.add_argument("--no-ema", dest="ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=DEFAULT_EMA_DECAY)
    parser.add_argument("--gating", action="store_true", default=False,
                        help="enable best-network gating (slower, more stable)")
    parser.add_argument("--gating-every", type=int, default=2000)
    parser.add_argument("--gating-games", type=int, default=40)
    parser.add_argument("--gating-sims", type=int, default=100)
    parser.add_argument("--gating-threshold", type=float, default=0.55)

    # Replay
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY)
    parser.add_argument("--recency-alpha", type=float, default=DEFAULT_RECENCY_ALPHA,
                        help="0 = uniform sampling; >0 biases toward recent samples")

    # Misc
    parser.add_argument("--ratio-3x3", type=float, default=DEFAULT_PRESET_RATIO_3X3)
    parser.add_argument("--ckpt-dir", type=str, default="runs")
    parser.add_argument("--ckpt-every", type=int, default=500)
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
