"""AlphaZero training loop for the 4x4-only DTTT pipeline.

Differences from :mod:`dttt_train.train` (which DO NOT change algorithm,
only data balance):

  * ``--grad-mult`` defaults to ``1.0`` (was 8). The legacy default produced
    severe replay-buffer overfitting because self-play was the wallclock
    bottleneck — the same buffer was resampled hundreds of times per refresh.
    Float values <1.0 are allowed (e.g. ``--grad-mult 0.5`` runs one grad step
    per two self-play games on the GPU side); we use ``ceil(n_parallel * gm)``.
  * ``--num-workers`` defaults to ``6`` (was 0). With the new fast 4x4 engine
    (~10-25x faster legal-mask / apply_move), the CPU workers should produce
    the bulk of the training data. ``--worker-blas-threads`` defaults to 1
    so that 6 workers + main + BLAS comfortably fit on an 8-core CPU.
  * ``--num-parallel`` (GPU main) defaults to ``64`` (was 256). The GPU loop
    no longer needs to be the primary data source.
  * ``--num-sims`` defaults to 200, ``--batch-size`` 512.
  * ``--replay-capacity`` defaults to 1,000,000 (was 500,000) so the buffer
    can absorb the higher inflow.
  * Output dir defaults to ``runs/run10`` (legacy runs run01..run09 are
    untouched).

IMPORTANT: this trainer uses :class:`dttt_train_4x4.network.DTTTNet` (96ch x
6 ResBlocks, ~1.0M params). The legacy :class:`dttt_train.network.DTTTNet` is
128ch x 8 ResBlocks (~5.5M params, ~38 MB checkpoints) and is **not weight
compatible**. Resuming from a legacy checkpoint will raise a clear error;
this trainer can only resume from a checkpoint produced by itself.

Usage
-----
  python -m dttt_train_4x4.train --total-steps 50000 --num-sims 200

  # Quick dry-run for timing validation (one outer loop, verbose):
  python -m dttt_train_4x4.train --dry-run --device cpu \\
      --num-parallel 4 --num-sims 16 --num-workers 0 --total-steps 4
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import multiprocessing as mp
import os
import queue as queue_mod
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .augment import augment_sample
from .network import DTTTNet
from .parallel_selfplay import play_games_parallel
from .replay_buffer import ReplayBuffer
from .rules import PRESET_4X4_XL, GameRules

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE: int = 512
DEFAULT_LR: float = 2e-3
DEFAULT_WD: float = 1e-4
DEFAULT_REPLAY_CAPACITY: int = 1_000_000   # was 500k for the universal trainer
DEFAULT_MCTS_SIMS: int = 200
# z / Q mixing for the value target: target_scalar = (1-Q_MIX)*z + Q_MIX*q
DEFAULT_Q_MIX: float = 0.25
# Linear warmup steps (cosine kicks in afterwards).
DEFAULT_WARMUP_STEPS: int = 500
# EMA decay for the slow eval model used in self-play.
DEFAULT_EMA_DECAY: float = 0.999
# Recency bias for replay sampling (0 -> uniform).
DEFAULT_RECENCY_ALPHA: float = 1.0

# New defaults — fixing the data-balance problem from the universal trainer.
DEFAULT_GRAD_MULT: float = 1.0     # was 8 in dttt_train.train
DEFAULT_NUM_PARALLEL: int = 64     # was 256
DEFAULT_NUM_WORKERS: int = 6       # was 0
DEFAULT_WORKER_GAMES_PER_BATCH: int = 16
DEFAULT_WORKER_BLAS_THREADS: int = 1


# Marker bytes that identify a 4x4 checkpoint vs the legacy 128ch / 8-block
# universal one.  These are state-dict tensor shapes that differ between the
# two architectures.
_FOURBYFOUR_STEM_OUT_CH: int = 96      # legacy is 128
_FOURBYFOUR_NUM_RES_BLOCKS: int = 6    # legacy is 8


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


# ---------------------------------------------------------------------------
# WDL helpers
# ---------------------------------------------------------------------------


def _wdl_target_from_z_q(
    values_z: torch.Tensor,
    q_values: torch.Tensor,
    q_mix: float,
) -> torch.Tensor:
    """Build a soft (W, D, L) probability target.

    scalar = (1 - q_mix) * z + q_mix * q   (in [-1, 1])
    P(win)  = max(0,  scalar)
    P(loss) = max(0, -scalar)
    P(draw) = 1 - P(win) - P(loss)
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
    """Simple parameter EMA. Copies the *full* state_dict (params + buffers)
    so the EMA model is a drop-in replacement for inference / self-play."""

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
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _is_4x4_state_dict(state_dict: dict) -> bool:
    """Return True if ``state_dict`` looks like a 4x4-only checkpoint.

    We probe the stem conv output channels: 96 for 4x4-XL, 128 for the legacy
    universal network. Either ``stem_conv.weight`` or its underlying parameter
    name should be present.
    """
    w = state_dict.get("stem_conv.weight")
    if w is None:
        return False
    # weight shape is (out_ch, in_ch, kH, kW)
    if hasattr(w, "shape") and len(w.shape) == 4:
        return int(w.shape[0]) == _FOURBYFOUR_STEM_OUT_CH
    return False


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
        "arch": "dttt_train_4x4",
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
    model_state = state.get("model_state") or state
    arch = state.get("arch")

    if arch != "dttt_train_4x4" and not _is_4x4_state_dict(model_state):
        raise RuntimeError(
            f"refusing to resume from {ckpt_path}: this looks like a legacy "
            "dttt_train (universal 128ch / 8-block) checkpoint, which is NOT "
            "weight-compatible with dttt_train_4x4 (96ch / 6-block, ~1.0M "
            "params). Train from scratch or pass a checkpoint produced by "
            "dttt_train_4x4.train."
        )

    try:
        net.load_state_dict(model_state)
    except RuntimeError as exc:
        logger.error("failed to load %s: %s", ckpt_path, exc)
        raise
    if "optim_state" in state:
        try:
            optim.load_state_dict(state["optim_state"])
        except (RuntimeError, ValueError) as exc:
            logger.warning("optimizer state ignored (%s) - starting AdamW fresh", exc)
    if ema is not None and "ema_state" in state:
        try:
            ema.shadow.load_state_dict(state["ema_state"])
        except RuntimeError as exc:
            logger.warning("EMA state ignored (%s)", exc)
    step = int(state.get("step", 0))
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
    """Run a tournament; if candidate beats best, copy weights and dump best.pt."""
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
        torch.save(
            {"step": train_step, "model_state": best.state_dict(),
             "arch": "dttt_train_4x4"},
            ckpt_dir / "best.pt",
        )
        logger.info("*** candidate PROMOTED to best.pt @ step %d ***", train_step)
        return True
    return False


# ---------------------------------------------------------------------------
# CPU self-play worker (augments the GPU main loop with extra games)
# ---------------------------------------------------------------------------


def _cpu_selfplay_worker(
    worker_id: int,
    latest_ckpt_path: str,
    num_sims: int,
    games_per_batch: int,
    pcr_prob: float,
    reduced_sim_frac: float,
    seed: int,
    out_queue,
    stop_event,
    blas_threads: int,
) -> None:
    # Limit BLAS threads so siblings don't fight for the same physical cores.
    try:
        torch.set_num_threads(max(1, blas_threads))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, blas_threads)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, blas_threads)))

    sd = seed + worker_id * 7919
    rng = random.Random(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)

    net = DTTTNet().to("cpu")
    net.eval()
    last_mtime: float = 0.0

    def _maybe_reload() -> None:
        nonlocal last_mtime
        try:
            mtime = os.path.getmtime(latest_ckpt_path)
        except (FileNotFoundError, OSError):
            return
        if mtime <= last_mtime:
            return
        try:
            blob = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
            state = blob.get("ema_state") or blob.get("model_state") or blob
            net.load_state_dict(state)
            last_mtime = mtime
        except Exception:
            return

    while not stop_event.is_set():
        _maybe_reload()
        rules_list = [PRESET_4X4_XL for _ in range(games_per_batch)]
        try:
            games = play_games_parallel(
                net, rules_list,
                num_sims=num_sims,
                pcr_prob=pcr_prob,
                reduced_sim_frac=reduced_sim_frac,
            )
        except Exception:
            continue
        for g in games:
            try:
                out_queue.put(g, timeout=10.0)
            except queue_mod.Full:
                pass
            if stop_event.is_set():
                break
        # Suppress unused-name warnings in some IDEs.
        _ = rng


def _spawn_cpu_workers(
    args: argparse.Namespace,
    ckpt_dir: Path,
):
    """Spawn ``args.num_workers`` CPU self-play workers."""
    if args.num_workers <= 0:
        return None, None, None

    latest = ckpt_dir / "latest.pt"
    if not latest.exists():
        torch.save(
            {"step": 0, "model_state": DTTTNet().state_dict(),
             "arch": "dttt_train_4x4"},
            latest,
        )

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue(maxsize=args.worker_queue_max)
    stop_event = ctx.Event()
    workers: list[mp.Process] = []
    for wid in range(args.num_workers):
        p = ctx.Process(
            target=_cpu_selfplay_worker,
            args=(
                wid,
                str(latest.resolve()),
                args.num_sims,
                args.worker_games_per_batch,
                args.pcr_prob,
                args.reduced_sim_frac,
                args.seed + 1000,
                out_queue,
                stop_event,
                args.worker_blas_threads,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)
    logger.info(
        "spawned %d CPU workers  (games_per_batch=%d, blas_threads=%d)",
        args.num_workers, args.worker_games_per_batch, args.worker_blas_threads,
    )
    return workers, out_queue, stop_event


def _drain_worker_queue(
    out_queue,
    buffer: ReplayBuffer,
    max_drain: int = 64,
) -> tuple[int, int]:
    """Pull any available worker games out of ``out_queue``, augment, add to buffer."""
    if out_queue is None:
        return 0, 0
    games = 0
    samples = 0
    for _ in range(max_drain):
        try:
            game = out_queue.get_nowait()
        except queue_mod.Empty:
            break
        for s in game:
            aug = augment_sample(s[0], s[1], s[2], s[3])
            buffer.extend(aug)
            samples += len(aug)
        games += 1
    return games, samples


def _save_latest_only(
    ckpt_dir: Path, net: DTTTNet, ema: _EMA | None, train_step: int,
) -> None:
    """Lightweight save of just ``latest.pt`` (no per-step ckpt)."""
    state = {"step": train_step, "model_state": net.state_dict(),
             "arch": "dttt_train_4x4"}
    if ema is not None:
        state["ema_state"] = ema.shadow.state_dict()
    torch.save(state, ckpt_dir / "latest.pt")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _grad_steps_for(num_parallel: int, grad_mult: float) -> int:
    """Convert (num_parallel, grad_mult) -> integer grad-step count.

    ``grad_mult`` may be < 1.0 (e.g. 0.5 -> half a grad step per game on
    average). We round up so a non-zero ``grad_mult`` always produces at least
    one grad step per outer loop.
    """
    if grad_mult <= 0:
        return 0
    return max(1, math.ceil(num_parallel * grad_mult))


def _run_loop(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    device_type = device.type
    use_amp = args.amp and device_type == "cuda"
    num_parallel = max(1, args.num_parallel)
    logger.info(
        "device=%s  amp=%s  parallel=%d  sims=%d  batch=%d  steps=%d  "
        "grad_mult=%.2f  workers=%d  dry_run=%s",
        device, use_amp, num_parallel, args.num_sims, args.batch_size,
        args.total_steps, args.grad_mult, args.num_workers, args.dry_run,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    net = DTTTNet().to(device)
    net.eval()
    optim = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device_type) if use_amp else None
    buffer = ReplayBuffer(capacity=args.replay_capacity)
    ema = _EMA(net, decay=args.ema_decay) if args.ema else None

    best = DTTTNet().to(device)
    best.eval()
    best.load_state_dict(net.state_dict())

    resume_path = (
        Path(args.resume_from) if args.resume_from
        else (ckpt_dir / "latest.pt" if args.resume else None)
    )
    start_step = _load_checkpoint(resume_path, net, optim, ema) if resume_path else 0
    if start_step > 0:
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            blob = torch.load(best_path, map_location="cpu", weights_only=False)
            best.load_state_dict(blob["model_state"])
            logger.info("loaded best.pt from %s", best_path)
        else:
            best.load_state_dict(net.state_dict())

    for g in optim.param_groups:
        g["lr"] = args.lr * _lr_factor(start_step, args.warmup_steps, args.total_steps)

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

    if args.num_workers > 0:
        _save_latest_only(ckpt_dir, net, ema, start_step)
    workers, worker_queue, worker_stop = _spawn_cpu_workers(args, ckpt_dir)
    worker_games_total = 0

    def _selfplay_net() -> DTTTNet:
        if args.gating:
            return best
        if ema is not None:
            return ema.shadow
        return net

    # Dry-run timing accumulators.
    timing_self = 0.0
    timing_grad = 0.0
    outer_iters = 0

    try:
     while train_step < args.total_steps:
        outer_t0 = time.perf_counter()

        wg, _ws = _drain_worker_queue(worker_queue, buffer)
        worker_games_total += wg

        # ---- self-play (GPU main loop) ----
        rules_list = [PRESET_4X4_XL for _ in range(num_parallel)]
        sp_t0 = time.perf_counter()
        all_game_samples = play_games_parallel(
            _selfplay_net(),
            rules_list,
            num_sims=sim_sched.sims,
            pcr_prob=args.pcr_prob,
            reduced_sim_frac=args.reduced_sim_frac,
        )
        sp_dt = time.perf_counter() - sp_t0
        timing_self += sp_dt

        total_moves = 0
        for game_samples in all_game_samples:
            for s in game_samples:
                aug = augment_sample(s[0], s[1], s[2], s[3])
                buffer.extend(aug)
            total_moves += len(game_samples)
        games_played += num_parallel

        wg2, _ = _drain_worker_queue(worker_queue, buffer)
        worker_games_total += wg2

        if len(buffer) < args.batch_size:
            logger.info(
                "warmup  games=%d  moves_this_batch=%d  buffer=%d/%d  worker_games=%d",
                games_played, total_moves, len(buffer), args.batch_size, worker_games_total,
            )
            if args.dry_run:
                outer_iters += 1
                logger.info(
                    "[dry-run] outer iter complete  selfplay=%.2fs  "
                    "buffer=%d  worker_games=%d  games/min=%.1f",
                    sp_dt, len(buffer), worker_games_total,
                    (num_parallel + wg + wg2) / max(1e-6, time.perf_counter() - outer_t0) * 60.0,
                )
                break
            continue

        # ---- gradient updates ----
        grad_steps = _grad_steps_for(num_parallel, args.grad_mult)
        batch_losses: list[float] = []
        gs_t0 = time.perf_counter()
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
                    "step %d/%d  total=%.4f  policy=%.4f  value=%.4f  lr=%.2e  "
                    "sims=%d  games=%d  worker_games=%d  buf=%d",
                    train_step, args.total_steps, total, pol, val,
                    lr, sim_sched.sims, games_played, worker_games_total, len(buffer),
                )
            if train_step % args.ckpt_every == 0 or train_step == args.total_steps:
                _save_checkpoint(ckpt_dir, net, optim, train_step, ema)
            elif args.num_workers > 0 and train_step % args.worker_sync_every == 0:
                _save_latest_only(ckpt_dir, net, ema, train_step)

            if args.gating and train_step > 0 and train_step % args.gating_every == 0:
                _maybe_promote_best(
                    net=ema.shadow if ema is not None else net,
                    best=best,
                    rules_list=[PRESET_4X4_XL],
                    args=args,
                    train_step=train_step,
                    ckpt_dir=ckpt_dir,
                )
        gs_dt = time.perf_counter() - gs_t0
        timing_grad += gs_dt

        if batch_losses:
            avg_loss = sum(batch_losses) / len(batch_losses)
            if sim_sched.update(avg_loss):
                logger.info(
                    "*** sim count increased to %d (loss plateaued at %.4f) ***",
                    sim_sched.sims, avg_loss,
                )

        outer_dt = time.perf_counter() - outer_t0
        outer_iters += 1
        if args.dry_run:
            wq_drain = wg + wg2
            est_gpm = (num_parallel + wq_drain) / max(1e-6, outer_dt) * 60.0
            logger.info(
                "[dry-run] outer iter complete  selfplay=%.2fs  grad=%.2fs  "
                "outer=%.2fs  steps=%d  loss_avg=%.4f  games/min~=%.1f  "
                "main_games=%d  worker_games=%d",
                sp_dt, gs_dt, outer_dt, len(batch_losses),
                (sum(batch_losses) / max(1, len(batch_losses))),
                est_gpm, num_parallel, wq_drain,
            )
            break
    finally:
        if worker_stop is not None:
            worker_stop.set()
        if workers:
            for p in workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
            logger.info("CPU workers stopped (received %d worker games total)",
                        worker_games_total)

        if args.dry_run and outer_iters > 0:
            logger.info(
                "[dry-run summary] selfplay_total=%.2fs  grad_total=%.2fs  "
                "outer_iters=%d  worker_games=%d",
                timing_self, timing_grad, outer_iters, worker_games_total,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DTTT 4x4-only AlphaZero training")
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="run one outer iteration with verbose timing then exit")

    parser.add_argument("--grad-mult", type=float, default=DEFAULT_GRAD_MULT,
                        help="gradient steps per outer self-play loop = "
                             "ceil(num_parallel * grad_mult). Float allowed; "
                             "values <1.0 starve the GPU on purpose so CPU "
                             "workers can catch up.")
    parser.add_argument("--num-parallel", type=int, default=DEFAULT_NUM_PARALLEL)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="N CPU self-play workers in addition to the GPU main loop. "
                             "Each worker runs play_games_parallel on CPU and "
                             "reloads weights from latest.pt periodically.")
    parser.add_argument("--worker-games-per-batch", type=int,
                        default=DEFAULT_WORKER_GAMES_PER_BATCH,
                        help="games batched per worker forward pass")
    parser.add_argument("--worker-blas-threads", type=int,
                        default=DEFAULT_WORKER_BLAS_THREADS,
                        help="BLAS threads per worker (set so N*workers ~= phys_cores)")
    parser.add_argument("--worker-sync-every", type=int, default=50,
                        help="save latest.pt every N gradient steps so workers "
                             "can pick up fresh weights")
    parser.add_argument("--worker-queue-max", type=int, default=512,
                        help="max games queued from workers before they block")

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
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=DEFAULT_EMA_DECAY)
    parser.add_argument("--gating", action="store_true", default=False)
    parser.add_argument("--gating-every", type=int, default=2000)
    parser.add_argument("--gating-games", type=int, default=40)
    parser.add_argument("--gating-sims", type=int, default=100)
    parser.add_argument("--gating-threshold", type=float, default=0.55)

    # Replay
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY)
    parser.add_argument("--recency-alpha", type=float, default=DEFAULT_RECENCY_ALPHA,
                        help="0 = uniform sampling; >0 biases toward recent samples")

    # Misc
    parser.add_argument("--ckpt-dir", type=str, default="runs/run10",
                        help="default 'runs/run10' so we don't touch run01..run09")
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _run_loop(args)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
