"""Board symmetry data augmentation for DTTT training samples.

A square board has 8 symmetries (dihedral group D4):
  4 rotations (0°, 90°, 180°, 270°) × 2 (identity, horizontal flip)

Each (state_tensor, policy, value) sample is expanded into 8 equivalent
samples, multiplying training data by 8 at zero cost.
"""

from __future__ import annotations

import numpy as np

from .encoding import PLACE_ACTIONS, TOTAL_ACTIONS

_MAX_BOARD = 4
_SPATIAL_CHANNELS = 16  # channels 0-15 encode board positions
_NUM_SYMMETRIES = 8

# 8 coordinate transforms (r, c, n) -> (r', c') for an n×n board.
# Transform IDs: 0=identity, 1=rot90CW, 2=rot180, 3=rot270CW,
#                4=flipH, 5=flipV, 6=flipDiag, 7=flipAntiDiag
def _t(t_id: int, r: int, c: int, n: int) -> tuple[int, int]:
    if t_id == 0: return r, c
    if t_id == 1: return c, n - 1 - r
    if t_id == 2: return n - 1 - r, n - 1 - c
    if t_id == 3: return n - 1 - c, r
    if t_id == 4: return r, n - 1 - c
    if t_id == 5: return n - 1 - r, c
    if t_id == 6: return c, r
    if t_id == 7: return n - 1 - c, n - 1 - r
    raise ValueError(f"unknown transform id: {t_id}")


# Cache policy permutation tables keyed by (t_id, board_size).
_policy_perm_cache: dict[tuple[int, int], np.ndarray] = {}


def _build_policy_perm(t_id: int, board_size: int) -> np.ndarray:
    key = (t_id, board_size)
    if key in _policy_perm_cache:
        return _policy_perm_cache[key]

    n = board_size
    perm = np.arange(TOTAL_ACTIONS, dtype=np.int32)

    # PLACE actions: size_idx * 16 + row * 4 + col
    for size_idx in range(4):
        for r in range(n):
            for c in range(n):
                old_a = size_idx * 16 + r * 4 + c
                nr, nc = _t(t_id, r, c, n)
                new_a = size_idx * 16 + nr * 4 + nc
                perm[old_a] = new_a

    # MOVE actions: 64 + to_idx * 16 + from_idx
    for tr in range(n):
        for tc in range(n):
            for fr in range(n):
                for fc in range(n):
                    to_idx = tr * 4 + tc
                    fr_idx = fr * 4 + fc
                    old_a = PLACE_ACTIONS + to_idx * 16 + fr_idx
                    ntr, ntc = _t(t_id, tr, tc, n)
                    nfr, nfc = _t(t_id, fr, fc, n)
                    new_a = PLACE_ACTIONS + (ntr * 4 + ntc) * 16 + (nfr * 4 + nfc)
                    perm[old_a] = new_a

    _policy_perm_cache[key] = perm
    return perm


def _transform_tensor(tensor: np.ndarray, t_id: int, board_size: int) -> np.ndarray:
    """Apply symmetry t_id to (27, 4, 4) float32 tensor."""
    if t_id == 0:
        return tensor.copy()
    result = tensor.copy()
    n = board_size
    for ch in range(_SPATIAL_CHANNELS):
        new_ch = np.zeros((_MAX_BOARD, _MAX_BOARD), dtype=np.float32)
        for r in range(n):
            for c in range(n):
                nr, nc = _t(t_id, r, c, n)
                new_ch[nr, nc] = tensor[ch, r, c]
        result[ch] = new_ch
    # Channels 16-26 are broadcast scalars or fixed masks — copy as-is.
    return result


def _board_size_from_tensor(tensor: np.ndarray) -> int:
    """Infer board size from channel 25 (out-of-board mask)."""
    return 3 if tensor[25, 3, 0] > 0.5 else 4


def augment_sample(
    tensor: np.ndarray,
    policy: np.ndarray,
    value: float,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Expand one training sample into 8 symmetry-equivalent samples."""
    board_size = _board_size_from_tensor(tensor)
    samples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for t_id in range(_NUM_SYMMETRIES):
        aug_tensor = _transform_tensor(tensor, t_id, board_size)
        perm = _build_policy_perm(t_id, board_size)
        aug_policy = np.zeros_like(policy)
        aug_policy[perm] = policy
        samples.append((aug_tensor, aug_policy, value))
    return samples


__all__ = ["augment_sample"]
