"""Board symmetry data augmentation for DTTT training samples.

Square board -> 8 symmetries (dihedral group D4):
  4 rotations (0, 90, 180, 270 deg) x 2 (identity, horizontal flip)

Each (state, policy, value_z, q) sample is expanded into 8 equivalent
samples. The spatial board channels (0..15) are rotated/flipped via numpy
in vectorised form; the broadcast-scalar channels (16..26) are passed
through unchanged. Policy permutations are pre-computed and cached per
``(t_id, board_size)``.
"""

from __future__ import annotations

import numpy as np

from .encoding import PLACE_ACTIONS, TOTAL_ACTIONS

_MAX_BOARD = 4
_SPATIAL_CHANNELS = 16  # channels 0-15 encode board positions
_NUM_SYMMETRIES = 8


def _spatial_transform(plane: np.ndarray, t_id: int) -> np.ndarray:
    """Apply symmetry t_id to a (..., H, W) array on the *full* 4x4 grid.

    Transform IDs (mapping (r, c) -> (r', c')):
      0=identity  1=rot90CCW  2=rot180  3=rot90CW
      4=flipH     5=flipV     6=flipDiag  7=flipAntiDiag

    NOTE: We rotate/flip on the whole 4x4 plane. For the 3x3 preset, the
    out-of-board cells are encoded as zeros in channels 0..15 (channel 25
    carries the out-of-board mask separately), so D4 transforms of the
    full 4x4 are not strictly equivalent. To handle 3x3 correctly we apply
    the transform only on the active sub-grid; the rest of the plane is
    already zero for spatial channels and is restored explicitly.
    """
    if t_id == 0:
        return plane
    if t_id == 1:
        return np.rot90(plane, k=1, axes=(-2, -1))
    if t_id == 2:
        return np.rot90(plane, k=2, axes=(-2, -1))
    if t_id == 3:
        return np.rot90(plane, k=3, axes=(-2, -1))
    if t_id == 4:
        return np.flip(plane, axis=-1)
    if t_id == 5:
        return np.flip(plane, axis=-2)
    if t_id == 6:
        return np.swapaxes(plane, -2, -1)
    if t_id == 7:
        return np.flip(np.flip(np.swapaxes(plane, -2, -1), axis=-2), axis=-1)
    raise ValueError(f"unknown transform id: {t_id}")


def _t_coord(t_id: int, r: int, c: int, n: int) -> tuple[int, int]:
    """Coordinate transform on an n x n grid (n in {3, 4})."""
    if t_id == 0: return r, c
    # rot90 CCW (np.rot90 default direction): (r, c) -> (n-1-c, r)
    if t_id == 1: return n - 1 - c, r
    if t_id == 2: return n - 1 - r, n - 1 - c
    if t_id == 3: return c, n - 1 - r
    if t_id == 4: return r, n - 1 - c
    if t_id == 5: return n - 1 - r, c
    if t_id == 6: return c, r
    if t_id == 7: return n - 1 - c, n - 1 - r
    raise ValueError(f"unknown transform id: {t_id}")


_policy_perm_cache: dict[tuple[int, int], np.ndarray] = {}


def _build_policy_perm(t_id: int, board_size: int) -> np.ndarray:
    key = (t_id, board_size)
    if key in _policy_perm_cache:
        return _policy_perm_cache[key]

    n = board_size
    perm = np.arange(TOTAL_ACTIONS, dtype=np.int32)

    for size_idx in range(4):
        for r in range(n):
            for c in range(n):
                old_a = size_idx * 16 + r * 4 + c
                nr, nc = _t_coord(t_id, r, c, n)
                new_a = size_idx * 16 + nr * 4 + nc
                perm[old_a] = new_a

    for tr in range(n):
        for tc in range(n):
            for fr in range(n):
                for fc in range(n):
                    to_idx = tr * 4 + tc
                    fr_idx = fr * 4 + fc
                    old_a = PLACE_ACTIONS + to_idx * 16 + fr_idx
                    ntr, ntc = _t_coord(t_id, tr, tc, n)
                    nfr, nfc = _t_coord(t_id, fr, fc, n)
                    new_a = PLACE_ACTIONS + (ntr * 4 + ntc) * 16 + (nfr * 4 + nfc)
                    perm[old_a] = new_a

    _policy_perm_cache[key] = perm
    return perm


def _board_size_from_tensor(tensor: np.ndarray) -> int:
    """Infer board size from channel 25 (out-of-board mask)."""
    return 3 if tensor[25, 3, 0] > 0.5 else 4


def _transform_tensor(tensor: np.ndarray, t_id: int, board_size: int) -> np.ndarray:
    """Apply symmetry t_id to a (27, 4, 4) float32 tensor."""
    if t_id == 0:
        return tensor
    result = tensor.copy()
    if board_size == 4:
        # Vectorised transform on the whole 4x4 plane.
        result[:_SPATIAL_CHANNELS] = _spatial_transform(
            tensor[:_SPATIAL_CHANNELS], t_id
        )
    else:
        # 3x3 preset: only transform the active 3x3 sub-grid.
        n = board_size
        sub = tensor[:_SPATIAL_CHANNELS, :n, :n]
        sub_t = _spatial_transform(sub, t_id)
        result[:_SPATIAL_CHANNELS, :, :] = 0.0
        result[:_SPATIAL_CHANNELS, :n, :n] = sub_t
    # Channels 16-26 are broadcast scalars or fixed masks - copy as-is.
    return result


def augment_sample(
    tensor: np.ndarray,
    policy: np.ndarray,
    value_z: float,
    q_target: float,
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Expand one training sample into 8 symmetry-equivalent samples.

    Both ``value_z`` and ``q_target`` are invariant under board symmetry,
    so they pass through unchanged.
    """
    board_size = _board_size_from_tensor(tensor)
    samples: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for t_id in range(_NUM_SYMMETRIES):
        aug_tensor = _transform_tensor(tensor, t_id, board_size)
        perm = _build_policy_perm(t_id, board_size)
        aug_policy = np.zeros_like(policy)
        aug_policy[perm] = policy
        samples.append((aug_tensor, aug_policy, value_z, q_target))
    return samples


__all__ = ["augment_sample"]
