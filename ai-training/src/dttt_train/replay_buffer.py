"""FIFO replay buffer for self-play samples.

Stores tuples of ``(state_tensor, policy_target, value_z, q_target)`` and
supports optional **recency-biased sampling** (newer samples are drawn
more often, controlled by ``recency_alpha``).

  * ``recency_alpha == 0``  -> uniform (default behaviour)
  * ``recency_alpha > 0``   -> sample[i] weight = (i / N) ** alpha after
                               sorting by insertion order. Typical: 1.0.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# (state_tensor, policy_target, value_z, q_target)
Sample = tuple[np.ndarray, np.ndarray, float, float]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity: int = capacity
        self._buf: deque[Sample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, sample: Sample) -> None:
        self._buf.append(sample)

    def extend(self, samples: list[Sample]) -> None:
        self._buf.extend(samples)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
        recency_alpha: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(states, policies, values_z, q_targets)`` numpy batches.

        ``states``    : float32 (B, 27, 4, 4)
        ``policies``  : float32 (B, 320)
        ``values_z``  : float32 (B,)
        ``q_targets`` : float32 (B,)
        """
        if len(self._buf) == 0:
            raise RuntimeError("replay buffer is empty")
        rng = rng or np.random.default_rng()
        n = len(self._buf)

        if recency_alpha > 0.0 and n > 1:
            # Linear position 0..1 (oldest -> newest), weighted (pos+eps)^alpha.
            pos = np.arange(1, n + 1, dtype=np.float64) / n
            weights = pos ** recency_alpha
            weights /= weights.sum()
            idx = rng.choice(n, size=batch_size, p=weights)
        else:
            idx = rng.integers(0, n, size=batch_size)

        states = np.stack([self._buf[int(i)][0] for i in idx], axis=0).astype(np.float32)
        policies = np.stack([self._buf[int(i)][1] for i in idx], axis=0).astype(np.float32)
        values = np.array([self._buf[int(i)][2] for i in idx], dtype=np.float32)
        q_targets = np.array([self._buf[int(i)][3] for i in idx], dtype=np.float32)
        return states, policies, values, q_targets


__all__ = ["ReplayBuffer", "Sample"]
