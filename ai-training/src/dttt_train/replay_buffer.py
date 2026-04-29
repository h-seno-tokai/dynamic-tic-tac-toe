"""FIFO replay buffer for self-play samples.

Stores tuples of ``(state_tensor, policy_target, value_target)``. When the
buffer exceeds ``capacity`` (in samples), the oldest entries are dropped.
"""

from __future__ import annotations

from collections import deque

import numpy as np

Sample = tuple[np.ndarray, np.ndarray, float]


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
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(states, policies, values)`` numpy batches.

        ``states``   : float32 (B, 27, 4, 4)
        ``policies`` : float32 (B, 320)
        ``values``   : float32 (B,)
        """
        if len(self._buf) == 0:
            raise RuntimeError("replay buffer is empty")
        rng = rng or np.random.default_rng()
        n = len(self._buf)
        idx = rng.integers(0, n, size=batch_size)
        states = np.stack([self._buf[i][0] for i in idx], axis=0).astype(np.float32)
        policies = np.stack([self._buf[i][1] for i in idx], axis=0).astype(np.float32)
        values = np.array([self._buf[i][2] for i in idx], dtype=np.float32)
        return states, policies, values


__all__ = ["ReplayBuffer", "Sample"]
