"""Replay buffer for the 4x4-only training pipeline.

The data format ``(state_tensor, policy, value_z, q_target)`` is identical to
:mod:`dttt_train.replay_buffer`, so we simply re-export. The legacy module is
pure NumPy/python-collections — it does NOT import the legacy engine — so this
re-export keeps the 4x4 training loop free of legacy engine dependencies.
"""

from __future__ import annotations

from dttt_train.replay_buffer import ReplayBuffer, Sample

__all__ = ["ReplayBuffer", "Sample"]
