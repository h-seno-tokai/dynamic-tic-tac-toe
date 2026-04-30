"""D4 symmetry augmentation for the 4x4-only pipeline.

The augmentation is purely tensor/policy permutation work and does not touch
either the legacy or the fast engine — both packages share the ``(27, 4, 4)``
encoding and the same 320-d action layout. We therefore re-export
:func:`dttt_train.augment.augment_sample` directly: importing it does NOT pull
the legacy engine into the 4x4 training loop (verified: ``dttt_train.augment``
only imports from ``dttt_train.encoding`` constants).

For 4x4-only data the augmentation always operates on the full 4x4 sub-grid
(channel 25 — the out-of-board mask — is always zero in our encoding), so the
``board_size`` inference inside :func:`augment_sample` returns 4 for every
sample produced by :mod:`dttt_train_4x4.encoding`.
"""

from __future__ import annotations

from dttt_train.augment import augment_sample

__all__ = ["augment_sample"]
