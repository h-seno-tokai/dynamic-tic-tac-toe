"""AlphaZero network for the 4x4-only DTTT training pipeline.

This network supersedes the universal ``dttt_train.network.DTTTNet`` for the
``PRESET_4X4_XL`` preset.  3x3 inference is handled by a separate exact
solver and no longer shares this trunk, which lets us shrink the network and
train faster.

Architecture (4x4-only)::

  Stem  : Conv3x3(27 -> 96) + BN + ReLU
  Body  : 6 x ResBlock
            Conv3x3(96 -> 96) + BN + ReLU
            Conv3x3(96 -> 96) + BN + skip + ReLU
  Heads :
    Place head : Conv1x1(96 -> 4)   -> flatten -> 64 logits
    Move head  : Conv1x1(96 -> 16)  -> flatten -> 256 logits
    -> concat to 320-d policy logits (NO mask in graph)
    Value head : Conv1x1(96 -> 32) + BN + ReLU
                 -> Global Average Pool
                 -> Linear(32 -> 64) + ReLU
                 -> Linear(64 -> 3)  -- WDL logits (Win, Draw, Loss)

Forward signature mirrors :class:`dttt_train.network.DTTTNet` so existing MCTS
and training code can call it identically::

    policy_logits, wdl_logits = net(x)
    # x:            (B, 27, 4, 4) float32
    # policy_logits:(B, 320)  -- raw logits, NO mask, NO softmax
    # wdl_logits:   (B, 3)    -- raw WDL logits  (Win, Draw, Loss)

Channel choice rationale
------------------------
The companion ``dttt_train_4x4/encoding.py`` (parallel agent) keeps the
universal ``(27, 4, 4)`` layout for v1 with channels 25 (out-of-board mask)
and 26 (unused-size mask) zero-filled - both are constants on every 4x4
sample and contribute no information.  We accept the two redundant input
channels because:

* it lets us reuse the universal encoding + augmentation utilities verbatim
  (no risk of off-by-one errors in symmetry orbits);
* the cost is exactly ``96 * 2 * 3 * 3 = 1728`` extra weights in the stem
  (~0.07% of total parameters) - well below noise of a fresh training run;
* a v2 can drop to a 25-channel input once the encoding is rev'd, without
  any change to the body / heads.

Trunk size: 6 blocks x 96 channels was picked over the universal 8 x 128 to
roughly halve the FLOPs (~0.32x parameters, ~0.42x conv FLOPs at 4x4) while
keeping enough capacity for the 4x4 search; this matches typical small-board
AlphaZero ablations where shrinking width helps more than shrinking depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# We deliberately import constants from the existing universal package so the
# action-space layout (PLACE_ACTIONS=64, MOVE_ACTIONS=256, TOTAL_ACTIONS=320,
# MAX_BOARD=4) cannot drift between trunks.  These constants are pure
# integers and do not import any 3x3 logic.
from dttt_train.rules import (
    MAX_BOARD,
    MOVE_ACTIONS,
    PLACE_ACTIONS,
    TOTAL_ACTIONS,
)

# Architecture constants - DO NOT change without updating the design doc.
INPUT_CHANNELS: int = 27          # keep universal encoding shape for v1
TRUNK_CHANNELS: int = 96
NUM_RES_BLOCKS: int = 6
PLACE_HEAD_CHANNELS: int = 4      # 4 sizes
MOVE_HEAD_CHANNELS: int = 16      # 16 from-cells (4x4)
VALUE_HEAD_CHANNELS: int = 32
VALUE_HIDDEN: int = 64
WDL_OUTPUTS: int = 3              # Win / Draw / Loss


class _ResBlock(nn.Module):
    def __init__(self, channels: int = TRUNK_CHANNELS) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


class DTTTNet4x4(nn.Module):
    """4x4-only AlphaZero network for the DTTT game.

    Forward signature::

        policy_logits, wdl_logits = net(x)
        # x:            (B, 27, 4, 4) float32
        # policy_logits:(B, 320)  -- raw logits, NO mask, NO softmax
        # wdl_logits:   (B, 3)    -- raw WDL logits  (Win, Draw, Loss)
    """

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        trunk_channels: int = TRUNK_CHANNELS,
        num_res_blocks: int = NUM_RES_BLOCKS,
    ) -> None:
        super().__init__()

        # Stem
        self.stem_conv = nn.Conv2d(
            input_channels, trunk_channels, kernel_size=3, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(trunk_channels)

        # Body
        self.res_blocks = nn.ModuleList(
            [_ResBlock(trunk_channels) for _ in range(num_res_blocks)]
        )

        # Policy heads
        self.place_head_conv = nn.Conv2d(
            trunk_channels, PLACE_HEAD_CHANNELS, kernel_size=1, bias=True
        )
        self.move_head_conv = nn.Conv2d(
            trunk_channels, MOVE_HEAD_CHANNELS, kernel_size=1, bias=True
        )

        # Value head -> 3-way WDL
        self.value_conv = nn.Conv2d(
            trunk_channels, VALUE_HEAD_CHANNELS, kernel_size=1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(VALUE_HEAD_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_HEAD_CHANNELS, VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, WDL_OUTPUTS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Stem
        h = F.relu(self.stem_bn(self.stem_conv(x)))

        # Body
        for block in self.res_blocks:
            h = block(h)

        # Policy: place (B, 4, 4, 4) -> (B, 64); move (B, 16, 4, 4) -> (B, 256)
        place = self.place_head_conv(h)
        place_logits = place.flatten(start_dim=1)
        assert place_logits.shape[1] == PLACE_ACTIONS, (
            f"place head produced {place_logits.shape[1]} logits, expected {PLACE_ACTIONS}"
        )

        move = self.move_head_conv(h)
        # Permute to (B, to_row, to_col, from_idx) to match the universal
        # encoding's ``action_index_to_move`` ordering.  The 4x4 encoding
        # agent inherits this convention.
        move_perm = move.permute(0, 2, 3, 1).contiguous()
        move_logits = move_perm.flatten(start_dim=1)
        assert move_logits.shape[1] == MOVE_ACTIONS, (
            f"move head produced {move_logits.shape[1]} logits, expected {MOVE_ACTIONS}"
        )

        policy_logits = torch.cat([place_logits, move_logits], dim=1)
        assert policy_logits.shape[1] == TOTAL_ACTIONS

        # Value (WDL)
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.mean(dim=(2, 3))                         # GAP -> (B, 32)
        v = F.relu(self.value_fc1(v))                  # -> (B, 64)
        wdl_logits = self.value_fc2(v)                 # -> (B, 3)

        return policy_logits, wdl_logits


# Back-compat alias so call sites that previously did
# ``from dttt_train.network import DTTTNet`` can switch to
# ``from dttt_train_4x4.network import DTTTNet`` with no other changes.
DTTTNet = DTTTNet4x4


def wdl_to_scalar(wdl_logits: torch.Tensor) -> torch.Tensor:
    """Collapse WDL logits to a scalar Q in [-1, 1].

    ``Q = P(win) - P(loss)``  (draw contributes 0 by construction).
    """
    probs = F.softmax(wdl_logits, dim=-1)
    # weight vector (Win, Draw, Loss)
    weights = wdl_logits.new_tensor([1.0, 0.0, -1.0])
    return (probs * weights).sum(dim=-1, keepdim=True)


def make_dummy_input(batch_size: int = 1) -> torch.Tensor:
    """Build a zero input tensor of the canonical shape ``(B, 27, 4, 4)``."""
    return torch.zeros((batch_size, INPUT_CHANNELS, MAX_BOARD, MAX_BOARD), dtype=torch.float32)


__all__ = [
    "DTTTNet",
    "DTTTNet4x4",
    "wdl_to_scalar",
    "make_dummy_input",
    "INPUT_CHANNELS",
    "TRUNK_CHANNELS",
    "NUM_RES_BLOCKS",
    "WDL_OUTPUTS",
    "PLACE_ACTIONS",
    "MOVE_ACTIONS",
    "TOTAL_ACTIONS",
]
